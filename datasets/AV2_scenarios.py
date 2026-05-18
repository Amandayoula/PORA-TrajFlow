"""
Scenario-grouped AV2 cache builder for RL / PPO.

Unlike ``datasets/AV2_parallel.py`` (which drops the ``track_id=="AV"`` row and
flattens every remaining track into a single ``(N, 110, 2)`` tensor), this
builder:

- keeps the AV ego track;
- groups tracks by scenario so PPO can sample a (AV, list-of-non-AV) tuple;
- stores both world-frame and agent-centric positions per track (world for
  PORA risk math, agent-centric for TrajFlow inference);
- reuses the same quality filters (MIN_OBS / MIN_FUT / MAX_GAP / p95 speed and
  accel) so heatmap-ready tracks are consistent with the flat cache.

Output schema::

    {
      "spatial_boundaries": (2, 2) float tensor,
      "feature_boundaries": (5, 2) float tensor | None,
      "normalize_data": bool,
      "agent_centric": True,
      "scenarios": [
        {
          "scenario_id": str,
          "av": TrackDict | None,
          "non_av": [TrackDict, ...],
        },
        ...
      ],
    }

    TrackDict = {
      "track_uid": str,
      "is_av": bool,
      "positions_world": (110, 2) float tensor,   # raw world meters
      "positions_agent": (110, 2) float tensor,   # agent-centric meters
      "features": (50, 6) float tensor,           # [heading_rel, vx, vy, ax, ay, t]
      "pose": (3,) float tensor,                  # [tx, ty, heading0] world frame
    }
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from datasets.AV2_parallel import (
    AGENT_CENTRIC,
    AV2_FILTER_MAX_GAP,
    AV2_FILTER_MIN_FUT,
    AV2_FILTER_MIN_OBS,
    AV2_FILTER_P95_ACCEL,
    AV2_FILTER_P95_SPEED,
    DT,
    EXCLUDED_TRACK_ID_AV,
    FUTURE_STEPS,
    HISTORY_STEPS,
    TOTAL_STEPS,
    _wrap_angle,
    normalize,
)


@dataclass(frozen=True)
class _ScenarioParseCfg:
    object_types: Tuple[str, ...]
    object_categories: Optional[Tuple[int, ...]]
    keep_av: bool = True


def _build_track_record(
    track_id: str,
    track_df: "pd.DataFrame",
) -> Optional[Dict[str, np.ndarray]]:
    """Apply AV2_parallel quality filters and return a per-track record.

    Returns ``None`` if the track does not satisfy the filters.
    """
    import pandas as pd

    track_df = track_df.sort_values("timestep").reset_index(drop=True)

    obs_count = int(((track_df["observed"] == True) & (track_df["timestep"] < HISTORY_STEPS)).sum())
    fut_count = int(((track_df["observed"] == False) & (track_df["timestep"] >= HISTORY_STEPS)).sum())
    if obs_count < AV2_FILTER_MIN_OBS or fut_count < AV2_FILTER_MIN_FUT:
        return None

    ts = np.asarray(track_df["timestep"].values, dtype=np.int64)
    ts = np.unique(ts[(ts >= 0) & (ts < TOTAL_STEPS)])
    if ts.size == 0:
        return None
    if int(np.max(np.diff(ts))) > AV2_FILTER_MAX_GAP:
        return None

    try:
        vx_raw = np.asarray(
            track_df.loc[track_df["observed"] == True, "velocity_x"].values, dtype=np.float32
        )
        vy_raw = np.asarray(
            track_df.loc[track_df["observed"] == True, "velocity_y"].values, dtype=np.float32
        )
        if vx_raw.size >= 5 and vy_raw.size >= 5:
            speed_raw = np.sqrt(vx_raw ** 2 + vy_raw ** 2)
            if float(np.percentile(speed_raw, 95)) > AV2_FILTER_P95_SPEED:
                return None
    except Exception:
        pass

    full_index = pd.Index(np.arange(TOTAL_STEPS), name="timestep")
    sdf = (
        track_df.set_index("timestep")[
            ["position_x", "position_y", "heading", "velocity_x", "velocity_y", "observed"]
        ]
        .sort_index()
    )
    sdf = sdf[~sdf.index.duplicated(keep="last")]
    sdf = sdf.reindex(full_index)

    sdf["observed"] = sdf["observed"].fillna(False).astype(bool)
    sdf.loc[: HISTORY_STEPS - 1, "observed"] = True
    sdf.loc[HISTORY_STEPS:, "observed"] = False

    num_cols = ["position_x", "position_y", "heading", "velocity_x", "velocity_y"]
    sdf[num_cols] = sdf[num_cols].astype("float32").interpolate(method="linear", limit_direction="both")
    sdf[num_cols] = sdf[num_cols].ffill().bfill()

    world_xy = sdf[["position_x", "position_y"]].to_numpy(dtype=np.float32)  # (110, 2)

    obs = sdf.iloc[:HISTORY_STEPS]
    heading = obs["heading"].to_numpy(dtype=np.float32)
    vx = obs["velocity_x"].to_numpy(dtype=np.float32)
    vy = obs["velocity_y"].to_numpy(dtype=np.float32)

    t0_xy = world_xy[HISTORY_STEPS - 1].astype(np.float32).copy()
    h0 = float(heading[-1])

    if AGENT_CENTRIC:
        xy_translated = world_xy - t0_xy[None, :]
        c, s = np.cos(h0), np.sin(h0)
        R_inv = np.array([[c, s], [-s, c]], dtype=np.float32)
        agent_xy = (xy_translated @ R_inv.T).astype(np.float32)
        v = np.stack([vx, vy], axis=-1)
        v_rot = (v @ R_inv.T).astype(np.float32)
        vx_a = v_rot[:, 0]
        vy_a = v_rot[:, 1]
        heading_a = _wrap_angle(heading - h0).astype(np.float32)
    else:
        agent_xy = world_xy.copy()
        vx_a = vx
        vy_a = vy
        heading_a = heading

    ax = np.gradient(vx_a, DT).astype(np.float32)
    ay = np.gradient(vy_a, DT).astype(np.float32)

    a_norm = np.sqrt(ax ** 2 + ay ** 2)
    if float(np.percentile(a_norm, 95)) > AV2_FILTER_P95_ACCEL:
        return None

    feat5 = np.stack([heading_a, vx_a, vy_a, ax, ay], axis=-1).astype(np.float32)  # (50, 5)
    t_channel = np.linspace(0.0, 2.0, HISTORY_STEPS, dtype=np.float32).reshape(HISTORY_STEPS, 1)
    feat6 = np.concatenate([feat5, t_channel], axis=-1).astype(np.float32)  # (50, 6)

    return {
        "track_uid": str(track_id),
        "is_av": str(track_id) == EXCLUDED_TRACK_ID_AV,
        "positions_world": world_xy,
        "positions_agent": agent_xy,
        "features": feat6,
        "pose": np.array([t0_xy[0], t0_xy[1], h0], dtype=np.float32),
    }


def _parse_one_scenario(parquet_path: str, cfg: _ScenarioParseCfg) -> Dict:
    import pandas as pd

    scenario_dir = os.path.dirname(parquet_path)
    scenario_id = os.path.basename(scenario_dir) or os.path.basename(parquet_path)

    df = pd.read_parquet(parquet_path)
    df = df[df["object_type"].isin(cfg.object_types)].copy()
    if cfg.object_categories is not None:
        if "object_category" not in df.columns:
            raise KeyError(
                "object_categories filtering requested, but parquet is missing 'object_category' column."
            )
        df = df[df["object_category"].isin(cfg.object_categories)].copy()

    av: Optional[Dict] = None
    non_av: List[Dict] = []
    for track_id, track_df in df.groupby("track_id"):
        rec = _build_track_record(track_id, track_df)
        if rec is None:
            continue
        if rec["is_av"]:
            if cfg.keep_av:
                av = rec
        else:
            non_av.append(rec)

    return {"scenario_id": scenario_id, "av": av, "non_av": non_av}


def _record_to_tensors(rec: Dict) -> Dict[str, torch.Tensor]:
    return {
        "track_uid": rec["track_uid"],
        "is_av": bool(rec["is_av"]),
        "positions_world": torch.from_numpy(rec["positions_world"]).float(),
        "positions_agent": torch.from_numpy(rec["positions_agent"]).float(),
        "features": torch.from_numpy(rec["features"]).float(),
        "pose": torch.from_numpy(rec["pose"]).float(),
    }


def _load_reference_boundaries(ref_cache_path: Optional[str]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[bool]]:
    """Read spatial/feature boundaries from an existing flat cache so the new scenario cache is directly comparable with the model's training domain."""
    if ref_cache_path is None:
        return None, None, None
    try:
        data = torch.load(ref_cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(ref_cache_path, map_location="cpu")
    sb = data.get("spatial_boundaries", None)
    fb = data.get("feature_boundaries", None)
    nd = data.get("normalize_data", None)
    if sb is not None and not torch.is_tensor(sb):
        sb = torch.as_tensor(sb)
    if fb is not None and not torch.is_tensor(fb):
        fb = torch.as_tensor(fb)
    if sb is not None:
        sb = sb.float()
    if fb is not None:
        fb = fb.float()
    return sb, fb, (None if nd is None else bool(nd))


class AV2Scenarios:
    """Scenario-grouped cache builder.

    Mirrors the interface of ``datasets/AV2_parallel.AV2`` but emits a single
    ``scenarios.pt`` file instead of split DataLoaders. The consumer is the PPO
    env in ``rl/scenario_env.py``.
    """

    DEFAULT_OBJECT_TYPES = ("vehicle",)
    DEFAULT_OBJECT_CATEGORIES: Tuple[int, ...] = (0, 1, 2, 3)

    def __init__(
        self,
        root: str,
        object_types: Optional[Iterable[str]] = None,
        object_categories: Optional[Iterable[int]] = DEFAULT_OBJECT_CATEGORIES,
        max_scenarios: Optional[int] = None,
        num_workers: Optional[int] = None,
        ref_cache_path: Optional[str] = None,
        keep_av: bool = True,
    ):
        self.root = root
        self.object_types = tuple(object_types) if object_types is not None else tuple(self.DEFAULT_OBJECT_TYPES)
        self.object_categories = (
            None if object_categories is None else tuple(int(x) for x in object_categories)
        )
        self.max_scenarios = max_scenarios
        self.num_workers = num_workers
        self.ref_cache_path = ref_cache_path
        self.keep_av = bool(keep_av)

    def _iter_parquet_paths(self) -> List[str]:
        train_dir = os.path.join(self.root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Expected train split at: {train_dir}")

        scenario_ids = sorted(os.listdir(train_dir))
        if self.max_scenarios is not None:
            scenario_ids = scenario_ids[: int(self.max_scenarios)]

        parquet_paths: List[str] = []
        for scenario_id in scenario_ids:
            scenario_dir = os.path.join(train_dir, scenario_id)
            if not os.path.isdir(scenario_dir):
                continue
            for fname in os.listdir(scenario_dir):
                if fname.endswith(".parquet"):
                    parquet_paths.append(os.path.join(scenario_dir, fname))
        return parquet_paths

    def build(self, out_path: str) -> str:
        parquet_paths = self._iter_parquet_paths()
        if not parquet_paths:
            raise ValueError("No parquet files found under train/")

        cfg = _ScenarioParseCfg(
            object_types=self.object_types,
            object_categories=self.object_categories,
            keep_av=self.keep_av,
        )

        num_workers = self.num_workers
        if num_workers is None:
            num_workers = max(1, min(8, (os.cpu_count() or 1)))

        print(
            f"[AV2Scenarios] parsing {len(parquet_paths)} parquet files with {num_workers} workers..."
        )
        scenarios: List[Dict] = []

        if num_workers == 1:
            for pp in parquet_paths:
                scenarios.append(_parse_one_scenario(pp, cfg))
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(_parse_one_scenario, p, cfg) for p in parquet_paths]
                for fut in as_completed(futures):
                    scenarios.append(fut.result())

        # Drop scenarios with no qualifying non-AV tracks (nothing to drive against).
        scenarios = [s for s in scenarios if s["non_av"]]
        # Sort for determinism.
        scenarios.sort(key=lambda s: s["scenario_id"])

        n_av = sum(1 for s in scenarios if s["av"] is not None)
        n_non_av = sum(len(s["non_av"]) for s in scenarios)
        print(
            f"[AV2Scenarios] kept {len(scenarios)} scenarios "
            f"(with-AV={n_av}, total non-AV tracks={n_non_av})"
        )

        sb_ref, fb_ref, nd_ref = _load_reference_boundaries(self.ref_cache_path)
        if sb_ref is None:
            all_world = np.concatenate(
                [t["positions_world"] for s in scenarios for t in (s["non_av"] + ([s["av"]] if s["av"] else []))],
                axis=0,
            )
            mins = all_world.min(axis=0)
            maxs = all_world.max(axis=0)
            sb_ref = torch.tensor(np.stack([mins, maxs], axis=1), dtype=torch.float32)
            nd_ref = False

        payload = {
            "spatial_boundaries": sb_ref,
            "feature_boundaries": fb_ref,
            "normalize_data": bool(nd_ref) if nd_ref is not None else False,
            "agent_centric": bool(AGENT_CENTRIC),
            "history_steps": int(HISTORY_STEPS),
            "future_steps": int(FUTURE_STEPS),
            "total_steps": int(TOTAL_STEPS),
            "dt": float(DT),
            "object_types": list(self.object_types),
            "object_categories": None if self.object_categories is None else list(self.object_categories),
            "scenarios": [
                {
                    "scenario_id": s["scenario_id"],
                    "av": None if s["av"] is None else _record_to_tensors(s["av"]),
                    "non_av": [_record_to_tensors(t) for t in s["non_av"]],
                }
                for s in scenarios
            ],
        }

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        print(f"[AV2Scenarios] saving to {out_path}")
        torch.save(payload, out_path)
        return out_path


@dataclass
class ScenarioTrack:
    track_uid: str
    is_av: bool
    positions_world: torch.Tensor  # (110, 2)
    positions_agent: torch.Tensor  # (110, 2)
    features: torch.Tensor         # (50, 6)
    pose: torch.Tensor             # (3,)


@dataclass
class Scenario:
    scenario_id: str
    av: Optional[ScenarioTrack]
    non_av: List[ScenarioTrack]


@dataclass
class ScenariosCache:
    spatial_boundaries: torch.Tensor
    feature_boundaries: Optional[torch.Tensor]
    normalize_data: bool
    agent_centric: bool
    history_steps: int
    future_steps: int
    total_steps: int
    dt: float
    scenarios: List[Scenario]


def _track_from_dict(d: Dict) -> ScenarioTrack:
    return ScenarioTrack(
        track_uid=str(d["track_uid"]),
        is_av=bool(d["is_av"]),
        positions_world=d["positions_world"].float(),
        positions_agent=d["positions_agent"].float(),
        features=d["features"].float(),
        pose=d["pose"].float(),
    )


def load_scenarios(path: str) -> ScenariosCache:
    """Load a scenarios .pt file into a typed object."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    scenarios: List[Scenario] = []
    for s in payload["scenarios"]:
        av = None if s.get("av") is None else _track_from_dict(s["av"])
        non_av = [_track_from_dict(t) for t in s["non_av"]]
        scenarios.append(Scenario(scenario_id=str(s["scenario_id"]), av=av, non_av=non_av))

    return ScenariosCache(
        spatial_boundaries=payload["spatial_boundaries"].float(),
        feature_boundaries=(
            None if payload.get("feature_boundaries") is None else payload["feature_boundaries"].float()
        ),
        normalize_data=bool(payload.get("normalize_data", False)),
        agent_centric=bool(payload.get("agent_centric", False)),
        history_steps=int(payload.get("history_steps", HISTORY_STEPS)),
        future_steps=int(payload.get("future_steps", FUTURE_STEPS)),
        total_steps=int(payload.get("total_steps", TOTAL_STEPS)),
        dt=float(payload.get("dt", DT)),
        scenarios=scenarios,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build scenario-grouped AV2 cache for PPO")
    p.add_argument("--root", type=str, default="data/av2_mf_tiny")
    p.add_argument(
        "--out",
        type=str,
        default="data/av2_mf_tiny/scenarios_rl.pt",
        help="Output .pt path.",
    )
    p.add_argument(
        "--ref_cache",
        type=str,
        default="data/av2_mf_tiny/with_fragment_no_normalization_change_boundary_all.pt",
        help="Existing flat cache to copy spatial/feature boundaries from (keeps "
        "domain consistent with the trained TrajFlow).",
    )
    p.add_argument("--max_scenarios", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    builder = AV2Scenarios(
        root=args.root,
        max_scenarios=args.max_scenarios,
        num_workers=args.num_workers,
        ref_cache_path=args.ref_cache if (args.ref_cache and os.path.isfile(args.ref_cache)) else None,
    )
    builder.build(args.out)


if __name__ == "__main__":
    main()
