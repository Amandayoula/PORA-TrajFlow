#!/usr/bin/env python3
"""Precompute per-non-AV TrajFlow heatmaps for a scenario-grouped AV2 cache.

For every non-AV track in every scenario we evaluate the model density over a
dense spatial grid in the non-AV's agent-centric frame, at each of
``FUTURE_STEPS`` future timesteps. The resulting ``(T, H, W)`` stack is saved
under ``{heatmap_root}/{scenario_id}/{track_uid}.pt`` so PPO can load it on
demand in ``rl/scenario_env.py``.

Grid is deterministic per-track and stored with the heatmap so env can map
world-frame ego positions back to grid indices via the track's pose.

Schema (per file)::

    {
      "scenario_id": str,
      "track_uid": str,
      "is_av": bool,                       # always False for non-AV files
      "pose": (3,) float,                  # non-AV world pose [tx, ty, h0]
      "grid_bounds_agent": (2, 2) float,   # [[x_min, x_max], [y_min, y_max]] in the track's agent frame
      "steps": int,
      "future_steps": int,
      "heatmap": (T, H, W) float16,        # per-frame max-normalized density in [0, 1]
      "heatmap_raw_max": (T,) float,       # per-frame max before normalization (for debugging)
    }

Usage::

    python scripts/precompute_scenario_heatmaps.py \\
        --scenarios data/av2_mf_tiny/scenarios_rl.pt \\
        --model trajflow_GRU_DNF_marginal_AV2_best_all.pt \\
        --flow DNF --encoder GRU \\
        --out data/heatmaps_rl --steps 64 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.AV2_parallel import FUTURE_STEPS, HISTORY_STEPS
from datasets.AV2_scenarios import Scenario, ScenarioTrack, load_scenarios
from model.TrajFlow import CausalEnocder, Flow, TrajFlow


def _build_grid(
    steps: int,
    bounds_agent: np.ndarray,
    normalize_data: bool,
    spatial_boundaries: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Build a ``(steps*steps, 2)`` grid of points in the *model input frame*.

    When the model was trained on raw (un-normalized) agent-frame coords,
    ``normalize_data=False`` and we return points directly in agent meters.

    When the model was trained on normalized coords, ``spatial_boundaries`` is
    used to map agent meters -> [0, 1]^2 (same as ``observation_site.normalize``).
    """
    xs = np.linspace(bounds_agent[0, 0], bounds_agent[0, 1], steps, dtype=np.float32)
    ys = np.linspace(bounds_agent[1, 0], bounds_agent[1, 1], steps, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.flatten(), yy.flatten()], axis=-1).astype(np.float32)  # (S*S, 2)

    if normalize_data and spatial_boundaries is not None:
        b = spatial_boundaries.cpu().numpy().astype(np.float32)
        pts = (pts - b[:, 0]) / (b[:, 1] - b[:, 0])

    return torch.from_numpy(pts).to(device)


def _compute_pzt1(
    model: TrajFlow,
    input_hist: torch.Tensor,  # (1, HISTORY_STEPS, 2)
    features: torch.Tensor,    # (1, HISTORY_STEPS, 6)
    grid: torch.Tensor,        # (S*S, 2)
    future_steps: int,
    batch_size: int = 500,
) -> torch.Tensor:
    """Evaluate the TrajFlow density grid used to build one heatmap stack.

    Returns a ``(S*S, future_steps)`` tensor of densities.
    """
    with torch.no_grad():
        embedding = model._embedding(input_hist, features)
        outs: List[torch.Tensor] = []
        for grid_batch in grid.split(batch_size, dim=0):
            emb = embedding.repeat(grid_batch.shape[0], 1)
            grid_batch = grid_batch.unsqueeze(1).expand(-1, future_steps, -1)
            z_t0, delta_logpz = model.flow(grid_batch, emb)
            _, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            outs.append(logpz_t1.exp())
        return torch.cat(outs, dim=0)


def _default_grid_bounds_agent(track: ScenarioTrack, x_pad: float, y_pad: float) -> np.ndarray:
    """Conservative square-ish bounding box around the full ground-truth track in agent frame.

    Used when --grid_mode track. Keeps the grid small enough for a CNN input
    while covering the plausible region where the AV could realistically be
    near this non-AV.
    """
    xy = track.positions_agent.cpu().numpy()
    x_min = float(xy[:, 0].min()) - x_pad
    x_max = float(xy[:, 0].max()) + x_pad
    y_min = float(xy[:, 1].min()) - y_pad
    y_max = float(xy[:, 1].max()) + y_pad
    return np.array([[x_min, x_max], [y_min, y_max]], dtype=np.float32)


def _fixed_grid_bounds(x_range: Tuple[float, float], y_range: Tuple[float, float]) -> np.ndarray:
    return np.array([[x_range[0], x_range[1]], [y_range[0], y_range[1]]], dtype=np.float32)


def _load_model(
    checkpoint: str,
    encoder: str,
    flow: str,
    marginal: bool,
    norm_rotate: bool,
    device: torch.device,
) -> TrajFlow:
    model = TrajFlow(
        seq_len=HISTORY_STEPS,
        input_dim=2,
        feature_dim=6,
        embedding_dim=128,
        hidden_dim=512,
        causal_encoder=CausalEnocder[encoder],
        flow=Flow[flow],
        marginal=marginal,
        norm_rotation=norm_rotate,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state.items() if k in model_state and getattr(v, "shape", None) == model_state[k].shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"[precompute_heatmaps] loaded {len(compatible)} keys "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    model.eval()
    return model


def _save_heatmap_file(
    out_path: Path,
    *,
    scenario_id: str,
    track: ScenarioTrack,
    bounds_agent: np.ndarray,
    steps: int,
    future_steps: int,
    heatmap_norm: np.ndarray,
    raw_max: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": scenario_id,
        "track_uid": track.track_uid,
        "is_av": bool(track.is_av),
        "pose": track.pose.clone(),
        "grid_bounds_agent": torch.from_numpy(bounds_agent).float(),
        "steps": int(steps),
        "future_steps": int(future_steps),
        "heatmap": torch.from_numpy(heatmap_norm.astype(np.float16)),
        "heatmap_raw_max": torch.from_numpy(raw_max.astype(np.float32)),
    }
    torch.save(payload, str(out_path))


def precompute_for_scenario(
    scenario: Scenario,
    model: TrajFlow,
    *,
    out_root: Path,
    steps: int,
    future_steps: int,
    grid_mode: str,
    x_pad: float,
    y_pad: float,
    fixed_x: Tuple[float, float],
    fixed_y: Tuple[float, float],
    normalize_data: bool,
    spatial_boundaries: Optional[torch.Tensor],
    device: torch.device,
    overwrite: bool,
) -> int:
    written = 0
    out_dir = out_root / scenario.scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for track in scenario.non_av:
        out_path = out_dir / f"{track.track_uid}.pt"
        if out_path.exists() and not overwrite:
            continue

        if grid_mode == "track":
            bounds_agent = _default_grid_bounds_agent(track, x_pad=x_pad, y_pad=y_pad)
        elif grid_mode == "fixed":
            bounds_agent = _fixed_grid_bounds(fixed_x, fixed_y)
        else:
            raise ValueError(f"Unknown grid_mode={grid_mode!r}")

        inp = track.positions_agent[:HISTORY_STEPS].to(device).unsqueeze(0)  # (1, 50, 2)
        feat = track.features.to(device).unsqueeze(0)                        # (1, 50, 6)

        grid = _build_grid(
            steps=steps,
            bounds_agent=bounds_agent,
            normalize_data=normalize_data,
            spatial_boundaries=spatial_boundaries,
            device=device,
        )

        pz = _compute_pzt1(model, inp, feat, grid, future_steps=future_steps)
        pz = pz[:, :future_steps].detach().cpu().numpy()  # (S*S, T)
        hm = pz.reshape(steps, steps, future_steps).transpose(2, 0, 1)  # (T, S, S)

        per_frame_max = hm.reshape(future_steps, -1).max(axis=1)
        denom = np.where(per_frame_max > 0, per_frame_max, 1.0)
        hm_norm = hm / denom[:, None, None]
        hm_norm = np.clip(hm_norm, 0.0, 1.0).astype(np.float32)

        _save_heatmap_file(
            out_path,
            scenario_id=scenario.scenario_id,
            track=track,
            bounds_agent=bounds_agent,
            steps=steps,
            future_steps=future_steps,
            heatmap_norm=hm_norm,
            raw_max=per_frame_max,
        )
        written += 1

    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute per-non-AV TrajFlow heatmaps.")
    p.add_argument("--scenarios", type=str, required=True, help="Path to scenarios_rl.pt")
    p.add_argument("--model", type=str, required=True, help="TrajFlow checkpoint .pt")
    p.add_argument("--encoder", type=str, default="GRU", choices=("GRU", "CDE"))
    p.add_argument("--flow", type=str, default="DNF", choices=("DNF", "CNF"))
    p.add_argument("--marginal", action="store_true", default=True)
    p.add_argument("--no_marginal", dest="marginal", action="store_false")
    p.add_argument("--norm_rotate", action="store_true", default=False)
    p.add_argument("--out", type=str, default="data/heatmaps_rl", help="Heatmap root dir")
    p.add_argument("--steps", type=int, default=64, help="Grid resolution per axis (H=W=steps)")
    p.add_argument(
        "--future_steps",
        type=int,
        default=None,
        help="Number of future steps (default: from scenarios cache)",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--grid_mode",
        type=str,
        default="fixed",
        choices=("fixed", "track"),
        help="fixed: use same (x_range, y_range) for every track (CNN-friendly). "
        "track: per-track bounding box around ground-truth.",
    )
    p.add_argument("--x_pad", type=float, default=20.0)
    p.add_argument("--y_pad", type=float, default=20.0)
    p.add_argument("--fixed_x_min", type=float, default=-40.0)
    p.add_argument("--fixed_x_max", type=float, default=80.0)
    p.add_argument("--fixed_y_min", type=float, default=-40.0)
    p.add_argument("--fixed_y_max", type=float, default=40.0)
    p.add_argument(
        "--only_scenarios",
        type=int,
        default=None,
        help="If set, only process the first N scenarios (quick sanity runs).",
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    cache = load_scenarios(args.scenarios)
    future_steps = int(args.future_steps or cache.future_steps or FUTURE_STEPS)
    print(
        f"[precompute_heatmaps] loaded {len(cache.scenarios)} scenarios "
        f"(future_steps={future_steps}, normalize_data={cache.normalize_data})"
    )

    model = _load_model(
        checkpoint=args.model,
        encoder=args.encoder,
        flow=args.flow,
        marginal=bool(args.marginal),
        norm_rotate=bool(args.norm_rotate),
        device=device,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    scenarios = cache.scenarios
    if args.only_scenarios is not None:
        scenarios = scenarios[: int(args.only_scenarios)]

    total_tracks = 0
    total_written = 0
    t0 = time.time()
    for i, scen in enumerate(scenarios):
        n = precompute_for_scenario(
            scen,
            model=model,
            out_root=out_root,
            steps=int(args.steps),
            future_steps=future_steps,
            grid_mode=args.grid_mode,
            x_pad=float(args.x_pad),
            y_pad=float(args.y_pad),
            fixed_x=(float(args.fixed_x_min), float(args.fixed_x_max)),
            fixed_y=(float(args.fixed_y_min), float(args.fixed_y_max)),
            normalize_data=cache.normalize_data,
            spatial_boundaries=cache.spatial_boundaries,
            device=device,
            overwrite=bool(args.overwrite),
        )
        total_tracks += len(scen.non_av)
        total_written += n
        if (i + 1) % max(1, len(scenarios) // 20) == 0 or i + 1 == len(scenarios):
            elapsed = time.time() - t0
            print(
                f"[precompute_heatmaps] {i+1}/{len(scenarios)} scenarios "
                f"(wrote {total_written}/{total_tracks} tracks, elapsed {elapsed:.1f}s)"
            )

    print(
        f"[precompute_heatmaps] done. Wrote {total_written} heatmap files "
        f"for {total_tracks} non-AV tracks under {out_root}"
    )


if __name__ == "__main__":
    main()
