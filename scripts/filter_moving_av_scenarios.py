#!/usr/bin/env python3
"""Filter scenario cache to AV scenarios that keep moving.

This script reads a ``datasets.AV2_scenarios`` cache, scores each scenario by
the AV ego trajectory, and writes a smaller scenario cache that can be passed
directly to ``main.py train`` via ``--scenarios``.

Default definition of "keeps moving":
  * evaluate the PPO future segment, timesteps 50..109;
  * every consecutive AV step has speed >= 0.5 m/s;
  * total AV path length in that segment is >= 10 m.

Example:
    python scripts/filter_moving_av_scenarios.py \
        --scenarios data/av2_mf_tiny/scenarios_rl_s100.pt \
        --out data/av2_mf_tiny/scenarios_rl_moving_av_100.pt \
        --manifest data/av2_mf_tiny/scenarios_rl_moving_av_100.csv \
        --limit 100 --allow_less
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_payload(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _segment_bounds(
    segment: str,
    *,
    history_steps: int,
    total_steps: int,
    start: int | None,
    end: int | None,
) -> Tuple[int, int]:
    if segment == "all":
        lo, hi = 0, total_steps
    elif segment == "history":
        lo, hi = 0, history_steps
    elif segment == "future":
        lo, hi = history_steps - 1, total_steps
    elif segment == "custom":
        if start is None or end is None:
            raise ValueError("--segment custom requires --start and --end")
        lo, hi = int(start), int(end)
    else:
        raise ValueError(f"unknown segment: {segment}")

    lo = max(0, int(lo))
    hi = min(total_steps, int(hi))
    if hi - lo < 2:
        raise ValueError(f"segment must contain at least 2 points, got [{lo}, {hi})")
    return lo, hi


def _as_numpy_xy(track: Dict) -> np.ndarray:
    xy = track["positions_world"]
    if hasattr(xy, "detach"):
        xy = xy.detach().cpu().numpy()
    return np.asarray(xy, dtype=np.float64)


def _score_scenario(
    scenario: Dict,
    *,
    lo: int,
    hi: int,
    dt: float,
    min_speed: float,
    min_step_m: float,
    moving_fraction: float,
    min_path_m: float,
) -> Dict:
    av = scenario.get("av")
    if av is None:
        return {"keep": False, "reason": "missing_av"}

    xy = _as_numpy_xy(av)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < hi:
        return {"keep": False, "reason": "bad_av_shape"}

    seg = xy[lo:hi]
    delta = np.diff(seg, axis=0)
    step_m = np.linalg.norm(delta, axis=1)
    speed = step_m / max(float(dt), 1e-6)

    moving = (speed >= float(min_speed)) & (step_m >= float(min_step_m))
    frac = float(np.mean(moving)) if moving.size else 0.0
    path_m = float(np.sum(step_m))
    min_speed_seen = float(np.min(speed)) if speed.size else 0.0
    mean_speed = float(np.mean(speed)) if speed.size else 0.0
    p05_speed = float(np.percentile(speed, 5)) if speed.size else 0.0

    keep = (
        frac >= float(moving_fraction)
        and path_m >= float(min_path_m)
        and scenario.get("non_av") is not None
        and len(scenario.get("non_av", [])) > 0
    )
    return {
        "keep": bool(keep),
        "reason": "ok" if keep else "not_continuously_moving",
        "path_m": path_m,
        "min_speed": min_speed_seen,
        "p05_speed": p05_speed,
        "mean_speed": mean_speed,
        "moving_fraction": frac,
        "num_non_av": int(len(scenario.get("non_av", []))),
    }


def _write_manifest(path: str, rows: Iterable[Dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "rank",
        "scenario_id",
        "path_m",
        "min_speed",
        "p05_speed",
        "mean_speed",
        "moving_fraction",
        "num_non_av",
        "reason",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select scenarios whose AV ego keeps moving.")
    p.add_argument("--scenarios", required=True, help="Input scenarios .pt cache.")
    p.add_argument("--out", required=True, help="Output filtered scenarios .pt cache.")
    p.add_argument("--manifest", default=None, help="Optional CSV listing selected scenarios.")
    p.add_argument("--limit", type=int, default=100, help="Maximum number of scenarios to keep.")
    p.add_argument(
        "--segment",
        choices=("future", "history", "all", "custom"),
        default="future",
        help="Trajectory segment used to decide whether the AV keeps moving.",
    )
    p.add_argument("--start", type=int, default=None, help="Inclusive timestep for --segment custom.")
    p.add_argument("--end", type=int, default=None, help="Exclusive timestep for --segment custom.")
    p.add_argument("--min_speed", type=float, default=0.5, help="Minimum AV speed in m/s.")
    p.add_argument(
        "--min_step_m",
        type=float,
        default=0.0,
        help="Minimum per-timestep displacement in meters. Usually min_speed is enough.",
    )
    p.add_argument(
        "--moving_fraction",
        type=float,
        default=1.0,
        help="Fraction of steps that must satisfy movement thresholds; 1.0 means every step.",
    )
    p.add_argument("--min_path_m", type=float, default=10.0, help="Minimum total AV path length.")
    p.add_argument(
        "--allow_less",
        action="store_true",
        help="Write fewer than --limit scenarios if not enough pass the filters.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    payload = _load_payload(args.scenarios)
    scenarios: List[Dict] = list(payload["scenarios"])
    history_steps = int(payload.get("history_steps", 50))
    total_steps = int(payload.get("total_steps", 110))
    dt = float(payload.get("dt", 0.1))
    lo, hi = _segment_bounds(
        args.segment,
        history_steps=history_steps,
        total_steps=total_steps,
        start=args.start,
        end=args.end,
    )

    scored: List[Dict] = []
    for scenario in scenarios:
        stats = _score_scenario(
            scenario,
            lo=lo,
            hi=hi,
            dt=dt,
            min_speed=args.min_speed,
            min_step_m=args.min_step_m,
            moving_fraction=args.moving_fraction,
            min_path_m=args.min_path_m,
        )
        stats["scenario_id"] = str(scenario.get("scenario_id", ""))
        stats["scenario"] = scenario
        scored.append(stats)

    kept = [r for r in scored if r["keep"]]
    kept.sort(
        key=lambda r: (
            float(r.get("moving_fraction", 0.0)),
            float(r.get("min_speed", 0.0)),
            float(r.get("path_m", 0.0)),
            str(r.get("scenario_id", "")),
        ),
        reverse=True,
    )

    limit = int(args.limit)
    selected = kept[:limit]
    if len(selected) < limit and not args.allow_less:
        raise SystemExit(
            f"Only found {len(selected)} scenarios passing the filters, fewer than --limit {limit}. "
            "Relax thresholds or pass --allow_less."
        )

    out_payload = dict(payload)
    out_payload["scenarios"] = [r["scenario"] for r in selected]
    out_payload["moving_av_filter"] = {
        "source": str(args.scenarios),
        "segment": str(args.segment),
        "start": int(lo),
        "end": int(hi),
        "min_speed": float(args.min_speed),
        "min_step_m": float(args.min_step_m),
        "moving_fraction": float(args.moving_fraction),
        "min_path_m": float(args.min_path_m),
        "limit": int(limit),
        "selected": int(len(selected)),
        "candidates": int(len(kept)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, str(out_path))

    manifest_rows = []
    for rank, row in enumerate(selected, start=1):
        manifest_rows.append({**row, "rank": rank})
    if args.manifest:
        _write_manifest(args.manifest, manifest_rows)

    print(
        f"[filter_moving_av_scenarios] selected {len(selected)}/{len(scenarios)} scenarios "
        f"(candidates={len(kept)}, segment=[{lo}, {hi}), min_speed={args.min_speed} m/s)"
    )
    print(f"[filter_moving_av_scenarios] wrote {out_path}")
    if args.manifest:
        print(f"[filter_moving_av_scenarios] wrote {args.manifest}")


if __name__ == "__main__":
    main()
