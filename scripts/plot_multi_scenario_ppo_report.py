#!/usr/bin/env python3
"""Plot multi-scenario PPO training/evaluation metrics.

Input is the JSON saved by ``scripts/train_ppo_multi_scenario_accel.py``.

Example:
    python scripts/plot_multi_scenario_ppo_report.py \
        --history runs/ppo_multi_scenario_accel.json \
        --out runs/ppo_multi_scenario_accel_report.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trajflow_ppo_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trajflow_ppo_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", required=True, help="JSON from train_ppo_multi_scenario_accel.py")
    p.add_argument("--out", required=True, help="Output figure path, e.g. .png")
    p.add_argument("--smooth", type=int, default=5, help="Rolling mean window for training curves.")
    p.add_argument("--title", default=None)
    return p.parse_args()


def _load(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    if "training_history" not in payload:
        raise SystemExit(f"{path} is missing 'training_history'")
    return payload


def _series(rows: Sequence[dict], key: str, fill: float = np.nan) -> np.ndarray:
    return np.asarray([float(r.get(key, fill)) for r in rows], dtype=float)


def _eval_conflict_rate(rows: Sequence[dict]) -> np.ndarray:
    vals = []
    for row in rows:
        if "average_episode_conflict_rate" in row:
            vals.append(float(row["average_episode_conflict_rate"]))
            continue
        conflicts = float(row.get("average_episode_conflicts", np.nan))
        length = float(row.get("average_episode_length", np.nan))
        vals.append(conflicts / length if np.isfinite(conflicts) and length > 0.0 else np.nan)
    return np.asarray(vals, dtype=float)


def _updates(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([float(r.get("update", i + 1)) for i, r in enumerate(rows)], dtype=float)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < 3:
        return y
    valid = np.isfinite(y)
    if not valid.any():
        return y
    w = min(int(window), int(y.size))
    kernel = np.ones(w, dtype=float)
    y0 = np.where(valid, y, 0.0)
    num = np.convolve(y0, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    return np.divide(num, den, out=np.full_like(y, np.nan), where=den > 0)


def _plot_train(ax, rows: Sequence[dict], key: str, title: str, smooth: int, *, zero: bool = False) -> None:
    x = _updates(rows)
    y = _series(rows, key)
    ax.plot(x, y, color="#9aa4b2", linewidth=1.0, alpha=0.45)
    ax.plot(x, _smooth(y, smooth), color="#1f77b4", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("update")
    ax.grid(True, alpha=0.3)
    if zero:
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)


def _plot_eval_conflict_rate(ax, rows: Sequence[dict]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "no evaluation_history", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("evaluation: average episode conflict rate")
        return
    x = _updates(rows)
    y = _eval_conflict_rate(rows)
    ax.plot(x, y, marker="o", linewidth=1.8, color="#d62728")
    ax.set_title("evaluation: average episode conflict rate")
    ax.set_xlabel("update")
    ax.set_ylabel("conflict timesteps / episode length")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)


def _plot_eval_pora_risk(ax, rows: Sequence[dict]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "no evaluation_history", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("evaluation: eval PORA risk")
        return
    x = _updates(rows)
    avg = _series(rows, "average_PORA_risk")
    ax.plot(x, avg, marker="o", linewidth=1.8, color="#2ca02c", label="average_PORA_risk")
    if any("maximum_PORA_risk" in r for r in rows):
        max_risk = _series(rows, "maximum_PORA_risk")
        ax.plot(x, max_risk, marker=".", linewidth=1.2, linestyle="--", color="#98df8a", label="maximum_PORA_risk")
    ax.set_title("evaluation: eval PORA risk")
    ax.set_xlabel("update")
    ax.set_ylabel("PORA risk")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main() -> None:
    args = _parse_args()
    payload = _load(args.history)
    train_rows = payload.get("training_history", [])
    eval_rows = payload.get("evaluation_history", [])
    if not train_rows:
        raise SystemExit("training_history is empty")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True)
    ax = axes.ravel()

    _plot_train(ax[0], train_rows, "avg_epoch_return", "training: Average Epoch Return", args.smooth, zero=True)
    _plot_train(ax[1], train_rows, "avg_epoch_conflicts", "training: Average Epoch Conflicts", args.smooth)
    _plot_train(ax[2], train_rows, "total_ppo_loss", "training: Total PPO Loss", args.smooth, zero=True)
    _plot_eval_conflict_rate(ax[3], eval_rows)
    _plot_eval_pora_risk(ax[4], eval_rows)
    ax[5].axis("off")

    title = args.title
    if title is None:
        n_scenarios = payload.get("num_scenarios", "?")
        title = f"multi-scenario PPO report ({n_scenarios} scenarios)"
    fig.suptitle(title, fontsize=13)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[plot_multi_scenario_ppo_report] saved {out}")


if __name__ == "__main__":
    main()
