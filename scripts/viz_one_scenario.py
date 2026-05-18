#!/usr/bin/env python3
"""Visualize a trained blank-slate PPO policy on ONE scenario.

Two output modes (auto-selected by ``--out`` extension):

  * ``.png`` (or any image extension): a single static snapshot of the run,
    with the heatmap shown at ``t = argmax(pora)``.
  * ``.gif``: an animation. Each frame shows
    - the heatmap of the FIXED non-AV at that timestep (this is the per-step
      changing risk landscape the policy actually saw during training);
    - the policy's path GROWING up to the current timestep, colored by PORA;
    - the fixed non-AV's recorded position at that timestep (red X);
    - a vertical cursor on the right-hand PORA panel.

  Static layers (drivable polygons, lane centerlines, AV recorded reference,
  other non-AVs faint, full fixed-non-AV trajectory, forward arrow) stay the
  same across frames.

Usage::

    # Static PNG snapshot
    python scripts/viz_one_scenario.py --policy runs/x.pt --out runs/x.png

    # Per-timestep GIF animation
    python scripts/viz_one_scenario.py --policy runs/x.pt --out runs/x.gif --fps 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.AV2_map import AV2Map, load_or_fallback
from datasets.AV2_scenarios import load_scenarios
from rl.dict_ppo import DictActorCritic
from rl.scenario_env import (
    BlankSlateControlEnvConfig,
    BlankSlateEnvConfig,
    BlankSlateScenarioEnv,
    BlankSlateScenarioEnvControl,
    HeatmapBundle,
    ScenarioBundle,
    _rot_inv,
    load_scenario_bundle,
    select_fixed_non_av_by_min_distance,
)


DEFAULT_SCENARIO_ID = "001bb9db-afa4-4ed6-bc2d-5b0266616da3"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenarios", type=str, default="data/av2_mf_tiny/scenarios_rl_s100.pt")
    p.add_argument("--heatmap_root", type=str, default="data/heatmaps_rl_s100")
    p.add_argument("--map_root", type=str, default="data/av2_mf_tiny/train")
    p.add_argument("--scenario_id", type=str, default=DEFAULT_SCENARIO_ID)
    p.add_argument("--non_av_uid", type=str, default=None)
    p.add_argument("--policy", type=str, required=True, help="Path to saved DictActorCritic state_dict.")
    p.add_argument(
        "--out", type=str, default="runs/viz_one/rollout.png",
        help="Output path. .png/.jpg -> static snapshot; .gif -> per-step animation.",
    )
    p.add_argument("--fps", type=int, default=4, help="Frames per second for GIF output.")
    p.add_argument("--device", type=str, default="cpu")

    # Env config (must match training).
    p.add_argument("--lambda_risk", type=float, default=1.0)
    p.add_argument("--action_scale_m", type=float, default=2.0)
    p.add_argument("--w_forward_progress", type=float, default=1.0)
    p.add_argument("--w_off_map", type=float, default=1.0)
    p.add_argument("--w_back", type=float, default=1.0)
    p.add_argument("--w_smooth", type=float, default=0.05)
    p.add_argument("--w_jerk", type=float, default=0.05)
    p.add_argument("--w_lane_lateral", type=float, default=0.0)
    p.add_argument("--lane_lateral_deadband", type=float, default=1.0)
    p.add_argument("--progress_cap_m", type=float, default=-1.0)
    p.add_argument(
        "--gate_progress_on_map",
        dest="gate_progress_on_map",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--no_gate_progress_on_map",
        dest="gate_progress_on_map",
        action="store_false",
    )
    p.add_argument("--backward_threshold", type=float, default=0.5)
    p.add_argument("--speed_max", type=float, default=40.0)
    p.add_argument("--reward_clip", type=float, default=5.0)
    p.add_argument("--invalid_penalty", type=float, default=-20.0)
    p.add_argument(
        "--terminate_on_invalid",
        dest="terminate_on_invalid",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no_terminate_on_invalid",
        dest="terminate_on_invalid",
        action="store_false",
        help="Roll out all future_steps regardless of soft violations.",
    )
    p.add_argument(
        "--dynamic_forward_unit",
        dest="dynamic_forward_unit",
        action="store_true",
        default=False,
        help="Use the lane-tangent dynamic forward direction (must match training).",
    )
    p.add_argument(
        "--no_dynamic_forward_unit",
        dest="dynamic_forward_unit",
        action="store_false",
    )
    p.add_argument("--lane_smoothing_k", type=int, default=3)
    p.add_argument("--lane_weight_eps", type=float, default=0.5)
    p.add_argument(
        "--aggregate_heatmaps",
        dest="aggregate_heatmaps",
        action="store_true",
        default=False,
        help="Use the per-pixel max over all non-AV heatmaps for both obs "
        "and PORA. Must match the value used during training.",
    )
    p.add_argument(
        "--no_aggregate_heatmaps",
        dest="aggregate_heatmaps",
        action="store_false",
    )
    p.add_argument("--heatmap_agg", type=str, default="max", choices=["max"])
    p.add_argument("--vehicle_length", type=float, default=5.0)
    p.add_argument("--vehicle_width", type=float, default=2.0)
    p.add_argument("--pora_resolution", type=float, default=0.5)
    p.add_argument("--future_steps", type=int, default=None)
    p.add_argument("--history_steps", type=int, default=50)
    p.add_argument("--fallback_half_width", type=float, default=8.0)

    # Control-mode env: action = [acceleration, heading_rate]
    p.add_argument(
        "--control_mode", action="store_true", default=False,
        help="Use BlankSlateScenarioEnvControl (action = accel + heading_rate). "
        "Required when visualizing policies trained by the final control-mode trainer.",
    )
    p.add_argument("--accel_max_mps2", type=float, default=3.0)
    p.add_argument("--heading_rate_max_radps", type=float, default=0.8)
    p.add_argument(
        "--initial_speed_mps", type=float, default=-1.0,
        help="Control-mode only. <0 -> auto-infer from AV history; >=0 -> override.",
    )
    p.add_argument("--initial_speed_smoothing_k", type=int, default=3)

    # Goal-conditioned (must match training).
    p.add_argument("--goal_world_x", type=float, default=None)
    p.add_argument("--goal_world_y", type=float, default=None)
    p.add_argument("--goal_from_av_endpoint", action="store_true", default=False)
    p.add_argument("--goal_lane_offset_m", type=float, default=0.0)
    p.add_argument("--w_goal", type=float, default=1.0)
    p.add_argument("--w_heading_goal", type=float, default=0.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _agent_to_world(xy_agent: np.ndarray, pose: np.ndarray) -> np.ndarray:
    R_inv = _rot_inv(float(pose[2]))  # = R(-h0)
    return xy_agent @ R_inv + np.asarray([float(pose[0]), float(pose[1])])


def _build_grid_world(
    bounds_agent: np.ndarray, pose: np.ndarray, shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = shape
    xs = np.linspace(float(bounds_agent[0, 0]), float(bounds_agent[0, 1]), W)
    ys = np.linspace(float(bounds_agent[1, 0]), float(bounds_agent[1, 1]), H)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float64)
    world = _agent_to_world(pts, pose)
    return world[:, 0].reshape(H, W), world[:, 1].reshape(H, W)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


def _run_policy(
    env: BlankSlateScenarioEnv,
    policy: DictActorCritic,
    device: torch.device,
):
    obs, _ = env.reset()
    xys = [env._ego_xy.copy()]
    poras = [0.0]
    progs = [0.0]
    rewards = [0.0]
    fwds = [env._forward_unit.copy()]
    invalid = False
    while True:
        oe = torch.as_tensor(obs["env"], dtype=torch.float32, device=device).unsqueeze(0)
        oh = torch.as_tensor(obs["heatmap"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, _std, _v = policy(oe, oh)
        a_np = mu.squeeze(0).cpu().numpy()
        obs, r, term, trunc, info = env.step(a_np)
        xys.append(env._ego_xy.copy())
        poras.append(float(info["pora"]))
        progs.append(float(info["forward_progress"]))
        rewards.append(float(r))
        fwds.append(np.array(
            [float(info.get("forward_unit_x", env._forward_unit[0])),
             float(info.get("forward_unit_y", env._forward_unit[1]))],
            dtype=np.float64,
        ))
        if bool(info.get("invalid", False)):
            invalid = True
        if bool(term or trunc):
            break
    return (
        np.asarray(xys),
        np.asarray(poras),
        np.asarray(progs),
        np.asarray(rewards),
        np.asarray(fwds),
        invalid,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _draw_map(ax, mp: AV2Map) -> None:
    for poly in mp.drivable_polygons:
        patch = MplPolygon(
            poly,
            closed=True,
            facecolor="#dddddd",
            edgecolor="#888",
            lw=0.8,
            alpha=0.5,
            zorder=0,
        )
        ax.add_patch(patch)
    for cl in mp.lane_centerlines:
        ax.plot(cl[:, 0], cl[:, 1], "--", color="#999", lw=0.6, alpha=0.6, zorder=1)


def _plot(
    bundle: ScenarioBundle,
    map_obj: AV2Map,
    fixed_k: int,
    env: BlankSlateScenarioEnv,
    policy: DictActorCritic,
    device: torch.device,
    out_path: Path,
):
    pol_xys, pol_poras, pol_progs, pol_rew, pol_fwds, invalid = _run_policy(env, policy, device)

    if pol_poras.max() > 1e-6:
        t_peak = int(np.argmax(pol_poras))
    else:
        t_peak = len(pol_poras) // 2
    t_peak = max(1, min(t_peak, len(pol_poras) - 1))

    hm_bundle: HeatmapBundle = bundle.non_av_heatmaps[fixed_k]
    agg_stack = env._agg_heatmap_stack
    hm_t_idx = max(0, min(t_peak - 1, agg_stack.shape[0] - 1))
    hm = agg_stack[hm_t_idx]
    X, Y = _build_grid_world(hm_bundle.grid_bounds_agent, hm_bundle.pose, hm.shape)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[2.0, 0.05, 0.05], wspace=0.35
    )
    ax = fig.add_subplot(gs[0, 0])
    cax_hm = fig.add_subplot(gs[0, 1])
    cax_pol = fig.add_subplot(gs[0, 2])

    _draw_map(ax, map_obj)

    vmax = max(float(np.quantile(hm, 0.995)), 0.05)
    pc = ax.pcolormesh(
        X, Y, hm, cmap="hot_r", shading="auto", alpha=0.65,
        vmin=0.0, vmax=vmax, zorder=2,
    )
    fig.colorbar(pc, cax=cax_hm, label="heatmap prob\n(P99.5 clipped)")

    for i, tr in enumerate(bundle.scenario.non_av):
        if i == fixed_k:
            continue
        p = tr.positions_world.numpy()
        ax.plot(p[:, 0], p[:, 1], "-", color="#8f9bbf", lw=0.7, alpha=0.45, zorder=3)

    sel_track = bundle.scenario.non_av[fixed_k]
    sel_world = sel_track.positions_world.numpy()
    ax.plot(
        sel_world[:, 0], sel_world[:, 1], "-",
        color="#c8372d", lw=1.6, alpha=0.9, zorder=4,
        label=f"fixed non-AV (k={fixed_k}, uid={sel_track.track_uid})",
    )

    assert bundle.scenario.av is not None
    av_world = bundle.scenario.av.positions_world.numpy()
    ax.plot(
        av_world[:, 0], av_world[:, 1], ":",
        color="#444", lw=1.6, alpha=0.7, zorder=4,
        label="AV recorded (reference, NOT used by policy)",
    )

    if getattr(env, "_goal_xy_world", None) is not None:
        gx, gy = float(env._goal_xy_world[0]), float(env._goal_xy_world[1])
        ax.plot([gx], [gy], marker="*", markersize=22, color="#ffcc00",
                markeredgecolor="black", markeredgewidth=1.5, zorder=10,
                label=f"goal ({gx:.1f}, {gy:.1f})")

    ax.plot(pol_xys[:, 0], pol_xys[:, 1], "-", color="white", lw=5.5, alpha=0.95, zorder=5)
    pts = pol_xys.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    pora_norm_max = max(0.5, float(pol_poras.max()))
    lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0.0, pora_norm_max))
    seg_vals = 0.5 * (pol_poras[:-1] + pol_poras[1:])
    lc.set_array(seg_vals)
    lc.set_linewidth(4.0)
    lc.set_zorder(6)
    ax.add_collection(lc)
    fig.colorbar(lc, cax=cax_pol, label="policy PORA along path")

    arrow_len = 6.0
    fwd0 = pol_fwds[0]
    ax.annotate(
        "",
        xy=(pol_xys[0, 0] + arrow_len * fwd0[0], pol_xys[0, 1] + arrow_len * fwd0[1]),
        xytext=(pol_xys[0, 0], pol_xys[0, 1]),
        arrowprops=dict(arrowstyle="->", color="#1b7e3e", lw=2.4),
        zorder=8,
    )
    if env.cfg.dynamic_forward_unit and len(pol_fwds) > 4:
        # Sample a few intermediate forward arrows (lighter color) so the
        # viewer can see how the road direction evolves along the path.
        step_n = max(1, (len(pol_fwds) - 1) // 6)
        for i in range(step_n, len(pol_fwds), step_n):
            fi = pol_fwds[i]
            xi = pol_xys[i]
            ax.annotate(
                "",
                xy=(xi[0] + arrow_len * fi[0], xi[1] + arrow_len * fi[1]),
                xytext=(xi[0], xi[1]),
                arrowprops=dict(arrowstyle="->", color="#1b7e3e", lw=1.2, alpha=0.45),
                zorder=7,
            )
    ax.scatter(
        pol_xys[0, 0], pol_xys[0, 1], marker="o",
        color="lime", s=130, ec="k", lw=1.5, zorder=9,
        label=f"start, h0={float(env._h0):.2f} rad",
    )
    ax.scatter(
        pol_xys[-1, 0], pol_xys[-1, 1], marker="s",
        color="cyan", s=120, ec="k", lw=1.5, zorder=9,
        label=f"policy end (t={len(pol_xys)-1})",
    )
    idx_non_av = min(49 + t_peak, sel_world.shape[0] - 1)
    ax.scatter(
        sel_world[idx_non_av, 0], sel_world[idx_non_av, 1], marker="X",
        color="#c8372d", s=180, ec="k", lw=1.5, zorder=9,
        label=f"fixed non-AV (anchor) @ t={t_peak}",
    )
    # Other non-AVs' current positions (smaller red X) so the viewer can see
    # which agents contribute to the aggregated heatmap.
    other_xs, other_ys = [], []
    for i, tr in enumerate(bundle.scenario.non_av):
        if i == fixed_k:
            continue
        p = tr.positions_world.numpy()
        idx = min(49 + t_peak, p.shape[0] - 1)
        other_xs.append(p[idx, 0])
        other_ys.append(p[idx, 1])
    if other_xs:
        ax.scatter(
            other_xs, other_ys, marker="X",
            color="#c8372d", s=80, ec="k", lw=1.0, alpha=0.65, zorder=8,
            label=f"other non-AVs @ t={t_peak} (n={len(other_xs)})",
        )

    all_pts = np.concatenate([pol_xys, av_world, sel_world], axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 10.0)
    pad = 0.2 * span
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * span + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xlabel("x (world, m)")
    ax.set_ylabel("y (world, m)")
    n_agg = int(getattr(env, "_n_aggregated", 1))
    agg_label = (
        f"aggregated max over {n_agg} non-AVs"
        if env.cfg.aggregate_heatmaps and n_agg > 1
        else f"fixed_k={fixed_k} only"
    )
    ax.set_title(
        f"map_source={map_obj.source}  drivable_polys={len(map_obj.drivable_polygons)}  "
        f"lanes={len(map_obj.lane_centerlines)}  |  heatmap: {agg_label}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.2)

    pol_mean_pora = float(np.mean(pol_poras[1:])) if len(pol_poras) > 1 else 0.0
    pol_total_progress = float(np.sum(pol_progs))
    pol_return = float(pol_rew.sum())
    status = "INVALID" if invalid else "OK"
    fig.suptitle(
        f"{bundle.scenario.scenario_id}   "
        f"return={pol_return:+.2f}  mean_pora={pol_mean_pora:.3f}  "
        f"total_forward={pol_total_progress:+.2f} m  len={len(pol_xys)-1}  {status}",
        fontsize=13,
        y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _animate(
    bundle: ScenarioBundle,
    map_obj: AV2Map,
    fixed_k: int,
    env: BlankSlateScenarioEnv,
    policy: DictActorCritic,
    device: torch.device,
    out_path: Path,
    *,
    fps: int = 4,
):
    """Animate one deterministic policy rollout, one frame per timestep.

    Per-frame dynamic elements:
      * heatmap at this t (the env's actual per-step risk map: aggregated
        over all non-AVs when ``cfg.aggregate_heatmaps``, else fixed_k only)
      * policy path grown up to step t, colored by per-step PORA
      * fixed non-AV recorded position at step t (large red X, anchor)
      * every other non-AV's recorded position at step t (small red X)
    """
    pol_xys, pol_poras, pol_progs, pol_rew, pol_fwds, invalid = _run_policy(env, policy, device)

    hm_bundle: HeatmapBundle = bundle.non_av_heatmaps[fixed_k]
    agg_stack = env._agg_heatmap_stack
    X, Y = _build_grid_world(
        hm_bundle.grid_bounds_agent, hm_bundle.pose, agg_stack[0].shape
    )
    # Use a single global vmax across all frames so colors are comparable across t.
    vmax = max(float(np.quantile(agg_stack, 0.995)), 0.05)
    pora_norm_max = max(0.5, float(pol_poras.max()))
    n_agg = int(getattr(env, "_n_aggregated", 1))
    agg_label = (
        f"agg max over {n_agg} non-AVs"
        if env.cfg.aggregate_heatmaps and n_agg > 1
        else f"fixed_k={fixed_k} only"
    )

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[2.0, 0.05, 0.05], wspace=0.35
    )
    ax = fig.add_subplot(gs[0, 0])
    cax_hm = fig.add_subplot(gs[0, 1])
    cax_pol = fig.add_subplot(gs[0, 2])

    # ---- static layers ----
    _draw_map(ax, map_obj)

    for i, tr in enumerate(bundle.scenario.non_av):
        if i == fixed_k:
            continue
        p = tr.positions_world.numpy()
        ax.plot(p[:, 0], p[:, 1], "-", color="#8f9bbf", lw=0.7, alpha=0.45, zorder=3)

    sel_track = bundle.scenario.non_av[fixed_k]
    sel_world = sel_track.positions_world.numpy()
    ax.plot(
        sel_world[:, 0], sel_world[:, 1], "-",
        color="#c8372d", lw=1.6, alpha=0.9, zorder=4,
        label=f"fixed non-AV (k={fixed_k}, uid={sel_track.track_uid})",
    )
    assert bundle.scenario.av is not None
    av_world = bundle.scenario.av.positions_world.numpy()
    ax.plot(
        av_world[:, 0], av_world[:, 1], ":",
        color="#444", lw=1.6, alpha=0.7, zorder=4,
        label="AV recorded (reference, NOT used by policy)",
    )

    if getattr(env, "_goal_xy_world", None) is not None:
        gx, gy = float(env._goal_xy_world[0]), float(env._goal_xy_world[1])
        ax.plot([gx], [gy], marker="*", markersize=22, color="#ffcc00",
                markeredgecolor="black", markeredgewidth=1.5, zorder=10,
                label=f"goal ({gx:.1f}, {gy:.1f})")

    arrow_len = 6.0
    ax.scatter(
        pol_xys[0, 0], pol_xys[0, 1], marker="o",
        color="lime", s=130, ec="k", lw=1.5, zorder=9,
        label=f"start, h0={float(env._h0):.2f} rad",
    )

    # Set view limits once based on the union of all paths.
    all_pts = np.concatenate([pol_xys, av_world, sel_world], axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 10.0)
    pad = 0.2 * span
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * span + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xlabel("x (world, m)")
    ax.set_ylabel("y (world, m)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    # ---- dynamic artists (recreated per frame for simplicity) ----
    state = {
        "pc": None, "lc": None, "halo": None,
        "sel_marker": None, "other_markers": None,
        "end_marker": None, "fwd_arrow": None,
    }
    other_non_av_paths = [
        tr.positions_world.numpy()
        for i, tr in enumerate(bundle.scenario.non_av) if i != fixed_k
    ]

    def _set_frame(t: int) -> None:
        # Heatmap snapshot at this timestep (aggregated stack the policy
        # actually consumes; falls back to fixed_k's stack when aggregate is off).
        hm_t_idx = max(0, min(t, agg_stack.shape[0] - 1))
        hm = agg_stack[hm_t_idx]
        if state["pc"] is not None:
            state["pc"].remove()
        state["pc"] = ax.pcolormesh(
            X, Y, hm, cmap="hot_r", shading="auto", alpha=0.65,
            vmin=0.0, vmax=vmax, zorder=2,
        )

        # Policy path up to step t (inclusive).
        if state["lc"] is not None:
            state["lc"].remove()
        if state["halo"] is not None:
            state["halo"].remove()
        n = max(2, t + 2)  # need >=2 points to make a segment
        n = min(n, len(pol_xys))
        path = pol_xys[:n]
        state["halo"], = ax.plot(
            path[:, 0], path[:, 1], "-", color="white", lw=5.5, alpha=0.95, zorder=5,
        )
        if n >= 2:
            pts = path.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_vals = 0.5 * (pol_poras[: n - 1] + pol_poras[1:n])
            lc = LineCollection(
                segs, cmap="viridis", norm=plt.Normalize(0.0, pora_norm_max),
            )
            lc.set_array(seg_vals)
            lc.set_linewidth(4.0)
            lc.set_zorder(6)
            ax.add_collection(lc)
            state["lc"] = lc
        else:
            state["lc"] = None

        # Markers: fixed non-AV current position (large X, anchor) + every
        # other non-AV's current position (small X) so the viewer can see
        # which agents contribute to the aggregated heatmap.
        idx_non_av = min(49 + t, sel_world.shape[0] - 1)
        if state["sel_marker"] is not None:
            state["sel_marker"].remove()
        state["sel_marker"] = ax.scatter(
            sel_world[idx_non_av, 0], sel_world[idx_non_av, 1], marker="X",
            color="#c8372d", s=180, ec="k", lw=1.5, zorder=9,
        )
        if state["other_markers"] is not None:
            state["other_markers"].remove()
        if other_non_av_paths:
            xs, ys = [], []
            for p in other_non_av_paths:
                idx = min(49 + t, p.shape[0] - 1)
                xs.append(p[idx, 0])
                ys.append(p[idx, 1])
            state["other_markers"] = ax.scatter(
                xs, ys, marker="X",
                color="#c8372d", s=80, ec="k", lw=1.0, alpha=0.65, zorder=8,
            )
        else:
            state["other_markers"] = None
        if state["end_marker"] is not None:
            state["end_marker"].remove()
        head_idx = min(t + 1, len(pol_xys) - 1)
        state["end_marker"] = ax.scatter(
            pol_xys[head_idx, 0], pol_xys[head_idx, 1], marker="s",
            color="cyan", s=120, ec="k", lw=1.5, zorder=9,
        )

        # Forward arrow at the policy head, pointing along the *current*
        # forward direction (lane tangent if dynamic, else h0).
        if state["fwd_arrow"] is not None:
            state["fwd_arrow"].remove()
        f_here = pol_fwds[head_idx]
        h_xy = pol_xys[head_idx]
        state["fwd_arrow"] = ax.annotate(
            "",
            xy=(h_xy[0] + arrow_len * f_here[0], h_xy[1] + arrow_len * f_here[1]),
            xytext=(h_xy[0], h_xy[1]),
            arrowprops=dict(arrowstyle="->", color="#1b7e3e", lw=2.4),
            zorder=10,
        )

        ax.set_title(
            f"map_source={map_obj.source}  drivable_polys={len(map_obj.drivable_polygons)}  "
            f"lanes={len(map_obj.lane_centerlines)}  |  heatmap: {agg_label}  |  "
            f"step t={t}/{len(pol_xys)-1}  |  non-AVs @ t={t}",
            fontsize=10,
        )

    # Initialize colorbars on the first frame, then animate.
    _set_frame(0)
    fig.colorbar(state["pc"], cax=cax_hm, label="heatmap prob (P99.5 clipped)")
    fig.colorbar(state["lc"] if state["lc"] is not None else state["pc"],
                 cax=cax_pol, label="policy PORA along path")

    pol_mean_pora = float(np.mean(pol_poras[1:])) if len(pol_poras) > 1 else 0.0
    pol_total_progress = float(np.sum(pol_progs))
    pol_return = float(pol_rew.sum())
    status = "INVALID" if invalid else "OK"
    fig.suptitle(
        f"{bundle.scenario.scenario_id}   "
        f"return={pol_return:+.2f}  mean_pora={pol_mean_pora:.3f}  "
        f"total_forward={pol_total_progress:+.2f} m  len={len(pol_xys)-1}  {status}",
        fontsize=13, y=0.995,
    )

    n_frames = len(pol_xys)

    def _update(frame: int):
        _set_frame(frame)
        return ()

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=int(1000 / max(fps, 1)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=int(max(fps, 1)))
    anim.save(str(out_path), writer=writer, dpi=90)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    cache = load_scenarios(args.scenarios)
    by_id = {s.scenario_id: s for s in cache.scenarios}
    if args.scenario_id not in by_id:
        raise SystemExit(f"scenario_id {args.scenario_id!r} not in {args.scenarios}")
    bundle = load_scenario_bundle(by_id[args.scenario_id], args.heatmap_root)
    if bundle is None:
        raise SystemExit(f"No heatmap bundle for {args.scenario_id} under {args.heatmap_root}")

    av_world = bundle.scenario.av.positions_world.cpu().numpy()
    non_av_worlds = [t.positions_world.cpu().numpy() for t in bundle.scenario.non_av]
    map_obj = load_or_fallback(
        map_root=args.map_root,
        scenario_id=args.scenario_id,
        av_world=av_world,
        non_av_worlds=non_av_worlds,
        fallback_half_width=float(args.fallback_half_width),
    )
    print(f"[map] source={map_obj.source} n_drivable={len(map_obj.drivable_polygons)}")

    if args.non_av_uid is not None:
        uid = str(args.non_av_uid)
        try:
            fixed_k = next(i for i, t in enumerate(bundle.scenario.non_av) if t.track_uid == uid)
        except StopIteration:
            raise SystemExit(f"--non_av_uid {uid!r} not found.")
    else:
        fixed_k = select_fixed_non_av_by_min_distance(av_world, non_av_worlds)
    print(f"[env] fixed_k={fixed_k} non_av_uid={bundle.scenario.non_av[fixed_k].track_uid}")

    reward_clip = float(args.reward_clip) if float(args.reward_clip) > 0.0 else None
    progress_cap = float(args.progress_cap_m) if float(args.progress_cap_m) > 0.0 else None

    # Resolve goal_xy_world (must match training).
    goal_xy_world = None
    if args.goal_world_x is not None and args.goal_world_y is not None:
        goal_xy_world = (float(args.goal_world_x), float(args.goal_world_y))
        print(f"[goal] explicit world=({goal_xy_world[0]:.3f}, {goal_xy_world[1]:.3f})")
    elif bool(args.goal_from_av_endpoint):
        import math as _math
        endpoint = av_world[-1]
        h0 = float(bundle.scenario.av.pose.cpu().numpy()[2])
        tangent = map_obj.nearest_lane_tangent(
            endpoint, h0_hint=h0, k=3, weight_eps=0.5,
            fallback=np.array([_math.cos(h0), _math.sin(h0)], dtype=np.float64),
        )
        tangent = np.asarray(tangent, dtype=np.float64).reshape(2)
        perp_left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        cand = endpoint + float(args.goal_lane_offset_m) * perp_left
        goal_xy_world = (float(cand[0]), float(cand[1]))
        print(f"[goal] av_endpoint + {args.goal_lane_offset_m:+.2f}m * perp_left -> "
              f"({goal_xy_world[0]:.3f}, {goal_xy_world[1]:.3f})")

    common_kwargs = dict(
        lambda_risk=float(args.lambda_risk),
        w_forward_progress=float(args.w_forward_progress),
        w_off_map=float(args.w_off_map),
        w_back=float(args.w_back),
        w_smooth=float(args.w_smooth),
        w_jerk=float(args.w_jerk),
        w_lane_lateral=float(args.w_lane_lateral),
        lane_lateral_deadband=float(args.lane_lateral_deadband),
        progress_cap_m=progress_cap,
        gate_progress_on_map=bool(args.gate_progress_on_map),
        backward_threshold=float(args.backward_threshold),
        speed_max=float(args.speed_max),
        reward_clip=reward_clip,
        invalid_penalty=float(args.invalid_penalty),
        terminate_on_invalid=bool(args.terminate_on_invalid),
        dynamic_forward_unit=bool(args.dynamic_forward_unit),
        lane_smoothing_k=int(args.lane_smoothing_k),
        lane_weight_eps=float(args.lane_weight_eps),
        aggregate_heatmaps=bool(args.aggregate_heatmaps),
        heatmap_agg=str(args.heatmap_agg),
        vehicle_length=float(args.vehicle_length),
        vehicle_width=float(args.vehicle_width),
        pora_resolution=float(args.pora_resolution),
        goal_xy_world=goal_xy_world,
        w_goal=float(args.w_goal),
        w_heading_goal=float(args.w_heading_goal),
    )

    history_steps = int(args.history_steps)
    if bool(args.control_mode):
        if float(args.initial_speed_mps) < 0.0:
            k = max(1, int(args.initial_speed_smoothing_k))
            i1 = history_steps
            i0 = max(1, i1 - k)
            deltas = av_world[i0:i1] - av_world[i0 - 1 : i1 - 1]
            speeds = np.linalg.norm(deltas, axis=-1) / 0.1
            initial_speed_mps = float(np.mean(speeds))
            print(
                f"[init_speed] auto-inferred={initial_speed_mps:.3f} m/s "
                f"(mean over last {k} history step(s))"
            )
        else:
            initial_speed_mps = float(args.initial_speed_mps)
            print(f"[init_speed] override={initial_speed_mps:.3f} m/s")
        env_cfg = BlankSlateControlEnvConfig(
            **common_kwargs,
            accel_max_mps2=float(args.accel_max_mps2),
            heading_rate_max_radps=float(args.heading_rate_max_radps),
            initial_speed_mps=initial_speed_mps,
        )
        env = BlankSlateScenarioEnvControl(
            bundle=bundle,
            map_obj=map_obj,
            fixed_k=fixed_k,
            cfg=env_cfg,
            history_steps=history_steps,
            future_steps=args.future_steps,
        )
        print("[viz_one] control_mode=ON (action = accel + heading_rate)")
    else:
        env_cfg = BlankSlateEnvConfig(
            **common_kwargs,
            action_scale_m=float(args.action_scale_m),
        )
        env = BlankSlateScenarioEnv(
            bundle=bundle,
            map_obj=map_obj,
            fixed_k=fixed_k,
            cfg=env_cfg,
            history_steps=history_steps,
            future_steps=args.future_steps,
        )

    policy = DictActorCritic(
        env_dim=int(env.observation_space.spaces["env"].shape[0]),
        action_dim=int(env.action_space.shape[0]),
        heatmap_channels=int(env.observation_space.spaces["heatmap"].shape[0]),
    ).to(device)
    try:
        state = torch.load(args.policy, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.policy, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    print(f"[viz_one] loaded policy from {args.policy}")

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".gif":
        _animate(bundle, map_obj, fixed_k, env, policy, device, out_path, fps=int(args.fps))
    else:
        _plot(bundle, map_obj, fixed_k, env, policy, device, out_path)
    print(f"[viz_one] wrote {out_path}")


if __name__ == "__main__":
    main()
