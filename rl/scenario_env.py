"""PPO environment for scenario-grouped AV2 data with per-non-AV heatmaps.

The environment emits a *dict* observation::

    obs = {
        "env": np.ndarray shape (D,)       ego state, goal delta, speed, heading, ...
        "heatmap": np.ndarray shape (K, H, W) currently K=1, heatmap of the nearest non-AV
    }

It uses the PORA v4 risk math (from ``pora_trajflow_risk_only_v4_faster.py``) in
a *streaming* form: the safety box is built each step, the heatmap is sampled,
and the percent-change boost uses a 1-frame ring buffer of the previous
reversed / rotated safety-box sample.

Modularity (per the plan spec):
  * ``select_nearest_non_av`` chooses which heatmap to use (pluggable).
  * ``StreamingPora`` computes ``risk_max_norm`` each step; subclass to replace.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from datasets.AV2_map import AV2Map
from datasets.AV2_scenarios import Scenario, ScenarioTrack


# ---------------------------------------------------------------------------
# Heatmap file I/O
# ---------------------------------------------------------------------------


@dataclass
class HeatmapBundle:
    """Per-non-AV heatmap stack + geometry metadata."""

    track_uid: str
    scenario_id: str
    pose: np.ndarray              # (3,) [tx, ty, h0] world
    grid_bounds_agent: np.ndarray # (2, 2) [[x_min, x_max], [y_min, y_max]]
    steps: int
    heatmap: np.ndarray           # (T, H, W) float32 in [0, 1]

    @property
    def future_steps(self) -> int:
        return int(self.heatmap.shape[0])

    @property
    def dx(self) -> float:
        return float((self.grid_bounds_agent[0, 1] - self.grid_bounds_agent[0, 0]) / max(self.steps - 1, 1))

    @property
    def dy(self) -> float:
        return float((self.grid_bounds_agent[1, 1] - self.grid_bounds_agent[1, 0]) / max(self.steps - 1, 1))


def load_heatmap_bundle(path: str) -> HeatmapBundle:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    hm = payload["heatmap"]
    if hasattr(hm, "numpy"):
        hm = hm.numpy()
    pose = payload["pose"]
    if hasattr(pose, "numpy"):
        pose = pose.numpy()
    bounds = payload["grid_bounds_agent"]
    if hasattr(bounds, "numpy"):
        bounds = bounds.numpy()
    return HeatmapBundle(
        track_uid=str(payload["track_uid"]),
        scenario_id=str(payload["scenario_id"]),
        pose=np.asarray(pose, dtype=np.float32),
        grid_bounds_agent=np.asarray(bounds, dtype=np.float32),
        steps=int(payload["steps"]),
        heatmap=np.asarray(hm, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Scenario bundle (scenario + preloaded heatmaps)
# ---------------------------------------------------------------------------


@dataclass
class ScenarioBundle:
    scenario: Scenario
    non_av_heatmaps: List[HeatmapBundle] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scenario.av is None:
            raise ValueError(f"Scenario {self.scenario.scenario_id!r} has no AV track")
        # Align heatmap bundles 1:1 with scenario.non_av by track_uid.
        by_uid = {h.track_uid: h for h in self.non_av_heatmaps}
        aligned: List[HeatmapBundle] = []
        kept_non_av: List[ScenarioTrack] = []
        for t in self.scenario.non_av:
            if t.track_uid in by_uid:
                aligned.append(by_uid[t.track_uid])
                kept_non_av.append(t)
        if not aligned:
            raise ValueError(
                f"No non-AV heatmaps available for scenario {self.scenario.scenario_id!r}"
            )
        self.non_av_heatmaps = aligned
        self.scenario = Scenario(
            scenario_id=self.scenario.scenario_id,
            av=self.scenario.av,
            non_av=kept_non_av,
        )


def load_scenario_bundle(scenario: Scenario, heatmap_root: str) -> Optional[ScenarioBundle]:
    """Load every per-non-AV heatmap for this scenario. Returns None if none found."""
    if scenario.av is None:
        return None
    scen_dir = os.path.join(heatmap_root, scenario.scenario_id)
    if not os.path.isdir(scen_dir):
        return None

    bundles: List[HeatmapBundle] = []
    for t in scenario.non_av:
        path = os.path.join(scen_dir, f"{t.track_uid}.pt")
        if not os.path.isfile(path):
            continue
        try:
            bundles.append(load_heatmap_bundle(path))
        except Exception as e:
            print(f"[scenario_env] skipping {path}: {e}")
    if not bundles:
        return None
    try:
        return ScenarioBundle(scenario=scenario, non_av_heatmaps=bundles)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Heatmap selection
# ---------------------------------------------------------------------------


def select_nearest_non_av(
    ego_xy_world: np.ndarray,
    non_av_positions_world: np.ndarray,
) -> int:
    """Pick the index of the closest non-AV to the ego at the current timestep.

    Parameters
    ----------
    ego_xy_world : (2,) float, world frame.
    non_av_positions_world : (K, 2) float, world frame.

    Notes
    -----
    Designed to be swapped for a fused-heatmap selector later (see plan Notes).
    """
    if non_av_positions_world.ndim != 2 or non_av_positions_world.shape[0] == 0:
        raise ValueError("non_av_positions_world must be non-empty (K, 2)")
    d2 = np.sum((non_av_positions_world - ego_xy_world[None, :]) ** 2, axis=-1)
    return int(np.argmin(d2))


# ---------------------------------------------------------------------------
# PORA streaming risk (math mirrors pora_trajflow_risk_only_v4_faster)
# ---------------------------------------------------------------------------


def _safety_box_points(
    car_xy: np.ndarray,
    car_angle: float,
    box_length: float,
    box_width: float,
    resolution: float,
) -> np.ndarray:
    """Dense (W, L, 2) rectangle of points covering the safety box in its own frame."""
    L_res = max(int(math.ceil(box_length / resolution)), 2)
    W_res = max(int(math.ceil(box_width / resolution)), 2)
    xs = np.linspace(-box_length / 2.0, box_length / 2.0, L_res, dtype=np.float64)
    ys = np.linspace(-box_width / 2.0, box_width / 2.0, W_res, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # both (W_res, L_res)
    local = np.stack([xx, yy], axis=-1)          # (W_res, L_res, 2)
    cos_a, sin_a = math.cos(car_angle), math.sin(car_angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    rotated = local @ R.T
    return rotated + np.array([car_xy[0], car_xy[1]], dtype=np.float64)


def _sample_heatmap_at_points(
    heatmap: np.ndarray,        # (H, W) float32 in agent frame (y along rows, x along cols)
    grid_bounds: np.ndarray,    # (2,2) [[x_min,x_max],[y_min,y_max]] in agent frame
    points_agent: np.ndarray,   # (..., 2)
) -> np.ndarray:
    """Bilinear-sample ``heatmap`` at ``points_agent``. Out-of-range -> 0."""
    H, W = heatmap.shape
    x_min, x_max = float(grid_bounds[0, 0]), float(grid_bounds[0, 1])
    y_min, y_max = float(grid_bounds[1, 0]), float(grid_bounds[1, 1])
    if W < 2 or H < 2 or x_max <= x_min or y_max <= y_min:
        return np.zeros(points_agent.shape[:-1], dtype=np.float32)

    xs = points_agent[..., 0]
    ys = points_agent[..., 1]
    u = (xs - x_min) / (x_max - x_min) * (W - 1)
    v = (ys - y_min) / (y_max - y_min) * (H - 1)

    i0 = np.floor(u).astype(np.int64)
    j0 = np.floor(v).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1

    inside = (i0 >= 0) & (i1 < W) & (j0 >= 0) & (j1 < H)
    i0c = np.clip(i0, 0, W - 1)
    i1c = np.clip(i1, 0, W - 1)
    j0c = np.clip(j0, 0, H - 1)
    j1c = np.clip(j1, 0, H - 1)

    h00 = heatmap[j0c, i0c]
    h01 = heatmap[j1c, i0c]
    h10 = heatmap[j0c, i1c]
    h11 = heatmap[j1c, i1c]

    ti = u - i0
    tj = v - j0
    val = (
        (1 - ti) * (1 - tj) * h00
        + (1 - ti) * tj * h01
        + ti * (1 - tj) * h10
        + ti * tj * h11
    )
    return np.where(inside, val, 0.0).astype(np.float32)


def _risk_weight_map(
    shape: Tuple[int, int],
    vehicle_length: float,
    vehicle_width: float,
    resolution: float,
) -> np.ndarray:
    """Replicate the quadratic-ring weight map from v4.compute_risk_from_time_series."""
    H, W = shape
    w = np.zeros((H, W), dtype=np.float64)
    cx, cy = W // 2, H // 2

    length_x0 = (2 * vehicle_width) * (1 / resolution)
    length_y0 = (vehicle_length + vehicle_width) * (1 / resolution)
    sx = cx - int(length_x0 // 2)
    ex = cx + int(length_x0 // 2)
    sy = cy - int(length_y0 // 2)
    ey = cy + int(length_y0 // 2)
    w[max(sy, 0):min(ey, H), max(sx, 0):min(ex, W)] = 1.0

    range_end = H + 1
    for i in range(range_end):
        lx_lo = (2 * vehicle_width) * (1 / resolution) + i
        ly_lo = (vehicle_length + vehicle_width) * (1 / resolution) + i
        sx_lo = cx - int(lx_lo // 2)
        ex_lo = cx + int(lx_lo // 2)
        sy_lo = cy - int(ly_lo // 2)
        ey_lo = cy + int(ly_lo // 2)
        lower = np.zeros((H, W), dtype=bool)
        lower[max(sy_lo, 0):min(ey_lo, H), max(sx_lo, 0):min(ex_lo, W)] = True

        lx_hi = lx_lo + 1
        ly_hi = ly_lo + 1
        sx_hi = cx - int(lx_hi // 2)
        ex_hi = cx + int(lx_hi // 2)
        sy_hi = cy - int(ly_hi // 2)
        ey_hi = cy + int(ly_hi // 2)
        upper = np.zeros((H, W), dtype=bool)
        upper[max(sy_hi, 0):min(ey_hi, H), max(sx_hi, 0):min(ex_hi, W)] = True

        ring = upper & ~lower
        w[ring] = 1.0 - i / range_end

    return w


class StreamingPora:
    """Streaming PORA risk using the same math as ``compute_risk_from_time_series``.

    The safety box shape is fixed at ``reset(max_vel)`` (max velocity known from
    the AV's recorded future trajectory), matching how v4 iterates the full
    trajectory upfront with ``max_velocity`` on the AV's whole traj.
    """

    def __init__(
        self,
        *,
        vehicle_length: float = 5.0,
        vehicle_width: float = 2.0,
        resolution: float = 0.5,
        reaction_time_s: float = 1.5,
        friction: float = 0.5,
    ):
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.resolution = float(resolution)
        self.reaction_time_s = float(reaction_time_s)
        self.friction = float(friction)

        self._max_vel = np.zeros(2, dtype=np.float64)
        self._box_length = 0.0
        self._box_width = 0.0
        self._L_res = 0
        self._W_res = 0
        self._risk_weight: Optional[np.ndarray] = None
        self._prev_rts: Optional[np.ndarray] = None
        self._max_boost = float(math.exp(1.0))

    # ---- lifecycle ------------------------------------------------------
    def reset(self, max_vel_world: np.ndarray) -> None:
        self._max_vel = np.asarray(max_vel_world, dtype=np.float64).reshape(2)
        vmax = float(np.hypot(self._max_vel[0], self._max_vel[1]))
        stop_d = (
            0.278 * self.reaction_time_s * 3.6 * vmax
            + (3.6 * vmax) ** 2 / (254.0 * self.friction)
        )
        extra = max(self.vehicle_length, self.vehicle_width)
        self._box_length = float(self.vehicle_length + extra + 2.0 * stop_d)
        self._box_width = float(self.vehicle_length + extra)
        self._L_res = max(int(math.ceil(self._box_length / self.resolution)), 2)
        self._W_res = max(int(math.ceil(self._box_width / self.resolution)), 2)

        # After np.rot90, the (W_res, L_res) safety box sample becomes (L_res, W_res)
        # with Y = long axis, X = short axis. Match v4 semantics.
        self._risk_weight = _risk_weight_map(
            shape=(self._L_res, self._W_res),
            vehicle_length=self.vehicle_length,
            vehicle_width=self.vehicle_width,
            resolution=self.resolution,
        )
        self._prev_rts = None

    # ---- per-step -------------------------------------------------------
    def _stop_distance(self, speed: float) -> float:
        return float(
            0.278 * self.reaction_time_s * 3.6 * speed
            + (3.6 * speed) ** 2 / (254.0 * self.friction)
        )

    def compute(
        self,
        *,
        ego_xy_agent: np.ndarray,
        ego_heading_agent: float,
        ego_speed: float,
        heatmap_t: np.ndarray,
        grid_bounds_agent: np.ndarray,
    ) -> float:
        """Return ``risk_max_norm`` in [0, 1] for this step. Sampling in agent frame."""
        if self._risk_weight is None:
            raise RuntimeError("StreamingPora.reset(max_vel_world) must be called first")

        # Dense (W_res, L_res, 2) points of the MAX box in agent frame.
        max_points = _safety_box_points(
            car_xy=ego_xy_agent,
            car_angle=float(ego_heading_agent),
            box_length=self._box_length,
            box_width=self._box_width,
            resolution=self.resolution,
        )[:self._W_res, :self._L_res, :]

        filled_max = _sample_heatmap_at_points(
            heatmap=heatmap_t,
            grid_bounds=grid_bounds_agent,
            points_agent=max_points,
        )
        # rot90 + reversal along Y -> ``reversed_time_series`` in v4.
        rts = np.rot90(filled_max)[::-1, :].astype(np.float64)

        if self._prev_rts is None or self._prev_rts.shape != rts.shape:
            percent_change = np.ones_like(rts)
        else:
            percent_change = np.exp(np.clip(rts - self._prev_rts, -50.0, 50.0))

        weighted = rts * self._risk_weight[:rts.shape[0], :rts.shape[1]]
        boosted = weighted * percent_change

        # Active window = between the "og" (stationary) box and the "dynamic" (current v) box.
        # v4 uses ``boosted[:, start_idx:end_idx]`` with a negative ``end_idx`` intended to be
        # interpreted as ``L_res + end_idx``. We convert to positive indices up front.
        stop_d = self._stop_distance(float(ego_speed))
        extra = max(self.vehicle_length, self.vehicle_width)
        og_len = self.vehicle_length + extra
        dyn_len = self.vehicle_length + extra + 2.0 * stop_d
        og_Lres = max(int(math.ceil(og_len / self.resolution)), 2)
        dyn_Lres = max(int(math.ceil(dyn_len / self.resolution)), 2)
        start_idx = int(max(0, (self._L_res - og_Lres) // 2))
        end_idx = int((self._L_res + dyn_Lres) // 2 - 1)
        end_idx = int(min(self._L_res, max(0, end_idx)))

        if end_idx <= start_idx:
            return 0.0

        # v4's ``boosted[t][start_idx:end_idx, :]`` slices the long (L_res) axis, which after
        # rot90 becomes axis 0 of our rts/boosted array.
        cur_max = float(np.max(boosted[start_idx:end_idx, :]))
        # Note: ``_prev_rts`` is advanced by the env wrapper (``_update_pora_prev``)
        # after the reward is consumed so diagnostics can inspect a clean compute() call.
        return float(max(0.0, min(1.0, cur_max / self._max_boost)))


# ---------------------------------------------------------------------------
# Agent <-> world transforms
# ---------------------------------------------------------------------------


def _rot_inv(h: float) -> np.ndarray:
    """Rotation by -h (world -> agent frame where +x aligns heading0)."""
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def world_to_agent(xy_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    tx, ty, h0 = float(pose[0]), float(pose[1]), float(pose[2])
    R = _rot_inv(h0)
    return (xy_world - np.array([tx, ty])) @ R.T


def world_heading_to_agent(h_world: float, pose: np.ndarray) -> float:
    return float(h_world - float(pose[2]))


def _agent_grid_world_xy(
    pose: np.ndarray,
    bounds: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """World-frame XY of every pixel center on a (H, W) agent-frame grid.

    Uses the same row=y / col=x convention as ``_sample_heatmap_at_points``.
    Returns ``(H, W, 2)`` float64.
    """
    H, W = int(shape[0]), int(shape[1])
    x_min, x_max = float(bounds[0, 0]), float(bounds[0, 1])
    y_min, y_max = float(bounds[1, 0]), float(bounds[1, 1])
    xs = np.linspace(x_min, x_max, W, dtype=np.float64)
    ys = np.linspace(y_min, y_max, H, dtype=np.float64)
    Xa, Ya = np.meshgrid(xs, ys)  # (H, W)
    pts_agent = np.stack([Xa, Ya], axis=-1)  # (H, W, 2)
    h0 = float(pose[2])
    c, s = math.cos(h0), math.sin(h0)
    # agent -> world: world = pose_xy + R(h0) @ agent_xy
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    flat = pts_agent.reshape(-1, 2) @ R.T + np.array([float(pose[0]), float(pose[1])])
    return flat.reshape(H, W, 2)


def _resample_into(
    src_heatmap: np.ndarray,        # (T, H_s, W_s)
    src_pose: np.ndarray,           # (3,) [tx, ty, h0] of source bundle
    src_bounds: np.ndarray,         # (2, 2) src grid bounds in src agent frame
    tgt_pose: np.ndarray,           # (3,) target pose
    tgt_bounds: np.ndarray,         # (2, 2) target grid bounds in target agent frame
    tgt_shape: Tuple[int, int, int],  # (T, H_t, W_t) — T must match src
) -> np.ndarray:
    """Bilinearly resample a per-non-AV ``(T, H_s, W_s)`` heatmap stack
    from the source bundle's agent frame into the target bundle's agent frame.

    The geometric transform target_agent -> world -> source_agent is constant
    across timesteps, so we precompute the source-frame pixel coordinates of
    every target pixel once, then sample per-frame.

    Pixels falling outside the source grid are filled with 0. Output shape
    matches ``tgt_shape``.
    """
    T_src = int(src_heatmap.shape[0])
    T_tgt, H_t, W_t = int(tgt_shape[0]), int(tgt_shape[1]), int(tgt_shape[2])
    if T_src != T_tgt:
        raise ValueError(f"resample expects matching T; got src T={T_src}, tgt T={T_tgt}")

    # World-frame coords of every target pixel center.
    world_xy = _agent_grid_world_xy(tgt_pose, tgt_bounds, (H_t, W_t))  # (H_t, W_t, 2)
    # Convert world -> src agent frame.
    pts_src_agent = world_to_agent(world_xy.reshape(-1, 2), src_pose).reshape(H_t, W_t, 2)

    out = np.zeros((T_tgt, H_t, W_t), dtype=np.float32)
    for t in range(T_tgt):
        out[t] = _sample_heatmap_at_points(
            heatmap=src_heatmap[t], grid_bounds=src_bounds, points_agent=pts_src_agent
        )
    return out


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


@dataclass
class SimpleBox:
    shape: Tuple[int, ...]


@dataclass
class DictSpace:
    spaces: Dict[str, SimpleBox]


@dataclass
class ScenarioEnvConfig:
    lambda_risk: float = 1.0
    base_reward: float = 0.0
    dt: float = 0.1                # AV2 step = 10 Hz
    action_scale_m: float = 2.0    # max world-meters per step when residual_action=False (blank-slate mode)
    # Task reward weights.
    w_progress: float = 1.0
    w_lane: float = 0.05           # L1 distance now; smaller weight to keep reward O(1)
    w_smooth: float = 0.05
    w_jerk: float = 0.05
    w_boundary: float = 0.0
    # Task termination.
    max_goal_dev: float = 20.0     # meters; if ego wanders >this from the AV-recorded goal line, truncate.
    speed_max: float = 40.0
    # Per-step reward clip to keep value targets bounded (None disables).
    reward_clip: Optional[float] = 5.0
    # Heatmap channel count K (spec: 1 for nearest).
    heatmap_channels: int = 1
    # PORA params.
    vehicle_length: float = 5.0
    vehicle_width: float = 2.0
    pora_resolution: float = 0.5
    # Residual action mode: action is a delta around the AV's recorded next step.
    # ``new_ego = av_recorded_next + action * residual_scale``.
    # Warm-starts PPO at an imitator-like baseline (reward starts near 0, not -300).
    residual_action: bool = True
    residual_scale: float = 1.5
    # Extra terminal reward applied on the step that triggers ``invalid=True``.
    # Applied AFTER ``reward_clip`` so that reward-hacking exploits (e.g. saturate
    # action -> speed > speed_max -> episode ends after 1 step with a clipped +5)
    # are made strictly unprofitable. Set to 0.0 to disable.
    invalid_penalty: float = -20.0


class PPOScenarioEnv:
    """Interactive PPO env over one ScenarioBundle. Spec: see module docstring."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle: ScenarioBundle,
        *,
        cfg: Optional[ScenarioEnvConfig] = None,
        select_fn: Callable[[np.ndarray, np.ndarray], int] = select_nearest_non_av,
        pora: Optional[StreamingPora] = None,
        history_steps: int = 50,
        future_steps: Optional[int] = None,
    ):
        self.bundle = bundle
        self.cfg = cfg or ScenarioEnvConfig()
        self.select_fn = select_fn
        self.history_steps = int(history_steps)
        if future_steps is None:
            future_steps = min(bundle.non_av_heatmaps[0].future_steps, 60)
        self.future_steps = int(future_steps)

        self.pora = pora or StreamingPora(
            vehicle_length=self.cfg.vehicle_length,
            vehicle_width=self.cfg.vehicle_width,
            resolution=self.cfg.pora_resolution,
        )

        av = bundle.scenario.av
        assert av is not None
        av_world = av.positions_world.cpu().numpy().astype(np.float64)
        self._av_world = av_world
        self._goal_xy = av_world[-1].copy()
        self._start_xy = av_world[self.history_steps - 1].copy()

        # Spaces: env vec is 8-D (ego_x, ego_y, speed, heading, goal_dx, goal_dy, t_frac, K_idx_norm)
        self._env_dim = 8
        sample_hm = bundle.non_av_heatmaps[0].heatmap[0]
        self._hm_shape = (self.cfg.heatmap_channels, sample_hm.shape[0], sample_hm.shape[1])

        self.observation_space = DictSpace(
            {
                "env": SimpleBox((self._env_dim,)),
                "heatmap": SimpleBox(self._hm_shape),
            }
        )
        self.action_space = SimpleBox((2,))

        self._t = 0
        self._ego_xy = self._start_xy.copy()
        self._ego_vel = np.zeros(2, dtype=np.float64)
        self._ego_heading = 0.0
        self._ego_speed = 0.0
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._last_sel_k = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _non_av_positions_at(self, t: int) -> np.ndarray:
        out = []
        idx = self.history_steps + t
        for tr in self.bundle.scenario.non_av:
            p = tr.positions_world.cpu().numpy()
            idx_clamped = int(np.clip(idx, 0, p.shape[0] - 1))
            out.append(p[idx_clamped])
        return np.asarray(out, dtype=np.float64)

    def _av_recorded_at(self, t: int) -> np.ndarray:
        idx = self.history_steps + t
        idx = int(np.clip(idx, 0, self._av_world.shape[0] - 1))
        return self._av_world[idx]

    def _pack_env_obs(self) -> np.ndarray:
        goal_dx = float(self._goal_xy[0] - self._ego_xy[0])
        goal_dy = float(self._goal_xy[1] - self._ego_xy[1])
        t_frac = float(self._t) / max(self.future_steps, 1)
        k_frac = (
            float(self._last_sel_k) / max(len(self.bundle.non_av_heatmaps) - 1, 1)
            if len(self.bundle.non_av_heatmaps) > 1
            else 0.0
        )
        return np.array(
            [
                self._ego_xy[0],
                self._ego_xy[1],
                self._ego_speed,
                self._ego_heading,
                goal_dx,
                goal_dy,
                t_frac,
                k_frac,
            ],
            dtype=np.float32,
        )

    def _pack_obs(self, heatmap_t: np.ndarray) -> Dict[str, np.ndarray]:
        K = self.cfg.heatmap_channels
        if K == 1:
            hm = heatmap_t[None, :, :].astype(np.float32)
        else:
            # Future-compatible: broadcast one heatmap across K channels until fusion arrives.
            hm = np.broadcast_to(heatmap_t[None, :, :], (K, *heatmap_t.shape)).astype(np.float32)
        return {"env": self._pack_env_obs(), "heatmap": hm}

    # ------------------------------------------------------------------
    # api
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        if seed is not None:
            np.random.seed(seed)
        self._t = 0
        self._ego_xy = self._start_xy.copy()
        self._ego_vel = np.zeros(2, dtype=np.float64)
        self._ego_heading = 0.0
        self._ego_speed = 0.0
        self._prev_action = np.zeros(2, dtype=np.float64)

        # Use the AV's recorded future velocities to set a fixed max-velocity safety box.
        fut = self._av_world[self.history_steps:]
        if fut.shape[0] >= 2:
            dv = np.diff(fut, axis=0) / max(self.cfg.dt, 1e-6)
            max_vel = np.max(np.abs(dv), axis=0)
        else:
            max_vel = np.array([self.cfg.speed_max, self.cfg.speed_max], dtype=np.float64)
        self.pora.reset(max_vel)

        # Select nearest non-AV at t=0 for the initial observation.
        non_av_pos = self._non_av_positions_at(0)
        self._last_sel_k = self.select_fn(self._ego_xy, non_av_pos)
        hm_bundle = self.bundle.non_av_heatmaps[self._last_sel_k]
        heatmap_t0 = hm_bundle.heatmap[0]

        return self._pack_obs(heatmap_t0), {"scenario_id": self.bundle.scenario.scenario_id}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)[:2]
        a = np.clip(a, -1.0, 1.0)

        # Integrate ego: either blank-slate (action = world delta)
        # or residual (action = offset from AV's recorded next step).
        if self.cfg.residual_action:
            av_next = self._av_recorded_at(self._t + 1)
            new_xy = av_next + a * self.cfg.residual_scale
            dxy = new_xy - self._ego_xy
        else:
            dxy = a * self.cfg.action_scale_m
            new_xy = self._ego_xy + dxy

        speed = float(np.linalg.norm(dxy) / max(self.cfg.dt, 1e-9))
        heading = float(np.arctan2(dxy[1], dxy[0])) if float(np.linalg.norm(dxy)) > 1e-9 else self._ego_heading

        prev_goal_dist = float(np.linalg.norm(self._goal_xy - self._ego_xy))
        self._ego_xy = new_xy
        self._ego_vel = dxy / max(self.cfg.dt, 1e-9)
        self._ego_speed = speed
        self._ego_heading = heading
        new_goal_dist = float(np.linalg.norm(self._goal_xy - self._ego_xy))

        # Select nearest non-AV at this step, load its heatmap for timestep self._t.
        t_eval = int(min(self._t, self.future_steps - 1))
        non_av_pos = self._non_av_positions_at(t_eval)
        self._last_sel_k = self.select_fn(self._ego_xy, non_av_pos)
        hm_bundle = self.bundle.non_av_heatmaps[self._last_sel_k]
        hm_t_idx = int(min(self._t, hm_bundle.future_steps - 1))
        heatmap_t = hm_bundle.heatmap[hm_t_idx]

        # PORA: transform ego state -> this non-AV's agent frame.
        pose = hm_bundle.pose
        ego_xy_agent = world_to_agent(np.asarray(self._ego_xy, dtype=np.float64), pose)
        ego_heading_agent = world_heading_to_agent(self._ego_heading, pose)
        pora_t = self.pora.compute(
            ego_xy_agent=ego_xy_agent,
            ego_heading_agent=ego_heading_agent,
            ego_speed=self._ego_speed,
            heatmap_t=heatmap_t,
            grid_bounds_agent=hm_bundle.grid_bounds_agent,
        )
        # Persist prev_rts for next-step percent_change (the reset() already cleared it).
        # We need StreamingPora.compute to also update prev_rts; update here since that method returns early.
        _update_pora_prev(self.pora, heatmap_t, hm_bundle.grid_bounds_agent, ego_xy_agent, ego_heading_agent)

        # Task reward.
        progress = prev_goal_dist - new_goal_dist
        ref_xy = self._av_recorded_at(self._t + 1)
        # L1 distance keeps the lane penalty in the same O(1) scale as progress and risk
        # (v1 used L2 which grew to -250/step once ego drifted 20 m).
        lane = -self.cfg.w_lane * float(np.linalg.norm(self._ego_xy - ref_xy))
        smooth = -self.cfg.w_smooth * float(np.sum(a ** 2))
        jerk = -self.cfg.w_jerk * float(np.sum((a - self._prev_action) ** 2))
        task_r = (
            self.cfg.w_progress * progress
            + lane
            + smooth
            + jerk
        )

        reward = float(self.cfg.base_reward + task_r - self.cfg.lambda_risk * pora_t)
        if self.cfg.reward_clip is not None:
            c = float(self.cfg.reward_clip)
            reward = float(np.clip(reward, -c, c))

        self._prev_action = a.copy()
        self._t += 1
        time_done = self._t >= self.future_steps
        # Terminate if the ego drifts too far from the AV-recorded reference at this step.
        # Using a per-step reference (rather than start-to-goal total slack) prevents the
        # agent from racking up huge negative reward on an obviously-bad trajectory.
        lane_dev = float(np.linalg.norm(self._ego_xy - ref_xy))
        invalid = (
            lane_dev > self.cfg.max_goal_dev
            or self._ego_speed > self.cfg.speed_max
            or not np.isfinite(reward)
            or not np.isfinite(self._ego_xy).all()
        )
        # Apply the invalid-termination penalty *after* reward_clip so it is not
        # absorbed by the clip — without this, a saturated action that blows the
        # speed check at step 1 got a clipped +5 and a cheap exit (see eval logs).
        if invalid and float(self.cfg.invalid_penalty) != 0.0:
            reward = float(reward + float(self.cfg.invalid_penalty))
        terminated = bool(time_done or invalid)
        truncated = False

        obs = self._pack_obs(heatmap_t)
        info = {
            "risk": float(pora_t),
            "pora": float(pora_t),
            "task_reward": float(task_r),
            "lane_dev": float(lane_dev),
            "selected_k": int(self._last_sel_k),
            "timestep": int(self._t),
            "invalid": bool(invalid),
            "scenario_id": self.bundle.scenario.scenario_id,
        }
        return obs, reward, terminated, truncated, info


def _update_pora_prev(
    pora: StreamingPora,
    heatmap_t: np.ndarray,
    grid_bounds: np.ndarray,
    ego_xy_agent: np.ndarray,
    ego_heading_agent: float,
) -> None:
    """Rebuild the reversed-time-series slice used as prev for next step.

    Kept outside ``StreamingPora.compute`` so that function remains a pure
    reader; the env is responsible for advancing the 1-frame ring buffer after
    reward use, keeping diagnostics (e.g. parity tests) independent of env.
    """
    if pora._risk_weight is None:
        return
    pts = _safety_box_points(
        car_xy=ego_xy_agent,
        car_angle=float(ego_heading_agent),
        box_length=pora._box_length,
        box_width=pora._box_width,
        resolution=pora.resolution,
    )[:pora._W_res, :pora._L_res, :]
    filled = _sample_heatmap_at_points(heatmap_t, grid_bounds, pts)
    pora._prev_rts = np.rot90(filled)[::-1, :].astype(np.float64)


# ---------------------------------------------------------------------------
# Multi-scenario wrapper
# ---------------------------------------------------------------------------


class MultiScenarioEnv:
    """Samples one ScenarioBundle on each ``reset()``."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bundles: Sequence[ScenarioBundle],
        *,
        cfg: Optional[ScenarioEnvConfig] = None,
        scenario_seed: Optional[int] = None,
        select_fn: Callable[[np.ndarray, np.ndarray], int] = select_nearest_non_av,
        history_steps: int = 50,
        future_steps: Optional[int] = None,
    ):
        if not bundles:
            raise ValueError("bundles must be non-empty")
        self.bundles = list(bundles)
        self.cfg = cfg or ScenarioEnvConfig()
        self.select_fn = select_fn
        self.history_steps = int(history_steps)
        self.future_steps = future_steps
        self._rng = np.random.default_rng(scenario_seed)
        self._env = self._make_env(self.bundles[0])

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _make_env(self, bundle: ScenarioBundle) -> PPOScenarioEnv:
        return PPOScenarioEnv(
            bundle,
            cfg=self.cfg,
            select_fn=self.select_fn,
            history_steps=self.history_steps,
            future_steps=self.future_steps,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(len(self.bundles)))
        self._env = self._make_env(self.bundles[idx])
        obs, info = self._env.reset()
        info["bundle_index"] = idx
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        return self._env.step(action)


# ---------------------------------------------------------------------------
# Blank-slate env (single scenario, fixed non-AV heatmap, no AV reference)
# ---------------------------------------------------------------------------


@dataclass
class BlankSlateEnvConfig:
    """Reward / termination knobs for ``BlankSlateScenarioEnv``.

    Differences vs ``ScenarioEnvConfig``:
      * action is always world-frame delta (no residual mode);
      * no goal_dist or AV-recorded reference is consumed by reward;
      * adds ``w_off_map`` and ``w_back`` weights and a ``backward_threshold``
        used both for shaping (penalty) and termination (invalid).
    """

    # Risk + base.
    lambda_risk: float = 1.0
    base_reward: float = 0.0
    dt: float = 0.1
    # Action.
    action_scale_m: float = 2.0          # max world-meters per step at |a|=1
    # Reward shaping (no AV reference!).
    w_forward_progress: float = 1.0      # +reward per meter along forward_unit
    w_off_map: float = 1.0               # -1.0 per step outside drivable area (also triggers invalid)
    w_back: float = 1.0                  # -reward per (m/s) of backward velocity
    w_smooth: float = 0.05
    w_jerk: float = 0.05
    # Continuous lane-keeping penalty: -w_lane_lateral * max(0, d_to_CL - deadband).
    # Provides a *gradient* toward the nearest centerline so the policy is
    # incentivised to stay on the lane, not just "anywhere inside the
    # drivable polygon". 0 disables. The deadband matches the AV's typical
    # cross-track jitter (~0.5–1 m on AV2) so well-centered driving pays no
    # penalty.
    w_lane_lateral: float = 0.0
    lane_lateral_deadband: float = 1.0
    # Per-step cap on the forward_progress reward TERM (in meters projected
    # onto forward_unit). Prevents the policy from "outrunning" the off-map
    # penalty by simply going faster. None disables the cap.
    progress_cap_m: Optional[float] = None
    # If True, forward_progress reward is zeroed when off_map. Combined with
    # ``w_off_map`` this makes leaving the drivable area strictly bad: no
    # forward_progress upside while incurring the per-step off_map penalty.
    gate_progress_on_map: bool = False
    # Termination.
    backward_threshold: float = 0.5      # m/s along -forward_unit -> invalid
    speed_max: float = 40.0
    reward_clip: Optional[float] = 5.0
    invalid_penalty: float = -20.0
    # If True (default), soft violations (off_map / backward / over_speed) end
    # the episode immediately. If False, only NaN states or ``time_done`` end
    # the episode — soft violations turn into pure reward shaping (per-step
    # penalties via ``w_off_map`` / ``w_back``) so the policy gets to play out
    # the full ``future_steps`` and PPO can learn recovery behaviour.
    terminate_on_invalid: bool = True
    # If True, the "forward direction" used by reward and obs is recomputed
    # every step from the nearest lane centerlines (``AV2Map.nearest_lane_tangent``).
    # If False, we keep the static initial heading ``[cos(h0), sin(h0)]``, so
    # ``forward_progress`` rewards driving in a fixed global direction. The
    # dynamic mode lets the policy follow curved roads without the trick of
    # leaving the drivable area to gain forward_progress.
    dynamic_forward_unit: bool = False
    lane_smoothing_k: int = 3            # top-k centerline vertices for tangent avg
    lane_weight_eps: float = 0.5         # softening eps for 1/(d+eps) weights
    # PORA.
    vehicle_length: float = 5.0
    vehicle_width: float = 2.0
    pora_resolution: float = 0.5
    # Heatmap channels (kept at 1; we only ever use one fixed non-AV heatmap).
    heatmap_channels: int = 1
    # Multi-non-AV aggregation. When True, the heatmap stack used by both
    # observation and PORA reward is built by resampling EVERY non-AV's
    # heatmap into the anchor (fixed_k) bundle's frame and aggregating
    # per-pixel via ``heatmap_agg``. When False (legacy v1-v8), only the
    # fixed_k bundle's stack is used.
    aggregate_heatmaps: bool = False
    heatmap_agg: str = "max"             # only "max" implemented; placeholder for "sum" / "prob_union"
    # Goal-conditioned reward (optional). When ``goal_xy_world`` is set:
    #   * obs gets 2 extra dims: [goal_dx_local, goal_dy_local] in ego's body frame
    #   * reward gets a "+w_goal * (prev_dist - new_dist)" term (positive when closing in)
    # When None, the env behaves exactly as before (8-D obs, no goal reward).
    goal_xy_world: Optional[Tuple[float, float]] = None
    w_goal: float = 0.0
    # Per-step cosine alignment between ego heading and goal direction.
    # Only active when goal_xy_world is set and distance > 1 m.
    # 0.0 disables (default = pure distance-reduction reward).
    w_heading_goal: float = 0.0


def select_fixed_non_av_by_min_distance(
    av_world: np.ndarray,
    non_av_worlds: Sequence[np.ndarray],
) -> int:
    """Pick the non-AV whose 110-step trajectory comes closest to the AV's.

    Deterministic, returns the index of the most-likely interaction partner.
    """
    if not non_av_worlds:
        raise ValueError("non_av_worlds must be non-empty")
    best_idx = 0
    best_min = float("inf")
    av = np.asarray(av_world, dtype=np.float64)
    for i, p in enumerate(non_av_worlds):
        q = np.asarray(p, dtype=np.float64)
        T = min(av.shape[0], q.shape[0])
        d = np.linalg.norm(av[:T] - q[:T], axis=-1)
        m = float(d.min()) if d.size > 0 else float("inf")
        if m < best_min:
            best_min = m
            best_idx = i
    return best_idx


class BlankSlateScenarioEnv:
    """Single-scenario, blank-slate PPO env.

    Differences vs ``PPOScenarioEnv``:
      * one ``ScenarioBundle`` only;
      * one **fixed** non-AV's heatmap stack (chosen at construction);
      * reward / termination never reference the AV recorded trajectory or
        an AV-derived goal point;
      * action is world-frame delta only;
      * map constraint via an ``AV2Map`` (drivable polygons) — exiting it
        terminates the episode with ``invalid_penalty``;
      * forward direction = ``[cos(h0), sin(h0)]`` from ``AV.pose[2]`` (only
        AV scalar surfaced to the policy, encoded in obs as ``forward_unit``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle: ScenarioBundle,
        map_obj: AV2Map,
        fixed_k: int,
        *,
        cfg: Optional[BlankSlateEnvConfig] = None,
        history_steps: int = 50,
        future_steps: Optional[int] = None,
        pora: Optional[StreamingPora] = None,
    ):
        if bundle.scenario.av is None:
            raise ValueError("BlankSlateScenarioEnv requires an AV track for h0")
        if not (0 <= fixed_k < len(bundle.non_av_heatmaps)):
            raise ValueError(
                f"fixed_k={fixed_k} out of range for {len(bundle.non_av_heatmaps)} heatmaps"
            )

        self.bundle = bundle
        self.map = map_obj
        self.fixed_k = int(fixed_k)
        self.cfg = cfg or BlankSlateEnvConfig()
        self.history_steps = int(history_steps)
        if future_steps is None:
            future_steps = min(bundle.non_av_heatmaps[fixed_k].future_steps, 60)
        self.future_steps = int(future_steps)

        self.pora = pora or StreamingPora(
            vehicle_length=self.cfg.vehicle_length,
            vehicle_width=self.cfg.vehicle_width,
            resolution=self.cfg.pora_resolution,
        )

        av = bundle.scenario.av
        av_world = av.positions_world.cpu().numpy().astype(np.float64)
        self._av_world = av_world  # kept ONLY for: max_vel reset and viz; never in reward/obs.
        self._start_xy = av_world[self.history_steps - 1].copy()
        # AV-pose heading h0 = initial forward direction (used when
        # ``dynamic_forward_unit`` is False, and as the fallback / sign hint
        # for the dynamic lane-tangent lookup).
        self._h0 = float(av.pose.cpu().numpy()[2])
        c, s = math.cos(self._h0), math.sin(self._h0)
        self._h0_forward_unit = np.array([c, s], dtype=np.float64)
        # ``_forward_unit`` is the *current* forward unit consumed by reward
        # and obs. It is refreshed every reset() and step() when
        # ``dynamic_forward_unit`` is True; otherwise it stays at h0.
        self._forward_unit = self._h0_forward_unit.copy()

        # Goal (optional). Stored as np.array(2,) or None.
        if self.cfg.goal_xy_world is not None:
            self._goal_xy_world = np.asarray(self.cfg.goal_xy_world, dtype=np.float64).reshape(2)
        else:
            self._goal_xy_world = None

        # Spaces. Obs is 8-D normally; +2 dims for goal_dx/dy in body frame
        # when goal_xy_world is configured.
        self._env_dim = 10 if self._goal_xy_world is not None else 8
        sample_hm = bundle.non_av_heatmaps[fixed_k].heatmap[0]
        self._hm_shape = (self.cfg.heatmap_channels, sample_hm.shape[0], sample_hm.shape[1])
        self.observation_space = DictSpace(
            {
                "env": SimpleBox((self._env_dim,)),
                "heatmap": SimpleBox(self._hm_shape),
            }
        )
        self.action_space = SimpleBox((2,))

        # ----- aggregated heatmap stack -------------------------------
        # Anchor frame = fixed_k bundle. When aggregate_heatmaps=True we
        # resample every other non-AV's heatmap into this frame and combine
        # via cfg.heatmap_agg. Pre-computed here once; reset()/step() just
        # index into ``self._agg_heatmap_stack[t]``.
        anchor = bundle.non_av_heatmaps[fixed_k]
        if self.cfg.aggregate_heatmaps and len(bundle.non_av_heatmaps) > 1:
            stacks = [anchor.heatmap.astype(np.float32, copy=False)]
            for k, hm in enumerate(bundle.non_av_heatmaps):
                if k == fixed_k:
                    continue
                stacks.append(
                    _resample_into(
                        src_heatmap=hm.heatmap,
                        src_pose=hm.pose,
                        src_bounds=hm.grid_bounds_agent,
                        tgt_pose=anchor.pose,
                        tgt_bounds=anchor.grid_bounds_agent,
                        tgt_shape=anchor.heatmap.shape,
                    )
                )
            agg = self.cfg.heatmap_agg
            if agg == "max":
                self._agg_heatmap_stack = np.maximum.reduce(stacks).astype(np.float32)
            else:
                raise NotImplementedError(
                    f"heatmap_agg={agg!r} not implemented (only 'max' for now)"
                )
            self._n_aggregated = len(stacks)
        else:
            self._agg_heatmap_stack = anchor.heatmap.astype(np.float32, copy=False)
            self._n_aggregated = 1

        # Mutable state (set in reset()).
        self._t = 0
        self._ego_xy = self._start_xy.copy()
        self._ego_xy_prev = self._start_xy.copy()
        self._ego_vel = np.zeros(2, dtype=np.float64)
        self._ego_heading = self._h0
        self._ego_speed = 0.0
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._prev_goal_dist = 0.0  # set in reset() if goal is configured

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _refresh_forward_unit(self) -> None:
        """Update ``self._forward_unit`` for the current ``self._ego_xy``.

        No-op when ``cfg.dynamic_forward_unit`` is False (we keep the static
        h0 vector). Otherwise we query the map for the nearest lane tangent,
        sign-aligned with ``self._h0`` so it returns a *forward* unit. We
        always use ``self._h0`` (not the current heading) as the sign hint to
        avoid bistable flipping if the policy briefly veers backwards."""
        if not self.cfg.dynamic_forward_unit:
            return
        fu = self.map.nearest_lane_tangent(
            self._ego_xy,
            h0_hint=self._h0,
            k=int(self.cfg.lane_smoothing_k),
            weight_eps=float(self.cfg.lane_weight_eps),
            fallback=self._h0_forward_unit,
        )
        self._forward_unit = np.asarray(fu, dtype=np.float64).reshape(2)

    def _pack_env_obs(self) -> np.ndarray:
        t_frac = float(self._t) / max(self.future_steps, 1)
        # Base 8-D: [ego_x, ego_y, speed, heading, forward_x, forward_y, t_frac, fixed_k_norm]
        # fixed_k_norm is a constant per env, kept for layout parity with PPOScenarioEnv.
        k_frac = (
            float(self.fixed_k) / max(len(self.bundle.non_av_heatmaps) - 1, 1)
            if len(self.bundle.non_av_heatmaps) > 1
            else 0.0
        )
        # ego position relative to episode start to keep values O(10m) not O(1000m)
        rel = self._ego_xy - self._start_xy
        feats = [
            float(rel[0]),
            float(rel[1]),
            self._ego_speed,
            self._ego_heading,
            self._forward_unit[0],
            self._forward_unit[1],
            t_frac,
            k_frac,
        ]
        if self._goal_xy_world is not None:
            # Goal vector in ego's body frame: forward = +x_local, left = +y_local.
            dx_w = float(self._goal_xy_world[0] - self._ego_xy[0])
            dy_w = float(self._goal_xy_world[1] - self._ego_xy[1])
            c = math.cos(self._ego_heading)
            s = math.sin(self._ego_heading)
            dx_local = c * dx_w + s * dy_w
            dy_local = -s * dx_w + c * dy_w
            feats.extend([dx_local, dy_local])
        return np.array(feats, dtype=np.float32)

    def _pack_obs(self, heatmap_t: np.ndarray) -> Dict[str, np.ndarray]:
        K = self.cfg.heatmap_channels
        if K == 1:
            hm = heatmap_t[None, :, :].astype(np.float32)
        else:
            hm = np.broadcast_to(heatmap_t[None, :, :], (K, *heatmap_t.shape)).astype(np.float32)
        return {"env": self._pack_env_obs(), "heatmap": hm}

    # ------------------------------------------------------------------
    # api
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        if seed is not None:
            np.random.seed(seed)
        self._t = 0
        self._ego_xy = self._start_xy.copy()
        self._ego_xy_prev = self._start_xy.copy()
        self._ego_vel = np.zeros(2, dtype=np.float64)
        self._ego_heading = self._h0
        self._ego_speed = 0.0
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._forward_unit = self._h0_forward_unit.copy()
        self._refresh_forward_unit()

        if self._goal_xy_world is not None:
            self._prev_goal_dist = float(np.linalg.norm(self._ego_xy - self._goal_xy_world))
        else:
            self._prev_goal_dist = 0.0

        # Match PPOScenarioEnv: use AV recorded future to seed the safety-box
        # max-velocity. AV trajectory is otherwise NEVER referenced. We could
        # alternatively use ``cfg.speed_max``; using the AV's own peak keeps
        # the safety box realistic for this scenario.
        fut = self._av_world[self.history_steps:]
        if fut.shape[0] >= 2:
            dv = np.diff(fut, axis=0) / max(self.cfg.dt, 1e-6)
            max_vel = np.max(np.abs(dv), axis=0)
        else:
            max_vel = np.array([self.cfg.speed_max, self.cfg.speed_max], dtype=np.float64)
        self.pora.reset(max_vel)

        hm_bundle = self.bundle.non_av_heatmaps[self.fixed_k]
        return self._pack_obs(self._agg_heatmap_stack[0]), {
            "scenario_id": self.bundle.scenario.scenario_id,
            "fixed_k": int(self.fixed_k),
            "fixed_track_uid": hm_bundle.track_uid,
            "h0": float(self._h0),
            "map_source": str(self.map.source),
            "aggregate_heatmaps": bool(self.cfg.aggregate_heatmaps),
            "n_non_av_aggregated": int(self._n_aggregated),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)[:2]
        a = np.clip(a, -1.0, 1.0)

        # Blank-slate kinematics.
        dxy = a * self.cfg.action_scale_m
        new_xy = self._ego_xy + dxy
        speed = float(np.linalg.norm(dxy) / max(self.cfg.dt, 1e-9))
        if float(np.linalg.norm(dxy)) > 1e-9:
            heading = float(np.arctan2(dxy[1], dxy[0]))
        else:
            heading = self._ego_heading

        self._ego_xy_prev = self._ego_xy.copy()
        self._ego_xy = new_xy
        self._ego_vel = dxy / max(self.cfg.dt, 1e-9)
        self._ego_speed = speed
        self._ego_heading = heading
        # Refresh the *forward* direction at the new position. With
        # ``dynamic_forward_unit=True`` this lets reward & obs follow road
        # curvature; with False it remains [cos(h0), sin(h0)].
        self._refresh_forward_unit()

        # PORA against the (optionally aggregated) heatmap at this step. The
        # anchor frame is the fixed_k bundle; ``self._agg_heatmap_stack`` is
        # either that bundle's stack (legacy) or the per-pixel max over all
        # non-AVs resampled into that frame (when aggregate_heatmaps=True).
        hm_bundle = self.bundle.non_av_heatmaps[self.fixed_k]
        hm_t_idx = int(min(self._t, self._agg_heatmap_stack.shape[0] - 1))
        heatmap_t = self._agg_heatmap_stack[hm_t_idx]
        pose = hm_bundle.pose
        ego_xy_agent = world_to_agent(np.asarray(self._ego_xy, dtype=np.float64), pose)
        ego_heading_agent = world_heading_to_agent(self._ego_heading, pose)
        pora_t = self.pora.compute(
            ego_xy_agent=ego_xy_agent,
            ego_heading_agent=ego_heading_agent,
            ego_speed=self._ego_speed,
            heatmap_t=heatmap_t,
            grid_bounds_agent=hm_bundle.grid_bounds_agent,
        )
        _update_pora_prev(self.pora, heatmap_t, hm_bundle.grid_bounds_agent, ego_xy_agent, ego_heading_agent)

        # Reward components (no AV reference).
        forward_progress = float((self._ego_xy - self._ego_xy_prev) @ self._forward_unit)
        in_drivable = self.map.is_inside_drivable_area(self._ego_xy)
        off_map_t = 0.0 if in_drivable else 1.0
        backward_violation = max(0.0, -float(self._ego_vel @ self._forward_unit))
        smooth = -self.cfg.w_smooth * float(np.sum(a ** 2))
        jerk = -self.cfg.w_jerk * float(np.sum((a - self._prev_action) ** 2))

        # Continuous lane-keeping penalty (0 unless w_lane_lateral > 0).
        lane_lateral_dist = 0.0
        lateral_excess = 0.0
        if self.cfg.w_lane_lateral > 0.0:
            lane_lateral_dist = float(self.map.nearest_lane_offset(self._ego_xy))
            lateral_excess = max(
                0.0, lane_lateral_dist - float(self.cfg.lane_lateral_deadband)
            )

        # Effective progress used for reward (the raw value is still kept in
        # ``info["forward_progress"]`` for diagnostics).
        progress_eff = forward_progress
        if self.cfg.gate_progress_on_map and off_map_t > 0.5:
            progress_eff = 0.0
        if self.cfg.progress_cap_m is not None:
            cap = float(self.cfg.progress_cap_m)
            progress_eff = float(np.clip(progress_eff, -cap, cap))

        # Goal-distance reward: positive when ego closes in on the configured
        # goal_xy_world (in meters of distance reduction per step).
        goal_dist = 0.0
        goal_reward = 0.0
        heading_goal_reward = 0.0
        if self._goal_xy_world is not None:
            goal_dist = float(np.linalg.norm(self._ego_xy - self._goal_xy_world))
            goal_reward = self.cfg.w_goal * (self._prev_goal_dist - goal_dist)
            self._prev_goal_dist = goal_dist
            if self.cfg.w_heading_goal > 0.0 and goal_dist > 1.0:
                goal_vec = self._goal_xy_world - self._ego_xy
                goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
                ego_fwd = np.array([math.cos(self._ego_heading), math.sin(self._ego_heading)], dtype=np.float64)
                heading_goal_reward = self.cfg.w_heading_goal * float(np.dot(ego_fwd, goal_dir))

        task_r = (
            self.cfg.w_forward_progress * progress_eff
            - self.cfg.w_off_map * off_map_t
            - self.cfg.w_back * backward_violation
            - self.cfg.w_lane_lateral * lateral_excess
            + goal_reward
            + heading_goal_reward
            + smooth
            + jerk
        )
        reward = float(self.cfg.base_reward + task_r - self.cfg.lambda_risk * pora_t)
        if self.cfg.reward_clip is not None:
            c = float(self.cfg.reward_clip)
            reward = float(np.clip(reward, -c, c))

        self._prev_action = a.copy()
        self._t += 1
        time_done = self._t >= self.future_steps

        # NaN / non-finite states must always terminate (otherwise we'd
        # propagate garbage into PPO buffers).
        nan_violation = (not np.isfinite(reward)) or (not np.isfinite(self._ego_xy).all())
        # "Soft" violations are the configurable behavioural ones. Whether
        # they terminate the episode is governed by ``terminate_on_invalid``.
        soft_violation = (
            (off_map_t > 0.5)
            or (backward_violation > self.cfg.backward_threshold)
            or (self._ego_speed > self.cfg.speed_max)
        )
        invalid = bool(soft_violation or nan_violation)
        if invalid and float(self.cfg.invalid_penalty) != 0.0:
            reward = float(reward + float(self.cfg.invalid_penalty))
        if self.cfg.terminate_on_invalid:
            terminated = bool(time_done or invalid)
        else:
            # Soft mode: only NaN or time_done ends the episode; off_map /
            # backward / over_speed stay in info as shaping signals only.
            terminated = bool(time_done or nan_violation)
        truncated = False

        obs = self._pack_obs(heatmap_t)
        info = {
            "risk": float(pora_t),
            "pora": float(pora_t),
            "task_reward": float(task_r),
            "forward_progress": float(forward_progress),
            "off_map": float(off_map_t),
            "backward_violation": float(backward_violation),
            "selected_k": int(self.fixed_k),
            "timestep": int(self._t),
            "invalid": bool(invalid),
            "soft_violation": bool(soft_violation),
            "nan_violation": bool(nan_violation),
            "scenario_id": self.bundle.scenario.scenario_id,
            "forward_unit_x": float(self._forward_unit[0]),
            "forward_unit_y": float(self._forward_unit[1]),
            "lane_lateral": float(lane_lateral_dist),
            "lateral_excess": float(lateral_excess),
            "aggregate_heatmaps": bool(self.cfg.aggregate_heatmaps),
            "n_non_av_aggregated": int(self._n_aggregated),
            "goal_dist": float(goal_dist),
            "goal_reward": float(goal_reward),
            "heading_goal_reward": float(heading_goal_reward),
        }
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Blank-slate env (control action: acceleration + steering_rate)
# ---------------------------------------------------------------------------


@dataclass
class BlankSlateControlEnvConfig(BlankSlateEnvConfig):
    """Blank-slate config with control-space actions.

    Action semantics:
      action[0] in [-1, 1] -> acceleration command in m/s^2
      action[1] in [-1, 1] -> heading-rate command in rad/s (direct, no bicycle model)
    """

    accel_max_mps2: float = 3.0
    heading_rate_max_radps: float = 0.8
    initial_speed_mps: float = 0.0


class BlankSlateScenarioEnvControl(BlankSlateScenarioEnv):
    """Blank-slate one-scenario env with direct accel + heading-rate control.

    Action semantics:
      action[0] in [-1, 1] -> acceleration in [-accel_max, +accel_max] m/s^2
      action[1] in [-1, 1] -> heading rate in [-heading_rate_max, +heading_rate_max] rad/s

    Heading is updated directly (no bicycle model / no wheelbase coupling),
    so the agent can steer even when stationary. Rewards / map / heatmap
    pipeline are inherited unchanged from BlankSlateScenarioEnv.
    """

    def __init__(
        self,
        bundle: ScenarioBundle,
        map_obj: AV2Map,
        fixed_k: int,
        *,
        cfg: Optional[BlankSlateControlEnvConfig] = None,
        history_steps: int = 50,
        future_steps: Optional[int] = None,
        pora: Optional[StreamingPora] = None,
    ):
        super().__init__(
            bundle=bundle,
            map_obj=map_obj,
            fixed_k=fixed_k,
            cfg=cfg or BlankSlateControlEnvConfig(),
            history_steps=history_steps,
            future_steps=future_steps,
            pora=pora,
        )
        self.cfg: BlankSlateControlEnvConfig

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        _obs, info = super().reset(seed=seed, options=options)
        self._ego_speed = float(max(0.0, self.cfg.initial_speed_mps))
        self._ego_vel = self._forward_unit * self._ego_speed
        obs = self._pack_obs(self._agg_heatmap_stack[0])
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)[:2]
        a = np.clip(a, -1.0, 1.0)

        dt = max(float(self.cfg.dt), 1e-9)
        accel_cmd = float(a[0]) * float(self.cfg.accel_max_mps2)
        heading_rate_cmd = float(a[1]) * float(self.cfg.heading_rate_max_radps)

        speed = max(0.0, float(self._ego_speed + accel_cmd * dt))
        heading = float(self._ego_heading + heading_rate_cmd * dt)

        dxy = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64) * (speed * dt)
        new_xy = self._ego_xy + dxy

        self._ego_xy_prev = self._ego_xy.copy()
        self._ego_xy = new_xy
        self._ego_vel = dxy / dt
        self._ego_speed = float(np.linalg.norm(self._ego_vel))
        self._ego_heading = heading
        self._refresh_forward_unit()

        hm_bundle = self.bundle.non_av_heatmaps[self.fixed_k]
        hm_t_idx = int(min(self._t, self._agg_heatmap_stack.shape[0] - 1))
        heatmap_t = self._agg_heatmap_stack[hm_t_idx]
        pose = hm_bundle.pose
        ego_xy_agent = world_to_agent(np.asarray(self._ego_xy, dtype=np.float64), pose)
        ego_heading_agent = world_heading_to_agent(self._ego_heading, pose)
        pora_t = self.pora.compute(
            ego_xy_agent=ego_xy_agent,
            ego_heading_agent=ego_heading_agent,
            ego_speed=self._ego_speed,
            heatmap_t=heatmap_t,
            grid_bounds_agent=hm_bundle.grid_bounds_agent,
        )
        _update_pora_prev(self.pora, heatmap_t, hm_bundle.grid_bounds_agent, ego_xy_agent, ego_heading_agent)

        forward_progress = float((self._ego_xy - self._ego_xy_prev) @ self._forward_unit)
        in_drivable = self.map.is_inside_drivable_area(self._ego_xy)
        off_map_t = 0.0 if in_drivable else 1.0
        backward_violation = max(0.0, -float(self._ego_vel @ self._forward_unit))
        smooth = -self.cfg.w_smooth * float(np.sum(a ** 2))
        jerk = -self.cfg.w_jerk * float(np.sum((a - self._prev_action) ** 2))

        lane_lateral_dist = 0.0
        lateral_excess = 0.0
        if self.cfg.w_lane_lateral > 0.0:
            lane_lateral_dist = float(self.map.nearest_lane_offset(self._ego_xy))
            lateral_excess = max(
                0.0, lane_lateral_dist - float(self.cfg.lane_lateral_deadband)
            )

        progress_eff = forward_progress
        if self.cfg.gate_progress_on_map and off_map_t > 0.5:
            progress_eff = 0.0
        if self.cfg.progress_cap_m is not None:
            cap = float(self.cfg.progress_cap_m)
            progress_eff = float(np.clip(progress_eff, -cap, cap))

        goal_dist = 0.0
        goal_reward = 0.0
        heading_goal_reward = 0.0
        if self._goal_xy_world is not None:
            goal_dist = float(np.linalg.norm(self._ego_xy - self._goal_xy_world))
            goal_reward = self.cfg.w_goal * (self._prev_goal_dist - goal_dist)
            self._prev_goal_dist = goal_dist
            if self.cfg.w_heading_goal > 0.0 and goal_dist > 1.0:
                goal_vec = self._goal_xy_world - self._ego_xy
                goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
                ego_fwd = np.array([math.cos(self._ego_heading), math.sin(self._ego_heading)], dtype=np.float64)
                heading_goal_reward = self.cfg.w_heading_goal * float(np.dot(ego_fwd, goal_dir))

        task_r = (
            self.cfg.w_forward_progress * progress_eff
            - self.cfg.w_off_map * off_map_t
            - self.cfg.w_back * backward_violation
            - self.cfg.w_lane_lateral * lateral_excess
            + goal_reward
            + heading_goal_reward
            + smooth
            + jerk
        )
        reward = float(self.cfg.base_reward + task_r - self.cfg.lambda_risk * pora_t)
        if self.cfg.reward_clip is not None:
            c = float(self.cfg.reward_clip)
            reward = float(np.clip(reward, -c, c))

        self._prev_action = a.copy()
        self._t += 1
        time_done = self._t >= self.future_steps

        nan_violation = (not np.isfinite(reward)) or (not np.isfinite(self._ego_xy).all())
        soft_violation = (
            (off_map_t > 0.5)
            or (backward_violation > self.cfg.backward_threshold)
            or (self._ego_speed > self.cfg.speed_max)
        )
        invalid = bool(soft_violation or nan_violation)
        if invalid and float(self.cfg.invalid_penalty) != 0.0:
            reward = float(reward + float(self.cfg.invalid_penalty))
        if self.cfg.terminate_on_invalid:
            terminated = bool(time_done or invalid)
        else:
            terminated = bool(time_done or nan_violation)
        truncated = False

        obs = self._pack_obs(heatmap_t)
        info = {
            "risk": float(pora_t),
            "pora": float(pora_t),
            "task_reward": float(task_r),
            "forward_progress": float(forward_progress),
            "off_map": float(off_map_t),
            "backward_violation": float(backward_violation),
            "selected_k": int(self.fixed_k),
            "timestep": int(self._t),
            "invalid": bool(invalid),
            "soft_violation": bool(soft_violation),
            "nan_violation": bool(nan_violation),
            "scenario_id": self.bundle.scenario.scenario_id,
            "forward_unit_x": float(self._forward_unit[0]),
            "forward_unit_y": float(self._forward_unit[1]),
            "lane_lateral": float(lane_lateral_dist),
            "lateral_excess": float(lateral_excess),
            "aggregate_heatmaps": bool(self.cfg.aggregate_heatmaps),
            "n_non_av_aggregated": int(self._n_aggregated),
            "accel_cmd": float(accel_cmd),
            "heading_rate_cmd": float(heading_rate_cmd),
            "goal_dist": float(goal_dist),
            "goal_reward": float(goal_reward),
            "heading_goal_reward": float(heading_goal_reward),
        }
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Multi-scenario wrapper (kept below blank-slate for import hygiene)
# ---------------------------------------------------------------------------


def load_multi_scenario_env(
    scenarios_pt: str,
    heatmap_root: str,
    *,
    cfg: Optional[ScenarioEnvConfig] = None,
    scenario_seed: Optional[int] = None,
    history_steps: int = 50,
    future_steps: Optional[int] = None,
) -> MultiScenarioEnv:
    """Build a MultiScenarioEnv from ``scenarios_rl.pt`` + precomputed heatmaps."""
    from datasets.AV2_scenarios import load_scenarios

    cache = load_scenarios(scenarios_pt)
    bundles: List[ScenarioBundle] = []
    for s in cache.scenarios:
        b = load_scenario_bundle(s, heatmap_root)
        if b is not None:
            bundles.append(b)
    if not bundles:
        raise RuntimeError(
            f"No usable scenario bundles under heatmap_root={heatmap_root!r}. "
            "Did you run scripts/precompute_scenario_heatmaps.py?"
        )
    print(f"[scenario_env] loaded {len(bundles)} scenario bundles from {scenarios_pt}")
    return MultiScenarioEnv(
        bundles,
        cfg=cfg,
        scenario_seed=scenario_seed,
        history_steps=history_steps,
        future_steps=future_steps,
    )
