#!/usr/bin/env python3
"""Train PPO on multiple moving-AV scenarios with accel + heading-rate control.

This is the final multi-scenario trainer. It reuses
``BlankSlateScenarioEnvControl`` but builds one env per scenario and samples a
random scenario on every episode reset.

Default data:
    data/av2_mf_tiny/scenarios_rl_moving_av_100.pt

The default cache currently contains 51 moving-AV scenarios from
``scripts/filter_moving_av_scenarios.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.AV2_map import load_or_fallback
from datasets.AV2_scenarios import load_scenarios
from rl.dict_ppo import DictActorCritic, DictPPOConfig, DictRolloutBuffer, compute_dict_advantages
from rl.scenario_env import (
    BlankSlateControlEnvConfig,
    BlankSlateScenarioEnvControl,
    load_scenario_bundle,
    select_fixed_non_av_by_min_distance,
)


@dataclass
class TTCSummary:
    min_ttc: float
    conflict: bool
    collision: bool


class RandomScenarioControlEnv:
    """Gym-like wrapper that samples one prebuilt scenario env on reset()."""

    metadata = {"render_modes": []}

    def __init__(self, envs: Sequence[BlankSlateScenarioEnvControl], seed: Optional[int] = None):
        if not envs:
            raise ValueError("envs must be non-empty")
        self.envs = list(envs)
        self._rng = np.random.default_rng(seed)
        self._env = self.envs[0]
        self._env_idx = 0
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def current_env(self) -> BlankSlateScenarioEnvControl:
        return self._env

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._env_idx = int(self._rng.integers(len(self.envs)))
        self._env = self.envs[self._env_idx]
        obs, info = self._env.reset(options=options)
        info["bundle_index"] = int(self._env_idx)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._env.step(action)
        info["bundle_index"] = int(self._env_idx)
        return obs, reward, terminated, truncated, info


def _shape_from_space(sp) -> Tuple[int, ...]:
    if hasattr(sp, "shape"):
        return tuple(sp.shape)
    raise TypeError(f"Unsupported space: {sp!r}")


def _auto_initial_speed(av_world: np.ndarray, *, history_steps: int, dt: float, k: int) -> float:
    k = max(1, int(k))
    i1 = int(history_steps)
    i0 = max(1, i1 - k)
    deltas = av_world[i0:i1] - av_world[i0 - 1 : i1 - 1]
    speeds = np.linalg.norm(deltas, axis=-1) / max(float(dt), 1e-9)
    return float(np.mean(speeds))


def _goal_from_endpoint(env_bundle, map_obj, lane_offset_m: float) -> Tuple[float, float]:
    av_world = env_bundle.scenario.av.positions_world.cpu().numpy()
    endpoint = av_world[-1]
    h0 = float(env_bundle.scenario.av.pose.cpu().numpy()[2])
    fallback = np.array([math.cos(h0), math.sin(h0)], dtype=np.float64)
    tangent = map_obj.nearest_lane_tangent(
        endpoint,
        h0_hint=h0,
        k=3,
        weight_eps=0.5,
        fallback=fallback,
    )
    tangent = np.asarray(tangent, dtype=np.float64).reshape(2)
    perp_left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    goal = endpoint + float(lane_offset_m) * perp_left
    return (float(goal[0]), float(goal[1]))


def _build_env_config(args: argparse.Namespace, *, initial_speed_mps: float, goal_xy_world) -> BlankSlateControlEnvConfig:
    reward_clip = float(args.reward_clip) if float(args.reward_clip) > 0.0 else None
    progress_cap = float(args.progress_cap_m) if float(args.progress_cap_m) > 0.0 else None
    return BlankSlateControlEnvConfig(
        lambda_risk=float(args.lambda_risk),
        base_reward=float(args.base_reward),
        dt=float(args.dt),
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
        accel_max_mps2=float(args.accel_max_mps2),
        heading_rate_max_radps=float(args.heading_rate_max_radps),
        initial_speed_mps=float(initial_speed_mps),
        goal_xy_world=goal_xy_world,
        w_goal=float(args.w_goal),
        w_heading_goal=float(args.w_heading_goal),
    )


def _build_scenario_envs(args: argparse.Namespace) -> List[BlankSlateScenarioEnvControl]:
    cache = load_scenarios(args.scenarios)
    envs: List[BlankSlateScenarioEnvControl] = []
    skipped: List[str] = []
    for scenario in cache.scenarios:
        bundle = load_scenario_bundle(scenario, args.heatmap_root)
        if bundle is None:
            skipped.append(scenario.scenario_id)
            continue

        av_world = bundle.scenario.av.positions_world.cpu().numpy()
        non_av_worlds = [t.positions_world.cpu().numpy() for t in bundle.scenario.non_av]
        map_obj = load_or_fallback(
            map_root=args.map_root,
            scenario_id=scenario.scenario_id,
            av_world=av_world,
            non_av_worlds=non_av_worlds,
            fallback_half_width=float(args.fallback_half_width),
        )

        if args.non_av_uid is not None:
            uid = str(args.non_av_uid)
            matches = [i for i, t in enumerate(bundle.scenario.non_av) if t.track_uid == uid]
            if not matches:
                skipped.append(scenario.scenario_id)
                continue
            fixed_k = int(matches[0])
        else:
            fixed_k = select_fixed_non_av_by_min_distance(av_world, non_av_worlds)

        if float(args.initial_speed_mps) < 0.0:
            initial_speed_mps = _auto_initial_speed(
                av_world,
                history_steps=int(args.history_steps),
                dt=float(args.dt),
                k=int(args.initial_speed_smoothing_k),
            )
        else:
            initial_speed_mps = float(args.initial_speed_mps)

        goal_xy_world = None
        if args.goal_world_x is not None and args.goal_world_y is not None:
            goal_xy_world = (float(args.goal_world_x), float(args.goal_world_y))
        elif bool(args.goal_from_av_endpoint):
            goal_xy_world = _goal_from_endpoint(bundle, map_obj, float(args.goal_lane_offset_m))

        cfg = _build_env_config(args, initial_speed_mps=initial_speed_mps, goal_xy_world=goal_xy_world)
        env = BlankSlateScenarioEnvControl(
            bundle=bundle,
            map_obj=map_obj,
            fixed_k=fixed_k,
            cfg=cfg,
            history_steps=int(args.history_steps),
            future_steps=args.future_steps,
        )
        envs.append(env)

    if not envs:
        raise RuntimeError("No usable scenario envs. Check --scenarios and --heatmap_root.")
    print(
        f"[multi_scenario] built {len(envs)} envs from {args.scenarios} "
        f"(skipped={len(skipped)})"
    )
    return envs


def _non_av_state_at(env: BlankSlateScenarioEnvControl, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return non-AV positions and velocities in world frame at PPO future index t."""
    pos: List[np.ndarray] = []
    vel: List[np.ndarray] = []
    idx = int(np.clip(env.history_steps + t, 0, env._av_world.shape[0] - 1))
    prev_idx = max(0, idx - 1)
    dt = max(float(env.cfg.dt), 1e-9)
    for track in env.bundle.scenario.non_av:
        xy = track.positions_world.cpu().numpy().astype(np.float64)
        idx_i = int(np.clip(idx, 0, xy.shape[0] - 1))
        prev_i = int(np.clip(prev_idx, 0, xy.shape[0] - 1))
        pos.append(xy[idx_i])
        vel.append((xy[idx_i] - xy[prev_i]) / dt)
    if not pos:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64)


def _ttc_to_radius(rel_pos: np.ndarray, rel_vel: np.ndarray, radius: float) -> float:
    """Time until ||rel_pos + rel_vel * t|| reaches radius; inf if no hit."""
    dist = float(np.linalg.norm(rel_pos))
    if dist <= radius:
        return 0.0
    a = float(rel_vel @ rel_vel)
    if a < 1e-9:
        return float("inf")
    b = 2.0 * float(rel_pos @ rel_vel)
    c = float(rel_pos @ rel_pos) - float(radius) ** 2
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return float("inf")
    sqrt_disc = math.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    roots = [r for r in roots if r >= 0.0]
    return float(min(roots)) if roots else float("inf")


def compute_ttc_summary(
    env: BlankSlateScenarioEnvControl,
    *,
    t: int,
    conflict_ttc_s: float,
    collision_distance_m: float,
) -> TTCSummary:
    other_pos, other_vel = _non_av_state_at(env, t)
    if other_pos.shape[0] == 0:
        return TTCSummary(min_ttc=float("inf"), conflict=False, collision=False)
    ego_xy = np.asarray(env._ego_xy, dtype=np.float64).reshape(2)
    ego_vel = np.asarray(env._ego_vel, dtype=np.float64).reshape(2)
    ttcs = [
        _ttc_to_radius(p - ego_xy, v - ego_vel, float(collision_distance_m))
        for p, v in zip(other_pos, other_vel)
    ]
    dists = np.linalg.norm(other_pos - ego_xy[None, :], axis=-1)
    min_ttc = float(min(ttcs)) if ttcs else float("inf")
    collision = bool(np.any(dists <= float(collision_distance_m)))
    conflict = bool(min_ttc < float(conflict_ttc_s))
    return TTCSummary(min_ttc=min_ttc, conflict=conflict, collision=collision)


def _init_policy_and_optimizer(env, cfg: DictPPOConfig, device: torch.device) -> Tuple[DictActorCritic, optim.Optimizer]:
    env_dim = int(_shape_from_space(env.observation_space.spaces["env"])[0])
    hm_shape = tuple(_shape_from_space(env.observation_space.spaces["heatmap"]))
    if len(hm_shape) != 3:
        raise ValueError(f"heatmap space must be (K, H, W); got {hm_shape}")
    action_dim = int(_shape_from_space(env.action_space)[0])

    policy = DictActorCritic(
        env_dim=env_dim,
        action_dim=action_dim,
        heatmap_channels=int(hm_shape[0]),
        env_hidden=int(cfg.env_hidden),
        cnn_hidden=int(cfg.cnn_hidden),
        trunk_hidden=int(cfg.trunk_hidden),
        log_std_init=float(cfg.log_std_init),
    ).to(device)

    actor_params = (
        list(policy.env_branch.parameters())
        + list(policy.cnn_branch.parameters())
        + list(policy.trunk.parameters())
        + list(policy.mu_head.parameters())
        + [policy.log_std]
    )
    critic_params = list(policy.v_head.parameters())
    optimizer = optim.Adam(
        [
            {"params": actor_params, "lr": float(cfg.policy_lr)},
            {"params": critic_params, "lr": float(cfg.value_lr)},
        ]
    )
    return policy, optimizer


def collect_rollout_with_metrics(
    env: RandomScenarioControlEnv,
    policy: DictActorCritic,
    device: torch.device,
    *,
    max_steps: int,
    conflict_ttc_s: float,
    collision_distance_m: float,
    ttc_conflict_penalty: float,
    collision_penalty: float,
    collision_penalty_once: bool,
    terminate_on_collision: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    buf = DictRolloutBuffer()
    obs, _ = env.reset()

    episode_return = 0.0
    episode_env_return = 0.0
    episode_conflicts = 0
    episode_collisions = 0
    episode_had_collision = False
    completed_returns: List[float] = []
    completed_env_returns: List[float] = []
    completed_conflicts: List[int] = []
    completed_collisions: List[int] = []
    step_conflicts = 0
    step_collisions = 0
    penalty_sum = 0.0

    for _ in range(int(max_steps)):
        oe = torch.as_tensor(obs["env"], dtype=torch.float32, device=device).unsqueeze(0)
        oh = torch.as_tensor(obs["heatmap"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logp, value, _mu = policy.act(oe, oh)

        action_np = action.squeeze(0).cpu().numpy()
        obs2, reward, terminated, truncated, info = env.step(action_np)
        done = bool(terminated or truncated)

        ttc = compute_ttc_summary(
            env.current_env,
            t=int(info.get("timestep", 1)) - 1,
            conflict_ttc_s=float(conflict_ttc_s),
            collision_distance_m=float(collision_distance_m),
        )
        conflict_i = int(ttc.conflict)
        collision_i = int(ttc.collision)
        new_collision_i = int(collision_i and not episode_had_collision)
        collision_penalty_i = new_collision_i if collision_penalty_once else collision_i
        ttc_penalty = (
            float(ttc_conflict_penalty) * float(conflict_i)
            + float(collision_penalty) * float(collision_penalty_i)
        )
        shaped_reward = float(reward) - ttc_penalty
        if collision_i:
            episode_had_collision = True
        if bool(terminate_on_collision) and collision_i:
            done = True
        step_conflicts += conflict_i
        step_collisions += collision_i
        episode_conflicts += conflict_i
        episode_collisions += collision_i
        episode_env_return += float(reward)
        episode_return += float(shaped_reward)
        penalty_sum += float(ttc_penalty)

        buf.add(
            obs_env=obs["env"],
            obs_hm=obs["heatmap"],
            action=action_np,
            reward=float(shaped_reward),
            value=float(value.item()),
            log_prob=float(logp.item()),
            done=float(done),
            risk=float(info.get("risk", info.get("pora", 0.0))),
        )

        obs = obs2
        if done:
            completed_returns.append(float(episode_return))
            completed_env_returns.append(float(episode_env_return))
            completed_conflicts.append(int(episode_conflicts))
            completed_collisions.append(int(episode_collisions))
            episode_return = 0.0
            episode_env_return = 0.0
            episode_conflicts = 0
            episode_collisions = 0
            episode_had_collision = False
            obs, _ = env.reset()

    # Include the open episode as a partial epoch signal.
    if episode_return != 0.0 or episode_conflicts != 0 or episode_collisions != 0:
        completed_returns.append(float(episode_return))
        completed_env_returns.append(float(episode_env_return))
        completed_conflicts.append(int(episode_conflicts))
        completed_collisions.append(int(episode_collisions))

    data = buf.finalize(device)
    metrics = {
        "avg_training_pora_risk": float(data["risks"].mean().item()),
        "avg_epoch_conflicts": float(np.mean(completed_conflicts)) if completed_conflicts else 0.0,
        "avg_epoch_collisions": float(np.mean(completed_collisions)) if completed_collisions else 0.0,
        "avg_epoch_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
        "avg_epoch_env_return": float(np.mean(completed_env_returns)) if completed_env_returns else 0.0,
        "avg_ttc_collision_penalty": float(penalty_sum / max(int(max_steps), 1)),
        "rollout_conflict_steps": float(step_conflicts),
        "rollout_collision_steps": float(step_collisions),
    }
    return data, metrics


def ppo_update_with_loss_logging(
    batch: Dict[str, torch.Tensor],
    policy: DictActorCritic,
    optimizer: optim.Optimizer,
    cfg: DictPPOConfig,
) -> Dict[str, float]:
    obs_env = batch["obs_env"]
    obs_hm = batch["obs_hm"]
    actions = batch["actions"]
    old_logp = batch["log_probs"]
    adv = batch["advantages"]
    ret = batch["returns"]

    if cfg.normalize_advantage:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_loss": 0.0,
        "total_ppo_loss": 0.0,
    }
    n_updates = 0
    idx = np.arange(obs_env.shape[0])
    for _ in range(int(cfg.n_epochs)):
        np.random.shuffle(idx)
        for start in range(0, obs_env.shape[0], int(cfg.batch_size)):
            mb = idx[start : start + int(cfg.batch_size)]
            if len(mb) == 0:
                continue
            mb_t = torch.as_tensor(mb, device=obs_env.device, dtype=torch.long)
            oe = obs_env.index_select(0, mb_t)
            oh = obs_hm.index_select(0, mb_t)
            a_old = actions.index_select(0, mb_t)
            ol = old_logp.index_select(0, mb_t)
            adv_b = adv.index_select(0, mb_t)
            ret_b = ret.index_select(0, mb_t)

            mu, std, value = policy(oe, oh)
            dist = Normal(mu, std)
            logp = dist.log_prob(a_old).sum(-1)
            ratio = torch.exp(logp - ol)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((value - ret_b) ** 2).mean()
            entropy = dist.entropy().sum(-1).mean()
            entropy_loss = -float(cfg.entropy_coef) * entropy
            total_loss = policy_loss + float(cfg.value_coef) * value_loss + entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
            optimizer.step()

            sums["policy_loss"] += float(policy_loss.item())
            sums["value_loss"] += float(value_loss.item())
            sums["entropy_loss"] += float(entropy_loss.item())
            sums["total_ppo_loss"] += float(total_loss.item())
            n_updates += 1

    n = max(1, n_updates)
    return {k: v / n for k, v in sums.items()}


def evaluate_policy(
    envs: Sequence[BlankSlateScenarioEnvControl],
    policy: DictActorCritic,
    device: torch.device,
    *,
    conflict_ttc_s: float,
    collision_distance_m: float,
    ttc_conflict_penalty: float,
    collision_penalty: float,
    collision_penalty_once: bool,
    terminate_on_collision: bool,
    deterministic: bool = True,
) -> Dict[str, float]:
    episode_returns: List[float] = []
    episode_env_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_conflicts: List[int] = []
    episode_conflict_rates: List[float] = []
    episode_collisions: List[int] = []
    all_risks: List[float] = []
    all_ttc: List[float] = []

    for env in envs:
        obs, _ = env.reset()
        done = False
        ret = 0.0
        env_ret = 0.0
        length = 0
        conflicts = 0
        collided = False
        episode_had_collision = False
        while not done:
            oe = torch.as_tensor(obs["env"], dtype=torch.float32, device=device).unsqueeze(0)
            oh = torch.as_tensor(obs["heatmap"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, std, _value = policy(oe, oh)
                if deterministic:
                    action = mu
                else:
                    action = Normal(mu, std).sample()

            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            done = bool(terminated or truncated)
            ttc = compute_ttc_summary(
                env,
                t=int(info.get("timestep", 1)) - 1,
                conflict_ttc_s=float(conflict_ttc_s),
                collision_distance_m=float(collision_distance_m),
            )
            conflict_i = int(ttc.conflict)
            collision_i = int(ttc.collision)
            new_collision_i = int(collision_i and not episode_had_collision)
            collision_penalty_i = new_collision_i if collision_penalty_once else collision_i
            ttc_penalty = (
                float(ttc_conflict_penalty) * float(conflict_i)
                + float(collision_penalty) * float(collision_penalty_i)
            )
            env_ret += float(reward)
            ret += float(reward) - ttc_penalty
            length += 1
            conflicts += conflict_i
            collided = bool(collided or collision_i)
            if collision_i:
                episode_had_collision = True
            if bool(terminate_on_collision) and collision_i:
                done = True
            all_risks.append(float(info.get("risk", info.get("pora", 0.0))))
            if np.isfinite(ttc.min_ttc):
                all_ttc.append(float(ttc.min_ttc))

        episode_returns.append(float(ret))
        episode_env_returns.append(float(env_ret))
        episode_lengths.append(int(length))
        episode_conflicts.append(int(conflicts))
        episode_conflict_rates.append(float(conflicts) / max(float(length), 1.0))
        episode_collisions.append(int(collided))

    return {
        "collision_rate": float(np.mean(episode_collisions)) if episode_collisions else 0.0,
        "average_episode_conflicts": float(np.mean(episode_conflicts)) if episode_conflicts else 0.0,
        "average_episode_conflict_rate": (
            float(np.mean(episode_conflict_rates)) if episode_conflict_rates else 0.0
        ),
        "minimum_TTC": float(np.min(all_ttc)) if all_ttc else float("inf"),
        "average_PORA_risk": float(np.mean(all_risks)) if all_risks else 0.0,
        "maximum_PORA_risk": float(np.max(all_risks)) if all_risks else 0.0,
        "evaluation_average_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "evaluation_average_env_return": float(np.mean(episode_env_returns)) if episode_env_returns else 0.0,
        "average_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "average_travel_time": float(np.mean(episode_lengths) * envs[0].cfg.dt) if episode_lengths else 0.0,
        "num_eval_scenarios": float(len(envs)),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenarios", type=str, default="data/av2_mf_tiny/scenarios_rl_moving_av_100.pt")
    p.add_argument("--heatmap_root", type=str, default="data/heatmaps_rl_s100")
    p.add_argument("--map_root", type=str, default="data/av2_mf_tiny/train")
    p.add_argument("--non_av_uid", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario_seed", type=int, default=None)
    p.add_argument("--history_steps", type=int, default=50)

    # PPO.
    p.add_argument("--total_updates", type=int, default=50)
    p.add_argument("--rollout_steps", type=int, default=1024)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--policy_lr", type=float, default=3e-4)
    p.add_argument("--value_lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.05)
    p.add_argument("--env_hidden", type=int, default=128)
    p.add_argument("--cnn_hidden", type=int, default=128)
    p.add_argument("--trunk_hidden", type=int, default=128)
    p.add_argument("--advantage_mode", choices=("gae", "paper"), default="gae")
    p.add_argument("--no_normalize_advantage", action="store_true")
    p.add_argument("--log_std_init", type=float, default=0.0)

    # Env / reward / termination.
    p.add_argument("--lambda_risk", type=float, default=1.0)
    p.add_argument("--base_reward", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--w_forward_progress", type=float, default=1.0)
    p.add_argument("--w_off_map", type=float, default=1.0)
    p.add_argument("--w_back", type=float, default=1.0)
    p.add_argument("--w_smooth", type=float, default=0.05)
    p.add_argument("--w_jerk", type=float, default=0.05)
    p.add_argument("--w_lane_lateral", type=float, default=0.0)
    p.add_argument("--lane_lateral_deadband", type=float, default=1.0)
    p.add_argument("--progress_cap_m", type=float, default=-1.0)
    p.add_argument("--gate_progress_on_map", dest="gate_progress_on_map", action="store_true", default=False)
    p.add_argument("--no_gate_progress_on_map", dest="gate_progress_on_map", action="store_false")
    p.add_argument("--backward_threshold", type=float, default=0.5)
    p.add_argument("--speed_max", type=float, default=40.0)
    p.add_argument("--reward_clip", type=float, default=5.0)
    p.add_argument("--invalid_penalty", type=float, default=-20.0)
    p.add_argument("--terminate_on_invalid", dest="terminate_on_invalid", action="store_true", default=True)
    p.add_argument("--no_terminate_on_invalid", dest="terminate_on_invalid", action="store_false")
    p.add_argument("--dynamic_forward_unit", dest="dynamic_forward_unit", action="store_true", default=False)
    p.add_argument("--no_dynamic_forward_unit", dest="dynamic_forward_unit", action="store_false")
    p.add_argument("--lane_smoothing_k", type=int, default=3)
    p.add_argument("--lane_weight_eps", type=float, default=0.5)
    p.add_argument("--aggregate_heatmaps", dest="aggregate_heatmaps", action="store_true", default=False)
    p.add_argument("--no_aggregate_heatmaps", dest="aggregate_heatmaps", action="store_false")
    p.add_argument("--heatmap_agg", type=str, default="max", choices=["max"])
    p.add_argument("--vehicle_length", type=float, default=5.0)
    p.add_argument("--vehicle_width", type=float, default=2.0)
    p.add_argument("--pora_resolution", type=float, default=0.5)
    p.add_argument("--future_steps", type=int, default=None)
    p.add_argument("--fallback_half_width", type=float, default=8.0)
    p.add_argument("--accel_max_mps2", type=float, default=3.0)
    p.add_argument("--heading_rate_max_radps", type=float, default=0.8)

    # Goal options copied from one-scenario script.
    p.add_argument("--goal_world_x", type=float, default=None)
    p.add_argument("--goal_world_y", type=float, default=None)
    p.add_argument("--goal_from_av_endpoint", action="store_true", default=False)
    p.add_argument("--goal_lane_offset_m", type=float, default=0.0)
    p.add_argument("--w_goal", type=float, default=1.0)
    p.add_argument("--w_heading_goal", type=float, default=0.0)
    p.add_argument("--initial_speed_mps", type=float, default=-1.0)
    p.add_argument("--initial_speed_smoothing_k", type=int, default=3)

    # Metric logging / eval.
    p.add_argument("--eval_every", type=int, default=5, help="Run evaluation every N updates; <=0 disables.")
    p.add_argument("--conflict_ttc_s", type=float, default=2.0)
    p.add_argument(
        "--collision_distance_m",
        type=float,
        default=None,
        help="Distance threshold for TTC/collision. Default = vehicle_length.",
    )
    p.add_argument(
        "--ttc_conflict_penalty",
        type=float,
        default=0.0,
        help="Extra reward penalty per step when TTC < --conflict_ttc_s. Default 0 keeps old experiments unchanged.",
    )
    p.add_argument(
        "--collision_penalty",
        type=float,
        default=0.0,
        help="Extra reward penalty per step when distance <= --collision_distance_m. Default 0 keeps old experiments unchanged.",
    )
    p.add_argument(
        "--collision_penalty_once",
        action="store_true",
        default=False,
        help="Apply --collision_penalty only on the first collision step of each episode.",
    )
    p.add_argument(
        "--terminate_on_collision",
        action="store_true",
        default=False,
        help="End the current PPO episode as soon as distance <= --collision_distance_m.",
    )
    p.add_argument("--save_policy", type=str, default=None)
    p.add_argument("--save_history", type=str, default="runs/ppo_multi_scenario_accel.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    collision_distance_m = (
        float(args.vehicle_length)
        if args.collision_distance_m is None
        else float(args.collision_distance_m)
    )

    scenario_envs = _build_scenario_envs(args)
    train_env = RandomScenarioControlEnv(
        scenario_envs,
        seed=args.scenario_seed if args.scenario_seed is not None else args.seed,
    )

    ppo_cfg = DictPPOConfig(
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_eps=float(args.clip_eps),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        policy_lr=float(args.policy_lr),
        value_lr=float(args.value_lr),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        rollout_steps=int(args.rollout_steps),
        env_hidden=int(args.env_hidden),
        cnn_hidden=int(args.cnn_hidden),
        trunk_hidden=int(args.trunk_hidden),
        advantage_mode=str(args.advantage_mode),
        normalize_advantage=not args.no_normalize_advantage,
        log_std_init=float(args.log_std_init),
    )

    policy, optimizer = _init_policy_and_optimizer(train_env, ppo_cfg, device)
    history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []

    for update in range(int(args.total_updates)):
        rollout, train_metrics = collect_rollout_with_metrics(
            train_env,
            policy,
            device,
            max_steps=int(args.rollout_steps),
            conflict_ttc_s=float(args.conflict_ttc_s),
            collision_distance_m=collision_distance_m,
            ttc_conflict_penalty=float(args.ttc_conflict_penalty),
            collision_penalty=float(args.collision_penalty),
            collision_penalty_once=bool(args.collision_penalty_once),
            terminate_on_collision=bool(args.terminate_on_collision),
        )
        advantages, returns = compute_dict_advantages(
            ppo_cfg,
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
        )
        batch = {
            "obs_env": rollout["obs_env"],
            "obs_hm": rollout["obs_hm"],
            "actions": rollout["actions"],
            "log_probs": rollout["log_probs"],
            "advantages": advantages,
            "returns": returns,
        }
        loss_metrics = ppo_update_with_loss_logging(batch, policy, optimizer, ppo_cfg)
        row = {
            "update": float(update + 1),
            **loss_metrics,
            **train_metrics,
        }
        history.append(row)
        print(
            f"update {update + 1}/{args.total_updates} "
            f"policy_loss={row['policy_loss']:.4f} "
            f"value_loss={row['value_loss']:.4f} "
            f"entropy_loss={row['entropy_loss']:.4f} "
            f"total_ppo_loss={row['total_ppo_loss']:.4f} "
            f"avg_training_pora_risk={row['avg_training_pora_risk']:.4f} "
            f"avg_epoch_conflicts={row['avg_epoch_conflicts']:.2f} "
            f"avg_epoch_collisions={row['avg_epoch_collisions']:.2f} "
            f"avg_ttc_collision_penalty={row['avg_ttc_collision_penalty']:.3f} "
            f"avg_epoch_return={row['avg_epoch_return']:.2f}",
            flush=True,
        )

        if int(args.eval_every) > 0 and ((update + 1) % int(args.eval_every) == 0):
            eval_metrics = evaluate_policy(
                scenario_envs,
                policy,
                device,
                conflict_ttc_s=float(args.conflict_ttc_s),
                collision_distance_m=collision_distance_m,
                ttc_conflict_penalty=float(args.ttc_conflict_penalty),
                collision_penalty=float(args.collision_penalty),
                collision_penalty_once=bool(args.collision_penalty_once),
                terminate_on_collision=bool(args.terminate_on_collision),
                deterministic=True,
            )
            eval_row = {"update": float(update + 1), **eval_metrics}
            eval_history.append(eval_row)
            print(
                f"eval update {update + 1}: "
                f"collision_rate={eval_metrics['collision_rate']:.3f} "
                f"average_episode_conflicts={eval_metrics['average_episode_conflicts']:.2f} "
                f"minimum_TTC={eval_metrics['minimum_TTC']:.3f} "
                f"average_PORA_risk={eval_metrics['average_PORA_risk']:.4f} "
                f"maximum_PORA_risk={eval_metrics['maximum_PORA_risk']:.4f} "
                f"evaluation_average_return={eval_metrics['evaluation_average_return']:.2f} "
                f"evaluation_average_env_return={eval_metrics['evaluation_average_env_return']:.2f} "
                f"average_episode_length={eval_metrics['average_episode_length']:.2f}",
                flush=True,
            )

    if args.save_policy:
        out = Path(args.save_policy)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), str(out))
        print(f"[train_ppo_multi_scenario_accel] saved policy to {out}")

    if args.save_history:
        out = Path(args.save_history)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(
                {
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "num_scenarios": len(scenario_envs),
                    "collision_distance_m": collision_distance_m,
                    "ttc_conflict_penalty": float(args.ttc_conflict_penalty),
                    "collision_penalty": float(args.collision_penalty),
                    "collision_penalty_once": bool(args.collision_penalty_once),
                    "terminate_on_collision": bool(args.terminate_on_collision),
                    "training_history": history,
                    "evaluation_history": eval_history,
                },
                f,
                indent=2,
            )
        print(f"[train_ppo_multi_scenario_accel] saved history to {out}")


if __name__ == "__main__":
    main()
