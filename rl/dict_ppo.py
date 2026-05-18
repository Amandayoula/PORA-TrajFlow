"""Dict-obs (env MLP + heatmap CNN) PPO.

The existing ``rl/ppo.py`` operates on flat vector observations; this module is
a self-contained parallel implementation that consumes the dict observation
produced by ``rl/scenario_env.PPOScenarioEnv``::

    obs = {"env": (D,), "heatmap": (K, H, W)}

Structure mirrors ``rl/ppo.py`` (rollout -> GAE -> clipped PPO update -> outer
loop) so operators familiar with it can read this one top-to-bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rl.ppo import compute_gae, compute_mc_returns


@dataclass
class DictPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    # value_coef default is deliberately small. Our shared-trunk actor-critic puts the value
    # loss gradient through the same feature extractor as the policy, so a large value
    # target (even after reward clipping) would dominate the update and kill policy
    # learning. 0.05 matches the scale-balanced update in rl/ppo.py on clipped rewards.
    value_coef: float = 0.05
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    policy_lr: float = 3e-4
    value_lr: float = 1e-3  # critic head uses 3× actor lr to converge faster
    n_epochs: int = 10
    batch_size: int = 64
    rollout_steps: int = 2048
    env_hidden: int = 128
    cnn_hidden: int = 128
    trunk_hidden: int = 128
    advantage_mode: str = "gae"   # "gae" | "paper"
    normalize_advantage: bool = True
    log_std_init: float = 0.0     # log of initial action std; 0 -> std=1, -0.7 -> std~=0.5


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class EnvMLP(nn.Module):
    """MLP branch for scalar env state."""

    def __init__(self, in_dim: int, hidden: int = 128, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeatmapCNN(nn.Module):
    """3-layer Conv2d + AdaptiveAvgPool -> flat embedding for the heatmap branch."""

    def __init__(self, in_channels: int, hidden: int = 128, out: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        z = z.flatten(1)
        return torch.tanh(self.proj(z))


class DictActorCritic(nn.Module):
    """Shared backbone -> (mu, log_std) actor head + value head."""

    def __init__(
        self,
        env_dim: int,
        action_dim: int,
        heatmap_channels: int,
        *,
        env_hidden: int = 128,
        cnn_hidden: int = 128,
        trunk_hidden: int = 128,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.env_branch = EnvMLP(env_dim, hidden=env_hidden, out=env_hidden)
        self.cnn_branch = HeatmapCNN(heatmap_channels, hidden=cnn_hidden, out=cnn_hidden)
        self.trunk = nn.Sequential(
            nn.Linear(env_hidden + cnn_hidden, trunk_hidden),
            nn.Tanh(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(trunk_hidden, action_dim)
        self.v_head = nn.Linear(trunk_hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def forward(
        self, obs_env: torch.Tensor, obs_heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_env = self.env_branch(obs_env)
        f_hm = self.cnn_branch(obs_heatmap)
        h = self.trunk(torch.cat([f_env, f_hm], dim=-1))
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        v = self.v_head(h).squeeze(-1)
        return mu, std, v

    def act(
        self, obs_env: torch.Tensor, obs_heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, v = self(obs_env, obs_heatmap)
        dist = Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a, logp, v, mu


# ---------------------------------------------------------------------------
# Rollout buffer + collector
# ---------------------------------------------------------------------------


class DictRolloutBuffer:
    """Stores one rollout worth of transitions in torch tensors on CPU.

    Fields (all length T after ``finalize``):
      obs_env    : (T, D)
      obs_hm     : (T, K, H, W)
      actions    : (T, A)
      rewards    : (T,)
      values     : (T,)
      log_probs  : (T,)
      dones      : (T,)
      risks      : (T,)
    """

    def __init__(self):
        self.obs_env: List[np.ndarray] = []
        self.obs_hm: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[float] = []
        self.risks: List[float] = []

    def add(
        self,
        obs_env: np.ndarray,
        obs_hm: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: float,
        risk: float,
    ) -> None:
        self.obs_env.append(np.asarray(obs_env, dtype=np.float32))
        self.obs_hm.append(np.asarray(obs_hm, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(float(done))
        self.risks.append(float(risk))

    def finalize(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "obs_env": torch.as_tensor(np.stack(self.obs_env, axis=0), dtype=torch.float32, device=device),
            "obs_hm": torch.as_tensor(np.stack(self.obs_hm, axis=0), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.stack(self.actions, axis=0), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            "values": torch.as_tensor(self.values, dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones, dtype=torch.float32, device=device),
            "risks": torch.as_tensor(self.risks, dtype=torch.float32, device=device),
        }


def collect_rollout_dict(
    env,
    policy: DictActorCritic,
    device: torch.device,
    max_steps: int,
) -> Dict[str, torch.Tensor]:
    """Run policy for ``max_steps`` environment steps, resetting on episode done."""
    buf = DictRolloutBuffer()
    obs, _ = env.reset()
    for _ in range(max_steps):
        oe = torch.as_tensor(obs["env"], dtype=torch.float32, device=device).unsqueeze(0)
        oh = torch.as_tensor(obs["heatmap"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, logp, v, _ = policy.act(oe, oh)

        a_np = a.squeeze(0).cpu().numpy()
        logp_val = float(logp.item())
        v_val = float(v.item())

        obs2, r, term, trunc, info = env.step(a_np)
        done = bool(term or trunc)

        buf.add(
            obs_env=obs["env"],
            obs_hm=obs["heatmap"],
            action=a_np,
            reward=float(r),
            value=v_val,
            log_prob=logp_val,
            done=float(done),
            risk=float(info.get("risk", info.get("pora", 0.0))),
        )

        obs = obs2
        if done:
            obs, _ = env.reset()

    return buf.finalize(device)


# ---------------------------------------------------------------------------
# Advantages + update
# ---------------------------------------------------------------------------


def compute_dict_advantages(
    cfg: DictPPOConfig,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cfg.advantage_mode == "paper":
        returns = compute_mc_returns(rewards, dones, cfg.gamma)
        adv = returns - values
        return adv, returns
    if cfg.advantage_mode == "gae":
        return compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
    raise ValueError(cfg.advantage_mode)


def ppo_update_dict(
    batch: Dict[str, torch.Tensor],
    policy: DictActorCritic,
    optimizer: optim.Optimizer,
    cfg: DictPPOConfig,
) -> Dict[str, float]:
    """Single-optimizer update: backbone is shared so policy/value losses must
    go through one optimizer step to keep the trunk gradients consistent."""
    obs_env = batch["obs_env"]
    obs_hm = batch["obs_hm"]
    actions = batch["actions"]
    old_logp = batch["log_probs"]
    adv = batch["advantages"]
    ret = batch["returns"]

    if cfg.normalize_advantage:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total_pi_loss = 0.0
    total_v_loss = 0.0
    total_ent = 0.0
    n_updates = 0

    idx = np.arange(obs_env.shape[0])
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, obs_env.shape[0], cfg.batch_size):
            end = start + cfg.batch_size
            mb = idx[start:end]
            if len(mb) == 0:
                continue
            mb_t = torch.as_tensor(mb, device=obs_env.device, dtype=torch.long)

            oe = obs_env.index_select(0, mb_t)
            oh = obs_hm.index_select(0, mb_t)
            a_old = actions.index_select(0, mb_t)
            ol = old_logp.index_select(0, mb_t)
            adv_b = adv.index_select(0, mb_t)
            ret_b = ret.index_select(0, mb_t)

            mu, std, v = policy(oe, oh)
            dist = Normal(mu, std)
            logp = dist.log_prob(a_old).sum(-1)
            ratio = torch.exp(logp - ol)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
            pi_loss = -torch.min(surr1, surr2).mean()
            ent = dist.entropy().sum(-1).mean()
            v_loss = 0.5 * ((v - ret_b) ** 2).mean()

            loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_pi_loss += float(pi_loss.item())
            total_v_loss += float(v_loss.item())
            total_ent += float(ent.item())
            n_updates += 1

    n = max(n_updates, 1)
    return {
        "pi_loss": total_pi_loss / n,
        "v_loss": total_v_loss / n,
        "entropy": total_ent / n,
    }


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------


def _shape_from_space(sp) -> Tuple[int, ...]:
    if hasattr(sp, "shape"):
        return tuple(sp.shape)
    raise TypeError(f"Unsupported space: {sp!r}")


def train_ppo_dict(
    env,
    cfg: DictPPOConfig,
    total_updates: int,
    device: torch.device,
    seed: int = 0,
    on_update: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Tuple[DictActorCritic, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_space = env.observation_space.spaces["env"]
    hm_space = env.observation_space.spaces["heatmap"]
    env_dim = int(_shape_from_space(env_space)[0])
    hm_shape = tuple(_shape_from_space(hm_space))
    if len(hm_shape) != 3:
        raise ValueError(f"heatmap space must be (K, H, W); got {hm_shape}")
    K, _H, _W = hm_shape
    action_dim = int(_shape_from_space(env.action_space)[0])

    policy = DictActorCritic(
        env_dim=env_dim,
        action_dim=action_dim,
        heatmap_channels=K,
        env_hidden=cfg.env_hidden,
        cnn_hidden=cfg.cnn_hidden,
        trunk_hidden=cfg.trunk_hidden,
        log_std_init=cfg.log_std_init,
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
            {"params": actor_params, "lr": cfg.policy_lr},
            {"params": critic_params, "lr": cfg.value_lr},
        ]
    )

    history: List[Dict[str, float]] = []
    for update in range(total_updates):
        data = collect_rollout_dict(env, policy, device, cfg.rollout_steps)
        adv, ret = compute_dict_advantages(cfg, data["rewards"], data["values"], data["dones"])
        batch = {
            "obs_env": data["obs_env"],
            "obs_hm": data["obs_hm"],
            "actions": data["actions"],
            "log_probs": data["log_probs"],
            "advantages": adv,
            "returns": ret,
        }
        stats = ppo_update_dict(batch, policy, optimizer, cfg)
        stats["mean_reward"] = float(data["rewards"].mean().item())
        stats["mean_risk"] = float(data["risks"].mean().item())
        stats["max_risk"] = float(data["risks"].max().item())
        history.append(stats)
        if on_update is not None:
            on_update(update, stats)

    return policy, history
