"""
PPO for HeatmapTrajectoryEnv — separate policy & value nets (see PORA_PPO_implementation_guide.md §9).

Supports:
  - clipped surrogate + value MSE L_V = mean((V - R)^2)  (guide §4.4–4.5)
  - advantage: GAE (default) or paper-style Â = R - V with Monte Carlo returns (guide §4.2–4.3)
  - distinct policy_lr / value_lr (guide §11)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

AdvantageMode = Literal["gae", "paper"]


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    rollout_steps: int = 2048
    hidden_dim: int = 128
    advantage_mode: AdvantageMode = "gae"
    normalize_advantage: bool = True


class PolicyNet(nn.Module):
    """Gaussian policy π_θ(a|o)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        return mu, std


class ValueNet(nn.Module):
    """State-value V_φ(o) for advantages and L_V."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def collect_rollout(
    env,
    policy: PolicyNet,
    value_net: ValueNet,
    device: torch.device,
    max_steps: int,
) -> Dict[str, torch.Tensor]:
    obs_list, act_list, rew_list, val_list, logp_list, done_list, risk_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    o, _ = env.reset()
    for _ in range(max_steps):
        obs_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, std = policy(obs_t)
            v = value_net(obs_t)
            dist = Normal(mu, std)
            a = dist.sample()
            logp = dist.log_prob(a).sum(-1)

        a_np = a.squeeze(0).cpu().numpy()
        o2, r, term, trunc, info = env.step(a_np)
        d = term or trunc

        obs_list.append(o)
        act_list.append(a_np)
        rew_list.append(r)
        val_list.append(v.item())
        logp_list.append(logp.item())
        done_list.append(float(d))
        risk_list.append(float(info.get("risk", info.get("pora", 0.0))))

        o = o2
        if d:
            o, _ = env.reset()

    return {
        "obs": torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.array(act_list), dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(rew_list, dtype=torch.float32, device=device),
        "values": torch.as_tensor(val_list, dtype=torch.float32, device=device),
        "log_probs": torch.as_tensor(logp_list, dtype=torch.float32, device=device),
        "dones": torch.as_tensor(done_list, dtype=torch.float32, device=device),
        "risk": torch.as_tensor(risk_list, dtype=torch.float32, device=device),
    }


def compute_mc_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Discounted rewards-to-go R_{t_k} = sum_{t'>=t} gamma^{t'-t} r_{t'} (guide §4.2)."""
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    G = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G * (1.0 - dones[t])
        returns[t] = G
    return returns


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    adv = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        next_value = values[t]
    ret = adv + values
    return adv, ret


def compute_advantages(
    cfg: PPOConfig,
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


def ppo_update(
    batch: Dict[str, torch.Tensor],
    policy: PolicyNet,
    value_net: ValueNet,
    policy_opt: optim.Optimizer,
    value_opt: optim.Optimizer,
    cfg: PPOConfig,
) -> Dict[str, float]:
    obs = batch["obs"]
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

    idx = np.arange(obs.shape[0])
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, obs.shape[0], cfg.batch_size):
            end = start + cfg.batch_size
            mb = idx[start:end]
            if len(mb) == 0:
                continue

            o = obs[mb]
            a_old = actions[mb]
            ol = old_logp[mb]
            adv_b = adv[mb]
            ret_b = ret[mb]

            mu, std = policy(o)
            dist = Normal(mu, std)
            logp = dist.log_prob(a_old).sum(-1)
            ratio = torch.exp(logp - ol)

            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
            pi_loss = -torch.min(surr1, surr2).mean()

            v = value_net(o)
            v_loss = 0.5 * ((v - ret_b) ** 2).mean()
            ent = dist.entropy().sum(-1).mean()

            policy_opt.zero_grad()
            (pi_loss - cfg.entropy_coef * ent).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            policy_opt.step()

            value_opt.zero_grad()
            (cfg.value_coef * v_loss).backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), cfg.max_grad_norm)
            value_opt.step()

            total_pi_loss += pi_loss.item()
            total_v_loss += v_loss.item()
            total_ent += ent.item()
            n_updates += 1

    n = max(n_updates, 1)
    return {
        "pi_loss": total_pi_loss / n,
        "v_loss": total_v_loss / n,
        "entropy": total_ent / n,
    }


def train_ppo(
    env,
    cfg: PPOConfig,
    total_updates: int,
    device: torch.device,
    seed: int = 0,
) -> List[Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = PolicyNet(obs_dim, action_dim, hidden=cfg.hidden_dim).to(device)
    value_net = ValueNet(obs_dim, hidden=cfg.hidden_dim).to(device)
    policy_opt = optim.Adam(policy.parameters(), lr=cfg.policy_lr)
    value_opt = optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    history: List[Dict[str, float]] = []
    for _ in range(total_updates):
        data = collect_rollout(env, policy, value_net, device, cfg.rollout_steps)
        adv, ret = compute_advantages(
            cfg,
            data["rewards"],
            data["values"],
            data["dones"],
        )
        batch = {
            "obs": data["obs"],
            "actions": data["actions"],
            "log_probs": data["log_probs"],
            "advantages": adv,
            "returns": ret,
        }
        stats = ppo_update(batch, policy, value_net, policy_opt, value_opt, cfg)
        stats["mean_reward"] = float(data["rewards"].mean().item())
        rsk = data["risk"]
        stats["mean_risk"] = float(rsk.mean().item())
        stats["max_risk"] = float(rsk.max().item())
        stats["mean_pora"] = stats["mean_risk"]
        history.append(stats)
    return history
