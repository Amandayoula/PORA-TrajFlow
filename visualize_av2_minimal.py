
"""
Minimal AV2 visualization export for PORA_TrajFlow.ipynb

This module exports ONLY the files that the notebook expects:
  - denormalized_grid_1.csv
  - pz_t1_{i}.csv
  - observed_traj_{i}.csv
  - unobserved_traj_{i}.csv

No matplotlib frames, no ffmpeg, no videos.

Assumptions:
  - observation_site provides:
      * test_loader yielding (input, features, target)
      * denormalize(np_array) -> real-world x/y
  - model provides:
      * _embedding(input, features)
      * flow(z, embedding) -> (z_t0, delta_logpz)
      * log_prob(z_t0, delta_logpz) -> (_, logpz_t1)
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Match your project setting. Notebook reshapes with its own "steps" anyway.
FUTURE_STEPS = 60  # AV2: often timestep 50-109


def compute_pzt1(model, input_tensor, features, grid_norm01, future_steps: int = FUTURE_STEPS,
                 batch_size: int = 500):
    """
    Evaluate predicted density over a dense 2D grid.

    Parameters
    ----------
    grid_norm01: (steps*steps, 2) in normalized [0,1] space
    Returns
    -------
    pz_t1: (steps*steps, future_steps) torch tensor on same device as grid_norm01
    """
    model.eval()
    with torch.inference_mode():
        embedding = model._embedding(input_tensor, features)

        # Important: use ONE embedding per grid point batch
        # Your original code repeats to match grid_batch size.
        pz_t1_chunks = []
        for grid_batch in grid_norm01.split(batch_size, dim=0):
            emb = embedding.repeat(grid_batch.shape[0], 1)
            grid_batch = grid_batch.unsqueeze(1).expand(-1, future_steps, -1)
            z_t0, delta_logpz = model.flow(grid_batch, emb)
            _, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            pz_t1_chunks.append(logpz_t1.exp())

        return torch.cat(pz_t1_chunks, dim=0)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_grid_csv(denorm_grid_xy: np.ndarray, output_dir: str, filename: str = "denormalized_grid_1.csv"):
    """
    Save denormalized grid as CSV with columns x,y.
    """
    df = pd.DataFrame({"x": denorm_grid_xy[:, 0], "y": denorm_grid_xy[:, 1]})
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path


def _save_traj_csv(xy: np.ndarray, output_dir: str, filename: str):
    """
    Save trajectory as CSV with columns x,y.
    """
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path


def _save_pzt1_csv(pz_t1: torch.Tensor, output_dir: str, filename: str):
    """
    Save p(z_{t+1}) as CSV with a header row (0..T-1), because notebook uses pd.read_csv() (default header=0).
    """
    arr = pz_t1.detach().cpu().numpy()
    df = pd.DataFrame(arr)  # columns auto: 0..T-1
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path


def export_for_notebook(
    observation_site,
    model,
    num_samples: int,
    steps: int,
    output_dir: str = "visualization",
    device: str = "cpu",
    prob_threshold: Optional[float] = None,
):
    """
    Export minimal files required by PORA_TrajFlow.ipynb.

    Parameters
    ----------
    steps: grid resolution per axis. Total grid points = steps*steps.
           NOTE: Must match the notebook's `steps` when it reshapes pz_t1[:, t] into (steps, steps).
    prob_threshold: kept only for API compatibility; not used here.
    """
    _ensure_dir(output_dir)
    model.eval()

    # 1) Build dense grid in normalized [0,1] space
    lin = torch.linspace(0, 1, steps, device=device)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack((gx.flatten(), gy.flatten()), dim=-1)  # (steps*steps, 2)

    # 2) Denormalize grid once and save
    denorm_grid = observation_site.denormalize(grid.detach().cpu().numpy())
    _save_grid_csv(denorm_grid, output_dir, "denormalized_grid_1.csv")

    # 3) Iterate samples WITHOUT recreating the iterator each time
    it = iter(observation_site.test_loader)

    for i in range(num_samples):
        inp, feat, target = next(it)
        inp = inp.to(device)
        feat = feat.to(device)
        target = target.to(device)

        # 3a) model density on grid
        pz_t1 = compute_pzt1(model, inp, feat, grid, future_steps=FUTURE_STEPS)

        # 3b) denormalize trajectories (keep raw x,y; notebook flips y itself when needed)
        obs_xy = observation_site.denormalize(inp[0].detach().cpu().numpy())
        fut_xy = observation_site.denormalize(target[0].detach().cpu().numpy())

        # 3c) write outputs
        _save_pzt1_csv(pz_t1, output_dir, f"pz_t1_{i}.csv")
        _save_traj_csv(obs_xy, output_dir, f"observed_traj_{i}.csv")
        _save_traj_csv(fut_xy, output_dir, f"unobserved_traj_{i}.csv")

        print(f"[export_for_notebook] wrote sample {i}: pz_t1_{i}.csv, observed_traj_{i}.csv, unobserved_traj_{i}.csv")

    return output_dir
