import os
import json
import subprocess
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

matplotlib.use('Agg')

FUTURE_STEPS = 60  # AV2: timestep 50-109


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_maps(map_root):
    """
    Scan map_root/train/*/log_map_archive_*.json and merge all map elements
    into a single dict for rendering.

    Returns
    -------
    dict with keys:
        'drivable_areas'      : list of (N, 2) numpy arrays  – polygon vertices
        'lane_centerlines'    : list of (N, 2) numpy arrays  – polylines
        'lane_left_bounds'    : list of (N, 2) numpy arrays
        'lane_right_bounds'   : list of (N, 2) numpy arrays
        'pedestrian_crossings': list of dict {'edge1': (2,2), 'edge2': (2,2)}
    """
    drivable_areas      = []
    lane_centerlines    = []
    lane_left_bounds    = []
    lane_right_bounds   = []
    pedestrian_crossings = []

    train_dir = os.path.join(map_root, 'train')
    if not os.path.isdir(train_dir):
        print(f"[visualize_av2] map_root/train not found: {train_dir}; rendering without map.")
        return _empty_map()

    for scenario_id in sorted(os.listdir(train_dir)):
        scenario_dir = os.path.join(train_dir, scenario_id)
        if not os.path.isdir(scenario_dir):
            continue

        json_files = [f for f in os.listdir(scenario_dir) if f.startswith('log_map_archive') and f.endswith('.json')]
        for jf in json_files:
            with open(os.path.join(scenario_dir, jf)) as f:
                m = json.load(f)

            for entry in m.get('drivable_areas', {}).values():
                pts = _pts(entry['area_boundary'])
                if pts is not None:
                    drivable_areas.append(pts)

            for entry in m.get('lane_segments', {}).values():
                cl = _pts(entry.get('centerline', []))
                if cl is not None:
                    lane_centerlines.append(cl)
                lb = _pts(entry.get('left_lane_boundary', []))
                if lb is not None:
                    lane_left_bounds.append(lb)
                rb = _pts(entry.get('right_lane_boundary', []))
                if rb is not None:
                    lane_right_bounds.append(rb)

            for entry in m.get('pedestrian_crossings', {}).values():
                e1 = _pts(entry.get('edge1', []))
                e2 = _pts(entry.get('edge2', []))
                if e1 is not None and e2 is not None:
                    pedestrian_crossings.append({'edge1': e1, 'edge2': e2})

    return {
        'drivable_areas':       drivable_areas,
        'lane_centerlines':     lane_centerlines,
        'lane_left_bounds':     lane_left_bounds,
        'lane_right_bounds':    lane_right_bounds,
        'pedestrian_crossings': pedestrian_crossings,
    }


def _pts(point_list):
    """Convert [{x, y, z}, ...] to (N, 2) float32 array; return None if empty."""
    if not point_list:
        return None
    arr = np.array([[p['x'], p['y']] for p in point_list], dtype=np.float32)
    return arr if len(arr) >= 2 else None


def _empty_map():
    return {k: [] for k in ('drivable_areas', 'lane_centerlines',
                             'lane_left_bounds', 'lane_right_bounds',
                             'pedestrian_crossings')}


# ---------------------------------------------------------------------------
# Model inference on dense grid
# ---------------------------------------------------------------------------

def compute_pzt1(model, input, features, grid):
    """
    Evaluate the model's predicted density over a dense 2-D grid.

    grid  : (steps*steps, 2)  normalised [0,1] positions
    Returns pz_t1 : (steps*steps, FUTURE_STEPS) probability tensor
    """
    with torch.no_grad():
        batch_size = 500

        embedding = model._embedding(input, features)
        embedding = embedding.repeat(batch_size, 1)

        pz_t1 = []
        for grid_batch in grid.split(batch_size, dim=0):
            grid_batch = grid_batch.unsqueeze(1).expand(-1, FUTURE_STEPS, -1)
            z_t0, delta_logpz = model.flow(grid_batch, embedding)
            _, logpz_t1 = model.log_prob(z_t0, delta_logpz)
            pz_t1.append(logpz_t1.exp())

        return torch.cat(pz_t1, dim=0)


# ---------------------------------------------------------------------------
# Frame / video generation
# ---------------------------------------------------------------------------

def _draw_map(ax, map_data, simple):
    """Render map elements onto a matplotlib Axes in real-world (x, -y) space."""
    # Drivable area – light gray fill
    for poly_pts in map_data['drivable_areas']:
        patch = Polygon(np.c_[poly_pts[:, 0], -poly_pts[:, 1]], closed=True,
                        facecolor='#d9d9d9', edgecolor='none', alpha=0.5, zorder=1)
        ax.add_patch(patch)

    # Lane left / right boundaries – thin mid-gray lines
    for pts in map_data['lane_left_bounds']:
        ax.plot(pts[:, 0], -pts[:, 1], color='#888888', linewidth=0.6,
                alpha=0.6, zorder=2)
    for pts in map_data['lane_right_bounds']:
        ax.plot(pts[:, 0], -pts[:, 1], color='#888888', linewidth=0.6,
                alpha=0.6, zorder=2)

    # Lane centerlines – slightly darker
    for pts in map_data['lane_centerlines']:
        ax.plot(pts[:, 0], -pts[:, 1], color='#555555', linewidth=0.8,
                linestyle='--', alpha=0.5, dashes=(4, 4), zorder=3)

    # Pedestrian crossings – yellow dashed
    for pc in map_data['pedestrian_crossings']:
        for edge_key in ('edge1', 'edge2'):
            pts = pc[edge_key]
            ax.plot(pts[:, 0], -pts[:, 1], color='#F5A623', linewidth=1.2,
                    linestyle='--', alpha=0.7, zorder=3)


def _make_dir(directory):
    try:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(directory)
        os.makedirs(directory)
    except OSError as e:
        print(f"[visualize_av2] Error preparing directory {directory}: {e}")


def generate_frame(map_data, x, y, likelihood,
                   observed_traj, unobserved_traj,
                   x_lim, y_lim, t, output_dir, simple):
    frame_path = os.path.join(output_dir, f'frame_{t:03d}.png')
    if os.path.exists(frame_path):
        os.remove(frame_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    if simple:
        ax.axis('off')

    # 1. Map background
    _draw_map(ax, map_data, simple)

    # 2. Density heatmap
    color_map = plt.cm.viridis
    color_map.set_bad(color='none')
    heat_map = ax.pcolormesh(x, y, likelihood, shading='auto',
                             cmap=color_map, alpha=0.55, vmin=0, vmax=1, zorder=4)
    if not simple:
        plt.colorbar(heat_map, ax=ax, label='Likelihood')

    # 3. Observed trajectory
    label_obs = None if simple else 'Observed'
    ax.plot(observed_traj[:, 0], observed_traj[:, 1],
            color='#5DA5DA', linewidth=1.8, zorder=5, label=label_obs)

    # 4. True future trajectory up to current timestep
    label_fut = None if simple else 'Ground Truth Future'
    ax.plot(unobserved_traj[:t + 1, 0], unobserved_traj[:t + 1, 1],
            color='#E69F00', linewidth=1.8, zorder=5, label=label_fut)

    if not simple:
        ax.set_title(f'Predicted Density  –  t = {t}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(frame_path, dpi=150)
    plt.close(fig)


def generate_video(map_data, grid, pz_t1, prob_threshold,
                   observed_traj, unobserved_traj,
                   x_lim, y_lim, steps, output_dir, i, simple):
    frames_dir = os.path.join(output_dir, 'frames', f'video{i}')
    _make_dir(frames_dir)

    x = grid[:, 0].reshape(steps, steps)
    y = -grid[:, 1].reshape(steps, steps)

    for t in range(FUTURE_STEPS):
        likelihood = pz_t1[:, t].cpu().numpy().reshape(steps, steps)
        likelihood = likelihood / (np.max(likelihood) + 1e-8)
        likelihood = np.where(likelihood < prob_threshold, np.nan, likelihood)
        generate_frame(map_data, x, y, likelihood,
                       observed_traj, unobserved_traj,
                       x_lim, y_lim, t, frames_dir, simple)

    frame_source      = os.path.join(frames_dir, 'frame_%03d.png')
    video_destination = os.path.join(output_dir, f'video{i}.mp4')
    subprocess.run(
        ['ffmpeg', '-y', '-r', '10', '-i', frame_source,
         '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', video_destination],
        check=True
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def visualize_av2(observation_site, model, map_root,
                  num_samples, steps, prob_threshold,
                  output_dir, simple, device):
    """
    Parameters
    ----------
    observation_site : AV2ObservationSite
    model            : TrajFlow model (already loaded with weights)
    map_root         : path to the AV2 root dir (contains train/<scenario_id>/log_map_archive_*.json)
    num_samples      : number of test trajectories to visualise
    steps            : grid resolution per axis (e.g. 200); total grid = steps*steps points
    prob_threshold   : mask out densities below this fraction of the max
    output_dir       : where to write video*.mp4 files
    simple           : if True, hide axes/labels/legend
    device           : 'cuda' or 'cpu'
    """
    _make_dir(output_dir)
    model.eval()

    # Load all map geometry
    map_data = load_maps(map_root)

    # Dense evaluation grid in normalised [0,1] space
    linspace = torch.linspace(0, 1, steps)
    gx, gy = torch.meshgrid(linspace, linspace)
    grid = torch.stack((gx.flatten(), gy.flatten()), dim=-1).to(device)

    # Denormalise grid for display
    denorm_grid = observation_site.denormalize(grid.cpu().numpy())  # real-world x/y
    x_lim = (denorm_grid[:, 0].min(), denorm_grid[:, 0].max())
    y_lim = (-denorm_grid[:, 1].max(), -denorm_grid[:, 1].min())   # flip y

    for i in range(num_samples):
        inp, feat, target = next(iter(observation_site.test_loader))
        inp    = inp.to(device)
        feat   = feat.to(device)
        target = target.to(device)

        pz_t1 = compute_pzt1(model, inp, feat, grid)

        # Denormalise trajectories and flip y for display
        obs_xy = observation_site.denormalize(inp[0].cpu().numpy())
        obs_xy = np.stack([obs_xy[:, 0], -obs_xy[:, 1]], axis=-1)

        fut_xy = observation_site.denormalize(target[0].cpu().numpy())
        fut_xy = np.stack([fut_xy[:, 0], -fut_xy[:, 1]], axis=-1)

        generate_video(map_data, denorm_grid, pz_t1, prob_threshold,
                       obs_xy, fut_xy,
                       x_lim, y_lim, steps, output_dir, i, simple)

        print(f'[visualize_av2] video {i + 1}/{num_samples} done → {output_dir}/video{i}.mp4')
