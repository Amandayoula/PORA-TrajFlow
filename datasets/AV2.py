import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


HISTORY_STEPS = 50   # timestep 0-49,  observed=True
FUTURE_STEPS  = 60   # timestep 50-109, observed=False
TOTAL_STEPS   = HISTORY_STEPS + FUTURE_STEPS  # 110
DT            = 0.1  # 10 Hz


def normalize(data, boundaries):
    return (data - boundaries[:, 0]) / (boundaries[:, 1] - boundaries[:, 0])


def denormalize(data, boundaries):
    return (data * (boundaries[:, 1] - boundaries[:, 0])) + boundaries[:, 0]


class AV2Dataset(Dataset):
    def __init__(self, input, feature):
        assert input.shape[0] == feature.shape[0]
        self.input   = input    # (N, TOTAL_STEPS, 2)
        self.feature = feature  # (N, HISTORY_STEPS, 6)
        self.data_size = input.size(0)

    def __getitem__(self, index):
        inp    = self.input[index][:HISTORY_STEPS, ...]   # (50, 2)
        feat   = self.feature[index]                      # (50, 6)
        target = self.input[index][HISTORY_STEPS:, ...]   # (60, 2)
        return inp, feat, target

    def __len__(self):
        return self.data_size


class AV2ObservationSite:
    """Same public interface as InDObservationSite so main.py / evaluate.py work unchanged."""

    def __init__(self, spatial_boundaries, train_loader, test_loader):
        self.boundaries      = spatial_boundaries  # (2, 2): [[x_min, x_max], [y_min, y_max]]
        self.train_loader    = train_loader
        self.test_loader     = test_loader
        self.background      = None   # AV2 has no overhead background image
        self.ortho_px_to_meter = 1.0  # placeholder (only used in visualize.py)

    def normalize(self, data):
        return normalize(data, self.boundaries)

    def denormalize(self, data):
        return denormalize(data, self.boundaries)


class AV2:
    """
    Loads AV2 motion-forecasting parquet files from:

        root/
          train/
            <scenario_id>/
              scenario_<scenario_id>.parquet

    Only tracks with exactly HISTORY_STEPS observed frames and FUTURE_STEPS future
    frames are kept (i.e. fully-labelled tracks).

    Feature vector per history timestep (6-dim):
        heading, velocity_x, velocity_y, accel_x, accel_y, time

    Usage in main.py
    ----------------
        seq_len       = 50
        feature_dim   = 5      (heading + vx + vy + ax + ay)
        embedding_dim = 128
        hidden_dim    = 512

        av2 = AV2(root="av2_mf_tiny", train_ratio=0.8,
                  train_batch_size=64, test_batch_size=1)
        observation_site = av2.observation_site
    """

    # Object types to include (set to None to keep all)
    DEFAULT_OBJECT_TYPES = ['vehicle']

    def __init__(self, root, train_ratio=0.8, train_batch_size=64, test_batch_size=1,
                 object_types=None):
        self.root              = root
        self.train_ratio       = train_ratio
        self.train_batch_size  = train_batch_size
        self.test_batch_size   = test_batch_size
        self.object_types      = object_types if object_types is not None else self.DEFAULT_OBJECT_TYPES
        self._observation_site = None

    @property
    def observation_site(self):
        if self._observation_site is None:
            self._observation_site = self._load()
        return self._observation_site

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self):
        all_positions = []  # list of (1, TOTAL_STEPS, 2) arrays
        all_features  = []  # list of (1, HISTORY_STEPS, 5) arrays

        train_dir = os.path.join(self.root, 'train')
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Expected train split at: {train_dir}")

        for scenario_id in sorted(os.listdir(train_dir)):
            scenario_dir = os.path.join(train_dir, scenario_id)
            if not os.path.isdir(scenario_dir):
                continue

            for fname in os.listdir(scenario_dir):
                if not fname.endswith('.parquet'):
                    continue
                positions, features = self._parse(os.path.join(scenario_dir, fname))
                if positions.shape[0] > 0:
                    all_positions.append(positions)
                    all_features.append(features)

        if not all_positions:
            raise ValueError(f"No valid tracks found under {train_dir}. "
                             "Check object_types filter and data completeness.")

        positions = np.concatenate(all_positions, axis=0)  # (N, 110, 2)
        features  = np.concatenate(all_features,  axis=0)  # (N, 50, 5)
        N = positions.shape[0]

        # ---- Compute normalisation boundaries from the whole dataset ----
        # Spatial: based on all 110 timesteps so observed + future are on same scale
        xy = positions.reshape(-1, 2)
        spatial_boundaries = np.stack([xy.min(axis=0), xy.max(axis=0)], axis=1)  # (2, 2)

        # Feature: based on history portion only
        feat_flat = features.reshape(-1, 5)
        feature_boundaries = np.stack([feat_flat.min(axis=0), feat_flat.max(axis=0)], axis=1)  # (5, 2)

        # Guard against zero-range dimensions (e.g. perfectly constant channel)
        eps = 1e-6
        spatial_boundaries[:, 1] = np.where(
            np.abs(spatial_boundaries[:, 1] - spatial_boundaries[:, 0]) < eps,
            spatial_boundaries[:, 0] + eps,
            spatial_boundaries[:, 1]
        )
        feature_boundaries[:, 1] = np.where(
            np.abs(feature_boundaries[:, 1] - feature_boundaries[:, 0]) < eps,
            feature_boundaries[:, 0] + eps,
            feature_boundaries[:, 1]
        )

        # ---- Normalise ----
        positions_norm = normalize(xy, spatial_boundaries).reshape(N, TOTAL_STEPS, 2)
        features_norm  = normalize(feat_flat, feature_boundaries).reshape(N, HISTORY_STEPS, 5)

        # ---- Add time feature [0, 2] over history steps ----
        t = torch.linspace(0., 2., HISTORY_STEPS).unsqueeze(0).unsqueeze(-1).expand(N, HISTORY_STEPS, 1)
        features_with_time = torch.cat(
            [torch.FloatTensor(features_norm), t], dim=-1
        )  # (N, 50, 6)

        # ---- Train / test split ----
        perm    = np.random.permutation(N)
        n_train = int(N * self.train_ratio)
        train_idx = perm[:n_train]
        test_idx  = perm[n_train:]

        train_input   = torch.FloatTensor(positions_norm[train_idx])
        test_input    = torch.FloatTensor(positions_norm[test_idx])
        train_feature = features_with_time[train_idx]
        test_feature  = features_with_time[test_idx]

        train_dataset = AV2Dataset(train_input,  train_feature)
        test_dataset  = AV2Dataset(test_input,   test_feature)

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=self.test_batch_size,  shuffle=True)

        return AV2ObservationSite(spatial_boundaries, train_loader, test_loader)

    def _parse(self, parquet_path):
        """
        Returns:
            positions : np.ndarray  (M, TOTAL_STEPS, 2)   – normalised later
            features  : np.ndarray  (M, HISTORY_STEPS, 5) – [heading, vx, vy, ax, ay]
        """
        df = pd.read_parquet(parquet_path)

        # Keep only requested object types
        df = df[df['object_type'].isin(self.object_types)].copy()

        positions_list = []
        features_list  = []

        for _, track_df in df.groupby('track_id'):
            track_df = track_df.sort_values('timestep').reset_index(drop=True)

            obs_df = track_df[track_df['observed'] == True]
            fut_df = track_df[track_df['observed'] == False]

            # Only keep fully-labelled tracks
            if len(obs_df) != HISTORY_STEPS or len(fut_df) != FUTURE_STEPS:
                continue

            # Verify timestep continuity
            if list(obs_df['timestep']) != list(range(HISTORY_STEPS)):
                continue
            if list(fut_df['timestep']) != list(range(HISTORY_STEPS, TOTAL_STEPS)):
                continue

            # Positions for entire track (observed + future)
            all_xy = track_df[['position_x', 'position_y']].values[:TOTAL_STEPS]  # (110, 2)

            # Features for history portion only
            heading = obs_df['heading'].values         # (50,)
            vx      = obs_df['velocity_x'].values      # (50,)
            vy      = obs_df['velocity_y'].values      # (50,)
            ax      = np.gradient(vx, DT)              # (50,)  central differences
            ay      = np.gradient(vy, DT)              # (50,)

            feat = np.stack([heading, vx, vy, ax, ay], axis=-1)  # (50, 5)

            positions_list.append(all_xy[np.newaxis])  # (1, 110, 2)
            features_list.append(feat[np.newaxis])     # (1, 50, 5)

        if not positions_list:
            return np.zeros((0, TOTAL_STEPS, 2)), np.zeros((0, HISTORY_STEPS, 5))

        return np.concatenate(positions_list, axis=0), np.concatenate(features_list, axis=0)
