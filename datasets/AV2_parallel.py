import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


HISTORY_STEPS = 50   # timestep 0-49,  observed=True
FUTURE_STEPS = 60    # timestep 50-109, observed=False
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS  # 110
DT = 0.1  # 10 Hz


def normalize(data, boundaries):
    if torch.is_tensor(data) or torch.is_tensor(boundaries):
        data_t = data if torch.is_tensor(data) else torch.as_tensor(data)
        b_t = boundaries if torch.is_tensor(boundaries) else torch.as_tensor(boundaries)
        return (data_t - b_t[:, 0]) / (b_t[:, 1] - b_t[:, 0])
    return (data - boundaries[:, 0]) / (boundaries[:, 1] - boundaries[:, 0])


def denormalize(data, boundaries):
    if torch.is_tensor(data) or torch.is_tensor(boundaries):
        data_t = data if torch.is_tensor(data) else torch.as_tensor(data)
        b_t = boundaries if torch.is_tensor(boundaries) else torch.as_tensor(boundaries)
        return (data_t * (b_t[:, 1] - b_t[:, 0])) + b_t[:, 0]
    return (data * (boundaries[:, 1] - boundaries[:, 0])) + boundaries[:, 0]


class AV2Dataset(Dataset):
    def __init__(self, input, feature):
        assert input.shape[0] == feature.shape[0]
        self.input = input    # (N, TOTAL_STEPS, 2)
        self.feature = feature  # (N, HISTORY_STEPS, 6) or (N, HISTORY_STEPS, 5)
        self.data_size = input.size(0)

    def __getitem__(self, index):
        inp = self.input[index][:HISTORY_STEPS, ...]     # (50, 2)
        feat = self.feature[index]                       # (50, 6)
        target = self.input[index][HISTORY_STEPS:, ...]  # (60, 2)
        return inp, feat, target

    def __len__(self):
        return self.data_size


class AV2ObservationSite:
    """Same public interface as InDObservationSite so main.py / evaluate.py work unchanged."""

    def __init__(self, spatial_boundaries, train_loader, test_loader):
        self.boundaries = spatial_boundaries  # (2, 2): [[x_min, x_max], [y_min, y_max]]
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.background = None
        self.ortho_px_to_meter = 1.0

    def normalize(self, data):
        return normalize(data, self.boundaries)

    def denormalize(self, data):
        return denormalize(data, self.boundaries)


@dataclass(frozen=True)
class _ParseConfig:
    object_types: Tuple[str, ...]


def _parse_one_parquet(parquet_path: str, cfg: _ParseConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        positions : np.ndarray  (M, TOTAL_STEPS, 2)   – absolute x/y (normalized later)
        features  : np.ndarray  (M, HISTORY_STEPS, 5) – [heading, vx, vy, ax, ay]
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["object_type"].isin(cfg.object_types)].copy()

    positions_list: List[np.ndarray] = []
    features_list: List[np.ndarray] = []

    for _, track_df in df.groupby("track_id"):
        track_df = track_df.sort_values("timestep").reset_index(drop=True)

        obs_df = track_df[track_df["observed"] == True]
        fut_df = track_df[track_df["observed"] == False]

        if len(obs_df) != HISTORY_STEPS or len(fut_df) != FUTURE_STEPS:
            continue
        if list(obs_df["timestep"]) != list(range(HISTORY_STEPS)):
            continue
        if list(fut_df["timestep"]) != list(range(HISTORY_STEPS, TOTAL_STEPS)):
            continue

        all_xy = track_df[["position_x", "position_y"]].values[:TOTAL_STEPS]  # (110, 2)

        heading = obs_df["heading"].values
        vx = obs_df["velocity_x"].values
        vy = obs_df["velocity_y"].values
        ax = np.gradient(vx, DT)
        ay = np.gradient(vy, DT)

        feat = np.stack([heading, vx, vy, ax, ay], axis=-1)  # (50, 5)

        positions_list.append(all_xy[np.newaxis])
        features_list.append(feat[np.newaxis])

    if not positions_list:
        return np.zeros((0, TOTAL_STEPS, 2), dtype=np.float32), np.zeros((0, HISTORY_STEPS, 5), dtype=np.float32)

    pos = np.concatenate(positions_list, axis=0)
    feat = np.concatenate(features_list, axis=0)
    return pos, feat


class AV2:
    """
    Parallel AV2 cache builder.

    - Parallelizes parsing at parquet-file granularity using ProcessPoolExecutor.
    - Caches normalized positions + features(with time channel) as tensors in .pt.
    """

    DEFAULT_OBJECT_TYPES = ["vehicle"]

    def __init__(
        self,
        root: str,
        train_ratio: float = 0.8,
        train_batch_size: int = 64,
        test_batch_size: int = 1,
        object_types: Optional[Iterable[str]] = None,
        max_scenarios: Optional[int] = None,
        num_workers: Optional[int] = None,
    ):
        self.root = root
        self.train_ratio = train_ratio
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.object_types = tuple(object_types) if object_types is not None else tuple(self.DEFAULT_OBJECT_TYPES)
        self.max_scenarios = max_scenarios
        self.num_workers = num_workers
        self._observation_site = None

    @property
    def observation_site(self):
        if self._observation_site is None:
            self._observation_site = self._load()
        return self._observation_site

    def _cache_path(self) -> str:
        cache_tag = "all" if self.max_scenarios is None else str(int(self.max_scenarios))
        return os.path.join(self.root, f"av2_cache_{cache_tag}_parallel.pt")

    def _iter_parquet_paths(self) -> List[str]:
        train_dir = os.path.join(self.root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Expected train split at: {train_dir}")

        scenario_ids = sorted(os.listdir(train_dir))
        if self.max_scenarios is not None:
            scenario_ids = scenario_ids[: int(self.max_scenarios)]

        parquet_paths: List[str] = []
        for scenario_id in scenario_ids:
            scenario_dir = os.path.join(train_dir, scenario_id)
            if not os.path.isdir(scenario_dir):
                continue
            for fname in os.listdir(scenario_dir):
                if fname.endswith(".parquet"):
                    parquet_paths.append(os.path.join(scenario_dir, fname))
        return parquet_paths

    def _load(self):
        cache_path = self._cache_path()

        if os.path.exists(cache_path):
            print("Loading AV2 from cache...")
            try:
                cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Failed to load AV2 cache ({e}). Rebuilding cache...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return self._load()

            positions = cache["positions"]
            features = cache["features"]
            spatial_boundaries = cache["spatial_boundaries"]

            if not torch.is_tensor(positions):
                positions = torch.as_tensor(positions)
            if not torch.is_tensor(features):
                features = torch.as_tensor(features)
            if not torch.is_tensor(spatial_boundaries):
                spatial_boundaries = torch.as_tensor(spatial_boundaries)

            positions = positions.float()
            features = features.float()
            spatial_boundaries = spatial_boundaries.float()

            if features.ndim == 3 and features.shape[-1] == 5:
                t = torch.linspace(0.0, 2.0, HISTORY_STEPS).unsqueeze(0).unsqueeze(-1)
                t = t.expand(features.shape[0], HISTORY_STEPS, 1)
                features = torch.cat([features, t], dim=-1)

            dataset = AV2Dataset(positions, features)
            train_size = int(len(dataset) * self.train_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)
            return AV2ObservationSite(spatial_boundaries, train_loader, test_loader)

        parquet_paths = self._iter_parquet_paths()
        if not parquet_paths:
            raise ValueError("No parquet files found under train/ (check dataset root).")

        cfg = _ParseConfig(object_types=self.object_types)
        all_positions: List[np.ndarray] = []
        all_features: List[np.ndarray] = []

        num_workers = self.num_workers
        if num_workers is None:
            num_workers = max(1, min(8, (os.cpu_count() or 1)))

        print(f"Parsing {len(parquet_paths)} parquet files with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_parse_one_parquet, p, cfg) for p in parquet_paths]
            for fut in as_completed(futures):
                pos, feat = fut.result()
                if pos.shape[0] > 0:
                    all_positions.append(pos)
                    all_features.append(feat)

        if not all_positions:
            raise ValueError("No valid tracks found. Check object_types filter and data completeness.")

        positions = np.concatenate(all_positions, axis=0)  # (N, 110, 2)
        features = np.concatenate(all_features, axis=0)    # (N, 50, 5)
        N = positions.shape[0]

        xy = positions.reshape(-1, 2)
        spatial_boundaries = np.stack([xy.min(axis=0), xy.max(axis=0)], axis=1)  # (2, 2)

        feat_flat = features.reshape(-1, 5)
        feature_boundaries = np.stack([feat_flat.min(axis=0), feat_flat.max(axis=0)], axis=1)  # (5, 2)

        eps = 1e-6
        spatial_boundaries[:, 1] = np.where(
            np.abs(spatial_boundaries[:, 1] - spatial_boundaries[:, 0]) < eps,
            spatial_boundaries[:, 0] + eps,
            spatial_boundaries[:, 1],
        )
        feature_boundaries[:, 1] = np.where(
            np.abs(feature_boundaries[:, 1] - feature_boundaries[:, 0]) < eps,
            feature_boundaries[:, 0] + eps,
            feature_boundaries[:, 1],
        )

        positions_norm = normalize(xy, spatial_boundaries).reshape(N, TOTAL_STEPS, 2)
        features_norm = normalize(feat_flat, feature_boundaries).reshape(N, HISTORY_STEPS, 5)

        t = torch.linspace(0.0, 2.0, HISTORY_STEPS).unsqueeze(0).unsqueeze(-1).expand(N, HISTORY_STEPS, 1)
        features_with_time = torch.cat([torch.as_tensor(features_norm).float(), t], dim=-1)  # (N, 50, 6)

        perm = np.random.permutation(N)
        n_train = int(N * self.train_ratio)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_input = torch.as_tensor(positions_norm[train_idx]).float()
        test_input = torch.as_tensor(positions_norm[test_idx]).float()
        train_feature = features_with_time[train_idx]
        test_feature = features_with_time[test_idx]

        train_dataset = AV2Dataset(train_input, train_feature)
        test_dataset = AV2Dataset(test_input, test_feature)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)

        print(f"Total valid tracks: {N}")
        print("Saving AV2 cache...")
        torch.save(
            {
                "positions": torch.as_tensor(positions_norm).float(),
                "features": features_with_time.float(),
                "spatial_boundaries": torch.as_tensor(spatial_boundaries).float(),
            },
            cache_path,
        )

        return AV2ObservationSite(torch.as_tensor(spatial_boundaries).float(), train_loader, test_loader)


if __name__ == "__main__":
    """
    Convenience entry-point to ONLY build the AV2 cache as a .pt file, without
    touching the TrajFlow model or training loop.

    Usage (from project root):
        python -m datasets.AV2_parallel

    You can tweak the defaults below (root / max_scenarios / num_workers)
    before running.
    """

    default_root = "data/av2_mf_tiny"
    default_train_ratio = 0.8
    default_train_batch_size = 64
    default_test_batch_size = 1
    default_max_scenarios = None  # e.g. 100 for a small subset
    default_num_workers = None    # will auto-select up to 8

    print(f"Building AV2 parallel cache under root='{default_root}' "
          f"(max_scenarios={default_max_scenarios}, num_workers={default_num_workers})...")

    av2 = AV2(
        root=default_root,
        train_ratio=default_train_ratio,
        train_batch_size=default_train_batch_size,
        test_batch_size=default_test_batch_size,
        max_scenarios=default_max_scenarios,
        num_workers=default_num_workers,
    )

    # Accessing observation_site triggers cache build if it does not exist.
    site = av2.observation_site
    print(f"Done. Train samples: {len(site.train_loader.dataset)}, "
          f"test samples: {len(site.test_loader.dataset)}")
