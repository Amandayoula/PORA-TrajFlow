import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


HISTORY_STEPS = 50   # timestep 0-49,  observed=True
FUTURE_STEPS = 60    # timestep 50-109, observed=False
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS  # 110
DT = 0.1  # 10 Hz

# Ego vehicle row id in AV2 parquet; excluded even when object_category includes FOCAL.
EXCLUDED_TRACK_ID_AV = "AV"

AV2_FILTER_MIN_OBS = 45     # minimum observed timesteps within [0, 49]
AV2_FILTER_MIN_FUT = 50     # minimum future timesteps within [50, 109]
AV2_FILTER_MAX_GAP = 2      # maximum timestep gap allowed (<=2 means at most 1 missing frame)
AV2_FILTER_P95_SPEED = 35.0 # m/s
AV2_FILTER_P95_ACCEL = 15.0 # m/s^2

# If True, each track is expressed in an "agent-centric" frame: the last observed
# position (t = HISTORY_STEPS-1) becomes the origin and the last observed heading
# aligns to +x. Positions & velocities of that track are rotated accordingly, and
# heading is stored as a small relative angle.
AGENT_CENTRIC = True


def _wrap_angle(a):
    """Wrap to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


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
    def __init__(self, input, feature, poses=None):
        assert input.shape[0] == feature.shape[0]
        self.input = input    # (N, TOTAL_STEPS, 2)
        self.feature = feature  # (N, HISTORY_STEPS, 6) or (N, HISTORY_STEPS, 5)
        # poses: (N, 3) = [tx, ty, heading0] in world frame, used only for
        # visualization / debugging when the data is agent-centric. Optional
        # for backward compatibility with older caches.
        self.poses = poses
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

    def __init__(self, spatial_boundaries, train_loader, test_loader,
                 normalize_data: bool = True, agent_centric: bool = False,
                 poses=None):
        self.boundaries = spatial_boundaries  # (2, 2): [[x_min, x_max], [y_min, y_max]]
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.background = None
        self.ortho_px_to_meter = 1.0
        self.normalize_data = normalize_data
        # When True, boundaries / input / target are all in each track's own
        # local agent frame (origin = last observed position, +x = heading).
        self.agent_centric = agent_centric
        # Per-sample world-frame pose: (N, 3) tensor [tx, ty, heading0]. Only
        # needed to project predictions back onto the world map.
        self.poses = poses

    def normalize(self, data):
        if not self.normalize_data:
            return data
        return normalize(data, self.boundaries)

    def denormalize(self, data):
        if not self.normalize_data:
            return data
        return denormalize(data, self.boundaries)

    def to_world(self, agent_xy, pose):
        """
        Transform points from agent frame to world frame.
        agent_xy : (..., 2) numpy array in agent meters.
        pose     : (3,) [tx, ty, heading0]
        """
        agent_xy = np.asarray(agent_xy, dtype=np.float32)
        tx, ty, h = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = np.cos(h), np.sin(h)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)  # rotate by +heading0
        flat = agent_xy.reshape(-1, 2)
        rotated = flat @ R.T
        return (rotated + np.array([tx, ty], dtype=np.float32)).reshape(agent_xy.shape)


@dataclass(frozen=True)
class _ParseConfig:
    object_types: Tuple[str, ...]
    object_categories: Optional[Tuple[int, ...]]


def _parse_one_parquet(parquet_path: str, cfg: _ParseConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        positions : np.ndarray  (M, TOTAL_STEPS, 2)   – x/y (agent-centric if AGENT_CENTRIC)
        features  : np.ndarray  (M, HISTORY_STEPS, 5) – [heading_rel, vx, vy, ax, ay]
        poses     : np.ndarray  (M, 3)                – world-frame [tx, ty, heading0] of each track
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    df = df[df["object_type"].isin(cfg.object_types)].copy()
    # Optional filtering by AV2 track category (e.g., 1=UNSCORED, 2=SCORED, 3=FOCAL, 0=FRAGMENT).
    # If your parquet doesn't have this column, leave object_categories=None.
    if cfg.object_categories is not None:
        if "object_category" not in df.columns:
            raise KeyError(
                "object_categories filtering requested, but parquet is missing 'object_category' column."
            )
        df = df[df["object_category"].isin(cfg.object_categories)].copy()

    positions_list: List[np.ndarray] = []
    features_list: List[np.ndarray] = []
    poses_list: List[np.ndarray] = []

    for track_id, track_df in df.groupby("track_id"):
        if str(track_id) == EXCLUDED_TRACK_ID_AV:
            continue
        track_df = track_df.sort_values("timestep").reset_index(drop=True)

        # ---- Quality filtering + interpolation to fixed [0..109] timesteps ----
        # 1) observed/future coverage
        obs_count = int(((track_df["observed"] == True) & (track_df["timestep"] < HISTORY_STEPS)).sum())
        fut_count = int(((track_df["observed"] == False) & (track_df["timestep"] >= HISTORY_STEPS)).sum())
        if obs_count < AV2_FILTER_MIN_OBS or fut_count < AV2_FILTER_MIN_FUT:
            continue

        # 2) max timestep gap (at most one missing frame between consecutive points)
        ts = np.asarray(track_df["timestep"].values, dtype=np.int64)
        ts = np.unique(ts[(ts >= 0) & (ts < TOTAL_STEPS)])
        if ts.size == 0:
            continue
        if int(np.max(np.diff(ts))) > AV2_FILTER_MAX_GAP:
            continue

        # 3) speed sanity (prefer observed velocity, if available)
        try:
            vx_raw = np.asarray(track_df.loc[track_df["observed"] == True, "velocity_x"].values, dtype=np.float32)
            vy_raw = np.asarray(track_df.loc[track_df["observed"] == True, "velocity_y"].values, dtype=np.float32)
            if vx_raw.size >= 5 and vy_raw.size >= 5:
                speed_raw = np.sqrt(vx_raw ** 2 + vy_raw ** 2)
                if float(np.percentile(speed_raw, 95)) > AV2_FILTER_P95_SPEED:
                    continue
        except Exception:
            pass

        # Reindex to all timesteps and interpolate missing values (gap already bounded above).
        full_index = pd.Index(np.arange(TOTAL_STEPS), name="timestep")
        sdf = (
            track_df.set_index("timestep")[
                ["position_x", "position_y", "heading", "velocity_x", "velocity_y", "observed"]
            ]
            .sort_index()
        )
        sdf = sdf[~sdf.index.duplicated(keep="last")]
        sdf = sdf.reindex(full_index)

        # Fill observed flag: missing -> False (future). Then enforce split 0..49 observed, 50..109 future.
        sdf["observed"] = sdf["observed"].fillna(False).astype(bool)
        sdf.loc[: HISTORY_STEPS - 1, "observed"] = True
        sdf.loc[HISTORY_STEPS:, "observed"] = False

        num_cols = ["position_x", "position_y", "heading", "velocity_x", "velocity_y"]
        sdf[num_cols] = sdf[num_cols].astype("float32").interpolate(method="linear", limit_direction="both")
        sdf[num_cols] = sdf[num_cols].ffill().bfill()

        all_xy = sdf[["position_x", "position_y"]].to_numpy(dtype=np.float32)  # (110, 2)

        obs = sdf.iloc[:HISTORY_STEPS]
        heading = obs["heading"].to_numpy(dtype=np.float32)
        vx = obs["velocity_x"].to_numpy(dtype=np.float32)
        vy = obs["velocity_y"].to_numpy(dtype=np.float32)

        # ---- Agent-centric transform (translate to last observed point, rotate so heading0 -> +x) ----
        t0_xy = all_xy[HISTORY_STEPS - 1].astype(np.float32).copy()   # world origin for this track
        h0 = float(heading[-1])                                         # world heading at t0
        if AGENT_CENTRIC:
            # Translate positions so t0 is origin in world frame.
            xy_translated = all_xy - t0_xy[None, :]
            # Rotate by -h0: R(-h0) = [[cos h0, sin h0], [-sin h0, cos h0]]
            c, s = np.cos(h0), np.sin(h0)
            R_inv = np.array([[c, s], [-s, c]], dtype=np.float32)
            all_xy = (xy_translated @ R_inv.T).astype(np.float32)       # agent-frame positions

            # Rotate velocities; heading becomes relative and wrapped.
            v = np.stack([vx, vy], axis=-1)                              # (50, 2)
            v_rot = (v @ R_inv.T).astype(np.float32)
            vx = v_rot[:, 0]
            vy = v_rot[:, 1]
            heading = _wrap_angle(heading - h0).astype(np.float32)

        # Compute accel in the (possibly rotated) velocity frame.
        ax = np.gradient(vx, DT).astype(np.float32)
        ay = np.gradient(vy, DT).astype(np.float32)

        # accel p95 on observed window
        a_norm = np.sqrt(ax ** 2 + ay ** 2)
        if float(np.percentile(a_norm, 95)) > AV2_FILTER_P95_ACCEL:
            continue

        feat = np.stack([heading, vx, vy, ax, ay], axis=-1).astype(np.float32)  # (50, 5)

        positions_list.append(all_xy[np.newaxis])
        features_list.append(feat[np.newaxis])
        poses_list.append(np.array([t0_xy[0], t0_xy[1], h0], dtype=np.float32)[np.newaxis])

    if not positions_list:
        return (
            np.zeros((0, TOTAL_STEPS, 2), dtype=np.float32),
            np.zeros((0, HISTORY_STEPS, 5), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    pos = np.concatenate(positions_list, axis=0)
    feat = np.concatenate(features_list, axis=0)
    poses = np.concatenate(poses_list, axis=0)
    return pos, feat, poses


class AV2:
    """
    Parallel AV2 cache builder.

    - Parallelizes parsing at parquet-file granularity using ProcessPoolExecutor.
    - Caches positions + features (with time channel) as tensors in .pt.
    - When ``normalize_data`` is True (default), scales x/y and kinematic features to [0, 1]
      using dataset-wide min–max bounds; otherwise stores raw values.
    - Optional ``cache_filename`` overrides the default ``av2_cache_{tag}_parallel.pt`` name
      (relative paths are resolved under ``root``; absolute paths are used as-is).
    """

    DEFAULT_OBJECT_TYPES = ["vehicle"]
    # UNSCORED(1), SCORED(2), FOCAL(3). Excludes FRAGMENT(0). Ego row track_id=="AV" is always skipped.
    DEFAULT_OBJECT_CATEGORIES = (1, 2, 3)

    def __init__(
        self,
        root: str,
        train_ratio: float = 0.8,
        train_batch_size: int = 64,
        test_batch_size: int = 1,
        object_types: Optional[Iterable[str]] = None,
        object_categories: Optional[Iterable[int]] = DEFAULT_OBJECT_CATEGORIES,
        max_scenarios: Optional[int] = None,
        num_workers: Optional[int] = None,
        normalize_data: bool = True,
        cache_filename: Optional[str] = None,
    ):
        self.root = root
        self.train_ratio = train_ratio
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.object_types = tuple(object_types) if object_types is not None else tuple(self.DEFAULT_OBJECT_TYPES)
        self.object_categories = (
            None if object_categories is None else tuple(int(x) for x in object_categories)
        )
        self.max_scenarios = max_scenarios
        self.num_workers = num_workers
        self.normalize_data = normalize_data
        self.cache_filename = cache_filename
        self._observation_site = None

    @property
    def observation_site(self):
        if self._observation_site is None:
            self._observation_site = self._load()
        return self._observation_site

    def _cache_path(self) -> str:
        if self.cache_filename is not None:
            if not self.cache_filename:
                raise ValueError("cache_filename must be a non-empty string when set")
            return (
                self.cache_filename
                if os.path.isabs(self.cache_filename)
                else os.path.join(self.root, self.cache_filename)
            )
        cache_tag = "all" if self.max_scenarios is None else str(int(self.max_scenarios))
        return os.path.join(self.root, f"av2_cache_{cache_tag}_parallel.pt")

    @property
    def cache_path(self) -> str:
        """Resolved path to the cache ``.pt`` file for this instance."""
        return self._cache_path()

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
            poses = cache.get("poses", None)
            cached_object_types = cache.get("object_types", None)
            cached_object_categories = cache.get("object_categories", None)
            cached_filters = cache.get("track_filters", None)
            cached_agent_centric = bool(cache.get("agent_centric", False))

            # If cache was built with different filtering, rebuild to avoid silent mismatches.
            if cached_object_types is not None and tuple(cached_object_types) != tuple(self.object_types):
                print("AV2 cache object_types mismatch. Rebuilding cache...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return self._load()
            if (
                cached_object_categories is not None
                and tuple(cached_object_categories) != (None if self.object_categories is None else tuple(self.object_categories))
            ):
                print("AV2 cache object_categories mismatch. Rebuilding cache...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return self._load()
            if not cache.get("exclude_track_id_av", False):
                print("AV2 cache missing exclude_track_id_av flag. Rebuilding cache...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return self._load()
            cached_normalize = cache.get("normalize_data", True)
            if bool(cached_normalize) != bool(self.normalize_data):
                print("AV2 cache normalize_data mismatch. Rebuilding cache...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return self._load()
            if cached_agent_centric != bool(AGENT_CENTRIC):
                print("AV2 cache agent_centric mismatch. Rebuilding cache...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return self._load()
            expected_filters = {
                "min_obs": AV2_FILTER_MIN_OBS,
                "min_fut": AV2_FILTER_MIN_FUT,
                "max_gap": AV2_FILTER_MAX_GAP,
                "p95_speed": AV2_FILTER_P95_SPEED,
                "p95_accel": AV2_FILTER_P95_ACCEL,
            }
            if cached_filters != expected_filters:
                print("AV2 cache track_filters mismatch. Rebuilding cache...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return self._load()

            if not torch.is_tensor(positions):
                positions = torch.as_tensor(positions)
            if not torch.is_tensor(features):
                features = torch.as_tensor(features)
            if not torch.is_tensor(spatial_boundaries):
                spatial_boundaries = torch.as_tensor(spatial_boundaries)
            if poses is not None and not torch.is_tensor(poses):
                poses = torch.as_tensor(poses)

            positions = positions.float()
            features = features.float()
            spatial_boundaries = spatial_boundaries.float()
            if poses is not None:
                poses = poses.float()

            if features.ndim == 3 and features.shape[-1] == 5:
                t = torch.linspace(0.0, 2.0, HISTORY_STEPS).unsqueeze(0).unsqueeze(-1)
                t = t.expand(features.shape[0], HISTORY_STEPS, 1)
                features = torch.cat([features, t], dim=-1)

            dataset = AV2Dataset(positions, features, poses=poses)
            train_size = int(len(dataset) * self.train_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)
            return AV2ObservationSite(
                spatial_boundaries, train_loader, test_loader,
                normalize_data=self.normalize_data,
                agent_centric=cached_agent_centric, poses=poses,
            )

        parquet_paths = self._iter_parquet_paths()
        if not parquet_paths:
            raise ValueError("No parquet files found under train/ (check dataset root).")

        cfg = _ParseConfig(object_types=self.object_types, object_categories=self.object_categories)
        all_positions: List[np.ndarray] = []
        all_features: List[np.ndarray] = []
        all_poses: List[np.ndarray] = []

        num_workers = self.num_workers
        if num_workers is None:
            num_workers = max(1, min(8, (os.cpu_count() or 1)))

        print(f"Parsing {len(parquet_paths)} parquet files with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_parse_one_parquet, p, cfg) for p in parquet_paths]
            for fut in as_completed(futures):
                pos, feat, poses = fut.result()
                if pos.shape[0] > 0:
                    all_positions.append(pos)
                    all_features.append(feat)
                    all_poses.append(poses)

        if not all_positions:
            raise ValueError("No valid tracks found. Check object_types filter and data completeness.")

        positions = np.concatenate(all_positions, axis=0)  # (N, 110, 2)
        features = np.concatenate(all_features, axis=0)    # (N, 50, 5)
        poses = np.concatenate(all_poses, axis=0)          # (N, 3)
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

        if self.normalize_data:
            positions_arr = normalize(xy, spatial_boundaries).reshape(N, TOTAL_STEPS, 2)
            features_arr = normalize(feat_flat, feature_boundaries).reshape(N, HISTORY_STEPS, 5)
        else:
            positions_arr = xy.reshape(N, TOTAL_STEPS, 2)
            features_arr = feat_flat.reshape(N, HISTORY_STEPS, 5)

        t = torch.linspace(0.0, 2.0, HISTORY_STEPS).unsqueeze(0).unsqueeze(-1).expand(N, HISTORY_STEPS, 1)
        features_with_time = torch.cat([torch.as_tensor(features_arr).float(), t], dim=-1)  # (N, 50, 6)

        poses_tensor = torch.as_tensor(poses).float()
        perm = np.random.permutation(N)
        n_train = int(N * self.train_ratio)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_input = torch.as_tensor(positions_arr[train_idx]).float()
        test_input = torch.as_tensor(positions_arr[test_idx]).float()
        train_feature = features_with_time[train_idx]
        test_feature = features_with_time[test_idx]

        # Build a single AV2Dataset over the full set so poses stay aligned with
        # `random_split` indices; then split into train/test with `random_split`
        # exactly as the cache-load branch does.
        full_input = torch.as_tensor(positions_arr).float()
        full_feature = features_with_time.float()
        full_dataset = AV2Dataset(full_input, full_feature, poses=poses_tensor)
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, N - n_train]
        )
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)

        print(f"Total valid tracks: {N}")
        print("Saving AV2 cache...")
        torch.save(
            {
                "positions": torch.as_tensor(positions_arr).float(),
                "features": features_with_time.float(),
                "spatial_boundaries": torch.as_tensor(spatial_boundaries).float(),
                "poses": poses_tensor,
                "object_types": list(self.object_types),
                "object_categories": (
                    None if self.object_categories is None else list(self.object_categories)
                ),
                "exclude_track_id_av": True,
                "normalize_data": self.normalize_data,
                "agent_centric": bool(AGENT_CENTRIC),
                "track_filters": {
                    "min_obs": AV2_FILTER_MIN_OBS,
                    "min_fut": AV2_FILTER_MIN_FUT,
                    "max_gap": AV2_FILTER_MAX_GAP,
                    "p95_speed": AV2_FILTER_P95_SPEED,
                    "p95_accel": AV2_FILTER_P95_ACCEL,
                },
            },
            cache_path,
        )

        return AV2ObservationSite(
            torch.as_tensor(spatial_boundaries).float(),
            train_loader,
            test_loader,
            normalize_data=self.normalize_data,
            agent_centric=bool(AGENT_CENTRIC),
            poses=poses_tensor,
        )


if __name__ == "__main__":
    """
    Convenience entry-point to ONLY build the AV2 cache as a .pt file, without
    touching the TrajFlow model or training loop.

    Usage (from project root):
        python -m datasets.AV2_parallel

    Edit the ``default_*`` variables below (root, output filename, max_scenarios, …)
    before running.
    """

    default_root = "data/av2_mf_tiny"
    default_train_ratio = 0.8
    default_train_batch_size = 64
    default_test_batch_size = 1
    default_max_scenarios = None  # e.g. 100 for a small subset
    default_num_workers = None    # will auto-select up to 8
    # Include fragments (0) but filter by coverage/gap/speed/accel thresholds above.
    default_object_categories = (0, 1, 2, 3)
    # Output .pt name: None -> av2_cache_{all|N}_parallel.pt under default_root.
    # Otherwise a basename (e.g. "my_run.pt") or relative path under root, or an absolute path.
    default_cache_filename = 'with_fragment_no_normalization.pt'

    print(
        f"Building AV2 parallel cache under root='{default_root}' "
        f"(max_scenarios={default_max_scenarios}, num_workers={default_num_workers})..."
    )

    av2 = AV2(
        root=default_root,
        train_ratio=default_train_ratio,
        train_batch_size=default_train_batch_size,
        test_batch_size=default_test_batch_size,
        object_categories=default_object_categories,
        max_scenarios=default_max_scenarios,
        num_workers=default_num_workers,
        cache_filename=default_cache_filename,
    )

    # Accessing observation_site triggers cache build if it does not exist.
    site = av2.observation_site
    print(
        f"Done. Cache file: {av2.cache_path}\n"
        f"Train samples: {len(site.train_loader.dataset)}, "
        f"test samples: {len(site.test_loader.dataset)}"
    )
