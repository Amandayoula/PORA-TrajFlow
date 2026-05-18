"""Lightweight AV2 map loader (no av2-api / argoverse dependency).

Reads ``log_map_archive_<scenario_id>.json`` produced by the Argoverse 2
Motion Forecasting dataset. The schema is::

    {
      "drivable_areas":      {id: {"area_boundary": [{x,y,z}, ...], "id": int}},
      "lane_segments":       {id: {"centerline":           [{x,y,z}, ...],
                                    "left_lane_boundary":  [{x,y,z}, ...],
                                    "right_lane_boundary": [{x,y,z}, ...],
                                    "is_intersection": bool,
                                    "lane_type": str,
                                    ...}},
      "pedestrian_crossings": {id: {...}}
    }

The world frame of the JSON matches ``positions_world`` in
``datasets/AV2_scenarios.py`` (city frame), so points are directly comparable.

Public surface:
  * ``load_av2_map(path) -> AV2Map``                      - parse JSON
  * ``AV2Map.is_inside_drivable_area(xy_world)``          - point-in-polygon
  * ``build_corridor_fallback(av_world, non_av_worlds)``  - synth fallback
  * ``load_or_fallback(map_root, scenario_id, bundle)``   - one-call helper
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np


def _points_xy(seq) -> np.ndarray:
    """``[{x, y, z}, ...]`` -> ``(N, 2)`` float32."""
    if not seq:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.empty((len(seq), 2), dtype=np.float32)
    for i, p in enumerate(seq):
        out[i, 0] = float(p["x"])
        out[i, 1] = float(p["y"])
    return out


@dataclass
class AV2Map:
    """Parsed AV2 map for one scenario, all in world (city) frame.

    drivable_polygons: list of (N_i, 2) closed polygons (last vertex may or
    may not equal the first; ``is_inside_drivable_area`` handles either).
    """

    drivable_polygons: List[np.ndarray] = field(default_factory=list)
    lane_centerlines: List[np.ndarray] = field(default_factory=list)
    lane_left_bounds: List[np.ndarray] = field(default_factory=list)
    lane_right_bounds: List[np.ndarray] = field(default_factory=list)
    pedestrian_polygons: List[np.ndarray] = field(default_factory=list)
    source: Literal["av2_json", "fallback"] = "av2_json"

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def is_inside_drivable_area(self, xy_world: np.ndarray) -> bool:
        """Even-odd ray-casting against any drivable polygon. Returns True if
        ``(x, y)`` is inside at least one polygon. Robust to polygons that
        are or are not explicitly closed."""
        if not self.drivable_polygons:
            return True  # treat "no map" as fully drivable
        x = float(xy_world[0])
        y = float(xy_world[1])
        for poly in self.drivable_polygons:
            if _point_in_polygon(x, y, poly):
                return True
        return False

    def world_bounds(self) -> np.ndarray:
        """Tight ``(2, 2) = [[x_min, x_max], [y_min, y_max]]`` over all polygons.
        Used by viz to set axis limits when no policy rollout exists yet."""
        if not self.drivable_polygons:
            return np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        all_pts = np.concatenate(self.drivable_polygons, axis=0)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        return np.array([[x_min, x_max], [y_min, y_max]], dtype=np.float32)

    # ------------------------------------------------------------------
    # lane geometry: lateral offset (continuous on/off-road signal)
    # ------------------------------------------------------------------
    def nearest_lane_offset(self, xy_world: np.ndarray) -> float:
        """Unsigned distance (m) from ``xy_world`` to the closest lane
        centerline vertex. Used for a soft "stay-near-the-lane-center"
        reward term. Returns 0.0 when no centerlines are available
        (e.g. fallback corridor map) so the term degrades gracefully."""
        if not self.lane_centerlines:
            return 0.0
        x = float(xy_world[0])
        y = float(xy_world[1])
        best = float("inf")
        for cl in self.lane_centerlines:
            if cl.shape[0] == 0:
                continue
            dx = cl[:, 0].astype(np.float64) - x
            dy = cl[:, 1].astype(np.float64) - y
            d2 = dx * dx + dy * dy
            m = float(d2.min())
            if m < best:
                best = m
        return float(np.sqrt(best)) if best < float("inf") else 0.0

    # ------------------------------------------------------------------
    # lane geometry: dynamic forward direction
    # ------------------------------------------------------------------
    def nearest_lane_tangent(
        self,
        xy_world: np.ndarray,
        *,
        h0_hint: Optional[float] = None,
        k: int = 3,
        weight_eps: float = 0.5,
        fallback: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the local lane forward direction at ``xy_world`` as a unit
        ``(2,)`` vector.

        Algorithm: gather every centerline vertex's local tangent, sort by
        squared distance to ``xy_world``, take the top-``k`` and return their
        ``1 / (dist + weight_eps)``-weighted, re-normalised average.

        If ``h0_hint`` is provided, each tangent is flipped so its dot product
        with ``[cos(h0_hint), sin(h0_hint)]`` is non-negative (centerlines have
        no canonical orientation). This is what makes the result a *forward*
        unit and not a *signed* unit.

        If no centerlines are available (e.g. fallback corridor map), returns
        ``fallback`` if provided, else ``[1, 0]``.
        """
        if not self.lane_centerlines:
            if fallback is not None:
                return np.asarray(fallback, dtype=np.float64).reshape(2)
            return np.array([1.0, 0.0], dtype=np.float64)

        x = float(xy_world[0])
        y = float(xy_world[1])

        candidates: List[Tuple[float, np.ndarray]] = []
        for cl in self.lane_centerlines:
            if cl.shape[0] < 2:
                continue
            diffs = np.diff(cl.astype(np.float64), axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1, keepdims=True)
            seg_lens = np.where(seg_lens < 1e-9, 1.0, seg_lens)
            tangs = diffs / seg_lens  # (N-1, 2) - unit tangent per segment
            dx = cl[:, 0].astype(np.float64) - x
            dy = cl[:, 1].astype(np.float64) - y
            d2 = dx * dx + dy * dy
            n_pts = cl.shape[0]
            n_seg = tangs.shape[0]
            for i in range(n_pts):
                # Vertex i takes the tangent of segment min(i, n_seg-1):
                # endpoints inherit their adjacent segment, mid-vertices use
                # the forward segment (good enough for short AV2 polylines).
                seg_idx = i if i < n_seg else n_seg - 1
                candidates.append((float(d2[i]), tangs[seg_idx].copy()))

        if not candidates:
            if fallback is not None:
                return np.asarray(fallback, dtype=np.float64).reshape(2)
            return np.array([1.0, 0.0], dtype=np.float64)

        candidates.sort(key=lambda c: c[0])
        top = candidates[: max(1, int(k))]

        if h0_hint is not None:
            c, s = math.cos(float(h0_hint)), math.sin(float(h0_hint))
            h0_vec = np.array([c, s], dtype=np.float64)
            for i, (d2, t) in enumerate(top):
                if float(np.dot(t, h0_vec)) < 0.0:
                    top[i] = (d2, -t)

        weights = np.array(
            [1.0 / (math.sqrt(d2) + float(weight_eps)) for d2, _ in top],
            dtype=np.float64,
        )
        wsum = float(weights.sum())
        if wsum < 1e-12:
            return top[0][1] / (np.linalg.norm(top[0][1]) + 1e-12)
        weights = weights / wsum
        avg = np.zeros(2, dtype=np.float64)
        for w, (_, t) in zip(weights, top):
            avg += float(w) * t
        nrm = float(np.linalg.norm(avg))
        if nrm < 1e-6:
            # Tangents canceled (e.g. opposite directions w/o h0_hint).
            t0 = top[0][1]
            return t0 / (np.linalg.norm(t0) + 1e-12)
        return avg / nrm


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Even-odd test. ``poly`` is ``(N, 2)``; auto-closes the ring."""
    n = poly.shape[0]
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def load_av2_map(path: str) -> AV2Map:
    """Parse an AV2 ``log_map_archive_<scenario_id>.json`` into an ``AV2Map``."""
    with open(path, "r") as f:
        data = json.load(f)

    drivable_polys: List[np.ndarray] = []
    for da in (data.get("drivable_areas") or {}).values():
        poly = _points_xy(da.get("area_boundary"))
        if poly.shape[0] >= 3:
            drivable_polys.append(poly)

    centerlines: List[np.ndarray] = []
    left_bounds: List[np.ndarray] = []
    right_bounds: List[np.ndarray] = []
    for ls in (data.get("lane_segments") or {}).values():
        c = _points_xy(ls.get("centerline"))
        if c.shape[0] >= 2:
            centerlines.append(c)
        lb = _points_xy(ls.get("left_lane_boundary"))
        if lb.shape[0] >= 2:
            left_bounds.append(lb)
        rb = _points_xy(ls.get("right_lane_boundary"))
        if rb.shape[0] >= 2:
            right_bounds.append(rb)

    ped_polys: List[np.ndarray] = []
    for pc in (data.get("pedestrian_crossings") or {}).values():
        # AV2 ped_crossing schema differs across releases; defensively try common keys.
        for key in ("polygon", "edge1", "edge2", "boundary"):
            seq = pc.get(key)
            if seq is None:
                continue
            poly = _points_xy(seq)
            if poly.shape[0] >= 3:
                ped_polys.append(poly)
                break

    return AV2Map(
        drivable_polygons=drivable_polys,
        lane_centerlines=centerlines,
        lane_left_bounds=left_bounds,
        lane_right_bounds=right_bounds,
        pedestrian_polygons=ped_polys,
        source="av2_json",
    )


# ---------------------------------------------------------------------------
# Fallback corridor (when the JSON is missing)
# ---------------------------------------------------------------------------


def build_corridor_fallback(
    av_world: np.ndarray,
    non_av_worlds: Sequence[np.ndarray],
    half_width: float = 8.0,
    sample_step: float = 1.0,
) -> AV2Map:
    """Buffered-trajectory polygon used when no AV2 map JSON is available.

    Builds a single polygon by walking the AV recorded path and emitting two
    parallel offsets ``±half_width`` along the local normal. We use the AV's
    path (not the union) because: (1) AV tends to stay on-road; (2) keeping
    the corridor independent of non-AV trajectories means the policy is not
    "guided" by the very trajectories we are trying to avoid risk against.
    """
    av = np.asarray(av_world, dtype=np.float32)
    if av.ndim != 2 or av.shape[0] < 2:
        raise ValueError("av_world must be (T, 2) with T >= 2")

    # Resample evenly along arc length so corner offsets stay clean.
    seg = np.diff(av, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1])
    n_samples = max(int(np.ceil(total / max(sample_step, 1e-3))) + 1, 8)
    s_grid = np.linspace(0.0, total, n_samples, dtype=np.float32)
    xs = np.interp(s_grid, arc, av[:, 0]).astype(np.float32)
    ys = np.interp(s_grid, arc, av[:, 1]).astype(np.float32)
    path = np.stack([xs, ys], axis=1)

    # Tangent (forward-difference at endpoints, central elsewhere).
    tangent = np.zeros_like(path)
    tangent[1:-1] = path[2:] - path[:-2]
    tangent[0] = path[1] - path[0]
    tangent[-1] = path[-1] - path[-2]
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    tangent = tangent / norms
    # 90 deg CCW = left normal.
    left_n = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    left = path + half_width * left_n
    right = path - half_width * left_n

    # Polygon: walk left side forward, then right side backward, close.
    poly = np.concatenate([left, right[::-1]], axis=0).astype(np.float32)

    return AV2Map(drivable_polygons=[poly], source="fallback")


# ---------------------------------------------------------------------------
# Convenience: prefer JSON, else fallback
# ---------------------------------------------------------------------------


def load_or_fallback(
    map_root: str,
    scenario_id: str,
    av_world: np.ndarray,
    non_av_worlds: Optional[Sequence[np.ndarray]] = None,
    *,
    fallback_half_width: float = 8.0,
) -> AV2Map:
    """Look for ``map_root/<scenario_id>/log_map_archive_<scenario_id>.json``.

    If present and parseable, return it; otherwise build a corridor from the
    AV recorded path and return that with ``source='fallback'``.
    """
    json_path = os.path.join(
        map_root, scenario_id, f"log_map_archive_{scenario_id}.json"
    )
    if os.path.isfile(json_path):
        try:
            mp = load_av2_map(json_path)
            if mp.drivable_polygons:
                return mp
        except Exception as e:
            print(f"[AV2_map] failed to parse {json_path}: {e}; falling back.")
    return build_corridor_fallback(
        av_world=av_world,
        non_av_worlds=non_av_worlds or [],
        half_width=fallback_half_width,
    )
