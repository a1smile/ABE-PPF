from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ppf_feature import PPFKey, angle_from_transformed_point, compute_pair_features, discretize_feature, to_internal_feature_g
from .transform_utils import rotation_matrix_from_axis_angle


@dataclass(frozen=True)
class ModelPairEntry:
    mr: int
    mi: int
    g: Tuple[float, float, float, float]
    bucket_key: PPFKey


@dataclass
class BucketStatistics:
    total_pairs: int
    bucket_counts: Dict[PPFKey, int]
    bucket_scores: Dict[PPFKey, float]


@dataclass
class ModelHashBundle:
    model_path: str
    points: np.ndarray
    normals: np.ndarray
    ref_rotations: np.ndarray
    alpha_m: np.ndarray
    model_diameter: float
    angle_step: float
    distance_step: float
    min_pair_distance: float
    table: Dict[PPFKey, List[ModelPairEntry]]
    stats: BucketStatistics

    def query(self, key: PPFKey) -> List[ModelPairEntry]:
        return self.table.get(key, [])

    def bucket_size(self, key: PPFKey) -> int:
        return int(self.stats.bucket_counts.get(key, 0))

    def bucket_score(self, key: PPFKey) -> float:
        return float(self.stats.bucket_scores.get(key, 0.0))


def _reference_rotations(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    rotations = np.zeros((normals.shape[0], 3, 3), dtype=np.float64)
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for index, normal in enumerate(normals):
        unit = normal / (np.linalg.norm(normal) + 1e-12)
        axis = np.cross(unit, ex)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            axis = ey.copy()
            axis_norm = 1.0
        axis /= axis_norm
        angle = math.acos(max(-1.0, min(1.0, float(unit @ ex))))
        rotations[index] = rotation_matrix_from_axis_angle(axis, angle)
    return rotations


def build_model_hash_bundle(
    model_path: str,
    points: np.ndarray,
    normals: np.ndarray,
    angle_step: float,
    distance_step: float,
    min_pair_distance: Optional[float] = None,
) -> ModelHashBundle:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.shape != normals.shape:
        raise ValueError("points and normals must have the same shape")
    if points.shape[0] == 0:
        raise ValueError("model point cloud is empty")

    pair_distance = float(distance_step * 0.5 if min_pair_distance is None else min_pair_distance)
    ref_rotations = _reference_rotations(normals)
    alpha_m = np.zeros((points.shape[0], points.shape[0]), dtype=np.float32)
    table: Dict[PPFKey, List[ModelPairEntry]] = {}
    model_diameter = 0.0

    for mr in range(points.shape[0]):
        point_r = points[mr]
        normal_r = normals[mr]
        rotation_r = ref_rotations[mr]
        for mi in range(points.shape[0]):
            if mr == mi:
                continue
            feature = compute_pair_features(point_r, normal_r, points[mi], normals[mi])
            if feature is None:
                continue
            f1, f2, f3, f4 = feature
            if f4 < pair_distance:
                continue
            feature_g = to_internal_feature_g(f1, f2, f3, f4)
            key = discretize_feature(feature_g, angle_step, distance_step)
            entry = ModelPairEntry(
                mr=mr,
                mi=mi,
                g=tuple(float(value) for value in feature_g),
                bucket_key=key,
            )
            table.setdefault(key, []).append(entry)

            local_point = rotation_r @ (points[mi] - point_r)
            alpha_m[mr, mi] = float(-angle_from_transformed_point(local_point))
            model_diameter = max(model_diameter, f4)

    total_pairs = sum(len(bucket) for bucket in table.values())
    bucket_counts = {key: len(bucket) for key, bucket in table.items()}
    bucket_scores = {
        key: float(math.log(max(1.0, total_pairs) / float(count + 1)))
        for key, count in bucket_counts.items()
    }
    stats = BucketStatistics(
        total_pairs=total_pairs,
        bucket_counts=bucket_counts,
        bucket_scores=bucket_scores,
    )
    return ModelHashBundle(
        model_path=model_path,
        points=points,
        normals=normals,
        ref_rotations=ref_rotations,
        alpha_m=alpha_m,
        model_diameter=float(model_diameter),
        angle_step=float(angle_step),
        distance_step=float(distance_step),
        min_pair_distance=float(pair_distance),
        table=table,
        stats=stats,
    )


def cache_path_for_model(cache_dir: str, model_path: str, cfg_fingerprint_payload: Dict[str, object]) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = Path(model_path).stem
    payload = dict(cfg_fingerprint_payload)
    payload["model_path"] = str(Path(model_path).resolve())
    digest = hashlib.sha1(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{stem}__{digest}.pkl"


def save_model_hash_bundle(cache_path: Path, bundle: ModelHashBundle) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_hash_bundle(cache_path: Path) -> ModelHashBundle:
    with cache_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, ModelHashBundle):
        raise TypeError(f"Invalid model hash cache: {cache_path}")
    return bundle
