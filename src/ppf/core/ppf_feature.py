from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


PPFKey = Tuple[int, int, int, int]


def compute_pair_features(
    p1: np.ndarray,
    n1: np.ndarray,
    p2: np.ndarray,
    n2: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    dp = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    distance = float(np.linalg.norm(dp))
    if distance <= 1e-9:
        return None

    direction = dp / distance
    n1n = np.asarray(n1, dtype=np.float64)
    n2n = np.asarray(n2, dtype=np.float64)
    n1n = n1n / (np.linalg.norm(n1n) + 1e-12)
    n2n = n2n / (np.linalg.norm(n2n) + 1e-12)

    if abs(float(n1n @ direction)) > 0.999:
        return None

    u = n1n
    v = np.cross(u, direction)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12:
        return None
    v = v / v_norm
    w = np.cross(u, v)

    f1 = math.atan2(float(w @ n2n), float(u @ n2n))
    f2 = float(v @ n2n)
    f3 = float(u @ direction)
    return f1, f2, f3, distance


def angle_from_transformed_point(point: np.ndarray) -> float:
    y = float(point[1])
    z = float(point[2])
    angle = math.atan2(-z, y)
    if math.sin(angle) * z < 0.0:
        angle *= -1.0
    return angle


def to_internal_feature_g(f1: float, f2: float, f3: float, f4: float) -> np.ndarray:
    f2 = max(-1.0, min(1.0, f2))
    f3 = max(-1.0, min(1.0, f3))
    return np.array(
        [
            f1 + math.pi,
            math.acos(f2),
            math.acos(f3),
            f4,
        ],
        dtype=np.float32,
    )


def discretize_feature(feature_g: np.ndarray, angle_step: float, distance_step: float) -> PPFKey:
    g = np.asarray(feature_g, dtype=np.float64)
    return (
        int(math.floor(g[0] / angle_step)),
        int(math.floor(g[1] / angle_step)),
        int(math.floor(g[2] / angle_step)),
        int(math.floor(g[3] / distance_step)),
    )


def feature_bin_bounds(
    key: PPFKey,
    angle_step: float,
    distance_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            key[0] * angle_step,
            key[1] * angle_step,
            key[2] * angle_step,
            key[3] * distance_step,
        ],
        dtype=np.float64,
    )
    upper = lower + np.array([angle_step, angle_step, angle_step, distance_step], dtype=np.float64)
    return lower, upper


def boundary_distances(
    feature_g: np.ndarray,
    key: PPFKey,
    angle_step: float,
    distance_step: float,
) -> np.ndarray:
    lower, upper = feature_bin_bounds(key, angle_step, distance_step)
    feature_g = np.asarray(feature_g, dtype=np.float64)
    return np.minimum(feature_g - lower, upper - feature_g)


def neighbor_key(key: PPFKey, dim: int, direction: int) -> PPFKey:
    values = list(key)
    values[dim] += int(direction)
    return values[0], values[1], values[2], values[3]
