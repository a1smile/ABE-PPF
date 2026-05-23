from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_to_pi(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def make_affine(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def invert_affine(transform: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    inv_rotation = rotation.T
    inv_translation = -(inv_rotation @ translation)
    return make_affine(inv_rotation, inv_translation)


def compose_affine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    return (rotation @ pts.T).T + translation


def so3_distance_rad(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(rotation_b, dtype=np.float64)
    trace = float(np.trace(delta))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return float(math.acos(cos_theta))


def rotation_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    return math.degrees(so3_distance_rad(rotation_a, rotation_b))


def translation_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector_a, dtype=np.float64) - np.asarray(vector_b, dtype=np.float64)))


def average_rotation(rotations: Iterable[np.ndarray]) -> np.ndarray:
    rotations = list(rotations)
    if not rotations:
        return np.eye(3, dtype=np.float64)
    if len(rotations) == 1:
        return np.asarray(rotations[0], dtype=np.float64).copy()

    matrix = np.zeros((3, 3), dtype=np.float64)
    for rotation in rotations:
        matrix += np.asarray(rotation, dtype=np.float64)

    U, _, Vt = np.linalg.svd(matrix)
    avg = U @ Vt
    if np.linalg.det(avg) < 0:
        U[:, -1] *= -1.0
        avg = U @ Vt
    return avg
