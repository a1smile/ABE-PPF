from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from src.ppf.core.transform_utils import so3_distance_rad, transform_points


def rotation_translation_error(transform_pred: np.ndarray, transform_gt: np.ndarray) -> tuple[float, float]:
    rotation_error = math.degrees(so3_distance_rad(transform_gt[:3, :3], transform_pred[:3, :3]))
    translation_error = float(np.linalg.norm(transform_gt[:3, 3] - transform_pred[:3, 3]))
    return rotation_error, translation_error


def add_metric(model_points: np.ndarray, transform_pred: np.ndarray, transform_gt: np.ndarray) -> float:
    pred = transform_points(transform_pred, model_points)
    gt = transform_points(transform_gt, model_points)
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def add_s_metric(model_points: np.ndarray, transform_pred: np.ndarray, transform_gt: np.ndarray) -> float:
    pred = transform_points(transform_pred, model_points)
    gt = transform_points(transform_gt, model_points)
    pairwise = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    return float(np.mean(np.min(pairwise, axis=1)))


def inlier_ratio_model_to_scene(model_points_transformed: np.ndarray, scene_points: np.ndarray, radius: float) -> float:
    if model_points_transformed.size == 0 or scene_points.size == 0:
        return 0.0
    pairwise = np.linalg.norm(
        model_points_transformed[:, None, :] - np.asarray(scene_points, dtype=np.float64)[None, :, :],
        axis=2,
    )
    min_dist = np.min(pairwise, axis=1)
    return float(np.mean(min_dist <= float(radius)))


def compute_metrics(
    model_points: np.ndarray,
    scene_points: np.ndarray,
    transform_pred: np.ndarray,
    transform_gt: Optional[np.ndarray],
    inlier_radius: float,
) -> Dict[str, float]:
    model_points = np.asarray(model_points, dtype=np.float64)
    scene_points = np.asarray(scene_points, dtype=np.float64)
    pred_points = transform_points(transform_pred, model_points)
    metrics: Dict[str, float] = {
        "inlier_ratio": inlier_ratio_model_to_scene(pred_points, scene_points, inlier_radius),
    }
    if transform_gt is None:
        metrics["ADD"] = float("nan")
        metrics["ADD_S"] = float("nan")
        metrics["rotation_error_deg"] = float("nan")
        metrics["translation_error"] = float("nan")
        return metrics

    metrics["ADD"] = add_metric(model_points, transform_pred, transform_gt)
    metrics["ADD_S"] = add_s_metric(model_points, transform_pred, transform_gt)
    rotation_error, translation_error = rotation_translation_error(transform_pred, transform_gt)
    metrics["rotation_error_deg"] = float(rotation_error)
    metrics["translation_error"] = float(translation_error)
    return metrics


def success_from_metrics(dataset_name: str, metrics: Dict[str, float], threshold: float) -> bool:
    dataset = dataset_name.strip().lower()
    if dataset == "lmo":
        value = metrics.get("ADD_S", float("inf"))
    else:
        value = metrics.get("ADD", float("inf"))
    return bool(value == value and float(value) <= float(threshold))
