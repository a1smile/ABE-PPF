from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class PoseRefinementConfig:
    enable: bool = False
    max_iterations: int = 15
    max_correspondence_distance: float = 0.01


def refine_pose(
    model_points: np.ndarray,
    scene_points: np.ndarray,
    initial_transform: np.ndarray,
    cfg: PoseRefinementConfig,
) -> np.ndarray:
    if not bool(cfg.enable):
        return np.asarray(initial_transform, dtype=np.float64)

    model = o3d.geometry.PointCloud()
    scene = o3d.geometry.PointCloud()
    model.points = o3d.utility.Vector3dVector(np.asarray(model_points, dtype=np.float64))
    scene.points = o3d.utility.Vector3dVector(np.asarray(scene_points, dtype=np.float64))

    result = o3d.pipelines.registration.registration_icp(
        source=model,
        target=scene,
        max_correspondence_distance=float(cfg.max_correspondence_distance),
        init=np.asarray(initial_transform, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(cfg.max_iterations)),
    )
    return np.asarray(result.transformation, dtype=np.float64)
