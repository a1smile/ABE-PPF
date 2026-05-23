import copy
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from .clustering import PoseWithVotes, cluster_poses
from .going_further_ppf import _estimate_normals_for_cloud, _method_specific_target_spacing
from .model_builder import BaselineHashTable, PPFModel
from .model_cache_io import load_ppf_model_cache
from .ppf_features import angle_from_transformed_point, compute_pair_features, to_internal_feature_g
from .preprocess import (
    adaptive_subsample_and_calculate_normals_model,
    adaptive_subsample_and_calculate_normals_scene,
    subsample_and_calculate_normals_model,
    subsample_and_calculate_normals_scene,
)
from .registration import RegistrationStats, compute_transform_sg
from .rsmrq_hash import PPFEntry
from .utils import (
    Timer,
    compose_affine,
    invert_affine,
    make_affine,
    rotation_matrix_from_axis_angle,
    set_global_seed,
    transform_points,
    wrap_to_pi,
)


def make_edge_enhanced_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = copy.deepcopy(cfg)
    runtime["enable_rsmrq"] = False
    runtime["enable_robust_vote"] = False
    runtime["enable_kde_refine"] = False
    runtime["store_pair_features"] = False

    icp_cfg = dict(runtime.get("icp_refine", {}) or {})
    icp_cfg["enable"] = False
    runtime["icp_refine"] = icp_cfg

    pose_sel = dict(runtime.get("pose_selection", {}) or {})
    pose_sel["enable"] = False
    runtime["pose_selection"] = pose_sel

    pose_cluster = dict(runtime.get("pose_clustering", {}) or {})
    pose_cluster["enable"] = False
    runtime["pose_clustering"] = pose_cluster

    runtime["final_pose_policy"] = str(runtime.get("final_pose_policy", "edge_verify"))

    ext_meta = dict(runtime.get("external_method", {}) or {})
    ext_meta.update(
        {
            "name": "edge_enhanced_ppf",
            "paper": "Liu et al., Computational Visual Media 2023/2024",
            "implementation_style": "pure_python_repo_reproduction",
            "comparison_mode": "stanford_base_consistent_plus_method_scene_sampling",
            "incremental_modules_over_drost": [
                "model_pair_angle_filtering",
                "edge_preserving_scene_sampling",
                "diameter_radius_intelligent_pair_sampling",
                "hierarchical_pose_clustering",
                "edge_matching_degree_verification",
                "verification_early_exit",
            ],
            "disabled_repo_enhancements": [
                "rsmrq",
                "robust_vote",
                "pose_selection",
                "mode_clustering",
                "kde_refine",
                "icp_refine",
            ],
        }
    )
    runtime["external_method"] = ext_meta
    return runtime


def _edge_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "model_pair_normal_angle_min_deg": 5.0,
        "model_pair_normal_angle_max_deg": 175.0,
        "scene_sampling_use_edge_preservation": True,
        "scene_candidate_voxel_ratio": 0.60,
        "scene_non_edge_voxel_scale": 1.15,
        "edge_normal_var_percentile": 75.0,
        "edge_curvature_percentile": 75.0,
        "edge_normal_var_abs_deg": 18.0,
        "edge_curvature_abs": 0.015,
        "edge_knn": 12,
        "min_edge_fraction": 0.15,
        "max_edge_fraction": 0.45,
        "intelligent_sampling_radius_scale": 1.0,
        "top_mr_per_reference": 24,
        "verification_top_n": 9,
        "verification_vote_split_ratio": 0.50,
        "verification_accept_high": 0.70,
        "verification_accept_low": 0.60,
        "verification_roi_scale": 1.40,
        "verification_edge_match_radius": 0.008,
        "verification_cluster_eps_scale": 0.08,
        "verification_cluster_min_points": 6,
        "final_selection_policy": "edge_verify",
    }
    out = dict(base)
    out.update(dict(cfg.get("edge_enhanced_ppf", {}) or {}))
    return out


def _safe_normal_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    n1n = n1 / (np.linalg.norm(n1) + 1.0e-12)
    n2n = n2 / (np.linalg.norm(n2) + 1.0e-12)
    dot = float(np.clip(np.dot(n1n, n2n), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _make_pcd(points: np.ndarray, normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None and np.asarray(normals).shape == np.asarray(points).shape:
        out.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return out


def _edge_descriptors(
    pts: np.ndarray,
    normals: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    kd = o3d.geometry.KDTreeFlann(pcd)
    normal_var = np.zeros(pts.shape[0], dtype=np.float64)
    curvature = np.zeros(pts.shape[0], dtype=np.float64)

    for i in range(pts.shape[0]):
        _, idxs, _ = kd.search_knn_vector_3d(pts[i], max(2, int(k) + 1))
        idxs = [int(x) for x in idxs if int(x) != int(i)]
        if not idxs:
            continue

        local_normals = normals[idxs]
        angles = [_safe_normal_angle_deg(normals[i], n) for n in local_normals]
        normal_var[i] = float(np.mean(angles)) if angles else 0.0

        local_pts = pts[[i] + idxs]
        centered = local_pts - np.mean(local_pts, axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, local_pts.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        s = float(np.sum(eigvals))
        curvature[i] = float(eigvals[0] / s) if s > 1.0e-12 else 0.0

    return normal_var, curvature


def _edge_mask_from_descriptors(
    normal_var: np.ndarray,
    curvature: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    params = _edge_cfg(cfg)
    if normal_var.size == 0:
        return np.zeros(0, dtype=bool), {
            "normal_var_threshold_deg": 0.0,
            "curvature_threshold": 0.0,
            "edge_fraction": 0.0,
        }

    nv_thr = max(
        float(params["edge_normal_var_abs_deg"]),
        float(np.percentile(normal_var, float(params["edge_normal_var_percentile"]))),
    )
    curv_thr = max(
        float(params["edge_curvature_abs"]),
        float(np.percentile(curvature, float(params["edge_curvature_percentile"]))),
    )
    mask = (normal_var >= nv_thr) | (curvature >= curv_thr)

    min_edge_fraction = float(params["min_edge_fraction"])
    max_edge_fraction = float(params["max_edge_fraction"])
    score = normal_var / max(1.0e-6, float(np.max(normal_var)))
    if np.max(curvature) > 0:
        score = score + curvature / float(np.max(curvature))

    n = int(mask.shape[0])
    min_keep = int(math.ceil(min_edge_fraction * n))
    max_keep = int(math.ceil(max_edge_fraction * n))
    order = np.argsort(-score)
    if int(np.count_nonzero(mask)) < min_keep:
        mask = np.zeros(n, dtype=bool)
        mask[order[:max(1, min_keep)]] = True
    elif int(np.count_nonzero(mask)) > max_keep:
        keep = np.zeros(n, dtype=bool)
        keep[order[:max(1, max_keep)]] = True
        mask = keep

    return mask, {
        "normal_var_threshold_deg": float(nv_thr),
        "curvature_threshold": float(curv_thr),
        "edge_fraction": float(np.mean(mask)) if mask.size > 0 else 0.0,
    }


def _extract_edge_points(
    pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, Dict[str, Any]]:
    pts = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if pts.shape[0] == 0 or normals.shape != pts.shape:
        empty = _make_pcd(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64))
        return empty, empty, {"edge_points": 0, "non_edge_points": 0}

    normal_var, curvature = _edge_descriptors(pts, normals, pcd, k=int(_edge_cfg(cfg)["edge_knn"]))
    mask, dbg = _edge_mask_from_descriptors(normal_var, curvature, cfg)
    edge_pts = pts[mask]
    edge_normals = normals[mask]
    non_edge_pts = pts[~mask]
    non_edge_normals = normals[~mask]

    edge_pcd = _make_pcd(edge_pts, edge_normals)
    non_edge_pcd = _make_pcd(non_edge_pts, non_edge_normals)
    dbg.update(
        {
            "edge_points": int(edge_pts.shape[0]),
            "non_edge_points": int(non_edge_pts.shape[0]),
            "mean_normal_var_deg": float(np.mean(normal_var)) if normal_var.size > 0 else 0.0,
            "mean_curvature": float(np.mean(curvature)) if curvature.size > 0 else 0.0,
        }
    )
    return edge_pcd, non_edge_pcd, dbg


def _combine_clouds(clouds: Sequence[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    pts_list: List[np.ndarray] = []
    normals_list: List[np.ndarray] = []
    for cloud in clouds:
        if len(cloud.points) == 0:
            continue
        pts = np.asarray(cloud.points, dtype=np.float64)
        normals = np.asarray(cloud.normals, dtype=np.float64)
        pts_list.append(pts)
        if normals.shape == pts.shape:
            normals_list.append(normals)
    if not pts_list:
        return _make_pcd(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64))
    pts = np.concatenate(pts_list, axis=0)
    normals = np.concatenate(normals_list, axis=0) if normals_list else None
    return _make_pcd(pts, normals)


def _edge_preserving_scene_sampling(
    cloud_scene: o3d.geometry.PointCloud,
    normal_k: int,
    sampling_leaf: float,
    adaptive_downsample: bool,
    adaptive_apply_to: str,
    adaptive_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, Dict[str, Any]]:
    params = _edge_cfg(cfg)
    if adaptive_downsample and adaptive_apply_to in ("scene", "both"):
        base_voxel, target_points = _method_specific_target_spacing(cloud_scene, adaptive_cfg)
        if base_voxel <= 0.0:
            base_voxel = float(sampling_leaf)
    else:
        base_voxel = float(sampling_leaf)
        target_points = None

    candidate_voxel = max(1.0e-4, float(base_voxel) * float(params["scene_candidate_voxel_ratio"]))
    candidate_raw = cloud_scene.voxel_down_sample(candidate_voxel)
    candidate = _estimate_normals_for_cloud(candidate_raw, k=normal_k, orient_model=False)

    edge_cloud, non_edge_cloud, edge_dbg = _extract_edge_points(candidate, cfg)
    non_edge_voxel = max(candidate_voxel, float(base_voxel) * float(params["scene_non_edge_voxel_scale"]))
    if len(non_edge_cloud.points) > 0:
        non_edge_coarse_raw = non_edge_cloud.voxel_down_sample(non_edge_voxel)
        non_edge_coarse = _estimate_normals_for_cloud(non_edge_coarse_raw, k=normal_k, orient_model=False)
    else:
        non_edge_coarse = _make_pcd(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64))

    scene_final = _combine_clouds([edge_cloud, non_edge_coarse])
    debug = {
        "raw_points": int(len(cloud_scene.points)),
        "target_points": target_points,
        "candidate_voxel_used": float(candidate_voxel),
        "base_voxel_used": float(base_voxel),
        "candidate_points": int(len(candidate.points)),
        "edge_points": int(len(edge_cloud.points)),
        "non_edge_points": int(len(non_edge_cloud.points)),
        "non_edge_coarse_points": int(len(non_edge_coarse.points)),
        "down_points": int(len(scene_final.points)),
        "edge_debug": edge_dbg,
    }
    return scene_final, edge_cloud, debug


def build_edge_enhanced_model(
    model_pcd: o3d.geometry.PointCloud,
    angle_step: float,
    distance_step: float,
    cfg: Dict[str, Any],
    logger=None,
) -> PPFModel:
    params = _edge_cfg(cfg)
    pts = np.asarray(model_pcd.points, dtype=np.float64)
    normals = np.asarray(model_pcd.normals, dtype=np.float64)
    n_pts = int(pts.shape[0])

    edge_cloud, _, edge_dbg = _extract_edge_points(model_pcd, cfg)
    edge_pts = np.asarray(edge_cloud.points, dtype=np.float64)
    edge_normals = np.asarray(edge_cloud.normals, dtype=np.float64)

    alpha_m = [[0.0 for _ in range(n_pts)] for __ in range(n_pts)]
    ref_R: List[np.ndarray] = [None] * n_pts  # type: ignore
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)
    for i in range(n_pts):
        ni = normals[i]
        ni = ni / (np.linalg.norm(ni) + 1.0e-12)
        axis = np.cross(ni, ex)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            axis = ey.copy()
            axis_norm = 1.0
        axis /= axis_norm
        angle = math.acos(max(-1.0, min(1.0, float(ni @ ex))))
        ref_R[i] = rotation_matrix_from_axis_angle(axis, angle)

    hash_table = BaselineHashTable(angle_step, distance_step)
    model_diameter = 0.0
    inserted_pairs = 0
    filtered_low_normal_angle = 0
    filtered_high_normal_angle = 0

    angle_min = float(params["model_pair_normal_angle_min_deg"])
    angle_max = float(params["model_pair_normal_angle_max_deg"])
    for i in range(n_pts):
        pi = pts[i]
        ni = normals[i]
        Ri = ref_R[i]
        for j in range(n_pts):
            if i == j:
                continue
            pj = pts[j]
            nj = normals[j]
            n_angle = _safe_normal_angle_deg(ni, nj)
            if n_angle < angle_min:
                filtered_low_normal_angle += 1
                continue
            if n_angle > angle_max:
                filtered_high_normal_angle += 1
                continue

            feat = compute_pair_features(pi, ni, pj, nj)
            if feat is None:
                continue
            f1, f2, f3, f4 = feat
            g = to_internal_feature_g(f1, f2, f3, f4)
            hash_table.add(g, PPFEntry(mr=i, mi=j, g=None))
            inserted_pairs += 1

            pj_mg = Ri @ (pj - pi)
            alpha_m[i][j] = -angle_from_transformed_point(pj_mg)
            if f4 > model_diameter:
                model_diameter = float(f4)

    model = PPFModel(
        angle_step=angle_step,
        distance_step=distance_step,
        alpha_m=alpha_m,
        model_diameter=float(model_diameter),
        ref_R=ref_R,
        pts=pts,
        normals=normals,
        hash_table=hash_table,
        enable_rsmrq=False,
        merge_mode="union",
    )
    model.edge_points = edge_pts
    model.edge_normals = edge_normals
    model.edge_enhanced_meta = {
        "inserted_pairs": int(inserted_pairs),
        "filtered_low_normal_angle": int(filtered_low_normal_angle),
        "filtered_high_normal_angle": int(filtered_high_normal_angle),
        "edge_points": int(edge_pts.shape[0]),
        "edge_sampling_debug": edge_dbg,
    }

    if logger:
        logger.info(
            "[EdgeEnhancedModel] "
            f"N={n_pts} inserted_pairs={inserted_pairs} model_diameter={model_diameter:.6f} "
            f"edge_points={edge_pts.shape[0]}"
        )
        logger.info(
            "[EdgeEnhancedModel] "
            f"filtered_low_normal_angle={filtered_low_normal_angle} "
            f"filtered_high_normal_angle={filtered_high_normal_angle}"
        )
    return model


def _model_edge_points(model: PPFModel) -> np.ndarray:
    edge_pts = getattr(model, "edge_points", None)
    if isinstance(edge_pts, np.ndarray) and edge_pts.ndim == 2 and edge_pts.shape[1] == 3 and edge_pts.shape[0] > 0:
        return edge_pts
    return model.pts


def _get_model_meta(model: PPFModel) -> Dict[str, Any]:
    meta = getattr(model, "edge_enhanced_meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _edge_match_score(
    T: np.ndarray,
    model: PPFModel,
    scene_edge_cloud: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    params = _edge_cfg(cfg)
    scene_edge_pts = np.asarray(scene_edge_cloud.points, dtype=np.float64)
    if scene_edge_pts.shape[0] == 0:
        return 0.0, {"n_roi": 0, "n_match": 0, "n_filtered_roi": 0}

    model_pts_tf = transform_points(T, model.pts)
    model_edge_tf = transform_points(T, _model_edge_points(model))

    bbox_min = model_pts_tf.min(axis=0)
    bbox_max = model_pts_tf.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    half = 0.5 * (bbox_max - bbox_min) * float(params["verification_roi_scale"])
    roi_mask = np.all((scene_edge_pts >= (center - half)) & (scene_edge_pts <= (center + half)), axis=1)
    roi_edge_pts = scene_edge_pts[roi_mask]
    if roi_edge_pts.shape[0] == 0:
        return 0.0, {"n_roi": 0, "n_match": 0, "n_filtered_roi": 0}

    roi_pcd = _make_pcd(roi_edge_pts)
    eps = max(0.004, float(model.model_diameter) * float(params["verification_cluster_eps_scale"]))
    labels = np.asarray(
        roi_pcd.cluster_dbscan(
            eps=eps,
            min_points=int(params["verification_cluster_min_points"]),
            print_progress=False,
        )
    )
    filtered_roi_pts = roi_edge_pts
    selected_cluster = -1
    if np.any(labels >= 0):
        clusters = []
        for label in sorted(set(int(x) for x in labels if int(x) >= 0)):
            idxs = np.where(labels == label)[0]
            if idxs.size == 0:
                continue
            centroid = np.mean(roi_edge_pts[idxs], axis=0)
            dist = float(np.linalg.norm(centroid - center))
            clusters.append((dist, label, idxs))
        if clusters:
            clusters.sort(key=lambda x: x[0])
            _, selected_cluster, idxs = clusters[0]
            filtered_roi_pts = roi_edge_pts[idxs]

    model_edge_pcd = _make_pcd(model_edge_tf)
    kd = o3d.geometry.KDTreeFlann(model_edge_pcd)
    match_radius = float(params["verification_edge_match_radius"])
    n_match = 0
    for i in range(filtered_roi_pts.shape[0]):
        k, _, _ = kd.search_radius_vector_3d(filtered_roi_pts[i], match_radius)
        if k > 0:
            n_match += 1

    score = float(n_match) / max(1, filtered_roi_pts.shape[0])
    return score, {
        "n_roi": int(roi_edge_pts.shape[0]),
        "n_filtered_roi": int(filtered_roi_pts.shape[0]),
        "n_match": int(n_match),
        "selected_cluster": int(selected_cluster),
        "eps": float(eps),
    }


def _verify_clustered_candidates(
    clustered: List[PoseWithVotes],
    model: PPFModel,
    scene_edge_cloud: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any], float]:
    params = _edge_cfg(cfg)
    if not clustered:
        return None, {"evaluated_candidates": 0, "final_source": "empty"}, 0.0

    t0 = time.perf_counter()
    top_n = max(1, int(params["verification_top_n"]))
    split_ratio = float(params["verification_vote_split_ratio"])
    accept_high = float(params["verification_accept_high"])
    accept_low = float(params["verification_accept_low"])

    vmax = max(float(c.votes) for c in clustered)
    first = [c for c in clustered if float(c.votes) >= split_ratio * vmax]
    second = [c for c in clustered if float(c.votes) < split_ratio * vmax]

    def _eval(cands: Sequence[PoseWithVotes], stage: str) -> List[Dict[str, Any]]:
        out = []
        for rank, cand in enumerate(cands[:top_n]):
            score, score_dbg = _edge_match_score(cand.T, model, scene_edge_cloud, cfg)
            out.append(
                {
                    "stage": stage,
                    "rank": int(rank),
                    "vote": float(cand.votes),
                    "score": float(score),
                    "debug": score_dbg,
                    "T": cand.T,
                }
            )
        out.sort(key=lambda x: (x["score"], x["vote"]), reverse=True)
        return out

    first_eval = _eval(first, "first")
    second_eval: List[Dict[str, Any]] = []
    final_T = None
    final_source = "cluster_fallback"

    if first_eval:
        best_first = first_eval[0]
        if float(best_first["score"]) > accept_high:
            final_T = best_first["T"]
            final_source = "first_stage_high_accept"
        elif float(best_first["score"]) > accept_low:
            final_T = best_first["T"]
            final_source = "first_stage_low_accept"

    if final_T is None:
        second_eval = _eval(second, "second")
        combined = sorted(first_eval + second_eval, key=lambda x: (x["score"], x["vote"]), reverse=True)
        if combined:
            final_T = combined[0]["T"]
            final_source = "best_verified_candidate"
        elif clustered:
            final_T = clustered[0].T
            final_source = "cluster_fallback"

    verify_time = time.perf_counter() - t0
    debug = {
        "vmax": float(vmax),
        "first_stage_count": int(len(first)),
        "second_stage_count": int(len(second)),
        "evaluated_candidates": int(len(first_eval) + len(second_eval)),
        "first_stage_top": [
            {
                "vote": float(x["vote"]),
                "score": float(x["score"]),
                "n_roi": int(x["debug"]["n_roi"]),
                "n_filtered_roi": int(x["debug"]["n_filtered_roi"]),
                "n_match": int(x["debug"]["n_match"]),
            }
            for x in first_eval[:5]
        ],
        "second_stage_top": [
            {
                "vote": float(x["vote"]),
                "score": float(x["score"]),
                "n_roi": int(x["debug"]["n_roi"]),
                "n_filtered_roi": int(x["debug"]["n_filtered_roi"]),
                "n_match": int(x["debug"]["n_match"]),
            }
            for x in second_eval[:5]
        ],
        "final_source": final_source,
        "verify_time": float(verify_time),
    }
    return final_T, debug, float(verify_time)


def _edge_enhanced_match(
    model: PPFModel,
    scene_pcd: o3d.geometry.PointCloud,
    scene_edge_cloud: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any], float, float, float]:
    params = _edge_cfg(cfg)
    scene_pts = np.asarray(scene_pcd.points, dtype=np.float64)
    scene_normals = np.asarray(scene_pcd.normals, dtype=np.float64)
    if scene_pts.shape[0] == 0 or scene_normals.shape != scene_pts.shape:
        return np.eye(4, dtype=float), {}, 0.0, 0.0, 0.0

    aux_size = int(math.ceil(2.0 * math.pi / model.angle_step))
    radius = float(model.model_diameter) * float(params["intelligent_sampling_radius_scale"])
    scene_ref_sampling_rate = max(1, int(cfg.get("scene_ref_sampling_rate", 20)))
    pos_thresh = float(cfg.get("pos_thresh", 0.005))
    rot_thresh_rad = math.radians(float(cfg.get("rot_thresh_deg", 30.0)))
    kdtree = o3d.geometry.KDTreeFlann(scene_pcd)
    accumulator = np.zeros((len(model.pts), aux_size), dtype=np.float32)
    top_mr_per_reference = max(1, int(params["top_mr_per_reference"]))

    candidate_inflations: List[float] = []
    voted_poses: List[PoseWithVotes] = []

    t_front = time.perf_counter()
    for sr in range(0, scene_pts.shape[0], scene_ref_sampling_rate):
        sr_p = scene_pts[int(sr)]
        sr_n = scene_normals[int(sr)]
        T_sg, _ = compute_transform_sg(sr_p, sr_n)
        _, idxs, _ = kdtree.search_radius_vector_3d(sr_p, radius)

        for si in idxs:
            if int(si) == int(sr):
                continue
            si_p = scene_pts[int(si)]
            si_n = scene_normals[int(si)]
            feat = compute_pair_features(sr_p, sr_n, si_p, si_n)
            if feat is None:
                continue

            f1, f2, f3, f4 = feat
            gs = to_internal_feature_g(f1, f2, f3, f4)
            si_trans = T_sg[:3, :3] @ (si_p - sr_p)
            alpha_s = -angle_from_transformed_point(si_trans)
            buckets = model.hash_table.query_buckets(gs)
            base_sz = len(buckets[0]) if buckets else 0
            candidate_inflations.append(1.0 if base_sz > 0 else 0.0)

            bucket = buckets[0] if buckets else []
            for e in bucket:
                alpha = wrap_to_pi(model.alpha_m[e.mr][e.mi] - alpha_s)
                bin_j = int(math.floor((alpha + math.pi) / model.angle_step))
                bin_j = max(0, min(aux_size - 1, bin_j))
                accumulator[e.mr, bin_j] += 1.0

        mr_votes = np.max(accumulator, axis=1) if accumulator.size > 0 else np.zeros(len(model.pts), dtype=np.float32)
        active_mrs = np.where(mr_votes > 0)[0]
        if active_mrs.size > top_mr_per_reference:
            active_scores = mr_votes[active_mrs]
            keep_local = np.argpartition(-active_scores, top_mr_per_reference - 1)[:top_mr_per_reference]
            active_mrs = active_mrs[keep_local]

        for mr in active_mrs.tolist():
            row = accumulator[mr]
            votes = float(np.max(row))
            if votes <= 0.0:
                continue
            peak_idx = int(np.argmax(row))
            theta = (float(peak_idx) + 0.5) * model.angle_step - math.pi
            R_x = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), float(theta))
            T_x = make_affine(R_x, np.zeros(3, dtype=float))
            T_mg = make_affine(model.ref_R[mr], -model.ref_R[mr] @ model.pts[mr])
            T = compose_affine(invert_affine(T_sg), compose_affine(T_x, T_mg))
            voted_poses.append(PoseWithVotes(T=T, votes=float(votes)))

        accumulator.fill(0)

    ppf_frontend_time = time.perf_counter() - t_front
    t_back = time.perf_counter()
    clustered = cluster_poses(voted_poses, pos_thresh, rot_thresh_rad)
    legacy_clustering_time = time.perf_counter() - t_back

    T_verified, verify_debug, verify_time = _verify_clustered_candidates(clustered, model, scene_edge_cloud, cfg)
    if T_verified is not None:
        T_best = T_verified
        final_source = str(verify_debug.get("final_source", "edge_verify"))
    elif clustered:
        T_best = clustered[0].T
        final_source = "cluster_fallback"
    elif voted_poses:
        T_best = max(voted_poses, key=lambda x: float(x.votes)).T
        final_source = "raw_vote_fallback"
    else:
        T_best = np.eye(4, dtype=float)
        final_source = "identity_empty"

    backend_time = time.perf_counter() - t_back
    debug = {
        "candidate_inflation_mean": float(np.mean(candidate_inflations)) if candidate_inflations else 1.0,
        "edge_enhanced_ppf": {
            "radius_used": float(radius),
            "raw_pose_count": int(len(voted_poses)),
            "clustered_pose_count": int(len(clustered)),
            "scene_edge_points_for_verification": int(len(scene_edge_cloud.points)),
            "verification": verify_debug,
            "final_source": final_source,
        },
        "model_build": _get_model_meta(model),
        "timing": {
            "ppf_frontend_time": float(ppf_frontend_time),
            "legacy_clustering_time": float(legacy_clustering_time),
            "backend_time": float(backend_time),
            "verification_time": float(verify_time),
        },
        "final_pose_policy": "edge_enhanced_verify",
        "final_pose_path": final_source,
    }
    return T_best, debug, float(ppf_frontend_time), float(legacy_clustering_time), float(backend_time)


def run_edge_enhanced_registration(
    model_path: str,
    scene_path: str,
    cfg: Dict[str, Any],
    logger=None,
    model_cache_path: Optional[str] = None,
    strict_cache: bool = True,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud, Dict[str, Any], RegistrationStats]:
    runtime_cfg = make_edge_enhanced_config(cfg)
    seed = int(runtime_cfg.get("seed", 0))
    set_global_seed(seed)

    sampling_leaf = float(runtime_cfg.get("sampling_leaf", 5.0))
    normal_k = int(runtime_cfg.get("normal_k", 5))
    angle_step = math.radians(float(runtime_cfg.get("angle_step_deg", 12.0)))
    distance_step = float(runtime_cfg.get("distance_step_ratio", 0.6)) * sampling_leaf
    adaptive_downsample = bool(runtime_cfg.get("adaptive_downsample", False))
    adaptive_apply_to = str(runtime_cfg.get("adaptive_downsample_apply_to", "scene")).lower()
    adaptive_cfg = dict(runtime_cfg.get("adaptive_downsample_cfg", {}) or {})

    t0 = time.perf_counter()
    cloud_model = o3d.io.read_point_cloud(model_path)
    if len(cloud_model.points) == 0:
        raise ValueError(f"Empty model: {model_path}")
    cloud_scene = o3d.io.read_point_cloud(scene_path)
    if len(cloud_scene.points) == 0:
        raise ValueError(f"Empty scene: {scene_path}")

    model_ds_info: Dict[str, Any] = {
        "raw_points": len(cloud_model.points),
        "down_points": None,
        "target_points": None,
        "voxel_used": sampling_leaf,
        "adaptive_enabled": adaptive_downsample and adaptive_apply_to in ("model", "both"),
        "apply_to": adaptive_apply_to,
    }

    if model_cache_path:
        with Timer("model_load_cache", logger=logger) as tm:
            ppf_model = load_ppf_model_cache(model_cache_path, cfg=runtime_cfg, strict=strict_cache)
        if not isinstance(_get_model_meta(ppf_model), dict):
            raise ValueError(f"Edge-enhanced cache missing method metadata: {model_cache_path}")
        model_ds_info["cached_model_meta"] = _get_model_meta(ppf_model)
    else:
        with Timer("model_build", logger=logger) as tm:
            if adaptive_downsample and adaptive_apply_to in ("model", "both"):
                model_down, model_ds_info = adaptive_subsample_and_calculate_normals_model(
                    pcd=cloud_model,
                    k=normal_k,
                    cfg=adaptive_cfg,
                )
                model_ds_info["adaptive_enabled"] = True
                model_ds_info["apply_to"] = adaptive_apply_to
            else:
                model_down = subsample_and_calculate_normals_model(
                    pcd=cloud_model,
                    voxel_size=sampling_leaf,
                    k=normal_k,
                )
                model_ds_info = {
                    "raw_points": len(cloud_model.points),
                    "down_points": len(model_down.points),
                    "target_points": None,
                    "voxel_used": sampling_leaf,
                    "adaptive_enabled": False,
                    "apply_to": adaptive_apply_to,
                }

            ppf_model = build_edge_enhanced_model(model_down, angle_step, distance_step, runtime_cfg, logger=logger)

    scene_preprocess_time = 0.0
    with Timer("registration", logger=logger) as tr:
        t_scene_pre = time.perf_counter()
        if bool(_edge_cfg(runtime_cfg).get("scene_sampling_use_edge_preservation", True)):
            scene_down, scene_edge_cloud, scene_ds_info = _edge_preserving_scene_sampling(
                cloud_scene=cloud_scene,
                normal_k=normal_k,
                sampling_leaf=sampling_leaf,
                adaptive_downsample=adaptive_downsample,
                adaptive_apply_to=adaptive_apply_to,
                adaptive_cfg=adaptive_cfg,
                cfg=runtime_cfg,
            )
        elif adaptive_downsample and adaptive_apply_to in ("scene", "both"):
            scene_down, scene_ds_info = adaptive_subsample_and_calculate_normals_scene(
                pcd=cloud_scene,
                k=normal_k,
                cfg=adaptive_cfg,
            )
            scene_edge_cloud, _, _ = _extract_edge_points(scene_down, runtime_cfg)
        else:
            scene_down = subsample_and_calculate_normals_scene(
                pcd=cloud_scene,
                voxel_size=sampling_leaf,
                k=normal_k,
            )
            scene_edge_cloud, _, _ = _extract_edge_points(scene_down, runtime_cfg)
            scene_ds_info = {
                "raw_points": len(cloud_scene.points),
                "down_points": len(scene_down.points),
                "target_points": None,
                "voxel_used": sampling_leaf,
                "adaptive_enabled": False,
                "apply_to": adaptive_apply_to,
            }
        scene_preprocess_time = time.perf_counter() - t_scene_pre

        T_pred, debug, ppf_frontend_time, legacy_clustering_time, backend_time = _edge_enhanced_match(
            model=ppf_model,
            scene_pcd=scene_down,
            scene_edge_cloud=scene_edge_cloud,
            cfg=runtime_cfg,
            logger=logger,
        )
        debug["model_downsample"] = model_ds_info
        debug["scene_downsample"] = scene_ds_info

    out_model = o3d.geometry.PointCloud()
    out_pts = (T_pred[:3, :3] @ ppf_model.pts.T).T + T_pred[:3, 3]
    out_model.points = o3d.utility.Vector3dVector(out_pts)
    total_time = time.perf_counter() - t0

    debug = copy.deepcopy(debug) if isinstance(debug, dict) else {}
    debug["external_method"] = dict(runtime_cfg.get("external_method", {}))
    debug["external_method"]["effective_config"] = {
        "adaptive_downsample": bool(runtime_cfg.get("adaptive_downsample", False)),
        "final_pose_policy": str(runtime_cfg.get("final_pose_policy", "")),
        "edge_enhanced_ppf": dict(runtime_cfg.get("edge_enhanced_ppf", {}) or {}),
    }
    debug["external_method"]["reproduction_notes"] = (
        "Edge-enhanced PPF round-1 reproduction on top of the repo's Stanford scaffold. "
        "This implementation adds model pair-angle filtering, edge-preserving scene sampling, "
        "diameter-radius pair search, and edge-based pose verification with early exit, while "
        "leaving the paper's ICP refinement and some engineering details as future work."
    )

    stats = RegistrationStats(
        model_build_time=float(tm.elapsed),
        registration_time=float(tr.elapsed),
        total_time=float(total_time),
        candidate_inflation_mean=float(debug.get("candidate_inflation_mean", 1.0)),
        robust_vote_summary={},
        kde_refine_calls=0,
        scene_preprocess_time=float(scene_preprocess_time),
        ppf_frontend_time=float(ppf_frontend_time),
        pose_selection_time=0.0,
        pose_clustering_time=0.0,
        legacy_clustering_time=float(legacy_clustering_time),
        backend_time=float(backend_time),
        final_pose_policy=str(debug.get("final_pose_policy", runtime_cfg.get("final_pose_policy", ""))),
        final_pose_path=str(debug.get("final_pose_path", "")),
    )
    return T_pred, out_model, debug, stats
