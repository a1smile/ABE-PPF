import copy
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from .clustering import PoseWithVotes, cluster_poses
from .model_builder import BaselineHashTable, PPFModel
from .model_cache_io import load_ppf_model_cache
from .ppf_features import (
    angle_from_transformed_point,
    compute_pair_features,
    discretize_baseline,
    to_internal_feature_g,
)
from .preprocess import (
    _downsample_by_target_points,
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
    wrap_to_pi,
)


def make_going_further_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
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

    runtime["final_pose_policy"] = str(runtime.get("final_pose_policy", "raw_top1_vote"))

    ext_meta = dict(runtime.get("external_method", {}) or {})
    ext_meta.update(
        {
            "name": "going_further_with_ppf",
            "paper": "Hinterstoisser et al., ECCV 2016",
            "implementation_style": "pure_python_repo_reproduction",
            "comparison_mode": "stanford_base_consistent",
            "incremental_modules_over_drost": [
                "normal_aware_close_pair_retention",
                "two_ball_scene_pair_sampling",
                "feature_spreading",
                "rotation_neighbor_voting",
                "quantized_feature_rotation_vote_deduplication",
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


def _going_further_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "keep_close_pairs_with_normal_angle_deg": 30.0,
        "close_pair_min_distance_ratio": 1.0,
        "two_ball_enable": True,
        "small_ball_radius_scale": 1.0,
        "large_ball_radius_scale": 1.0,
        "feature_neighbor_steps": 1,
        "vote_angle_neighbor_steps": 1,
        "deduplicate_quantized_votes": True,
        "final_selection_policy": "raw_top1_vote",
        "use_method_specific_sampling": True,
        "method_specific_sampling_apply_to": "model",
        "candidate_voxel_ratio": 0.5,
        "max_close_retain_per_anchor": 2,
        "min_close_extra_distance_ratio": 0.20,
    }
    out = dict(base)
    out.update(dict(cfg.get("going_further_ppf", {}) or {}))
    return out


def _safe_normal_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    n1n = n1 / (np.linalg.norm(n1) + 1.0e-12)
    n2n = n2 / (np.linalg.norm(n2) + 1.0e-12)
    dot = float(np.clip(np.dot(n1n, n2n), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _flip_model_normals_like_repo(
    pts: np.ndarray,
    normals: np.ndarray,
    orig_pts: np.ndarray,
) -> np.ndarray:
    if pts.shape[0] == 0:
        return normals

    centroid = pts.mean(axis=0)
    out = normals.copy()
    for i in range(out.shape[0]):
        if i < orig_pts.shape[0]:
            orientation_reference = centroid - orig_pts[i]
        else:
            orientation_reference = centroid

        n = out[i]
        n_norm = np.linalg.norm(n)
        if n_norm == 0.0:
            n = orientation_reference
            n_norm = np.linalg.norm(n)
            if n_norm == 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm
        else:
            if float(n @ orientation_reference) > 0.0:
                n = -n
        out[i] = n
    return out


def _estimate_normals_for_cloud(
    pcd: o3d.geometry.PointCloud,
    k: int,
    orient_model: bool = False,
    orig_pts: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    out = copy.deepcopy(pcd)
    if len(out.points) == 0:
        return out
    out.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(k)))
    if orient_model:
        pts = np.asarray(out.points, dtype=np.float64)
        normals = np.asarray(out.normals, dtype=np.float64)
        ref_pts = orig_pts if orig_pts is not None else pts
        normals = _flip_model_normals_like_repo(pts, normals, ref_pts)
        out.normals = o3d.utility.Vector3dVector(normals)
    return out


def _method_specific_target_spacing(
    pcd: o3d.geometry.PointCloud,
    adaptive_cfg: Dict[str, Any],
) -> Tuple[float, Optional[int]]:
    target = None
    try:
        from .preprocess import get_target_points_by_raw_count

        target = get_target_points_by_raw_count(
            n_raw=len(pcd.points),
            no_downsample_thresh=int(adaptive_cfg.get("no_downsample_thresh", 450)),
            mid_thresh=int(adaptive_cfg.get("mid_thresh", 1500)),
            large_thresh=int(adaptive_cfg.get("large_thresh", 3000)),
            target_mid=int(adaptive_cfg.get("target_mid", 500)),
            target_large=int(adaptive_cfg.get("target_large", 450)),
            target_xlarge=int(adaptive_cfg.get("target_xlarge", 400)),
        )
    except Exception:
        target = None

    if target is None:
        return 0.0, None

    _, voxel_used, _ = _downsample_by_target_points(
        pcd=pcd,
        target_points=int(target),
        min_points_keep=int(adaptive_cfg.get("no_downsample_thresh", 450)),
        search_steps=int(adaptive_cfg.get("search_steps", 12)),
        voxel_min=float(adaptive_cfg.get("voxel_min", 1.0e-3)),
        voxel_max_scale=float(adaptive_cfg.get("voxel_max_scale", 0.20)),
    )
    return float(voxel_used), int(target)


def _apply_close_pair_retention_sampling(
    pcd: o3d.geometry.PointCloud,
    k: int,
    base_voxel: float,
    cfg: Dict[str, Any],
    orient_model: bool = False,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    params = _going_further_cfg(cfg)
    adaptive_cfg = dict(cfg.get("adaptive_downsample_cfg", {}) or {})
    voxel_min = float(adaptive_cfg.get("voxel_min", 1.0e-4))
    candidate_ratio = float(params.get("candidate_voxel_ratio", 0.5))
    candidate_voxel = max(voxel_min, float(base_voxel) * max(1.0e-3, candidate_ratio)) if base_voxel > 0.0 else 0.0
    max_close_retain_per_anchor = max(1, int(params.get("max_close_retain_per_anchor", 2)))
    keep_angle_deg = float(params.get("keep_close_pairs_with_normal_angle_deg", 30.0))
    min_extra_distance = float(base_voxel) * float(params.get("min_close_extra_distance_ratio", 0.20))

    orig_pts = np.asarray(pcd.points, dtype=np.float64)
    if candidate_voxel > 0.0:
        candidate_raw = pcd.voxel_down_sample(candidate_voxel)
    else:
        candidate_raw = copy.deepcopy(pcd)
    candidate = _estimate_normals_for_cloud(candidate_raw, k=k, orient_model=orient_model, orig_pts=orig_pts)

    if len(candidate.points) == 0:
        return candidate, {
            "candidate_points": 0,
            "base_points": 0,
            "retained_close_points": 0,
            "base_voxel_used": float(base_voxel),
            "candidate_voxel_used": float(candidate_voxel),
            "target_points": None,
        }

    if base_voxel > 0.0:
        base_raw = candidate.voxel_down_sample(base_voxel)
    else:
        base_raw = copy.deepcopy(candidate)
    base = _estimate_normals_for_cloud(base_raw, k=k, orient_model=orient_model, orig_pts=orig_pts)

    cand_pts = np.asarray(candidate.points, dtype=np.float64)
    cand_normals = np.asarray(candidate.normals, dtype=np.float64)
    base_pts = np.asarray(base.points, dtype=np.float64)
    base_normals = np.asarray(base.normals, dtype=np.float64)
    if base_pts.shape[0] == 0:
        return candidate, {
            "candidate_points": int(cand_pts.shape[0]),
            "base_points": 0,
            "retained_close_points": 0,
            "base_voxel_used": float(base_voxel),
            "candidate_voxel_used": float(candidate_voxel),
            "target_points": None,
        }

    base_kd = o3d.geometry.KDTreeFlann(base)
    retained_by_anchor: Dict[int, List[Tuple[float, float, int]]] = {}
    retained_count = 0

    for ci in range(cand_pts.shape[0]):
        _, idxs, d2 = base_kd.search_knn_vector_3d(cand_pts[ci], 1)
        if not idxs or not d2:
            continue
        anchor = int(idxs[0])
        dist = math.sqrt(float(d2[0]))
        if dist <= 1.0e-12:
            continue
        if base_voxel > 0.0 and dist > float(base_voxel):
            continue
        if dist < min_extra_distance:
            continue
        angle_deg = _safe_normal_angle_deg(cand_normals[ci], base_normals[anchor])
        if angle_deg < keep_angle_deg:
            continue
        retained_by_anchor.setdefault(anchor, []).append((float(angle_deg), float(dist), int(ci)))

    extra_pts: List[np.ndarray] = []
    extra_normals: List[np.ndarray] = []
    for anchor, items in retained_by_anchor.items():
        items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for angle_deg, dist, ci in items[:max_close_retain_per_anchor]:
            extra_pts.append(cand_pts[ci])
            extra_normals.append(cand_normals[ci])
            retained_count += 1

    if extra_pts:
        merged_pts = np.concatenate([base_pts, np.asarray(extra_pts, dtype=np.float64)], axis=0)
        merged_normals = np.concatenate([base_normals, np.asarray(extra_normals, dtype=np.float64)], axis=0)
    else:
        merged_pts = base_pts
        merged_normals = base_normals

    sampled = o3d.geometry.PointCloud()
    sampled.points = o3d.utility.Vector3dVector(merged_pts)
    sampled.normals = o3d.utility.Vector3dVector(merged_normals)
    if orient_model:
        merged_normals = _flip_model_normals_like_repo(merged_pts, merged_normals, orig_pts)
        sampled.normals = o3d.utility.Vector3dVector(merged_normals)

    return sampled, {
        "candidate_points": int(cand_pts.shape[0]),
        "base_points": int(base_pts.shape[0]),
        "retained_close_points": int(retained_count),
        "base_voxel_used": float(base_voxel),
        "candidate_voxel_used": float(candidate_voxel),
        "target_points": None,
    }


def build_going_further_model(
    model_pcd: o3d.geometry.PointCloud,
    angle_step: float,
    distance_step: float,
    cfg: Dict[str, Any],
    sampling_min_distance: float,
    logger=None,
) -> PPFModel:
    params = _going_further_cfg(cfg)
    pts = np.asarray(model_pcd.points, dtype=np.float64)
    normals = np.asarray(model_pcd.normals, dtype=np.float64)
    n_pts = int(pts.shape[0])

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
    min_pair_dist = float(params["close_pair_min_distance_ratio"]) * float(max(sampling_min_distance, 1.0e-12))
    keep_close_angle = float(params["keep_close_pairs_with_normal_angle_deg"])

    inserted_pairs = 0
    skipped_close_pairs = 0
    kept_close_pairs = 0
    model_diameter = 0.0

    for i in range(n_pts):
        pi = pts[i]
        ni = normals[i]
        Ri = ref_R[i]
        for j in range(n_pts):
            if i == j:
                continue
            pj = pts[j]
            nj = normals[j]

            feat = compute_pair_features(pi, ni, pj, nj)
            if feat is None:
                continue

            f1, f2, f3, f4 = feat
            if f4 < min_pair_dist:
                if _safe_normal_angle_deg(ni, nj) < keep_close_angle:
                    skipped_close_pairs += 1
                    continue
                kept_close_pairs += 1

            g = to_internal_feature_g(f1, f2, f3, f4)
            hash_table.add(g, PPFEntry(mr=i, mi=j, g=None))
            inserted_pairs += 1

            pj_mg = Ri @ (pj - pi)
            alpha_m[i][j] = -angle_from_transformed_point(pj_mg)
            if f4 > model_diameter:
                model_diameter = float(f4)

    bbox_min = pts.min(axis=0) if pts.shape[0] > 0 else np.zeros(3, dtype=np.float64)
    bbox_max = pts.max(axis=0) if pts.shape[0] > 0 else np.zeros(3, dtype=np.float64)
    bbox_dims = np.sort((bbox_max - bbox_min).astype(np.float64))
    d_min = float(bbox_dims[0]) if bbox_dims.size >= 1 else 0.0
    d_med = float(bbox_dims[1]) if bbox_dims.size >= 2 else d_min
    small_ball_radius = float(math.sqrt(d_min * d_min + d_med * d_med)) * float(params["small_ball_radius_scale"])
    large_ball_radius = float(model_diameter) * float(params["large_ball_radius_scale"])
    if small_ball_radius <= 0.0:
        small_ball_radius = min(large_ball_radius, float(model_diameter) / 2.0)

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
    model.going_further_meta = {
        "bbox_dims_sorted": [float(x) for x in bbox_dims.tolist()],
        "small_ball_radius": float(small_ball_radius),
        "large_ball_radius": float(large_ball_radius),
        "inserted_pairs": int(inserted_pairs),
        "skipped_close_pairs": int(skipped_close_pairs),
        "kept_close_pairs": int(kept_close_pairs),
        "keep_close_pairs_with_normal_angle_deg": float(keep_close_angle),
        "close_pair_min_distance_ratio": float(params["close_pair_min_distance_ratio"]),
        "min_pair_distance_used": float(min_pair_dist),
        "sampling_min_distance": float(sampling_min_distance),
    }

    if logger:
        logger.info(
            "[GoingFurtherModel] "
            f"N={n_pts} inserted_pairs={inserted_pairs} model_diameter={model_diameter:.6f} "
            f"small_ball={small_ball_radius:.6f} large_ball={large_ball_radius:.6f}"
        )
        logger.info(
            "[GoingFurtherModel] "
            f"kept_close_pairs={kept_close_pairs} skipped_close_pairs={skipped_close_pairs} "
            f"keep_angle_deg={keep_close_angle:.1f}"
        )
    return model


def _get_model_meta(model: PPFModel) -> Dict[str, Any]:
    meta = getattr(model, "going_further_meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _angle_bin_count(span: float, angle_step: float) -> int:
    return max(1, int(math.ceil(float(span) / float(angle_step))))


def _feature_neighbor_keys(
    key: Tuple[int, int, int, int],
    angle_step: float,
    feature_neighbor_steps: int,
) -> List[Tuple[int, int, int, int]]:
    if int(feature_neighbor_steps) <= 0:
        return [key]

    k1, k2, k3, k4 = [int(x) for x in key]
    n1 = _angle_bin_count(2.0 * math.pi, angle_step)
    n23 = _angle_bin_count(math.pi, angle_step)

    out: List[Tuple[int, int, int, int]] = []
    seen = set()
    for d1 in range(-feature_neighbor_steps, feature_neighbor_steps + 1):
        for d2 in range(-feature_neighbor_steps, feature_neighbor_steps + 1):
            for d3 in range(-feature_neighbor_steps, feature_neighbor_steps + 1):
                for d4 in range(-feature_neighbor_steps, feature_neighbor_steps + 1):
                    nk1 = (k1 + d1) % n1
                    nk2 = k2 + d2
                    nk3 = k3 + d3
                    nk4 = k4 + d4
                    if nk2 < 0 or nk2 >= n23 or nk3 < 0 or nk3 >= n23 or nk4 < 0:
                        continue
                    nk = (int(nk1), int(nk2), int(nk3), int(nk4))
                    if nk in seen:
                        continue
                    seen.add(nk)
                    out.append(nk)
    return out


def _quantize_alpha_bin(alpha: float, angle_step: float, aux_size: int) -> int:
    bin_j = int(math.floor((float(alpha) + math.pi) / float(angle_step)))
    return max(0, min(int(aux_size) - 1, int(bin_j)))


def _iter_angle_neighbor_bins(bin_j: int, steps: int, aux_size: int) -> Iterable[int]:
    if int(steps) <= 0:
        yield int(bin_j)
        return
    for dj in range(-steps, steps + 1):
        yield int((int(bin_j) + int(dj)) % int(aux_size))


def _extract_pass_hypotheses(
    accumulator: np.ndarray,
    model: PPFModel,
    T_sg: np.ndarray,
    pass_name: str,
    voted_poses: List[PoseWithVotes],
) -> int:
    count_before = len(voted_poses)
    for mr in range(len(model.pts)):
        row = accumulator[mr]
        votes = float(np.max(row))
        if votes <= 0.0:
            continue
        bj = int(np.argmax(row))
        theta = (float(bj) + 0.5) * model.angle_step - math.pi
        R_x = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), float(theta))
        T_x = make_affine(R_x, np.zeros(3, dtype=float))
        T_mg = make_affine(model.ref_R[mr], -model.ref_R[mr] @ model.pts[mr])
        T = compose_affine(invert_affine(T_sg), compose_affine(T_x, T_mg))
        voted_poses.append(PoseWithVotes(T=T, votes=float(votes)))
    return int(len(voted_poses) - count_before)


def _vote_scene_pairs(
    model: PPFModel,
    seg_pts: np.ndarray,
    seg_normals: np.ndarray,
    sr: int,
    pair_indices: Sequence[int],
    accumulator: np.ndarray,
    seen_feature_rot: set,
    cfg: Dict[str, Any],
    debug_acc: Dict[str, Any],
) -> None:
    params = _going_further_cfg(cfg)
    aux_size = int(accumulator.shape[1])
    feature_steps = int(params["feature_neighbor_steps"])
    angle_steps = int(params["vote_angle_neighbor_steps"])
    dedup_enable = bool(params["deduplicate_quantized_votes"])
    table = model.hash_table.table  # type: ignore[attr-defined]

    sr_p = seg_pts[int(sr)]
    sr_n = seg_normals[int(sr)]
    _, R_sg = compute_transform_sg(sr_p, sr_n)

    for si in pair_indices:
        if int(si) == int(sr):
            continue
        si_p = seg_pts[int(si)]
        si_n = seg_normals[int(si)]
        feat = compute_pair_features(sr_p, sr_n, si_p, si_n)
        if feat is None:
            continue

        f1, f2, f3, f4 = feat
        gs = to_internal_feature_g(f1, f2, f3, f4)
        base_key = discretize_baseline(gs, model.angle_step, model.distance_step)
        feature_keys = _feature_neighbor_keys(base_key, model.angle_step, feature_steps)

        si_trans = R_sg @ (si_p - sr_p)
        alpha_s = -angle_from_transformed_point(si_trans)
        alpha_s_bin = _quantize_alpha_bin(alpha_s, model.angle_step, aux_size)

        base_bucket_len = len(table.get(base_key, []))
        total_bucket_len = int(sum(len(table.get(k, [])) for k in feature_keys))
        if base_bucket_len > 0:
            infl = float(total_bucket_len) / float(base_bucket_len)
        else:
            infl = float(total_bucket_len) if total_bucket_len > 0 else 1.0
        debug_acc["candidate_inflations"].append(float(infl))

        for fkey in feature_keys:
            dedup_key = (fkey, int(alpha_s_bin))
            if dedup_enable and dedup_key in seen_feature_rot:
                debug_acc["dedup_suppressed"] += 1
                continue
            if dedup_enable:
                seen_feature_rot.add(dedup_key)

            bucket = table.get(fkey, [])
            if not bucket:
                continue

            debug_acc["accepted_quantized_pairs"] += 1
            debug_acc["accepted_bucket_entries"] += int(len(bucket))
            for e in bucket:
                alpha = wrap_to_pi(model.alpha_m[e.mr][e.mi] - alpha_s)
                bin_j = _quantize_alpha_bin(alpha, model.angle_step, aux_size)
                for bj in _iter_angle_neighbor_bins(bin_j, angle_steps, aux_size):
                    accumulator[e.mr, int(bj)] += 1.0


def _run_going_further_frontend(
    model: PPFModel,
    scene_pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any], float, float, float]:
    params = _going_further_cfg(cfg)
    scene_pts = np.asarray(scene_pcd.points, dtype=np.float64)
    scene_normals = np.asarray(scene_pcd.normals, dtype=np.float64)
    if scene_pts.shape[0] == 0 or scene_normals.shape != scene_pts.shape:
        return np.eye(4, dtype=float), {}, 0.0, 0.0, 0.0

    aux_size = int(math.ceil(2.0 * math.pi / model.angle_step))
    meta = _get_model_meta(model)
    small_ball_radius = float(meta.get("small_ball_radius", float(model.model_diameter) / 2.0))
    large_ball_radius = float(meta.get("large_ball_radius", float(model.model_diameter)))
    if not bool(params.get("two_ball_enable", True)):
        small_ball_radius = large_ball_radius

    scene_ref_sampling_rate = max(1, int(cfg.get("scene_ref_sampling_rate", 20)))
    pos_thresh = float(cfg.get("pos_thresh", 0.005))
    rot_thresh_rad = math.radians(float(cfg.get("rot_thresh_deg", 30.0)))
    kdtree = o3d.geometry.KDTreeFlann(scene_pcd)
    accumulator = np.zeros((len(model.pts), aux_size), dtype=np.float32)
    voted_poses: List[PoseWithVotes] = []

    pass_pose_counts = {"small_ball": 0, "large_ball": 0}
    pass_pair_counts = {"small_ball": 0, "large_ball": 0}
    candidate_inflations: List[float] = []
    dedup_suppressed_total = 0
    accepted_quantized_pairs_total = 0
    accepted_bucket_entries_total = 0
    sr_debug_rows: List[Dict[str, Any]] = []

    t_front = time.perf_counter()
    for sr in range(0, scene_pts.shape[0], scene_ref_sampling_rate):
        sr_p = scene_pts[int(sr)]
        sr_n = scene_normals[int(sr)]
        T_sg, _ = compute_transform_sg(sr_p, sr_n)

        _, idxs, d2 = kdtree.search_radius_vector_3d(sr_p, max(small_ball_radius, large_ball_radius))
        if not idxs:
            continue

        small_pairs: List[int] = []
        large_only_pairs: List[int] = []
        for si, dist2 in zip(idxs, d2):
            if int(si) == int(sr):
                continue
            dist = math.sqrt(float(dist2))
            if dist <= float(small_ball_radius):
                small_pairs.append(int(si))
            elif dist <= float(large_ball_radius):
                large_only_pairs.append(int(si))

        seen_feature_rot: set = set()
        sr_stats = {
            "candidate_inflations": [],
            "dedup_suppressed": 0,
            "accepted_quantized_pairs": 0,
            "accepted_bucket_entries": 0,
        }

        if small_pairs:
            pass_pair_counts["small_ball"] += int(len(small_pairs))
            _vote_scene_pairs(model, scene_pts, scene_normals, sr, small_pairs, accumulator, seen_feature_rot, cfg, sr_stats)
            pass_pose_counts["small_ball"] += _extract_pass_hypotheses(accumulator, model, T_sg, "small_ball", voted_poses)

        if large_only_pairs:
            pass_pair_counts["large_ball"] += int(len(large_only_pairs))
            _vote_scene_pairs(model, scene_pts, scene_normals, sr, large_only_pairs, accumulator, seen_feature_rot, cfg, sr_stats)
            pass_pose_counts["large_ball"] += _extract_pass_hypotheses(accumulator, model, T_sg, "large_ball", voted_poses)

        candidate_inflations.extend(float(x) for x in sr_stats["candidate_inflations"])
        dedup_suppressed_total += int(sr_stats["dedup_suppressed"])
        accepted_quantized_pairs_total += int(sr_stats["accepted_quantized_pairs"])
        accepted_bucket_entries_total += int(sr_stats["accepted_bucket_entries"])
        sr_debug_rows.append(
            {
                "sr": int(sr),
                "small_pairs": int(len(small_pairs)),
                "large_only_pairs": int(len(large_only_pairs)),
                "dedup_suppressed": int(sr_stats["dedup_suppressed"]),
                "accepted_quantized_pairs": int(sr_stats["accepted_quantized_pairs"]),
                "accepted_bucket_entries": int(sr_stats["accepted_bucket_entries"]),
                "candidate_inflation_mean": float(np.mean(sr_stats["candidate_inflations"])) if sr_stats["candidate_inflations"] else 1.0,
            }
        )
        accumulator.fill(0)

    ppf_frontend_time = time.perf_counter() - t_front

    t_back = time.perf_counter()
    final_policy = str(params.get("final_selection_policy", "raw_top1_vote")).lower()
    legacy_clustering_time = 0.0
    if final_policy == "legacy_cluster":
        clustered = cluster_poses(voted_poses, pos_thresh, rot_thresh_rad)
        legacy_clustering_time = time.perf_counter() - t_back
        T_best = clustered[0].T if clustered else np.eye(4, dtype=float)
        final_source = "legacy_cluster"
    else:
        if voted_poses:
            best = max(voted_poses, key=lambda x: float(x.votes))
            T_best = best.T
            final_source = "raw_top1_vote"
        else:
            T_best = np.eye(4, dtype=float)
            final_source = "identity_empty"
    backend_time = time.perf_counter() - t_back

    debug = {
        "candidate_inflation_mean": float(np.mean(candidate_inflations)) if candidate_inflations else 1.0,
        "going_further_ppf": {
            "small_ball_radius": float(small_ball_radius),
            "large_ball_radius": float(large_ball_radius),
            "feature_neighbor_steps": int(params["feature_neighbor_steps"]),
            "vote_angle_neighbor_steps": int(params["vote_angle_neighbor_steps"]),
            "deduplicate_quantized_votes": bool(params["deduplicate_quantized_votes"]),
            "two_ball_enable": bool(params["two_ball_enable"]),
            "pass_pose_counts": {k: int(v) for k, v in pass_pose_counts.items()},
            "pass_pair_counts": {k: int(v) for k, v in pass_pair_counts.items()},
            "raw_pose_count": int(len(voted_poses)),
            "dedup_suppressed_total": int(dedup_suppressed_total),
            "accepted_quantized_pairs_total": int(accepted_quantized_pairs_total),
            "accepted_bucket_entries_total": int(accepted_bucket_entries_total),
            "scene_reference_debug": sr_debug_rows[:50],
            "final_selection_policy": final_policy,
            "final_source": final_source,
        },
        "model_build": meta,
        "timing": {
            "ppf_frontend_time": float(ppf_frontend_time),
            "legacy_clustering_time": float(legacy_clustering_time),
            "backend_time": float(backend_time),
        },
        "final_pose_policy": f"going_further_{final_policy}",
        "final_pose_path": final_source,
    }
    return T_best, debug, float(ppf_frontend_time), float(legacy_clustering_time), float(backend_time)


def run_going_further_registration(
    model_path: str,
    scene_path: str,
    cfg: Dict[str, Any],
    logger=None,
    model_cache_path: Optional[str] = None,
    strict_cache: bool = True,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud, Dict[str, Any], RegistrationStats]:
    runtime_cfg = make_going_further_config(cfg)
    seed = int(runtime_cfg.get("seed", 0))
    set_global_seed(seed)

    sampling_leaf = float(runtime_cfg.get("sampling_leaf", 5.0))
    normal_k = int(runtime_cfg.get("normal_k", 5))
    angle_step = math.radians(float(runtime_cfg.get("angle_step_deg", 12.0)))
    distance_step = float(runtime_cfg.get("distance_step_ratio", 0.6)) * sampling_leaf
    adaptive_downsample = bool(runtime_cfg.get("adaptive_downsample", False))
    adaptive_apply_to = str(runtime_cfg.get("adaptive_downsample_apply_to", "scene")).lower()
    adaptive_cfg = dict(runtime_cfg.get("adaptive_downsample_cfg", {}) or {})
    gf_params = _going_further_cfg(runtime_cfg)
    use_method_specific_sampling = bool(gf_params.get("use_method_specific_sampling", True))
    method_specific_sampling_apply_to = str(gf_params.get("method_specific_sampling_apply_to", "model")).lower()

    t0 = time.perf_counter()
    cloud_model = o3d.io.read_point_cloud(model_path)
    if len(cloud_model.points) == 0:
        raise ValueError(f"Empty model: {model_path}")
    cloud_scene = o3d.io.read_point_cloud(scene_path)
    if len(cloud_scene.points) == 0:
        raise ValueError(f"Empty scene: {scene_path}")

    model_down = None
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
            raise ValueError(f"Going Further cache missing method metadata: {model_cache_path}")
        model_ds_info["method_specific_sampling"] = bool(use_method_specific_sampling)
        model_ds_info["cached_model_meta"] = _get_model_meta(ppf_model)
    else:
        with Timer("model_build", logger=logger) as tm:
            if use_method_specific_sampling and method_specific_sampling_apply_to in ("model", "both"):
                if adaptive_downsample and adaptive_apply_to in ("model", "both"):
                    base_voxel, target_points = _method_specific_target_spacing(cloud_model, adaptive_cfg)
                else:
                    base_voxel, target_points = float(sampling_leaf), None
                model_down, sample_info = _apply_close_pair_retention_sampling(
                    pcd=cloud_model,
                    k=normal_k,
                    base_voxel=max(float(base_voxel), float(sampling_leaf)),
                    cfg=runtime_cfg,
                    orient_model=True,
                )
                model_ds_info = {
                    "raw_points": len(cloud_model.points),
                    "down_points": len(model_down.points),
                    "target_points": target_points,
                    "voxel_used": float(sample_info.get("base_voxel_used", max(float(base_voxel), float(sampling_leaf)))),
                    "candidate_voxel_used": float(sample_info.get("candidate_voxel_used", 0.0)),
                    "candidate_points": int(sample_info.get("candidate_points", 0)),
                    "base_points": int(sample_info.get("base_points", 0)),
                    "retained_close_points": int(sample_info.get("retained_close_points", 0)),
                    "adaptive_enabled": adaptive_downsample and adaptive_apply_to in ("model", "both"),
                    "apply_to": adaptive_apply_to,
                    "method_specific_sampling": True,
                }
            elif adaptive_downsample and adaptive_apply_to in ("model", "both"):
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

            ppf_model = build_going_further_model(
                model_down,
                angle_step,
                distance_step,
                runtime_cfg,
                sampling_min_distance=float(model_ds_info.get("voxel_used", sampling_leaf)),
                logger=logger,
            )

    scene_preprocess_time = 0.0
    with Timer("registration", logger=logger) as tr:
        t_scene_pre = time.perf_counter()
        if use_method_specific_sampling and method_specific_sampling_apply_to in ("scene", "both"):
            if adaptive_downsample and adaptive_apply_to in ("scene", "both"):
                base_voxel_scene, target_points_scene = _method_specific_target_spacing(cloud_scene, adaptive_cfg)
            else:
                base_voxel_scene, target_points_scene = float(sampling_leaf), None
            scene_down, sample_info = _apply_close_pair_retention_sampling(
                pcd=cloud_scene,
                k=normal_k,
                base_voxel=max(float(base_voxel_scene), float(sampling_leaf)),
                cfg=runtime_cfg,
                orient_model=False,
            )
            scene_ds_info = {
                "raw_points": len(cloud_scene.points),
                "down_points": len(scene_down.points),
                "target_points": target_points_scene,
                "voxel_used": float(sample_info.get("base_voxel_used", max(float(base_voxel_scene), float(sampling_leaf)))),
                "candidate_voxel_used": float(sample_info.get("candidate_voxel_used", 0.0)),
                "candidate_points": int(sample_info.get("candidate_points", 0)),
                "base_points": int(sample_info.get("base_points", 0)),
                "retained_close_points": int(sample_info.get("retained_close_points", 0)),
                "adaptive_enabled": adaptive_downsample and adaptive_apply_to in ("scene", "both"),
                "apply_to": adaptive_apply_to,
                "method_specific_sampling": True,
            }
        elif adaptive_downsample and adaptive_apply_to in ("scene", "both"):
            scene_down, scene_ds_info = adaptive_subsample_and_calculate_normals_scene(
                pcd=cloud_scene,
                k=normal_k,
                cfg=adaptive_cfg,
            )
            scene_ds_info["adaptive_enabled"] = True
            scene_ds_info["apply_to"] = adaptive_apply_to
        else:
            scene_down = subsample_and_calculate_normals_scene(
                pcd=cloud_scene,
                voxel_size=sampling_leaf,
                k=normal_k,
            )
            scene_ds_info = {
                "raw_points": len(cloud_scene.points),
                "down_points": len(scene_down.points),
                "target_points": None,
                "voxel_used": sampling_leaf,
                "adaptive_enabled": False,
                "apply_to": adaptive_apply_to,
            }
        scene_preprocess_time = time.perf_counter() - t_scene_pre

        T_pred, debug, ppf_frontend_time, legacy_clustering_time, backend_time = _run_going_further_frontend(
            model=ppf_model,
            scene_pcd=scene_down,
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
        "going_further_ppf": dict(runtime_cfg.get("going_further_ppf", {}) or {}),
        "use_method_specific_sampling": bool(use_method_specific_sampling),
        "method_specific_sampling_apply_to": str(method_specific_sampling_apply_to),
    }
    debug["external_method"]["reproduction_notes"] = (
        "Going Further round-1 reproduction on top of the repo's Stanford-consistent Drost baseline. "
        "This implementation focuses on method-specific normal-aware sampling, two-ball scene pair sampling, "
        "feature/rotation spreading, and quantized vote deduplication, while leaving the paper's full "
        "ICP and rendering-based post checks as future work."
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
