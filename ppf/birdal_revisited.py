import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from .clustering import PoseWithVotes, cluster_poses
from .model_builder import PPFModel, build_ppf_model
from .model_cache_io import load_ppf_model_cache
from .pose_selection import VisibilityCfg, _compute_visibility_support
from .ppf_features import compute_pair_features, angle_from_transformed_point, to_internal_feature_g
from .preprocess import (
    adaptive_subsample_and_calculate_normals_model,
    adaptive_subsample_and_calculate_normals_scene,
    subsample_and_calculate_normals_model,
    subsample_and_calculate_normals_scene,
)
from .registration import RegistrationStats, compute_transform_sg
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


@dataclass
class BirdalHypothesis:
    T: np.ndarray
    vote: float
    vote_norm: float
    support_ratio: float
    visibility_support: float
    normal_consistency: float
    score: float
    source_segment: str

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "vote": float(self.vote),
            "vote_norm": float(self.vote_norm),
            "support_ratio": float(self.support_ratio),
            "visibility_support": float(self.visibility_support),
            "normal_consistency": float(self.normal_consistency),
            "score": float(self.score),
            "source_segment": str(self.source_segment),
        }


def make_birdal_revisited_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = copy.deepcopy(cfg)
    runtime["enable_rsmrq"] = False
    runtime["enable_robust_vote"] = False
    runtime["enable_kde_refine"] = False
    runtime["store_pair_features"] = True

    icp_cfg = dict(runtime.get("icp_refine", {}) or {})
    icp_cfg["enable"] = False
    runtime["icp_refine"] = icp_cfg

    pose_sel = dict(runtime.get("pose_selection", {}) or {})
    pose_sel["enable"] = False
    runtime["pose_selection"] = pose_sel

    pose_cluster = dict(runtime.get("pose_clustering", {}) or {})
    pose_cluster["enable"] = False
    runtime["pose_clustering"] = pose_cluster

    runtime["final_pose_policy"] = "birdal_revisited_ranked"

    ext_meta = dict(runtime.get("external_method", {}) or {})
    ext_meta.update(
        {
            "name": "birdal_revisited",
            "paper": "Birdal and Ilic, 3DV 2015",
            "implementation_style": "pure_python_repo_reproduction",
            "comparison_mode": "stanford_base_consistent",
            "incremental_modules_over_drost": [
                "scene_segmentation",
                "weighted_hough_voting",
                "alpha_peak_interpolation",
                "lightweight_visibility_aware_ranking",
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


def _birdal_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "segmentation_enable": True,
        "segmentation_eps_factor": 0.10,
        "segmentation_min_eps": 0.010,
        "segmentation_min_points": 18,
        "segmentation_max_segments": 6,
        "include_full_scene": True,
        "weighted_vote_sigma": 0.20,
        "interpolate_alpha": True,
        "rank_top_k": 16,
        "ranking_inlier_radius": 0.008,
        "ranking_normal_dot_thresh": 0.10,
        "ranking_vote_weight": 0.55,
        "ranking_support_weight": 0.30,
        "ranking_visibility_weight": 0.15,
        "visibility_radius": 0.008,
        "visibility_normal_dot_thresh": 0.10,
        "visibility_scene_normal_dot_thresh": 0.10,
        "visibility_require_normal_agreement": True,
        "final_selection_policy": "ranked_cluster",
        "min_ranked_vote_ratio": 0.90,
        "min_ranked_support": 0.18,
        "min_ranked_visibility": 0.04,
        "min_ranked_score_margin": 0.03,
    }
    out = dict(base)
    out.update(dict(cfg.get("birdal_revisited", {}) or {}))
    return out


def _segment_scene(
    scene_pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
) -> Tuple[List[Tuple[str, o3d.geometry.PointCloud]], Dict[str, Any]]:
    params = _birdal_cfg(cfg)
    segments: List[Tuple[str, o3d.geometry.PointCloud]] = []
    debug: Dict[str, Any] = {
        "segmentation_enable": bool(params["segmentation_enable"]),
        "eps": None,
        "num_clusters_raw": 0,
        "segment_names": [],
        "segment_sizes": [],
    }

    if len(scene_pcd.points) == 0:
        return segments, debug

    if bool(params.get("include_full_scene", True)):
        segments.append(("full_scene", scene_pcd))

    if not bool(params.get("segmentation_enable", True)):
        debug["num_clusters_raw"] = 0
        debug["segment_names"] = [name for name, _ in segments]
        debug["segment_sizes"] = [len(seg.points) for _, seg in segments]
        return segments, debug

    pts = np.asarray(scene_pcd.points, dtype=np.float64)
    bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) if pts.shape[0] > 0 else 0.0
    eps = max(float(params["segmentation_min_eps"]), float(params["segmentation_eps_factor"]) * bbox_diag)
    debug["eps"] = float(eps)

    labels = np.asarray(scene_pcd.cluster_dbscan(eps=eps, min_points=int(params["segmentation_min_points"]), print_progress=False))
    keep_labels = [int(x) for x in sorted(np.unique(labels)) if int(x) >= 0]
    debug["num_clusters_raw"] = len(keep_labels)

    cluster_sizes = []
    for label in keep_labels:
        idxs = np.where(labels == label)[0]
        cluster_sizes.append((label, int(idxs.shape[0])))

    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    max_segments = int(params["segmentation_max_segments"])
    for label, _ in cluster_sizes[:max_segments]:
        idxs = np.where(labels == label)[0].tolist()
        if len(idxs) < int(params["segmentation_min_points"]):
            continue
        seg = scene_pcd.select_by_index(idxs)
        segments.append((f"cluster_{label}", seg))

    seen = set()
    unique_segments: List[Tuple[str, o3d.geometry.PointCloud]] = []
    for name, seg in segments:
        if name in seen:
            continue
        seen.add(name)
        unique_segments.append((name, seg))

    debug["segment_names"] = [name for name, _ in unique_segments]
    debug["segment_sizes"] = [len(seg.points) for _, seg in unique_segments]
    return unique_segments, debug


def _weighted_vote_weight(
    gs: np.ndarray,
    gm: Optional[np.ndarray],
    model_diameter: float,
    sigma: float,
) -> float:
    if gm is None:
        return 1.0
    scale = np.array([math.pi, math.pi, math.pi, max(1e-12, model_diameter)], dtype=np.float64)
    residual = float(np.linalg.norm((gm.astype(np.float64) - gs.astype(np.float64)) / scale))
    sigma = max(1.0e-6, float(sigma))
    return float(math.exp(-0.5 * (residual / sigma) ** 2))


def _refine_theta_from_row(row: np.ndarray, peak_idx: int, angle_step: float) -> Tuple[float, float]:
    n_bins = int(row.shape[0])
    base_theta = (float(peak_idx) + 0.5) * float(angle_step) - math.pi
    if n_bins < 3:
        return base_theta, 0.0

    prev_idx = (int(peak_idx) - 1) % n_bins
    next_idx = (int(peak_idx) + 1) % n_bins
    y0 = float(row[prev_idx])
    y1 = float(row[int(peak_idx)])
    y2 = float(row[next_idx])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1.0e-12:
        return base_theta, 0.0

    delta = 0.5 * (y0 - y2) / denom
    delta = max(-1.0, min(1.0, float(delta)))
    theta = (float(peak_idx) + 0.5 + delta) * float(angle_step) - math.pi
    return theta, float(delta)


def _evaluate_hypothesis(
    T: np.ndarray,
    vote: float,
    max_vote: float,
    model: PPFModel,
    scene_pts: np.ndarray,
    scene_normals: np.ndarray,
    scene_kd: o3d.geometry.KDTreeFlann,
    cfg: Dict[str, Any],
    source_segment: str,
) -> BirdalHypothesis:
    params = _birdal_cfg(cfg)
    inlier_radius = float(params["ranking_inlier_radius"])
    normal_thresh = float(params["ranking_normal_dot_thresh"])
    vote_weight = float(params["ranking_vote_weight"])
    support_weight = float(params["ranking_support_weight"])
    visibility_weight = float(params["ranking_visibility_weight"])

    transformed_pts = transform_points(T, model.pts)
    transformed_normals = (T[:3, :3] @ model.normals.T).T

    radius2 = inlier_radius * inlier_radius
    support_hits = 0
    normal_hits = 0.0
    valid = 0

    for i in range(transformed_pts.shape[0]):
        _, idxs, d2 = scene_kd.search_knn_vector_3d(transformed_pts[i], 1)
        if not idxs or not d2:
            continue
        valid += 1
        if float(d2[0]) > radius2:
            continue
        support_hits += 1

        if scene_normals.shape == scene_pts.shape and scene_normals.shape[0] > idxs[0]:
            dot = float(np.dot(transformed_normals[i], scene_normals[idxs[0]]))
            normal_hits += max(0.0, dot)

    support_ratio = float(support_hits) / max(1, transformed_pts.shape[0])
    normal_consistency = float(normal_hits) / max(1, support_hits)
    if normal_consistency < normal_thresh:
        normal_consistency *= 0.5

    vis_cfg = VisibilityCfg(
        enable=True,
        radius=float(params["visibility_radius"]),
        normal_dot_thresh=float(params["visibility_normal_dot_thresh"]),
        require_normal_agreement=bool(params["visibility_require_normal_agreement"]),
        scene_normal_dot_thresh=float(params["visibility_scene_normal_dot_thresh"]),
    )
    visibility_support = _compute_visibility_support(
        T=T,
        model_pts=model.pts,
        model_normals=model.normals,
        scene_pts=scene_pts,
        scene_normals=scene_normals,
        scene_kd=scene_kd,
        vis_cfg=vis_cfg,
    )

    vote_norm = float(vote) / max(1.0e-12, float(max_vote))
    support_term = 0.7 * support_ratio + 0.3 * normal_consistency
    score = vote_weight * vote_norm + support_weight * support_term + visibility_weight * visibility_support

    return BirdalHypothesis(
        T=T,
        vote=float(vote),
        vote_norm=float(vote_norm),
        support_ratio=float(support_ratio),
        visibility_support=float(visibility_support),
        normal_consistency=float(normal_consistency),
        score=float(score),
        source_segment=str(source_segment),
    )


def _birdal_match_segments(
    model: PPFModel,
    scene_pcd: o3d.geometry.PointCloud,
    cfg: Dict[str, Any],
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any], float, float, float]:
    params = _birdal_cfg(cfg)
    scene_ref_sampling_rate = int(cfg.get("scene_ref_sampling_rate", 20))
    pos_thresh = float(cfg.get("pos_thresh", 0.005))
    rot_thresh_rad = math.radians(float(cfg.get("rot_thresh_deg", 30.0)))
    sigma = float(params["weighted_vote_sigma"])
    aux_size = int(math.ceil(2.0 * math.pi / model.angle_step))
    radius = float(model.model_diameter) / 2.0

    candidate_inflations: List[float] = []
    raw_pose_count = 0
    segment_pose_counts: Dict[str, int] = {}
    interpolation_deltas: List[float] = []
    voted_poses: List[PoseWithVotes] = []

    t_front = time.perf_counter()
    segments, seg_debug = _segment_scene(scene_pcd, cfg)
    for seg_name, seg_pcd in segments:
        seg_pts = np.asarray(seg_pcd.points, dtype=np.float64)
        seg_normals = np.asarray(seg_pcd.normals, dtype=np.float64)
        if seg_pts.shape[0] < 2 or seg_normals.shape != seg_pts.shape:
            continue

        kdtree = o3d.geometry.KDTreeFlann(seg_pcd)
        accumulator = np.zeros((len(model.pts), aux_size), dtype=np.float32)

        for sr in range(0, seg_pts.shape[0], max(1, scene_ref_sampling_rate)):
            sr_p = seg_pts[sr]
            sr_n = seg_normals[sr]
            T_sg, R_sg = compute_transform_sg(sr_p, sr_n)
            _, idxs, _ = kdtree.search_radius_vector_3d(sr_p, radius)

            for si in idxs:
                if int(si) == int(sr):
                    continue
                si_p = seg_pts[int(si)]
                si_n = seg_normals[int(si)]
                feat = compute_pair_features(sr_p, sr_n, si_p, si_n)
                if feat is None:
                    continue

                f1, f2, f3, f4 = feat
                gs = to_internal_feature_g(f1, f2, f3, f4)
                si_trans = R_sg @ (si_p - sr_p)
                alpha_s = -angle_from_transformed_point(si_trans)
                buckets = model.hash_table.query_buckets(gs)

                base_sz = len(buckets[0]) if buckets else 0
                tot_sz = sum(len(b) for b in buckets)
                infl = (tot_sz / max(1, base_sz)) if base_sz > 0 else (float(tot_sz) if tot_sz > 0 else 1.0)
                candidate_inflations.append(float(infl))

                bucket = buckets[0] if buckets else []
                for e in bucket:
                    alpha = wrap_to_pi(model.alpha_m[e.mr][e.mi] - alpha_s)
                    bin_j = int(math.floor((alpha + math.pi) / model.angle_step))
                    bin_j = max(0, min(aux_size - 1, bin_j))
                    gm = None if e.g is None else np.asarray(e.g, dtype=np.float64)
                    accumulator[e.mr, bin_j] += float(_weighted_vote_weight(gs, gm, model.model_diameter, sigma))
            for mr in range(len(model.pts)):
                row = accumulator[mr]
                votes = float(np.max(row))
                if votes <= 0.0:
                    continue

                peak_idx = int(np.argmax(row))
                theta, delta = _refine_theta_from_row(row, peak_idx, model.angle_step)
                if not bool(params.get("interpolate_alpha", True)):
                    theta = (float(peak_idx) + 0.5) * model.angle_step - math.pi
                    delta = 0.0
                interpolation_deltas.append(float(delta))

                R_x = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), float(theta))
                T_x = make_affine(R_x, np.zeros(3, dtype=float))
                T_mg = make_affine(model.ref_R[mr], -model.ref_R[mr] @ model.pts[mr])
                T = compose_affine(invert_affine(T_sg), compose_affine(T_x, T_mg))
                voted_poses.append(PoseWithVotes(T=T, votes=float(votes)))
                raw_pose_count += 1
                segment_pose_counts[seg_name] = int(segment_pose_counts.get(seg_name, 0) + 1)

            accumulator.fill(0)

    ppf_frontend_time = time.perf_counter() - t_front

    t_back = time.perf_counter()
    final_selection_policy = str(params.get("final_selection_policy", "ranked_cluster")).lower()
    raw_best = max(voted_poses, key=lambda x: float(x.votes)) if voted_poses else None
    clustered: List[PoseWithVotes] = []
    ranked: List[BirdalHypothesis] = []
    legacy_clustering_time = 0.0

    if final_selection_policy != "raw_top1_vote":
        clustered = cluster_poses(voted_poses, pos_thresh, rot_thresh_rad)
        legacy_clustering_time = time.perf_counter() - t_back

        if final_selection_policy in ("ranked_cluster", "ranked_with_guard"):
            max_vote = max((float(p.votes) for p in clustered), default=1.0)
            rank_top_k = int(params["rank_top_k"])
            scene_pts = np.asarray(scene_pcd.points, dtype=np.float64)
            scene_normals = np.asarray(scene_pcd.normals, dtype=np.float64)
            scene_kd = o3d.geometry.KDTreeFlann(scene_pcd)

            for pv in clustered[:rank_top_k]:
                ranked.append(
                    _evaluate_hypothesis(
                        T=pv.T,
                        vote=float(pv.votes),
                        max_vote=float(max_vote),
                        model=model,
                        scene_pts=scene_pts,
                        scene_normals=scene_normals,
                        scene_kd=scene_kd,
                        cfg=cfg,
                        source_segment="clustered",
                    )
                )
            ranked.sort(key=lambda x: (x.score, x.vote), reverse=True)

    cluster_best = clustered[0] if clustered else None
    ranked_best = ranked[0] if ranked else None
    ranked_score_margin = (
        float(ranked[0].score - ranked[1].score)
        if len(ranked) >= 2
        else (float(ranked[0].score) if ranked else 0.0)
    )
    ranked_guard_ok = bool(
        ranked_best is not None
        and float(ranked_best.vote_norm) >= float(params.get("min_ranked_vote_ratio", 0.90))
        and float(ranked_best.support_ratio) >= float(params.get("min_ranked_support", 0.18))
        and float(ranked_best.visibility_support) >= float(params.get("min_ranked_visibility", 0.04))
        and float(ranked_score_margin) >= float(params.get("min_ranked_score_margin", 0.03))
    )

    if final_selection_policy == "raw_top1_vote" and raw_best is not None:
        T_best = raw_best.T
        final_source = "raw_top1_vote"
    elif final_selection_policy == "cluster_top1" and cluster_best is not None:
        T_best = cluster_best.T
        final_source = "cluster_top1"
    elif final_selection_policy == "ranked_with_guard":
        if ranked_best is not None and ranked_guard_ok:
            T_best = ranked_best.T
            final_source = "ranked_cluster_guarded"
        elif raw_best is not None:
            T_best = raw_best.T
            final_source = "raw_top1_fallback_after_guard"
        elif cluster_best is not None:
            T_best = cluster_best.T
            final_source = "cluster_fallback_after_guard"
        else:
            T_best = np.eye(4, dtype=float)
            final_source = "identity_empty"
    elif ranked_best is not None:
        T_best = ranked_best.T
        final_source = "ranked_cluster"
    elif cluster_best is not None:
        T_best = cluster_best.T
        final_source = "cluster_fallback"
    elif raw_best is not None:
        T_best = raw_best.T
        final_source = "raw_vote_fallback"
    else:
        T_best = np.eye(4, dtype=float)
        final_source = "identity_empty"

    backend_time = time.perf_counter() - t_back
    debug = {
        "candidate_inflation_mean": float(np.mean(candidate_inflations)) if candidate_inflations else 1.0,
        "segmentation": seg_debug,
        "birdal_revisited": {
            "num_segments_used": len(segments),
            "segment_pose_counts": {k: int(v) for k, v in segment_pose_counts.items()},
            "raw_pose_count": int(raw_pose_count),
            "clustered_pose_count": int(len(clustered)),
            "ranked_pose_count": int(len(ranked)),
            "final_source": final_source,
            "final_selection_policy": final_selection_policy,
            "ranked_guard_ok": bool(ranked_guard_ok),
            "ranked_score_margin": float(ranked_score_margin),
            "raw_top_vote": float(raw_best.votes) if raw_best is not None else 0.0,
            "cluster_top_vote": float(cluster_best.votes) if cluster_best is not None else 0.0,
            "top_scores": [float(h.score) for h in ranked[:5]],
            "top_votes": [float(h.vote) for h in ranked[:5]],
            "top_supports": [float(h.support_ratio) for h in ranked[:5]],
            "top_visibility_supports": [float(h.visibility_support) for h in ranked[:5]],
            "top_normal_consistency": [float(h.normal_consistency) for h in ranked[:5]],
            "top_hypotheses": [h.to_debug_dict() for h in ranked[:5]],
            "interpolation_delta_mean": float(np.mean(interpolation_deltas)) if interpolation_deltas else 0.0,
        },
        "timing": {
            "ppf_frontend_time": float(ppf_frontend_time),
            "legacy_clustering_time": float(legacy_clustering_time),
            "backend_time": float(backend_time),
        },
        "final_pose_policy": f"birdal_revisited_{final_selection_policy}",
        "final_pose_path": final_source,
    }
    return T_best, debug, float(ppf_frontend_time), float(legacy_clustering_time), float(backend_time)


def run_birdal_revisited_registration(
    model_path: str,
    scene_path: str,
    cfg: Dict[str, Any],
    logger=None,
    model_cache_path: Optional[str] = None,
    strict_cache: bool = True,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud, Dict[str, Any], RegistrationStats]:
    runtime_cfg = make_birdal_revisited_config(cfg)
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

            ppf_model = build_ppf_model(model_down, angle_step, distance_step, runtime_cfg, logger=logger)

    scene_preprocess_time = 0.0
    with Timer("registration", logger=logger) as tr:
        t_scene_pre = time.perf_counter()
        if adaptive_downsample and adaptive_apply_to in ("scene", "both"):
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

        T_pred, debug, ppf_frontend_time, legacy_clustering_time, backend_time = _birdal_match_segments(
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
        "enable_rsmrq": bool(runtime_cfg.get("enable_rsmrq", False)),
        "enable_robust_vote": bool(runtime_cfg.get("enable_robust_vote", False)),
        "store_pair_features": bool(runtime_cfg.get("store_pair_features", False)),
        "adaptive_downsample": bool(runtime_cfg.get("adaptive_downsample", False)),
        "final_pose_policy": str(runtime_cfg.get("final_pose_policy", "")),
    }
    debug["external_method"]["reproduction_notes"] = (
        "Birdal round-1 reproduction on top of the repo's Stanford-consistent Drost baseline. "
        "It adds coarse scene segmentation, weighted voting, local alpha interpolation, and "
        "a lightweight visibility-aware ranking stage."
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
