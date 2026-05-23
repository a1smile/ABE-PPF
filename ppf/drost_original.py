import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d

from .registration import RegistrationStats, run_registration


def make_drost_original_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = copy.deepcopy(cfg)
    runtime["enable_rsmrq"] = False
    runtime["enable_robust_vote"] = False
    runtime["enable_kde_refine"] = False

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
            "name": "drost_original_ppf",
            "paper": "Drost et al., CVPR 2010",
            "implementation_style": "pure_python_repo_reproduction",
            "comparison_mode": "stanford_base_consistent",
            "exact_components": [
                "baseline_hash_table",
                "classic_point_pair_feature_quantization",
                "2d_hough_voting",
                "raw_vote_pose_generation",
            ],
            "approximate_components": [
                "repo_default_normal_estimation",
                "repo_default_scene_reference_sampling",
                "repo_stanford_preprocess_and_downsampling",
                "repo_raw_top1_vote_finalization",
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


def run_drost_registration(
    model_path: str,
    scene_path: str,
    cfg: Dict[str, Any],
    logger=None,
    model_cache_path: Optional[str] = None,
    strict_cache: bool = True,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud, Dict[str, Any], RegistrationStats]:
    runtime_cfg = make_drost_original_config(cfg)
    T_pred, out_model, debug, stats = run_registration(
        model_path=model_path,
        scene_path=scene_path,
        cfg=runtime_cfg,
        logger=logger,
        model_cache_path=model_cache_path,
        strict_cache=strict_cache,
    )

    debug = copy.deepcopy(debug) if isinstance(debug, dict) else {}
    debug["external_method"] = dict(runtime_cfg.get("external_method", {}))
    debug["external_method"]["effective_config"] = {
        "enable_rsmrq": bool(runtime_cfg.get("enable_rsmrq", False)),
        "enable_robust_vote": bool(runtime_cfg.get("enable_robust_vote", False)),
        "enable_kde_refine": bool(runtime_cfg.get("enable_kde_refine", False)),
        "adaptive_downsample": bool(runtime_cfg.get("adaptive_downsample", False)),
        "final_pose_policy": str(runtime_cfg.get("final_pose_policy", "")),
    }
    debug["external_method"]["reproduction_notes"] = (
        "Pure Python Drost-style baseline built on the repo's baseline PPF hash/vote/legacy-cluster path. "
        "This intentionally disables all repo-specific front-end and back-end enhancements."
    )
    return T_pred, out_model, debug, stats
