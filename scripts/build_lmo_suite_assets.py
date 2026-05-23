import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
LMO_CONFIG_DIR = CONFIGS_DIR / "LMO"
TABLES_DIR = REPO_ROOT / "experiments" / "tables" / "lmo"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

LMO_ROOT = REPO_ROOT / "data" / "LM-O (Linemod-Occluded)"
LMO_TEST_ROOT = LMO_ROOT / "lmo_test_all" / "test"
LMO_PREPROC_ROOT = LMO_TEST_ROOT / "_preprocessed_lmo" / "inst_pcd_visib"
LMO_MODELS_DIR = LMO_ROOT / "lmo_models" / "models_eval"
LMO_SCENE_ID = "000002"
LMO_FULL_CSV = TABLES_DIR / "lmo_scene000002_full.csv"
LMO_SUBSET_SRC_CSV = REPO_ROOT / "lmo_subset_csvs" / "lmo_scene000002_eval_500.csv"
LMO_SUBSET_CSV = TABLES_DIR / "lmo_scene000002_eval_500.csv"


OBJ_IDS = [1, 5, 6, 8, 9, 10, 11, 12]


GROUPS = {
    "ablation": [
        "same_backbone",
        "no_rsmrq",
        "no_robust_vote",
        "no_backend",
        "ours_full",
    ],
    "tradeoff": [
        "adaptive",
        "ours_rv20",
        "ours_rv10",
        "ours_cap64",
        "ours_cap128",
        "ours_cap256",
    ],
    "external": [
        "drost",
        "going_further",
        "birdal",
        "edge_enhanced_ppf",
    ],
}


METHOD_SPECS = {
    "same_backbone": {
        "label": "Same-backbone Baseline",
        "group": "ablation",
        "config": "same_backbone_baseline_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": True,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "no_rsmrq": {
        "label": "No RS-MRQ",
        "group": "ablation",
        "config": "ablation_no_rsmrq_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": True,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "no_robust_vote": {
        "label": "No Robust Vote",
        "group": "ablation",
        "config": "ablation_no_robust_vote_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "no_backend": {
        "label": "No Backend",
        "group": "ablation",
        "config": "ablation_no_backend_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "ours_full": {
        "label": "Ours (Full)",
        "group": "ablation",
        "config": "ablation_ours_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "adaptive": {
        "label": "Adaptive Two-Stage",
        "group": "tradeoff",
        "config": "adaptive_two_stage_lmo.yaml",
        "runner_kind": "adaptive",
        "runner_script": "scripts/run_batch_stanford_adaptive.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "ours_rv20": {
        "label": "Ours + RV20",
        "group": "tradeoff",
        "config": "ablation_ours_rv20_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "ours_rv10": {
        "label": "Ours + RV10",
        "group": "tradeoff",
        "config": "ablation_ours_rv10_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "ours_cap64": {
        "label": "Ours + RS-MRQ cap64",
        "group": "tradeoff",
        "config": "ablation_ours_cap64_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "ours_cap128": {
        "label": "Ours + RS-MRQ cap128",
        "group": "tradeoff",
        "config": "ablation_ours_cap128_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "ours_cap256": {
        "label": "Ours + RS-MRQ cap256",
        "group": "tradeoff",
        "config": "ablation_ours_cap256_lmo.yaml",
        "runner_kind": "internal",
        "runner_script": "scripts/run_batch_stanford.py",
        "rebuild_cache": False,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_internal",
    },
    "drost": {
        "label": "Drost",
        "group": "external",
        "config": "drost_lmo.yaml",
        "runner_kind": "external_drost",
        "runner_script": "scripts/run_batch_stanford_drost.py",
        "rebuild_cache": True,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_drost",
    },
    "going_further": {
        "label": "Going Further",
        "group": "external",
        "config": "going_further_lmo.yaml",
        "runner_kind": "external_going_further",
        "runner_script": "scripts/run_batch_stanford_going_further.py",
        "rebuild_cache": True,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_going_further",
    },
    "birdal": {
        "label": "Birdal Revisited",
        "group": "external",
        "config": "birdal_revisited_lmo.yaml",
        "runner_kind": "external_birdal",
        "runner_script": "scripts/run_batch_stanford_birdal.py",
        "rebuild_cache": True,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_birdal",
    },
    "edge_enhanced_ppf": {
        "label": "Edge-enhanced PPF",
        "group": "external",
        "config": "edge_enhanced_ppf_lmo.yaml",
        "runner_kind": "external_edge",
        "runner_script": "scripts/run_batch_stanford_edge_enhanced.py",
        "rebuild_cache": True,
        "cache_dir": "data/LM-O (Linemod-Occluded)/model_cache_lmo_edge_enhanced",
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_yaml_fallback(*paths: Path) -> Dict[str, Any]:
    for path in paths:
        if path.exists():
            return _load_yaml(path)
    raise FileNotFoundError(f"None of the config paths exist: {[str(path) for path in paths]}")


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _set_output(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(cfg)
    out["output"] = {
        "root": "experiments",
        "results_dir": "experiments/results",
        "logs_dir": "experiments/logs",
        "tables_dir": "experiments/tables",
        "figures_dir": "experiments/figures",
    }
    out.setdefault("bop_gt", {})["t_scale"] = 1.0
    return out


def _base_internal_cfg() -> Dict[str, Any]:
    return _set_output(
        _load_yaml_fallback(
            CONFIGS_DIR / "ablation_ours.yaml",
            LMO_CONFIG_DIR / "ablation_ours_lmo.yaml",
        )
    )


def _base_baseline_cfg() -> Dict[str, Any]:
    return _set_output(
        _load_yaml_fallback(
            CONFIGS_DIR / "ablation_baseline.yaml",
            LMO_CONFIG_DIR / "same_backbone_baseline_lmo.yaml",
        )
    )


def _make_same_backbone() -> Dict[str, Any]:
    cfg = _base_baseline_cfg()
    cfg["final_pose_policy"] = "raw_top1_vote"
    return cfg


def _make_no_rsmrq() -> Dict[str, Any]:
    cfg = _set_output(
        _load_yaml_fallback(
            CONFIGS_DIR / "ablation_ours_no_rsmrq.yaml",
            LMO_CONFIG_DIR / "ablation_no_rsmrq_lmo.yaml",
        )
    )
    cfg["final_pose_policy"] = "selected_plus_mode_cluster"
    return cfg


def _make_no_robust_vote() -> Dict[str, Any]:
    cfg = _set_output(
        _load_yaml_fallback(
            CONFIGS_DIR / "ablation_ours_no_robustvote.yaml",
            LMO_CONFIG_DIR / "ablation_no_robust_vote_lmo.yaml",
        )
    )
    cfg["final_pose_policy"] = "selected_plus_mode_cluster"
    return cfg


def _make_no_backend() -> Dict[str, Any]:
    cfg = _base_internal_cfg()
    pose_selection = cfg.setdefault("pose_selection", {})
    pose_selection["enable"] = True
    pose_selection["pre_top_m_by_vote"] = 1
    pose_selection["candidate_top_k"] = 1
    pose_selection["refine_top_k"] = 0
    pose_selection["inlier_radius"] = 8.0
    pose_selection["max_correspondence_distance"] = 15.0
    pose_selection["residual_sigma"] = 6.0
    pose_selection["coverage_grid_size"] = 4
    pose_selection["normal_use_abs_dot"] = False
    pose_selection["keep_original_if_refine_worse"] = True
    pose_selection["weights"] = {
        "vote": 1.0,
        "inlier": 0.0,
        "coverage": 0.0,
        "normal": 0.0,
        "residual": 0.0,
        "visibility": 0.0,
    }
    visibility = pose_selection.setdefault("visibility", {})
    visibility["enable"] = False
    visibility["radius"] = 8.0
    visibility["normal_dot_thresh"] = 0.10
    visibility["require_normal_agreement"] = True
    visibility["scene_normal_dot_thresh"] = 0.20
    candidate_veto = pose_selection.setdefault("candidate_veto", {})
    candidate_veto["enable"] = False
    candidate_veto["min_visibility_and_inlier_visibility"] = 0.12
    candidate_veto["min_visibility_and_inlier_inlier"] = 0.22
    candidate_veto["min_visibility_and_coverage_visibility"] = 0.10
    candidate_veto["min_visibility_and_coverage_coverage"] = 0.30
    candidate_veto["relative_visibility_ratio"] = 0.55
    candidate_veto["relative_inlier_ratio"] = 0.60
    candidate_veto["relative_coverage_ratio"] = 0.60
    candidate_veto["min_keep_candidates"] = 1
    light_refine = pose_selection.setdefault("light_refine", {})
    light_refine["enable"] = False
    light_refine["max_iter"] = 5
    light_refine["distance_threshold"] = 8.0
    cfg.setdefault("pose_clustering", {})["enable"] = False
    cfg["final_pose_policy"] = "selected_top1"
    return cfg


def _make_ours_full() -> Dict[str, Any]:
    cfg = _base_internal_cfg()
    cfg["final_pose_policy"] = "selected_plus_mode_cluster"
    return cfg


def _make_rv(top_m: int) -> Dict[str, Any]:
    cfg = _make_ours_full()
    cfg.setdefault("robust_vote", {})["top_m_per_bucket"] = int(top_m)
    return cfg


def _make_cap(cap: int) -> Dict[str, Any]:
    cfg = _make_ours_full()
    cfg.setdefault("rsmrq", {})["global_candidate_cap"] = int(cap)
    return cfg


def _scale_stanford_external_cfg(path: Path) -> Dict[str, Any]:
    cfg = _set_output(_load_yaml(path))
    # switch to mm-scale base parameters for LMO
    cfg["sampling_leaf"] = 15.0
    cfg["normal_k"] = 5
    cfg["scene_ref_sampling_rate"] = 20
    cfg["pos_thresh"] = 20.0
    cfg["rot_thresh_deg"] = 20.0
    cfg["distance_step_ratio"] = 0.6
    cfg["adaptive_downsample"] = True
    cfg["adaptive_downsample_apply_to"] = "scene"
    base_cfg = _load_yaml_fallback(
        CONFIGS_DIR / "ablation_ours.yaml",
        LMO_CONFIG_DIR / "ablation_ours_lmo.yaml",
    )
    cfg["adaptive_downsample_cfg"] = deepcopy(base_cfg["adaptive_downsample_cfg"])
    cfg.setdefault("icp_refine", {})["distance_threshold"] = 5.0
    cfg.setdefault("bop_gt", {})["t_scale"] = 1.0
    return cfg


def _make_drost() -> Dict[str, Any]:
    cfg = _scale_stanford_external_cfg(CONFIGS_DIR / "Stanford" / "drost_original_stanford.yaml")
    cfg["final_pose_policy"] = "raw_top1_vote"
    cfg["adaptive_downsample_apply_to"] = "scene"
    cfg.setdefault("external_method", {})["comparison_mode"] = "lmo_base_consistent"
    return cfg


def _make_birdal() -> Dict[str, Any]:
    cfg = _scale_stanford_external_cfg(CONFIGS_DIR / "Stanford" / "birdal_revisited_stanford.yaml")
    birdal = cfg.setdefault("birdal_revisited", {})
    birdal["segmentation_min_eps"] = 10.0
    birdal["ranking_inlier_radius"] = 8.0
    birdal["visibility_radius"] = 8.0
    cfg.setdefault("external_method", {})["comparison_mode"] = "lmo_base_consistent"
    return cfg


def _make_going_further() -> Dict[str, Any]:
    cfg = _scale_stanford_external_cfg(CONFIGS_DIR / "Stanford" / "going_further_ppf_stanford.yaml")
    cfg.setdefault("external_method", {})["comparison_mode"] = "lmo_base_consistent"
    return cfg


def _make_edge_enhanced() -> Dict[str, Any]:
    cfg = _scale_stanford_external_cfg(CONFIGS_DIR / "Stanford" / "edge_enhanced_ppf_stanford.yaml")
    edge = cfg.setdefault("edge_enhanced_ppf", {})
    edge["verification_edge_match_radius"] = 8.0
    cfg.setdefault("external_method", {})["comparison_mode"] = "lmo_base_consistent_plus_method_scene_sampling"
    return cfg


def _make_adaptive() -> Dict[str, Any]:
    return {
        "stage1_config": "configs/LMO/ablation_no_rsmrq_lmo.yaml",
        "stage2_config": "configs/LMO/ablation_ours_lmo.yaml",
        "stage1_method_name": "no_rsmrq",
        "stage2_method_name": "ours_full",
        "adaptive_trigger_thresholds": {
            "best_score": 0.72,
            "top_score_margin": 0.02,
            "best_visibility_support": 0.28,
        },
        "output": {
            "root": "experiments",
            "results_dir": "experiments/results",
            "logs_dir": "experiments/logs",
            "tables_dir": "experiments/tables",
            "figures_dir": "experiments/figures",
        },
    }


CONFIG_BUILDERS = {
    "same_backbone_baseline_lmo.yaml": _make_same_backbone,
    "ablation_no_rsmrq_lmo.yaml": _make_no_rsmrq,
    "ablation_no_robust_vote_lmo.yaml": _make_no_robust_vote,
    "ablation_no_backend_lmo.yaml": _make_no_backend,
    "ablation_ours_lmo.yaml": _make_ours_full,
    "adaptive_two_stage_lmo.yaml": _make_adaptive,
    "ablation_ours_rv20_lmo.yaml": lambda: _make_rv(20),
    "ablation_ours_rv10_lmo.yaml": lambda: _make_rv(10),
    "ablation_ours_cap64_lmo.yaml": lambda: _make_cap(64),
    "ablation_ours_cap128_lmo.yaml": lambda: _make_cap(128),
    "ablation_ours_cap256_lmo.yaml": lambda: _make_cap(256),
    "drost_lmo.yaml": _make_drost,
    "going_further_lmo.yaml": _make_going_further,
    "birdal_revisited_lmo.yaml": _make_birdal,
    "edge_enhanced_ppf_lmo.yaml": _make_edge_enhanced,
}


def build_full_csv() -> int:
    inst_root = LMO_PREPROC_ROOT / LMO_SCENE_ID
    depth_root = LMO_TEST_ROOT / LMO_SCENE_ID / "depth"
    rows: List[Dict[str, Any]] = []
    frame_paths = sorted(inst_root.glob("*.ply"))
    grouped: Dict[int, Dict[int, Path]] = {}
    for path in frame_paths:
        frame_id = int(path.stem.split("_")[0])
        obj_token = int(path.stem.split("_")[1])
        grouped.setdefault(frame_id, {})[obj_token] = path

    for frame_id in sorted(grouped):
        depth_path = depth_root / f"{frame_id:06d}.png"
        for obj_token, obj_id in enumerate(OBJ_IDS):
            pcd_path = grouped.get(frame_id, {}).get(obj_token)
            if pcd_path is None:
                continue
            rows.append(
                {
                    "pcd_path": _repo_rel(pcd_path),
                    "obj_token": obj_token,
                    "frame_id": frame_id,
                    "depth_path": _repo_rel(depth_path),
                    "expected_model_path": _repo_rel(LMO_MODELS_DIR / f"obj_{obj_id:06d}.ply"),
                    "expected_obj_id": obj_id,
                    "scene_id": int(LMO_SCENE_ID),
                    "scene_name": f"LMO_{LMO_SCENE_ID}",
                    "scene_variant": frame_id,
                }
            )

    LMO_FULL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LMO_FULL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pcd_path",
                "obj_token",
                "frame_id",
                "depth_path",
                "expected_model_path",
                "expected_obj_id",
                "scene_id",
                "scene_name",
                "scene_variant",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _normalize_repo_like_path(raw: str) -> str:
    raw_norm = str(raw).strip().replace("\\", "/")
    if not raw_norm:
        return raw_norm

    if raw_norm.startswith("data/"):
        return Path(*[part for part in raw_norm.split("/") if part]).as_posix()

    raw_path = Path(raw_norm)
    if raw_path.is_absolute():
        try:
            return raw_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except Exception:
            return raw_path.as_posix()

    return raw_path.as_posix()


def build_subset_csv() -> int:
    rows: List[Dict[str, Any]] = []
    with LMO_SUBSET_SRC_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            rows.append(
                {
                    "pcd_path": _normalize_repo_like_path(row["pcd_path"]),
                    "obj_token": int(row["obj_token"]),
                    "frame_id": frame_id,
                    "depth_path": _normalize_repo_like_path(row["depth_path"]),
                    "expected_model_path": _normalize_repo_like_path(row["expected_model_path"]),
                    "expected_obj_id": int(row["expected_obj_id"]),
                    "scene_id": int(LMO_SCENE_ID),
                    "scene_name": f"LMO_{LMO_SCENE_ID}",
                    "scene_variant": frame_id,
                }
            )

    with LMO_SUBSET_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pcd_path",
                "obj_token",
                "frame_id",
                "depth_path",
                "expected_model_path",
                "expected_obj_id",
                "scene_id",
                "scene_name",
                "scene_variant",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_configs() -> None:
    LMO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in CONFIG_BUILDERS.items():
        _dump_yaml(builder(), LMO_CONFIG_DIR / filename)


def _get_nested(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_config_audit(rows_count: int, subset_rows_count: int) -> None:
    lines = [
        "# LMO Config Audit",
        "",
        f"- Default subset CSV: `{LMO_SUBSET_CSV.relative_to(REPO_ROOT)}`",
        f"- Default subset rows: `{subset_rows_count}`",
        f"- Full CSV: `{LMO_FULL_CSV.relative_to(REPO_ROOT)}`",
        f"- Full instance rows: `{rows_count}`",
        f"- Models dir: `{LMO_MODELS_DIR.relative_to(REPO_ROOT)}`",
        "",
        "| Method | Config | enable_rsmrq | enable_robust_vote | pose_selection.enable | pose_clustering.enable | final_pose_policy | rebuild_cache | cache_dir | output_ok | scale_note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for key, spec in METHOD_SPECS.items():
        cfg_path = LMO_CONFIG_DIR / spec["config"]
        cfg = _load_yaml(cfg_path)
        if spec["runner_kind"] == "adaptive":
            enable_rsmrq = "-"
            enable_robust = "-"
            pose_sel = "-"
            pose_cluster = "-"
            final_pose = "adaptive(stage1->stage2)"
            scale_note = "LMO mm-scale via stage1/stage2 configs"
        else:
            enable_rsmrq = _get_nested(cfg, "enable_rsmrq", "")
            enable_robust = _get_nested(cfg, "enable_robust_vote", "")
            pose_sel = _get_nested(cfg, "pose_selection.enable", False)
            pose_cluster = _get_nested(cfg, "pose_clustering.enable", False)
            final_pose = _get_nested(cfg, "final_pose_policy", "")
            if spec["group"] == "external":
                scale_note = "Stanford external config rescaled to mm for LMO"
            else:
                scale_note = "Native mm-scale LMO config"
        output_ok = _get_nested(cfg, "output.results_dir", "") == "experiments/results"
        lines.append(
            "| "
            + " | ".join(
                [
                    spec["label"],
                    str(cfg_path.relative_to(REPO_ROOT)),
                    str(enable_rsmrq),
                    str(enable_robust),
                    str(pose_sel),
                    str(pose_cluster),
                    str(final_pose),
                    "Yes" if spec["rebuild_cache"] else "No",
                    f"`{spec['cache_dir']}`",
                    "Yes" if output_ok else "No",
                    scale_note,
                ]
            )
            + " |"
        )
    (TABLES_DIR / "lmo_config_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_full_plan() -> None:
    lines = [
        "# LMO Full Plan",
        "",
        "- Benchmark scope: `LMO scene 000002 eval_500 subset`",
        f"- Default CSV: `{LMO_SUBSET_CSV.relative_to(REPO_ROOT)}`",
        "",
        "## Group A: Full Ablation",
        "",
    ]
    for key in GROUPS["ablation"]:
        lines.append(f"- {METHOD_SPECS[key]['label']}")
    lines.extend(
        [
            "",
            "## Group B: Trade-off Variants",
            "",
        ]
    )
    for key in GROUPS["tradeoff"]:
        lines.append(f"- {METHOD_SPECS[key]['label']}")
    lines.extend(
        [
            "",
            "## Group C: External Baselines",
            "",
        ]
    )
    for key in GROUPS["external"]:
        lines.append(f"- {METHOD_SPECS[key]['label']}")
    (TABLES_DIR / "lmo_full_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    build_configs()
    rows_count = build_full_csv()
    subset_rows_count = build_subset_csv()
    build_config_audit(rows_count, subset_rows_count)
    build_full_plan()
    print(f"Wrote configs under {LMO_CONFIG_DIR}")
    print(f"Wrote {LMO_SUBSET_CSV} with {subset_rows_count} rows")
    print(f"Wrote {LMO_FULL_CSV} with {rows_count} rows")
    print(f"Wrote {TABLES_DIR / 'lmo_config_audit.md'}")
    print(f"Wrote {TABLES_DIR / 'lmo_full_plan.md'}")


if __name__ == "__main__":
    main()
