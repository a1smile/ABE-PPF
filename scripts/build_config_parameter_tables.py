from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from nature_table_utils import export_table_bundle


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
OUTPUT_DIR = ROOT / "experiments" / "tables" / "nature_style"

DATASET_DISPLAY = {"LMO": "LM-O", "Stanford": "Stanford"}
GROUP_DISPLAY = {
    "Main": "Main",
    "Ablation": "Ablation",
    "Trade-off": "Trade-off",
    "External baseline": "External Baseline",
}
METHOD_DISPLAY = {
    "Ours (full)": "Ours (Full)",
    "Ours Full": "Ours (Full)",
    "Same-backbone baseline": "Same-backbone Baseline",
    "No Rsmrq": "No RS-MRQ",
    "Adaptive two-stage": "Adaptive Two-Stage",
}
FINAL_POLICY_DISPLAY = {
    "selected_plus_mode_cluster": "Selected + mode cluster",
    "selected_top1": "Selected top-1",
    "raw_top1_vote": "Raw top-1 vote",
    "birdal_revisited_ranked": "Birdal ranked",
    "edge_verify": "Edge verification",
}


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def nested_get(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def flatten_dict(data: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_dict(value, new_prefix))
    else:
        flat[prefix] = data
    return flat


def fmt_scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def pretty_number(value: float, digits: int = 3) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def to_mm(value: Any, dataset: str) -> float:
    scalar = float(value)
    return scalar * 1000.0 if dataset == "Stanford" else scalar


def fmt_mm(value: Any, dataset: str, digits: int = 1) -> str:
    return f"{pretty_number(to_mm(value, dataset), digits)} mm"


def fmt_yes_no(value: Any) -> str:
    if value is None:
        return "-"
    return "Yes" if bool(value) else "No"


def display_dataset(name: str) -> str:
    return DATASET_DISPLAY.get(name, name)


def display_group(name: str) -> str:
    return GROUP_DISPLAY.get(name, name)


def display_method(name: str) -> str:
    return METHOD_DISPLAY.get(name, name)


def display_policy(name: Any) -> str:
    if name is None:
        return "-"
    text = str(name)
    return FINAL_POLICY_DISPLAY.get(text, text.replace("_", " "))


def fmt_value(data: dict[str, Any], path: str) -> str:
    return fmt_scalar(nested_get(data, path))


def format_w_levels(data: dict[str, Any], dataset: str) -> str:
    levels = nested_get(data, "rsmrq.w_levels") or []
    parts = []
    for idx, level in enumerate(levels, start=1):
        if not isinstance(level, list) or len(level) != 4:
            continue
        rot = ", ".join(f"{v:.3f}" for v in level[:3])
        parts.append(f"L{idx}: ({rot}; {pretty_number(to_mm(level[3], dataset), 1)} mm)")
    return "; ".join(parts) if parts else "-"


def format_adaptive_schedule(data: dict[str, Any]) -> str:
    cfg = nested_get(data, "adaptive_downsample_cfg") or {}
    return (
        f"{cfg.get('no_downsample_thresh')} / {cfg.get('mid_thresh')} / {cfg.get('large_thresh')} pts"
        f" -> {cfg.get('target_mid')} / {cfg.get('target_large')} / {cfg.get('target_xlarge')} pts"
    )


def format_weight_vector(data: dict[str, Any]) -> str:
    w = nested_get(data, "pose_selection.weights") or {}
    order = ["vote", "inlier", "coverage", "normal", "residual", "visibility"]
    return ", ".join(f"{k}={w.get(k)}" for k in order if k in w) or "-"


def format_visibility_rule(data: dict[str, Any], dataset: str) -> str:
    vis = nested_get(data, "pose_selection.visibility") or {}
    return (
        f"r={fmt_mm(vis.get('radius'), dataset)}, n={pretty_number(float(vis.get('normal_dot_thresh')), 2)}, "
        f"scene n={pretty_number(float(vis.get('scene_normal_dot_thresh')), 2)}"
    )


MAIN_ROW_SPECS = [
    {
        "section": "Pre-processing",
        "parameter": "Sampling voxel size",
        "path": "sampling_leaf",
        "distance_mm": True,
        "notes": "Voxel size for model and scene downsampling.",
    },
    {
        "section": "Pre-processing",
        "parameter": "Scene reference sampling stride",
        "path": "scene_ref_sampling_rate",
        "notes": "Keep one reference point every N scene points.",
    },
    {
        "section": "Pre-processing",
        "parameter": "Normal estimation k",
        "path": "normal_k",
        "notes": "kNN size for normal estimation.",
    },
    {
        "section": "PPF quantization",
        "parameter": "Angle quantization (deg)",
        "path": "angle_step_deg",
        "notes": "Angular step for PPF discretization.",
    },
    {
        "section": "PPF quantization",
        "parameter": "Distance step ratio",
        "path": "distance_step_ratio",
        "notes": "Relative distance quantization step.",
    },
    {
        "section": "Adaptive downsampling",
        "parameter": "Enable adaptive downsampling",
        "path": "adaptive_downsample",
        "notes": "Adaptive voxel sizing before matching.",
    },
    {
        "section": "Adaptive downsampling",
        "parameter": "Apply adaptive downsampling to",
        "path": "adaptive_downsample_apply_to",
        "notes": "Target cloud(s) that use the adaptive policy.",
    },
    {
        "section": "Adaptive downsampling",
        "parameter": "Adaptive point schedule",
        "kind": "adaptive_schedule",
        "notes": "Point-count thresholds and target counts.",
    },
    {
        "section": "RS-MRQ",
        "parameter": "Hierarchy levels L",
        "path": "rsmrq.L",
        "notes": "Number of coarse-to-fine query levels.",
    },
    {
        "section": "RS-MRQ",
        "parameter": "Hash tables per level",
        "path": "rsmrq.T_tables",
        "notes": "Tables used for repeated random queries.",
    },
    {
        "section": "RS-MRQ",
        "parameter": "Query widths [rot_x, rot_y, rot_z, trans]",
        "kind": "w_levels",
        "notes": "Angular widths are in radians; translation width follows dataset units.",
    },
    {
        "section": "RS-MRQ",
        "parameter": "Merge mode",
        "path": "rsmrq.merge_mode",
        "notes": "How repeated query candidates are merged.",
    },
    {
        "section": "Robust vote",
        "parameter": "Vote kernel (sigma, tau, B)",
        "kind": "robust_vote_kernel",
        "notes": "Gaussian robust vote parameters.",
    },
    {
        "section": "Robust vote",
        "parameter": "Top hypotheses per bucket",
        "path": "robust_vote.top_m_per_bucket",
        "notes": "Maximum retained hypotheses for robust vote aggregation.",
    },
    {
        "section": "Pose selection",
        "parameter": "Pre-top-M by vote",
        "path": "pose_selection.pre_top_m_by_vote",
        "notes": "Initial vote-ranked candidates before refinement.",
    },
    {
        "section": "Pose selection",
        "parameter": "Candidate top-k / refine top-k",
        "kind": "candidate_refine",
        "notes": "Candidates kept for scoring and lightweight refinement.",
    },
    {
        "section": "Pose selection",
        "parameter": "Inlier radius",
        "path": "pose_selection.inlier_radius",
        "distance_mm": True,
        "notes": "Radius for inlier support evaluation.",
    },
    {
        "section": "Pose selection",
        "parameter": "Max correspondence distance",
        "path": "pose_selection.max_correspondence_distance",
        "distance_mm": True,
        "notes": "Distance cutoff for correspondence residuals.",
    },
    {
        "section": "Pose selection",
        "parameter": "Residual sigma",
        "path": "pose_selection.residual_sigma",
        "distance_mm": True,
        "notes": "Scale used in residual-based candidate scoring.",
    },
    {
        "section": "Pose selection",
        "parameter": "Coverage grid size",
        "path": "pose_selection.coverage_grid_size",
        "notes": "Grid resolution used to measure coverage.",
    },
    {
        "section": "Pose selection",
        "parameter": "Score weights",
        "kind": "weights",
        "notes": "Weights for vote, inlier, coverage, normal, residual, and visibility terms.",
    },
    {
        "section": "Pose selection",
        "parameter": "Visibility rule",
        "kind": "visibility_rule",
        "notes": "Visibility support radius and normal-agreement thresholds.",
    },
    {
        "section": "Pose clustering",
        "parameter": "Mode clustering thresholds (trans / rot deg)",
        "kind": "cluster_thresholds",
        "notes": "Thresholds for post-selection mode clustering.",
    },
    {
        "section": "Refinement",
        "parameter": "ICP refine",
        "kind": "icp",
        "notes": "Disabled in the default experiments.",
    },
    {
        "section": "Output policy",
        "parameter": "Final pose policy",
        "path": "final_pose_policy",
        "notes": "How the final pose is chosen after selection/clustering.",
    },
]


VARIANT_ROWS = [
    ("Stanford", "Main", "Ours (full)", "ablation_ours_stanford.yaml"),
    ("Stanford", "Ablation", "Same-backbone baseline", "same_backbone_baseline_stanford.yaml"),
    ("Stanford", "Ablation", "No RS-MRQ", "ablation_no_rsmrq_stanford.yaml"),
    ("Stanford", "Ablation", "No Robust Vote", "ablation_no_robust_vote_stanford.yaml"),
    ("Stanford", "Ablation", "No Backend", "ablation_no_backend_stanford.yaml"),
    ("Stanford", "Trade-off", "Ours-RV10", "ablation_ours_rv10_stanford.yaml"),
    ("Stanford", "Trade-off", "Ours-RV20", "ablation_ours_rv20_stanford.yaml"),
    ("Stanford", "Trade-off", "Ours-Cap64", "ablation_ours_rsmrq_cap64_stanford.yaml"),
    ("Stanford", "Trade-off", "Ours-Cap128", "ablation_ours_rsmrq_cap128_stanford.yaml"),
    ("Stanford", "Trade-off", "Ours-Cap256", "ablation_ours_rsmrq_cap256_stanford.yaml"),
    ("Stanford", "External baseline", "Drost", "drost_original_stanford.yaml"),
    ("Stanford", "External baseline", "Birdal Revisited", "birdal_revisited_stanford.yaml"),
    ("Stanford", "External baseline", "Edge-enhanced PPF", "edge_enhanced_ppf_stanford.yaml"),
    ("Stanford", "External baseline", "Going Further", "going_further_ppf_stanford.yaml"),
    ("LMO", "Main", "Ours (full)", "ablation_ours_lmo.yaml"),
    ("LMO", "Ablation", "Same-backbone baseline", "same_backbone_baseline_lmo.yaml"),
    ("LMO", "Ablation", "No RS-MRQ", "ablation_no_rsmrq_lmo.yaml"),
    ("LMO", "Ablation", "No Robust Vote", "ablation_no_robust_vote_lmo.yaml"),
    ("LMO", "Ablation", "No Backend", "ablation_no_backend_lmo.yaml"),
    ("LMO", "Trade-off", "Ours-RV10", "ablation_ours_rv10_lmo.yaml"),
    ("LMO", "Trade-off", "Ours-RV20", "ablation_ours_rv20_lmo.yaml"),
    ("LMO", "Trade-off", "Ours-Cap64", "ablation_ours_cap64_lmo.yaml"),
    ("LMO", "Trade-off", "Ours-Cap128", "ablation_ours_cap128_lmo.yaml"),
    ("LMO", "Trade-off", "Ours-Cap256", "ablation_ours_cap256_lmo.yaml"),
    ("LMO", "Trade-off", "Adaptive two-stage", "adaptive_two_stage_lmo.yaml"),
    ("LMO", "External baseline", "Drost", "drost_lmo.yaml"),
    ("LMO", "External baseline", "Birdal Revisited", "birdal_revisited_lmo.yaml"),
    ("LMO", "External baseline", "Edge-enhanced PPF", "edge_enhanced_ppf_lmo.yaml"),
    ("LMO", "External baseline", "Going Further", "going_further_lmo.yaml"),
]


DIFF_KEYS = [
    "enable_rsmrq",
    "enable_robust_vote",
    "pose_selection.enable",
    "pose_clustering.enable",
    "final_pose_policy",
    "robust_vote.top_m_per_bucket",
    "rsmrq.global_candidate_cap",
    "rsmrq.merge_mode",
    "pos_thresh",
    "rot_thresh_deg",
    "pose_selection.pre_top_m_by_vote",
    "pose_selection.candidate_top_k",
    "pose_selection.refine_top_k",
]


def main_value(row_spec: dict[str, Any], data: dict[str, Any], dataset: str) -> str:
    kind = row_spec.get("kind")
    if kind == "adaptive_schedule":
        return format_adaptive_schedule(data)
    if kind == "w_levels":
        return format_w_levels(data, dataset)
    if kind == "robust_vote_kernel":
        sigma = nested_get(data, "robust_vote.sigma")
        tau = nested_get(data, "robust_vote.tau")
        b_val = nested_get(data, "robust_vote.B")
        return f"gaussian ({sigma}, {tau}, {b_val})"
    if kind == "candidate_refine":
        c_top = nested_get(data, "pose_selection.candidate_top_k")
        r_top = nested_get(data, "pose_selection.refine_top_k")
        return f"{c_top} / {r_top}"
    if kind == "weights":
        return format_weight_vector(data)
    if kind == "visibility_rule":
        return format_visibility_rule(data, dataset)
    if kind == "cluster_thresholds":
        pos_val = nested_get(data, "pose_clustering.pos_thresh")
        rot_val = nested_get(data, "pose_clustering.rot_thresh_deg")
        return f"{fmt_mm(pos_val, dataset)} / {pretty_number(float(rot_val), 1)} deg"
    if kind == "icp":
        enabled = nested_get(data, "icp_refine.enable")
        dist = nested_get(data, "icp_refine.distance_threshold")
        return f"{fmt_yes_no(enabled)} (dist={fmt_mm(dist, dataset)})"
    if row_spec.get("distance_mm"):
        return fmt_mm(nested_get(data, row_spec["path"]), dataset)
    if row_spec.get("path") == "final_pose_policy":
        return display_policy(nested_get(data, row_spec["path"]))
    return fmt_value(data, row_spec["path"])


def method_specific_summary(config: dict[str, Any]) -> str:
    name = nested_get(config, "external_method.name")
    if name == "birdal_revisited":
        block = nested_get(config, "birdal_revisited") or {}
        return (
            f"segmentation={fmt_yes_no(block.get('segmentation_enable'))}; "
            f"rank top-k={block.get('rank_top_k')}; "
            f"weighted vote sigma={block.get('weighted_vote_sigma')}"
        )
    if name == "edge_enhanced_ppf":
        block = nested_get(config, "edge_enhanced_ppf") or {}
        return (
            f"edge-preserving sampling={fmt_yes_no(block.get('scene_sampling_use_edge_preservation'))}; "
            f"verification top-n={block.get('verification_top_n')}; "
            f"accept={block.get('verification_accept_low')}/{block.get('verification_accept_high')}"
        )
    if name == "going_further_with_ppf":
        block = nested_get(config, "going_further_ppf") or {}
        return (
            f"method-specific sampling={fmt_yes_no(block.get('use_method_specific_sampling'))}; "
            f"candidate voxel ratio={block.get('candidate_voxel_ratio')}; "
            f"two-ball={fmt_yes_no(block.get('two_ball_enable'))}"
        )
    if name == "drost_original_ppf":
        return "raw top-1 vote without post-selection"
    if "adaptive_trigger_thresholds" in config:
        block = config["adaptive_trigger_thresholds"]
        stage1 = str(config.get("stage1_method_name", "")).replace("_", " ").title()
        stage2 = str(config.get("stage2_method_name", "")).replace("_", " ").title()
        return (
            f"stage 1={display_method(stage1)}; stage 2={display_method(stage2)}; "
            f"best score>={block.get('best_score')}; "
            f"margin>={block.get('top_score_margin')}; "
            f"visibility>={block.get('best_visibility_support')}"
        )
    return "-"


def diff_summary(dataset: str, config: dict[str, Any], reference: dict[str, Any]) -> str:
    if "adaptive_trigger_thresholds" in config:
        return method_specific_summary(config)

    flat_cfg = flatten_dict(config)
    flat_ref = flatten_dict(reference)
    diffs = []

    if flat_cfg.get("enable_rsmrq") != flat_ref.get("enable_rsmrq"):
        diffs.append(f"RS-MRQ={fmt_yes_no(flat_cfg.get('enable_rsmrq'))}")
    if flat_cfg.get("enable_robust_vote") != flat_ref.get("enable_robust_vote"):
        diffs.append(f"robust vote={fmt_yes_no(flat_cfg.get('enable_robust_vote'))}")
    if flat_cfg.get("pose_selection.enable") != flat_ref.get("pose_selection.enable"):
        diffs.append(f"pose selection={fmt_yes_no(flat_cfg.get('pose_selection.enable'))}")
    if flat_cfg.get("pose_clustering.enable") != flat_ref.get("pose_clustering.enable"):
        diffs.append(f"mode clustering={fmt_yes_no(flat_cfg.get('pose_clustering.enable'))}")
    if flat_cfg.get("final_pose_policy") != flat_ref.get("final_pose_policy"):
        diffs.append(f"final policy={display_policy(flat_cfg.get('final_pose_policy'))}")
    if flat_cfg.get("robust_vote.top_m_per_bucket") != flat_ref.get("robust_vote.top_m_per_bucket"):
        diffs.append(f"RV top-M={flat_cfg.get('robust_vote.top_m_per_bucket')}")
    if flat_cfg.get("rsmrq.global_candidate_cap") != flat_ref.get("rsmrq.global_candidate_cap"):
        diffs.append(f"RS-MRQ cap={flat_cfg.get('rsmrq.global_candidate_cap')}")
    if flat_cfg.get("rsmrq.merge_mode") != flat_ref.get("rsmrq.merge_mode"):
        diffs.append(f"merge={flat_cfg.get('rsmrq.merge_mode')}")

    pos_thresh = flat_cfg.get("pos_thresh")
    rot_thresh = flat_cfg.get("rot_thresh_deg")
    if pos_thresh != flat_ref.get("pos_thresh") or rot_thresh != flat_ref.get("rot_thresh_deg"):
        if pos_thresh is not None and rot_thresh is not None:
            diffs.append(f"cluster={fmt_mm(pos_thresh, dataset, 3)} / {pretty_number(float(rot_thresh), 1)} deg")

    pre_top_m = flat_cfg.get("pose_selection.pre_top_m_by_vote")
    if pre_top_m is not None and pre_top_m != flat_ref.get("pose_selection.pre_top_m_by_vote"):
        diffs.append(f"pre-top-M={pre_top_m}")

    candidate_top = flat_cfg.get("pose_selection.candidate_top_k")
    refine_top = flat_cfg.get("pose_selection.refine_top_k")
    if (
        candidate_top is not None
        and refine_top is not None
        and (
            candidate_top != flat_ref.get("pose_selection.candidate_top_k")
            or refine_top != flat_ref.get("pose_selection.refine_top_k")
        )
    ):
        diffs.append(
            f"backend top-k={candidate_top} / {refine_top}"
        )

    method_note = method_specific_summary(config)
    if method_note != "-":
        diffs.append(method_note)

    return "; ".join(diffs) if diffs else "same as full config"


def build_main_rows() -> list[dict[str, str]]:
    stanford = load_yaml(CONFIG_ROOT / "Stanford" / "ablation_ours_stanford.yaml")
    lmo = load_yaml(CONFIG_ROOT / "LMO" / "ablation_ours_lmo.yaml")
    rows = []
    for spec in MAIN_ROW_SPECS:
        rows.append(
            {
                "Module": spec["section"],
                "Parameter": spec["parameter"],
                "Stanford": main_value(spec, stanford, "Stanford"),
                "LM-O": main_value(spec, lmo, "LMO"),
                "Description": spec["notes"],
            }
        )
    return rows


def build_variant_rows() -> list[dict[str, str]]:
    refs = {
        "Stanford": load_yaml(CONFIG_ROOT / "Stanford" / "ablation_ours_stanford.yaml"),
        "LMO": load_yaml(CONFIG_ROOT / "LMO" / "ablation_ours_lmo.yaml"),
    }
    rows = []
    for dataset, group, method, file_name in VARIANT_ROWS:
        cfg_dir = CONFIG_ROOT / dataset
        cfg = load_yaml(cfg_dir / file_name)
        rows.append(
            {
                "Dataset": display_dataset(dataset),
                "Group": display_group(group),
                "Method": display_method(method),
                "Config": file_name,
                "RS-MRQ": fmt_yes_no(nested_get(cfg, "enable_rsmrq")),
                "Robust Vote": fmt_yes_no(nested_get(cfg, "enable_robust_vote")),
                "Final Policy": display_policy(nested_get(cfg, "final_pose_policy")),
                "Key Differences vs Ours": diff_summary(dataset, cfg, refs[dataset]),
            }
        )
    return rows


def build_note_file(path: Path) -> None:
    note = """# Configuration Table Notes

- Main table values come from `configs/Stanford/ablation_ours_stanford.yaml` and `configs/LMO/ablation_ours_lmo.yaml`.
- Geometric thresholds for Stanford are converted from metres to millimetres in the paper-facing table.
- Variant rows summarize only the effective overrides relative to the dataset-specific full configuration.
- `configs/stanford_retrieval/*.ini` were not merged into the main paper table because they are scene path manifests rather than algorithm hyper-parameters.
"""
    path.write_text(note, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_rows = build_main_rows()
    variant_rows = build_variant_rows()

    main_columns = [
        {"header": "Module", "align": "left"},
        {"header": "Parameter", "align": "left"},
        {"header": "Stanford", "align": "left"},
        {"header": "LM-O", "align": "left"},
        {"header": "Description", "align": "left"},
    ]
    variant_columns = [
        {"header": "Dataset", "align": "left"},
        {"header": "Group", "align": "left"},
        {"header": "Method", "align": "left"},
        {"header": "Config", "align": "left"},
        {"header": "RS-MRQ", "align": "center"},
        {"header": "Robust Vote", "align": "center"},
        {"header": "Final Policy", "align": "left"},
        {"header": "Key Differences vs Ours", "align": "left"},
    ]

    export_table_bundle(
        OUTPUT_DIR / "config_parameter_table_main",
        main_rows,
        main_columns,
        caption="Dataset-specific default parameters used in the main experiments. Stanford distances are reported in millimetres for readability.",
        label="tab:config-main",
        col_spec="@{}llp{2.4cm}p{2.4cm}p{6.2cm}@{}",
    )
    export_table_bundle(
        OUTPUT_DIR / "config_parameter_table_variants",
        variant_rows,
        variant_columns,
        caption="Ablation, trade-off and external-baseline settings derived from the experiment configuration files.",
        label="tab:config-variants",
        longtable=True,
        col_spec="@{}llp{2.4cm}p{2.9cm}ccp{2.6cm}p{6.0cm}@{}",
    )

    build_note_file(OUTPUT_DIR / "config_parameter_table_notes.md")

    print(f"Wrote files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
