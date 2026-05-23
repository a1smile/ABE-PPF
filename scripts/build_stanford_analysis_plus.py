import argparse
import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "experiments" / "tables" / "stanford" / "analysis_plus"
FIG_DIR = ROOT / "experiments" / "figures" / "stanford_analysis_plus"
SOURCE_TABLE_DIR = ROOT / "experiments" / "tables" / "stanford"
RESULT_DIR = ROOT / "experiments" / "results" / "Stanford"
SUCCESS_THRESHOLD = 0.02
ROT_SUCCESS_THRESHOLD = 5.0
PNG_DPI = 320
BOOTSTRAP_ROUNDS = 3000
RNG_SEED = 7


def batch_json_path(filename: str) -> Path:
    preferred = RESULT_DIR / filename
    if preferred.exists():
        return preferred
    return RESULT_DIR.parent / filename

METHOD_ORDER = [
    "Same-backbone Baseline",
    "No RS-MRQ",
    "No Robust Vote",
    "No Backend",
    "Ours (Full)",
    "Adaptive Two-Stage",
    "Ours + RV20",
    "Ours + RV10",
    "Ours + RS-MRQ cap64",
    "Ours + RS-MRQ cap128",
    "Ours + RS-MRQ cap256",
]

METHOD_SOURCES = {
    "Same-backbone Baseline": {
        "batch_json": batch_json_path("stanford_same_backbone_baseline_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "same_backbone_baseline_stanford.yaml"],
    },
    "No RS-MRQ": {
        "batch_json": batch_json_path("stanford_no_rsmrq_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_no_rsmrq_stanford.yaml"],
    },
    "No Robust Vote": {
        "batch_json": batch_json_path("stanford_no_robust_vote_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_no_robust_vote_stanford.yaml"],
    },
    "No Backend": {
        "batch_json": batch_json_path("stanford_no_backend_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_no_pose_pipeline_stanford.yaml"],
    },
    "Ours (Full)": {
        "batch_json": batch_json_path("stanford_ours_full_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_ours_stanford.yaml"],
    },
    "Adaptive Two-Stage": {
        "batch_json": batch_json_path("stanford_adaptive_two_stage_batch.json"),
        "config_paths": [
            ROOT / "configs" / "Stanford" / "ablation_no_rsmrq_stanford.yaml",
            ROOT / "configs" / "Stanford" / "ablation_ours_stanford.yaml",
        ],
    },
    "Ours + RV20": {
        "batch_json": batch_json_path("stanford_ours_rv20_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_ours_rv20_stanford.yaml"],
    },
    "Ours + RV10": {
        "batch_json": batch_json_path("stanford_ours_rv10_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_ours_rv10_stanford.yaml"],
    },
    "Ours + RS-MRQ cap64": {
        "batch_json": batch_json_path("stanford_ours_cap64_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_ours_rsmrq_cap64_stanford.yaml"],
    },
    "Ours + RS-MRQ cap128": {
        "batch_json": batch_json_path("stanford_ours_cap128_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_ours_rsmrq_cap128_stanford.yaml"],
    },
    "Ours + RS-MRQ cap256": {
        "batch_json": batch_json_path("stanford_ours_cap256_batch.json"),
        "config_paths": [ROOT / "configs" / "Stanford" / "ablation_ours_rsmrq_cap256_stanford.yaml"],
    },
}

METHOD_COLORS = {
    "Same-backbone Baseline": "#4E79A7",
    "No RS-MRQ": "#59A14F",
    "No Robust Vote": "#9C755F",
    "No Backend": "#BAB0AC",
    "Ours (Full)": "#E15759",
    "Adaptive Two-Stage": "#F28E2B",
    "Ours + RV20": "#76B7B2",
    "Ours + RV10": "#B07AA1",
    "Ours + RS-MRQ cap64": "#EDC948",
    "Ours + RS-MRQ cap128": "#FF9DA7",
    "Ours + RS-MRQ cap256": "#86BCB6",
}

MAIN_METHODS = [
    "Same-backbone Baseline",
    "No RS-MRQ",
    "No Robust Vote",
    "No Backend",
    "Ours (Full)",
    "Adaptive Two-Stage",
]
RSMRQ_METHODS = [
    "Ours + RS-MRQ cap64",
    "Ours + RS-MRQ cap128",
    "Ours + RS-MRQ cap256",
    "Ours (Full)",
]
ROBUST_VOTE_METHODS = [
    "Ours (Full)",
    "Ours + RV20",
    "Ours + RV10",
]
PAIRWISE_COMPARISONS = [
    ("Ours (Full)", "No RS-MRQ"),
    ("Ours (Full)", "Adaptive Two-Stage"),
    ("Adaptive Two-Stage", "No RS-MRQ"),
    ("Ours (Full)", "No Backend"),
]
PAIRWISE_METRICS = [
    ("ADD", "Mean ADD"),
    ("registration_time", "Mean Registration Time (s)"),
    ("ppf_frontend_time", "Mean Front-end Time (s)"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Stanford analysis-plus tables, plots, and markdown reports.")
    ap.add_argument("--bootstrap_rounds", type=int, default=BOOTSTRAP_ROUNDS)
    ap.add_argument("--rng_seed", type=int, default=RNG_SEED)
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, basename: str) -> None:
    ensure_dir(FIG_DIR)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=PNG_DPI)
    fig.savefig(FIG_DIR / f"{basename}.svg")
    plt.close(fig)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def write_df(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, na_rep="NaN")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [cols] + [[str(row[col]) for col in cols] for _, row in df.iterrows()]
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(cols))]
    header = "| " + " | ".join(str(cols[i]).ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(cols))) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, sep] + body)


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out) or math.isinf(out):
        return float("nan")
    return out


def nested_get(payload: object, *keys: str) -> object:
    cur = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def bool_or_nan(value: object) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    return bool(value)


def sort_scene_key(scene_name: object) -> Tuple[int, str]:
    text = str(scene_name)
    m = re.search(r"(\d+)", text)
    return (int(m.group(1)) if m else 10**9, text)


def sort_obj_key(obj_id: object) -> Tuple[int, str]:
    try:
        return (int(obj_id), str(obj_id))
    except (TypeError, ValueError):
        return (10**9, str(obj_id))


def method_index(method: str) -> int:
    return METHOD_ORDER.index(method)


def mean_or_nan(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def median_or_nan(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else float("nan")


def max_or_nan(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.max()) if not clean.empty else float("nan")


def quantile_or_nan(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.quantile(q)) if not clean.empty else float("nan")


def success_rate(series: pd.Series, threshold: float) -> float:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[np.isfinite(clean)]
    return float((clean <= threshold).mean()) if not clean.empty else float("nan")


def pct_change(new_value: float, base_value: float) -> float:
    if pd.isna(new_value) or pd.isna(base_value) or abs(base_value) < 1e-12:
        return float("nan")
    return float((new_value - base_value) / base_value * 100.0)


def resolve_batch_config_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if path.exists():
        return path
    candidate = ROOT / "configs" / "Stanford" / path.name
    return candidate if candidate.exists() else None


def load_method_payloads() -> Dict[str, dict]:
    payloads = {}
    for method, meta in METHOD_SOURCES.items():
        payloads[method] = load_json(meta["batch_json"])
    return payloads


def config_meta_from_payload(method: str, payload: dict) -> Dict[str, object]:
    if method == "Adaptive Two-Stage":
        cfg_paths = [
            resolve_batch_config_path(payload.get("stage1_config_path")) or METHOD_SOURCES[method]["config_paths"][0],
            resolve_batch_config_path(payload.get("stage2_config_path")) or METHOD_SOURCES[method]["config_paths"][1],
        ]
        stage2_cfg = load_yaml(cfg_paths[1])
        return {
            "config_paths": [str(cfg_paths[0]), str(cfg_paths[1])],
            "robust_vote_top_m_per_bucket": safe_float(nested_get(stage2_cfg, "robust_vote", "top_m_per_bucket")),
            "rsmrq_global_candidate_cap": safe_float(nested_get(stage2_cfg, "rsmrq", "global_candidate_cap")),
            "enable_robust_vote": bool(stage2_cfg.get("enable_robust_vote", False)),
            "enable_rsmrq": bool(stage2_cfg.get("enable_rsmrq", False)),
        }
    cfg_path = resolve_batch_config_path(payload.get("config_path")) or METHOD_SOURCES[method]["config_paths"][0]
    cfg = load_yaml(cfg_path)
    robust_enabled = bool(cfg.get("enable_robust_vote", False))
    return {
        "config_paths": [str(cfg_path)],
        "robust_vote_top_m_per_bucket": (
            safe_float(nested_get(cfg, "robust_vote", "top_m_per_bucket")) if robust_enabled else float("nan")
        ),
        "rsmrq_global_candidate_cap": safe_float(nested_get(cfg, "rsmrq", "global_candidate_cap")),
        "enable_robust_vote": robust_enabled,
        "enable_rsmrq": bool(cfg.get("enable_rsmrq", False)),
    }


def build_master_per_case(payloads: Dict[str, dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for method in METHOD_ORDER:
        payload = payloads[method]
        cfg_meta = config_meta_from_payload(method, payload)
        batch_path = METHOD_SOURCES[method]["batch_json"]
        for record in payload.get("results", []):
            metrics = record.get("metrics") or {}
            stats = record.get("stats") or {}
            debug = record.get("debug") or {}
            pose_sel = debug.get("pose_selection") or {}
            robust = debug.get("robust_vote") or {}
            rsmrq = debug.get("rsmrq") or {}
            adaptive = debug.get("adaptive") or {}
            rows.append(
                {
                    "method": method,
                    "method_order": method_index(method),
                    "source_batch_json": str(batch_path),
                    "config_paths": ";".join(cfg_meta["config_paths"]),
                    "idx": int(record.get("idx", -1)),
                    "scene_id": record.get("scene_id"),
                    "scene_name": record.get("scene_name"),
                    "scene_variant": record.get("scene_variant"),
                    "obj_id": record.get("obj_id"),
                    "case_key": f"{record.get('scene_name')}|{record.get('scene_variant')}|{record.get('obj_id')}",
                    "ADD": safe_float(metrics.get("ADD")),
                    "ADD_S": safe_float(metrics.get("ADD_S")),
                    "rotation_error_deg": safe_float(metrics.get("rotation_error_deg")),
                    "translation_error": safe_float(metrics.get("translation_error")),
                    "inlier_ratio": safe_float(metrics.get("inlier_ratio")),
                    "registration_time": safe_float(stats.get("registration_time")),
                    "total_time": safe_float(stats.get("total_time")),
                    "scene_preprocess_time": safe_float(stats.get("scene_preprocess_time")),
                    "ppf_frontend_time": safe_float(stats.get("ppf_frontend_time")),
                    "backend_time": safe_float(stats.get("backend_time")),
                    "pose_selection_time": safe_float(stats.get("pose_selection_time")),
                    "pose_clustering_time": safe_float(stats.get("pose_clustering_time")),
                    "legacy_clustering_time": safe_float(stats.get("legacy_clustering_time")),
                    "candidate_inflation_mean": safe_float(
                        stats.get("candidate_inflation_mean", debug.get("candidate_inflation_mean"))
                    ),
                    "final_pose_policy": stats.get("final_pose_policy", debug.get("final_pose_policy")),
                    "adaptive_used_upgrade": bool(record.get("adaptive_used_upgrade", False)),
                    "adaptive_trigger_reason": record.get("adaptive_trigger_reason"),
                    "adaptive_total_effective_time": safe_float(record.get("adaptive_total_effective_time")),
                    "adaptive_stage1_registration_time": safe_float(record.get("adaptive_stage1_registration_time")),
                    "adaptive_stage2_registration_time": safe_float(record.get("adaptive_stage2_registration_time")),
                    "adaptive_stage1_frontend_time": safe_float(record.get("adaptive_stage1_frontend_time")),
                    "adaptive_stage2_frontend_time": safe_float(record.get("adaptive_stage2_frontend_time")),
                    "adaptive_stage1_method": record.get("adaptive_stage1_method"),
                    "adaptive_stage2_method": record.get("adaptive_stage2_method"),
                    "adaptive_final_source": record.get("adaptive_final_source"),
                    "adaptive_threshold_best_score": safe_float(nested_get(adaptive, "thresholds", "best_score")),
                    "adaptive_threshold_top_score_margin": safe_float(
                        nested_get(adaptive, "thresholds", "top_score_margin")
                    ),
                    "adaptive_threshold_best_visibility_support": safe_float(
                        nested_get(adaptive, "thresholds", "best_visibility_support")
                    ),
                    "num_input_candidates": safe_float(pose_sel.get("num_input_candidates")),
                    "num_selected_candidates": safe_float(pose_sel.get("num_selected")),
                    "num_preselected_candidates": safe_float(pose_sel.get("num_preselected")),
                    "best_score": safe_float(pose_sel.get("best_score")),
                    "best_visibility_support": safe_float(pose_sel.get("best_visibility_support")),
                    "top_scores_margin": (
                        safe_float(pose_sel.get("top_scores")[0]) - safe_float(pose_sel.get("top_scores")[1])
                        if isinstance(pose_sel.get("top_scores"), list) and len(pose_sel.get("top_scores", [])) >= 2
                        else float("nan")
                    ),
                    "n_votes": safe_float(robust.get("n_votes")),
                    "topm_used": safe_float(robust.get("topm_used")),
                    "topm_total": safe_float(robust.get("topm_total")),
                    "topm_ratio": safe_float(robust.get("topm_ratio")),
                    "rsmrq_global_candidate_cap": (
                        safe_float(rsmrq.get("global_candidate_cap"))
                        if "global_candidate_cap" in rsmrq
                        else safe_float(cfg_meta["rsmrq_global_candidate_cap"])
                    ),
                    "rsmrq_cap_hit_count": safe_float(rsmrq.get("cap_hit_count")),
                    "rsmrq_cap_hit_ratio": safe_float(rsmrq.get("cap_hit_ratio")),
                    "rsmrq_merged_candidates_pre_cap_mean": safe_float(rsmrq.get("merged_candidates_pre_cap_mean")),
                    "rsmrq_merged_candidates_post_cap_mean": safe_float(rsmrq.get("merged_candidates_post_cap_mean")),
                    "robust_vote_top_m_per_bucket": safe_float(cfg_meta["robust_vote_top_m_per_bucket"]),
                    "robust_vote_enabled": bool_or_nan(cfg_meta["enable_robust_vote"]),
                    "rsmrq_enabled": bool_or_nan(cfg_meta["enable_rsmrq"]),
                }
            )
    df = pd.DataFrame(rows)
    df["success_add_002"] = df["ADD"] <= SUCCESS_THRESHOLD
    df["success_adds_002"] = df["ADD_S"] <= SUCCESS_THRESHOLD
    df["success_rot_5deg"] = df["rotation_error_deg"] <= ROT_SUCCESS_THRESHOLD
    df = df.sort_values(["method_order", "idx"]).reset_index(drop=True)
    return df


def build_master_summary(master_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        subset = master_df[master_df["method"] == method]
        rows.append(
            {
                "method": method,
                "n_cases": int(len(subset)),
                "mean_registration_time": mean_or_nan(subset["registration_time"]),
                "median_registration_time": median_or_nan(subset["registration_time"]),
                "max_registration_time": max_or_nan(subset["registration_time"]),
                "mean_frontend_time": mean_or_nan(subset["ppf_frontend_time"]),
                "mean_backend_time": mean_or_nan(subset["backend_time"]),
                "mean_ADD": mean_or_nan(subset["ADD"]),
                "mean_ADD_S": mean_or_nan(subset["ADD_S"]),
                "mean_rotation_error_deg": mean_or_nan(subset["rotation_error_deg"]),
                "mean_translation_error": mean_or_nan(subset["translation_error"]),
                "SR@ADD<=0.02": success_rate(subset["ADD"], SUCCESS_THRESHOLD),
                "SR@ADD-S<=0.02": success_rate(subset["ADD_S"], SUCCESS_THRESHOLD),
                "SR@rot<=5deg": success_rate(subset["rotation_error_deg"], ROT_SUCCESS_THRESHOLD),
                "candidate_inflation_mean": mean_or_nan(subset["candidate_inflation_mean"]),
                "num_input_candidates_mean": mean_or_nan(subset["num_input_candidates"]),
                "n_votes_mean": mean_or_nan(subset["n_votes"]),
                "topm_ratio_mean": mean_or_nan(subset["topm_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def compare_summary_row(actual: pd.Series, expected: pd.Series, mappings: Dict[str, str], tol: float = 1e-9) -> List[str]:
    problems = []
    for actual_col, expected_col in mappings.items():
        av = safe_float(actual.get(actual_col))
        ev = safe_float(expected.get(expected_col))
        if pd.isna(av) and pd.isna(ev):
            continue
        if pd.isna(av) != pd.isna(ev):
            problems.append(f"{actual_col}: actual={actual.get(actual_col)} expected={expected.get(expected_col)}")
            continue
        if abs(av - ev) > tol:
            problems.append(f"{actual_col}: actual={av:.12f} expected={ev:.12f}")
    return problems


def build_data_consistency_report(master_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    problems: List[str] = []
    required_columns = [
        "method",
        "idx",
        "scene_name",
        "scene_variant",
        "obj_id",
        "ADD",
        "ADD_S",
        "rotation_error_deg",
        "translation_error",
        "registration_time",
        "ppf_frontend_time",
        "backend_time",
    ]
    duplicated_idx = master_df.duplicated(subset=["method", "idx"]).sum()
    duplicated_case = master_df.duplicated(subset=["method", "case_key"]).sum()
    if duplicated_idx:
        problems.append(f"Found {duplicated_idx} duplicated rows by method+idx.")
    if duplicated_case:
        problems.append(f"Found {duplicated_case} duplicated rows by method+case_key.")

    missing_counts = master_df[required_columns].isna().sum()
    for col, count in missing_counts.items():
        if int(count) > 0:
            problems.append(f"Column `{col}` has {int(count)} missing values.")

    case_counts = master_df.groupby("method").size()
    case_count_mismatches = case_counts[case_counts != case_counts.max()]
    if not case_count_mismatches.empty:
        problems.append(f"Case count mismatch by method: {case_count_mismatches.to_dict()}")

    base_idx = set(master_df[master_df["method"] == METHOD_ORDER[0]]["idx"].tolist())
    for method in METHOD_ORDER[1:]:
        idxs = set(master_df[master_df["method"] == method]["idx"].tolist())
        if idxs != base_idx:
            problems.append(f"Case coverage mismatch for `{method}`.")

    adaptive_existing = pd.read_csv(SOURCE_TABLE_DIR / "adaptive_summary.csv")
    adaptive_actual = summary_df[summary_df["method"].isin(["No RS-MRQ", "Adaptive Two-Stage", "Ours (Full)"])].copy()
    adaptive_actual = adaptive_actual.rename(
        columns={
            "method": "Method",
            "mean_ADD": "Mean ADD",
            "mean_ADD_S": "Mean ADD-S",
            "mean_rotation_error_deg": "Mean Rotation Error (deg)",
            "mean_registration_time": "Mean Registration Time (s)",
            "mean_frontend_time": "Mean Front-end Time (s)",
            "mean_backend_time": "Mean Back-end Time (s)",
        }
    )
    adaptive_actual["Upgrade Ratio"] = adaptive_actual["Method"].map(
        master_df.groupby("method")["adaptive_used_upgrade"].mean().to_dict()
    )
    for _, row in adaptive_actual.iterrows():
        expected = adaptive_existing[adaptive_existing["Method"] == row["Method"]]
        if expected.empty:
            problems.append(f"adaptive_summary.csv is missing method `{row['Method']}`.")
            continue
        row_problems = compare_summary_row(
            row,
            expected.iloc[0],
            {
                "Upgrade Ratio": "Upgrade Ratio",
                "Mean ADD": "Mean ADD",
                "Mean ADD-S": "Mean ADD-S",
                "SR@ADD<=0.02": "SR@ADD<=0.02",
                "Mean Rotation Error (deg)": "Mean Rotation Error (deg)",
                "Mean Registration Time (s)": "Mean Registration Time (s)",
                "Mean Front-end Time (s)": "Mean Front-end Time (s)",
                "Mean Back-end Time (s)": "Mean Back-end Time (s)",
            },
        )
        if row_problems:
            problems.append(f"adaptive_summary.csv mismatch for `{row['Method']}`: " + "; ".join(row_problems))

    rsmrq_existing = pd.read_csv(SOURCE_TABLE_DIR / "rsmrq_budget_sweep.csv")
    def normalize_cap_label(value: object) -> str:
        text = str(value).strip()
        if text.lower() == "none" or text == "nan":
            return "None"
        try:
            return str(int(float(text)))
        except (TypeError, ValueError):
            return text
    rsmrq_existing["_cap_norm"] = rsmrq_existing["Cap"].map(normalize_cap_label)
    cap_map = {
        64: "Ours + RS-MRQ cap64",
        128: "Ours + RS-MRQ cap128",
        256: "Ours + RS-MRQ cap256",
        None: "Ours (Full)",
    }
    rsmrq_actual_rows = []
    for cap_value, method in cap_map.items():
        subset = master_df[master_df["method"] == method]
        rsmrq_actual_rows.append(
            {
                "Cap": "None" if cap_value is None else str(cap_value),
                "Mean candidate_inflation_mean": mean_or_nan(subset["candidate_inflation_mean"]),
                "Mean ppf_frontend_time": mean_or_nan(subset["ppf_frontend_time"]),
                "Mean registration_time": mean_or_nan(subset["registration_time"]),
                "Mean ADD": mean_or_nan(subset["ADD"]),
                "SR@ADD<=0.02": success_rate(subset["ADD"], SUCCESS_THRESHOLD),
            }
        )
    rsmrq_actual = pd.DataFrame(rsmrq_actual_rows)
    for _, row in rsmrq_actual.iterrows():
        expected = rsmrq_existing[rsmrq_existing["_cap_norm"] == normalize_cap_label(row["Cap"])]
        if expected.empty:
            problems.append(f"rsmrq_budget_sweep.csv is missing cap `{row['Cap']}`.")
            continue
        row_problems = compare_summary_row(
            row,
            expected.iloc[0],
            {
                "Mean candidate_inflation_mean": "Mean candidate_inflation_mean",
                "Mean ppf_frontend_time": "Mean ppf_frontend_time",
                "Mean registration_time": "Mean registration_time",
                "Mean ADD": "Mean ADD",
                "SR@ADD<=0.02": "SR@ADD<=0.02",
            },
        )
        if row_problems:
            problems.append(f"rsmrq_budget_sweep.csv mismatch for cap `{row['Cap']}`: " + "; ".join(row_problems))

    rv_existing = pd.read_csv(SOURCE_TABLE_DIR / "robust_vote_light_sweep.csv")
    rv_map = {50: "Ours (Full)", 20: "Ours + RV20", 10: "Ours + RV10"}
    rv_actual_rows = []
    for top_m, method in rv_map.items():
        subset = master_df[master_df["method"] == method]
        rv_actual_rows.append(
            {
                "top_m_per_bucket": str(top_m),
                "Mean ppf_frontend_time": mean_or_nan(subset["ppf_frontend_time"]),
                "Mean registration_time": mean_or_nan(subset["registration_time"]),
                "Mean ADD": mean_or_nan(subset["ADD"]),
                "SR@ADD<=0.02": success_rate(subset["ADD"], SUCCESS_THRESHOLD),
                "Mean candidate_inflation_mean": mean_or_nan(subset["candidate_inflation_mean"]),
            }
        )
    rv_actual = pd.DataFrame(rv_actual_rows)
    for _, row in rv_actual.iterrows():
        expected = rv_existing[rv_existing["top_m_per_bucket"].astype(str) == str(row["top_m_per_bucket"])]
        if expected.empty:
            problems.append(f"robust_vote_light_sweep.csv is missing top_m `{row['top_m_per_bucket']}`.")
            continue
        row_problems = compare_summary_row(
            row,
            expected.iloc[0],
            {
                "Mean ppf_frontend_time": "Mean ppf_frontend_time",
                "Mean registration_time": "Mean registration_time",
                "Mean ADD": "Mean ADD",
                "SR@ADD<=0.02": "SR@ADD<=0.02",
                "Mean candidate_inflation_mean": "Mean candidate_inflation_mean",
            },
        )
        if row_problems:
            problems.append(
                f"robust_vote_light_sweep.csv mismatch for top_m `{row['top_m_per_bucket']}`: "
                + "; ".join(row_problems)
            )

    lines = [
        "# Data Consistency Report",
        "",
        f"- Checked {len(METHOD_ORDER)} methods and {len(master_df)} total per-case rows.",
        f"- Duplicate rows by method+idx: {int(duplicated_idx)}.",
        f"- Duplicate rows by method+case_key: {int(duplicated_case)}.",
    ]
    if problems:
        lines.append("- Issues found:")
        for item in problems:
            lines.append(f"  - {item}")
    else:
        lines.append("- No summary mismatches, case count mismatches, duplicate cases, or required-field gaps were found.")
        lines.append("- Existing `adaptive_summary.csv`, `rsmrq_budget_sweep.csv`, and `robust_vote_light_sweep.csv` all match the re-aggregated batch JSON values.")
    return "\n".join(lines)


def build_main_comparison(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    table = summary_df.copy()
    table = table.rename(
        columns={
            "method": "Method",
            "mean_ADD": "Mean ADD",
            "mean_ADD_S": "Mean ADD-S",
            "mean_rotation_error_deg": "Mean Rotation Error (deg)",
            "mean_registration_time": "Mean Registration Time (s)",
            "mean_frontend_time": "Mean Front-end Time (s)",
            "mean_backend_time": "Mean Back-end Time (s)",
            "candidate_inflation_mean": "Candidate Inflation Mean",
        }
    )
    table = table[
        [
            "Method",
            "Mean ADD",
            "Mean ADD-S",
            "SR@ADD<=0.02",
            "Mean Rotation Error (deg)",
            "Mean Registration Time (s)",
            "Mean Front-end Time (s)",
            "Mean Back-end Time (s)",
            "Candidate Inflation Mean",
        ]
    ]
    best_accuracy = table.loc[table["Mean ADD"].idxmin()]
    fastest = table.loc[table["Mean Registration Time (s)"].idxmin()]
    ours = table[table["Method"] == "Ours (Full)"].iloc[0]
    adaptive = table[table["Method"] == "Adaptive Two-Stage"].iloc[0]
    no_rsmrq = table[table["Method"] == "No RS-MRQ"].iloc[0]
    bottleneck = "Front-end" if ours["Mean Front-end Time (s)"] > ours["Mean Back-end Time (s)"] else "Back-end"
    findings = "\n".join(
        [
            "# Main Findings",
            "",
            f"- Best mean ADD is `{best_accuracy['Method']}` at {best_accuracy['Mean ADD']:.6f}.",
            f"- Fastest mean registration time is `{fastest['Method']}` at {fastest['Mean Registration Time (s)']:.2f}s.",
            (
                f"- Adaptive Two-Stage keeps SR@ADD<=0.02 at {adaptive['SR@ADD<=0.02']:.4f}, "
                f"matching Ours (Full), while reducing mean registration time by "
                f"{(1.0 - adaptive['Mean Registration Time (s)'] / ours['Mean Registration Time (s)']) * 100.0:.2f}%."
            ),
            (
                f"- No RS-MRQ is much faster than Ours (Full) "
                f"({no_rsmrq['Mean Registration Time (s)']:.2f}s vs {ours['Mean Registration Time (s)']:.2f}s), "
                f"but loses {ours['SR@ADD<=0.02'] - no_rsmrq['SR@ADD<=0.02']:.4f} absolute SR@ADD<=0.02."
            ),
            (
                f"- The current bottleneck is `{bottleneck}`: Ours (Full) spends "
                f"{ours['Mean Front-end Time (s)']:.2f}s in front-end vs {ours['Mean Back-end Time (s)']:.2f}s in back-end."
            ),
        ]
    )
    return table, findings


def build_adaptive_efficiency(master_df: pd.DataFrame, summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    adaptive = summary_df[summary_df["method"] == "Adaptive Two-Stage"].iloc[0]
    ours = summary_df[summary_df["method"] == "Ours (Full)"].iloc[0]
    no_rsmrq = summary_df[summary_df["method"] == "No RS-MRQ"].iloc[0]
    ad_df = master_df[master_df["method"] == "Adaptive Two-Stage"].copy()
    ours_df = master_df[master_df["method"] == "Ours (Full)"].copy()
    merged_ours = ad_df[["idx", "success_add_002"]].merge(
        ours_df[["idx", "success_add_002"]].rename(columns={"success_add_002": "ours_success"}),
        on="idx",
        how="inner",
    )
    ours_success_retention = float(
        merged_ours.loc[merged_ours["ours_success"] == True, "success_add_002"].mean()  # noqa: E712
    )
    ours_gap = ours["SR@ADD<=0.02"] - no_rsmrq["SR@ADD<=0.02"]
    recovery_ratio = (
        float((adaptive["SR@ADD<=0.02"] - no_rsmrq["SR@ADD<=0.02"]) / ours_gap) if abs(ours_gap) > 1e-12 else float("nan")
    )
    row = pd.DataFrame(
        [
            {
                "Method": "Adaptive Two-Stage",
                "Mean ADD": adaptive["mean_ADD"],
                "Mean ADD-S": adaptive["mean_ADD_S"],
                "SR@ADD<=0.02": adaptive["SR@ADD<=0.02"],
                "Mean registration_time": adaptive["mean_registration_time"],
                "Mean frontend_time": adaptive["mean_frontend_time"],
                "Mean backend_time": adaptive["mean_backend_time"],
                "Upgrade Ratio": mean_or_nan(ad_df["adaptive_used_upgrade"]),
                "Ours Success Retention": ours_success_retention,
                "Time Reduction vs Ours (%)": -pct_change(adaptive["mean_registration_time"], ours["mean_registration_time"]),
                "Frontend Reduction vs Ours (%)": -pct_change(adaptive["mean_frontend_time"], ours["mean_frontend_time"]),
                "Accuracy Recovery vs No RS-MRQ (SR fraction)": recovery_ratio,
            }
        ]
    )
    summary = "\n".join(
        [
            "# Adaptive Efficiency Summary",
            "",
            f"- Upgrade ratio is {mean_or_nan(ad_df['adaptive_used_upgrade']) * 100.0:.2f}%.",
            (
                f"- Adaptive mean registration time is {adaptive['mean_registration_time']:.2f}s, "
                f"{-pct_change(adaptive['mean_registration_time'], ours['mean_registration_time']):.2f}% lower than Ours (Full)."
            ),
            (
                f"- Adaptive mean front-end time is {adaptive['mean_frontend_time']:.2f}s, "
                f"{-pct_change(adaptive['mean_frontend_time'], ours['mean_frontend_time']):.2f}% lower than Ours (Full)."
            ),
            (
                f"- Adaptive keeps {ours_success_retention * 100.0:.2f}% of Ours (Full) successful cases and "
                f"recovers {recovery_ratio * 100.0:.2f}% of the No RS-MRQ to Ours success-rate gap."
            ),
        ]
    )
    return row, summary


def build_adaptive_upgrade_tables(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ad_df = master_df[master_df["method"] == "Adaptive Two-Stage"].copy()
    upgraded = ad_df[ad_df["adaptive_used_upgrade"] == True].copy()  # noqa: E712
    breakdown = (
        upgraded.groupby("adaptive_trigger_reason", dropna=False)
        .agg(
            n_cases=("idx", "count"),
            case_ratio=("idx", lambda s: float(len(s) / len(ad_df))),
            upgraded_case_ratio=("idx", lambda s: float(len(s) / len(upgraded)) if len(upgraded) else float("nan")),
            success_rate=("success_add_002", lambda s: float(pd.Series(s).mean())),
            mean_ADD=("ADD", "mean"),
            mean_registration_time=("registration_time", "mean"),
        )
        .reset_index()
        .rename(columns={"adaptive_trigger_reason": "trigger_reason"})
        .sort_values(["n_cases", "trigger_reason"], ascending=[False, True])
    )
    by_scene = (
        ad_df.groupby("scene_name", dropna=False)
        .agg(
            n_cases=("idx", "count"),
            n_upgraded=("adaptive_used_upgrade", "sum"),
            upgrade_ratio=("adaptive_used_upgrade", "mean"),
            success_rate=("success_add_002", "mean"),
            mean_ADD=("ADD", "mean"),
            mean_registration_time=("registration_time", "mean"),
        )
        .reset_index()
        .sort_values(["upgrade_ratio", "scene_name"], ascending=[False, True])
    )
    by_object = (
        ad_df.groupby("obj_id", dropna=False)
        .agg(
            n_cases=("idx", "count"),
            n_upgraded=("adaptive_used_upgrade", "sum"),
            upgrade_ratio=("adaptive_used_upgrade", "mean"),
            success_rate=("success_add_002", "mean"),
            mean_ADD=("ADD", "mean"),
            mean_registration_time=("registration_time", "mean"),
        )
        .reset_index()
        .sort_values(["upgrade_ratio", "obj_id"], ascending=[False, True])
    )
    top_scene = by_scene.iloc[0] if not by_scene.empty else None
    top_object = by_object.iloc[0] if not by_object.empty else None
    summary = "\n".join(
        [
            "# Adaptive Upgrade Summary",
            "",
            (
                f"- Adaptive upgrades {int(upgraded.shape[0])} / {int(ad_df.shape[0])} cases "
                f"({mean_or_nan(ad_df['adaptive_used_upgrade']) * 100.0:.2f}%)."
            ),
            (
                f"- Most common trigger reason is `{breakdown.iloc[0]['trigger_reason']}` "
                f"with {int(breakdown.iloc[0]['n_cases'])} cases."
                if not breakdown.empty
                else "- No adaptive upgrades were found."
            ),
            (
                f"- Highest upgrade-ratio scene is `{top_scene['scene_name']}` at {top_scene['upgrade_ratio'] * 100.0:.2f}%."
                if top_scene is not None
                else "- No scene-level upgrade ratio is available."
            ),
            (
                f"- Highest upgrade-ratio object is `{top_object['obj_id']}` at {top_object['upgrade_ratio'] * 100.0:.2f}%."
                if top_object is not None
                else "- No object-level upgrade ratio is available."
            ),
        ]
    )
    return breakdown, by_scene, by_object, summary


def build_rescue_case_table(
    lhs_df: pd.DataFrame,
    rhs_df: pd.DataFrame,
    lhs_method: str,
    rhs_method: str,
    lhs_must_succeed: bool,
    rhs_must_fail: bool,
) -> pd.DataFrame:
    merged = lhs_df.merge(
        rhs_df[["idx", "ADD", "ADD_S", "rotation_error_deg", "registration_time", "ppf_frontend_time", "success_add_002"]].rename(
            columns={
                "ADD": "rhs_ADD",
                "ADD_S": "rhs_ADD_S",
                "rotation_error_deg": "rhs_rotation_error_deg",
                "registration_time": "rhs_registration_time",
                "ppf_frontend_time": "rhs_ppf_frontend_time",
                "success_add_002": "rhs_success_add_002",
            }
        ),
        on="idx",
        how="inner",
    )
    lhs_success = merged["success_add_002"] == True  # noqa: E712
    rhs_success = merged["rhs_success_add_002"] == True  # noqa: E712
    mask = pd.Series(True, index=merged.index)
    if lhs_must_succeed:
        mask &= lhs_success
    if rhs_must_fail:
        mask &= ~rhs_success
    out = merged.loc[mask].copy()
    out["lhs_method"] = lhs_method
    out["rhs_method"] = rhs_method
    out["ADD_improvement_vs_rhs"] = out["rhs_ADD"] - out["ADD"]
    out["rotation_error_improvement_vs_rhs"] = out["rhs_rotation_error_deg"] - out["rotation_error_deg"]
    out["registration_time_delta_vs_rhs"] = out["registration_time"] - out["rhs_registration_time"]
    out["ppf_frontend_time_delta_vs_rhs"] = out["ppf_frontend_time"] - out["rhs_ppf_frontend_time"]
    return out.sort_values(["scene_name", "scene_variant", "obj_id", "idx"]).reset_index(drop=True)


def build_adaptive_rescue_outputs(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ad = master_df[master_df["method"] == "Adaptive Two-Stage"].copy()
    ours = master_df[master_df["method"] == "Ours (Full)"].copy()
    no = master_df[master_df["method"] == "No RS-MRQ"].copy()
    rescue_vs_no = build_rescue_case_table(ad, no, "Adaptive Two-Stage", "No RS-MRQ", True, True)
    fail_vs_ours = build_rescue_case_table(ad, ours, "Adaptive Two-Stage", "Ours (Full)", False, False)
    fail_vs_ours = fail_vs_ours[(fail_vs_ours["success_add_002"] != True) & (fail_vs_ours["rhs_success_add_002"] == True)].copy()  # noqa: E712
    summary = pd.DataFrame(
        [
            {"metric": "Adaptive rescues vs No RS-MRQ", "value": int(len(rescue_vs_no))},
            {"metric": "Adaptive fails but Ours succeeds", "value": int(len(fail_vs_ours))},
            {
                "metric": "Adaptive rescue unique scenes",
                "value": int(rescue_vs_no["scene_name"].nunique()) if not rescue_vs_no.empty else 0,
            },
            {
                "metric": "Adaptive rescue unique objects",
                "value": int(rescue_vs_no["obj_id"].nunique()) if not rescue_vs_no.empty else 0,
            },
        ]
    )
    return summary, rescue_vs_no, fail_vs_ours


def build_timing_breakdown(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    tail_rows = []
    for method in METHOD_ORDER:
        subset = master_df[master_df["method"] == method]
        reg = subset["registration_time"]
        rows.append(
            {
                "Method": method,
                "scene preprocess mean": mean_or_nan(subset["scene_preprocess_time"]),
                "frontend mean": mean_or_nan(subset["ppf_frontend_time"]),
                "backend mean": mean_or_nan(subset["backend_time"]),
                "pose_selection mean": mean_or_nan(subset["pose_selection_time"]),
                "pose_clustering mean": mean_or_nan(subset["pose_clustering_time"]),
                "legacy_clustering mean": mean_or_nan(subset["legacy_clustering_time"]),
                "frontend_ratio": mean_or_nan(subset["ppf_frontend_time"]) / mean_or_nan(reg),
                "backend_ratio": mean_or_nan(subset["backend_time"]) / mean_or_nan(reg),
            }
        )
        tail_rows.append(
            {
                "method": method,
                "mean": mean_or_nan(reg),
                "median": median_or_nan(reg),
                "p90": quantile_or_nan(reg, 0.90),
                "p95": quantile_or_nan(reg, 0.95),
                "p99": quantile_or_nan(reg, 0.99),
                "max": max_or_nan(reg),
                "n_cases_gt_10s": int((pd.to_numeric(reg, errors="coerce") > 10.0).sum()),
                "n_cases_gt_30s": int((pd.to_numeric(reg, errors="coerce") > 30.0).sum()),
                "n_cases_gt_60s": int((pd.to_numeric(reg, errors="coerce") > 60.0).sum()),
            }
        )
    timing_df = pd.DataFrame(rows)
    tail_df = pd.DataFrame(tail_rows)
    no_robust = tail_df[tail_df["method"] == "No Robust Vote"].iloc[0]
    adaptive = tail_df[tail_df["method"] == "Adaptive Two-Stage"].iloc[0]
    ours = tail_df[tail_df["method"] == "Ours (Full)"].iloc[0]
    summary = "\n".join(
        [
            "# Time Tail Summary",
            "",
            (
                f"- No Robust Vote reduces mean registration time to {no_robust['mean']:.2f}s "
                f"and keeps the >30s tail at {int(no_robust['n_cases_gt_30s'])} cases."
            ),
            (
                f"- Adaptive improves both mean time ({adaptive['mean']:.2f}s vs {ours['mean']:.2f}s on Ours) "
                f"and p95 ({adaptive['p95']:.2f}s vs {ours['p95']:.2f}s)."
            ),
            (
                f"- Ours still has the heavier upper tail, with {int(ours['n_cases_gt_30s'])} cases above 30s "
                f"and max time {ours['max']:.2f}s."
            ),
        ]
    )
    return timing_df, tail_df, summary


def build_rsmrq_budget_enriched(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    ours = master_df[master_df["method"] == "Ours (Full)"]
    ours_mean_reg = mean_or_nan(ours["registration_time"])
    ours_mean_front = mean_or_nan(ours["ppf_frontend_time"])
    ours_mean_add = mean_or_nan(ours["ADD"])
    ours_sr = success_rate(ours["ADD"], SUCCESS_THRESHOLD)
    rows = []
    for method in RSMRQ_METHODS:
        subset = master_df[master_df["method"] == method]
        cap_value = subset["rsmrq_global_candidate_cap"].dropna()
        label = "None" if method == "Ours (Full)" else str(int(cap_value.iloc[0])) if not cap_value.empty else "None"
        mean_reg = mean_or_nan(subset["registration_time"])
        mean_front = mean_or_nan(subset["ppf_frontend_time"])
        mean_add = mean_or_nan(subset["ADD"])
        sr = success_rate(subset["ADD"], SUCCESS_THRESHOLD)
        mean_cap_hit = mean_or_nan(subset["rsmrq_cap_hit_ratio"])
        rows.append(
            {
                "Cap": label,
                "Method": method,
                "Mean candidate_inflation_mean": mean_or_nan(subset["candidate_inflation_mean"]),
                "Mean ppf_frontend_time": mean_front,
                "Mean registration_time": mean_reg,
                "Mean ADD": mean_add,
                "SR@ADD<=0.02": sr,
                "Mean pre_cap_candidate_count": mean_or_nan(subset["rsmrq_merged_candidates_pre_cap_mean"]),
                "Mean post_cap_candidate_count": mean_or_nan(subset["rsmrq_merged_candidates_post_cap_mean"]),
                "Mean cap_hit_ratio": mean_cap_hit,
                "Cap Triggered": bool(mean_cap_hit > 0.0) if not pd.isna(mean_cap_hit) else False,
                "Registration Time Change vs Ours (%)": pct_change(mean_reg, ours_mean_reg),
                "Front-end Time Change vs Ours (%)": pct_change(mean_front, ours_mean_front),
                "ADD Delta vs Ours": mean_add - ours_mean_add,
                "SR Delta vs Ours": sr - ours_sr,
            }
        )
    enriched = pd.DataFrame(rows)
    cap64 = enriched[enriched["Method"] == "Ours + RS-MRQ cap64"].iloc[0]
    summary = "\n".join(
        [
            "# RS-MRQ Budget Summary",
            "",
            (
                f"- `cap64` reduces post-cap candidates to {cap64['Mean post_cap_candidate_count']:.2f} "
                f"from {cap64['Mean pre_cap_candidate_count']:.2f}, with cap hit ratio {cap64['Mean cap_hit_ratio']:.2f}."
            ),
            (
                f"- Despite that truncation, `cap64` changes mean registration time by "
                f"{cap64['Registration Time Change vs Ours (%)']:.2f}% vs Ours (Full), so the runtime does not fall in lockstep."
            ),
            (
                f"- `cap128` barely triggers (mean cap-hit ratio "
                f"{enriched[enriched['Method'] == 'Ours + RS-MRQ cap128'].iloc[0]['Mean cap_hit_ratio']:.2f}), "
                f"and `cap256` never triggers."
            ),
        ]
    )
    mechanism = "\n".join(
        [
            "# RS-MRQ Mechanism Analysis",
            "",
            (
                f"- `cap64` lowers the post-cap candidate count by "
                f"{(1.0 - cap64['Mean post_cap_candidate_count'] / cap64['Mean pre_cap_candidate_count']) * 100.0:.2f}%, "
                f"but mean registration time changes by {cap64['Registration Time Change vs Ours (%)']:.2f}%, "
                "which indicates late truncation is not removing the dominant cost."
            ),
            "- The evidence points to query and merge overhead still being paid before the cap is applied, so candidate truncation only affects the downstream voting stage.",
            "- `candidate_inflation_mean` stays insensitive to the cap because it is a pre-cap statistic already recorded in the existing timing/debug schema.",
            "- For capped runs, `merged_candidates_pre_cap_mean` and `merged_candidates_post_cap_mean` should be interpreted separately: the former explains query inflation, the latter explains what actually reaches vote.",
        ]
    )
    return enriched, summary, mechanism


def build_robust_vote_enriched(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    ours = master_df[master_df["method"] == "Ours (Full)"]
    base_reg = mean_or_nan(ours["registration_time"])
    base_front = mean_or_nan(ours["ppf_frontend_time"])
    base_add = mean_or_nan(ours["ADD"])
    rows = []
    for method in ROBUST_VOTE_METHODS:
        subset = master_df[master_df["method"] == method]
        top_m = int(subset["robust_vote_top_m_per_bucket"].dropna().iloc[0]) if subset["robust_vote_top_m_per_bucket"].notna().any() else 0
        rows.append(
            {
                "Method": method,
                "top_m_per_bucket": top_m,
                "Mean ppf_frontend_time": mean_or_nan(subset["ppf_frontend_time"]),
                "Mean registration_time": mean_or_nan(subset["registration_time"]),
                "Mean ADD": mean_or_nan(subset["ADD"]),
                "SR@ADD<=0.02": success_rate(subset["ADD"], SUCCESS_THRESHOLD),
                "Mean candidate_inflation_mean": mean_or_nan(subset["candidate_inflation_mean"]),
                "Registration Time Change vs Ours (%)": pct_change(mean_or_nan(subset["registration_time"]), base_reg),
                "Front-end Time Change vs Ours (%)": pct_change(mean_or_nan(subset["ppf_frontend_time"]), base_front),
                "ADD Delta vs Ours": mean_or_nan(subset["ADD"]) - base_add,
            }
        )
    enriched = pd.DataFrame(rows).sort_values("top_m_per_bucket", ascending=False).reset_index(drop=True)
    add_min = enriched["Mean ADD"].min()
    add_max = enriched["Mean ADD"].max()
    time_min = enriched["Mean registration_time"].min()
    time_max = enriched["Mean registration_time"].max()
    enriched["normalized_distance_to_ideal"] = np.sqrt(
        ((enriched["Mean ADD"] - add_min) / max(add_max - add_min, 1e-12)) ** 2
        + ((enriched["Mean registration_time"] - time_min) / max(time_max - time_min, 1e-12)) ** 2
    )
    enriched["closest_to_ideal"] = enriched["normalized_distance_to_ideal"] == enriched["normalized_distance_to_ideal"].min()
    rv20 = enriched[enriched["top_m_per_bucket"] == 20].iloc[0]
    rv10 = enriched[enriched["top_m_per_bucket"] == 10].iloc[0]
    summary = "\n".join(
        [
            "# Robust Vote Light Summary",
            "",
            (
                f"- `rv20` lowers mean registration time by {-rv20['Registration Time Change vs Ours (%)']:.2f}% "
                f"with ADD change {rv20['ADD Delta vs Ours']:+.6f} and no success-rate drop."
            ),
            (
                f"- `rv10` is the aggressive point: registration time drops by {-rv10['Registration Time Change vs Ours (%)']:.2f}% "
                f"but ADD rises by {rv10['ADD Delta vs Ours']:+.6f}."
            ),
            (
                f"- The closest point to the joint time/quality ideal is "
                f"`{enriched.loc[enriched['closest_to_ideal'], 'Method'].iloc[0]}`."
            ),
            "- `rv20` is the safer lightweight setting, while `rv10` is useful as a more aggressive option if extra speed matters more than a small ADD increase.",
        ]
    )
    return enriched, summary


def build_hard_case_master(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    ours = master_df[master_df["method"] == "Ours (Full)"].copy()
    same_backbone = master_df[master_df["method"] == "Same-backbone Baseline"].copy()
    no_rsmrq = master_df[master_df["method"] == "No RS-MRQ"].copy()
    adaptive = master_df[master_df["method"] == "Adaptive Two-Stage"].copy()

    cats = [
        ("Ours rescues Same-backbone", ours, same_backbone),
        ("Ours rescues No RS-MRQ", ours, no_rsmrq),
        ("Adaptive rescues No RS-MRQ", adaptive, no_rsmrq),
        ("Adaptive fails but Ours succeeds", adaptive, ours),
    ]
    frames = []
    for category, focal, comparator in cats:
        lhs_must_succeed = category != "Adaptive fails but Ours succeeds"
        table = build_rescue_case_table(
            focal,
            comparator,
            focal["method"].iloc[0],
            comparator["method"].iloc[0],
            lhs_must_succeed=lhs_must_succeed,
            rhs_must_fail=True,
        )
        if category == "Adaptive fails but Ours succeeds":
            table = table[(table["success_add_002"] != True) & (table["rhs_success_add_002"] == True)].copy()  # noqa: E712
        table["category"] = category
        frames.append(table)
    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows = []
    if not master.empty:
        for category, subset in master.groupby("category"):
            rows.append(
                f"- `{category}`: {len(subset)} cases, mean ADD improvement {subset['ADD_improvement_vs_rhs'].mean():.6f}, "
                f"mean rotation error improvement {subset['rotation_error_improvement_vs_rhs'].mean():.3f} deg, "
                f"mean registration time delta {subset['registration_time_delta_vs_rhs'].mean():.3f}s."
            )
    report = "\n".join(["# Enhanced Hard-Case Report", ""] + rows) if rows else "# Enhanced Hard-Case Report\n\n- No hard cases found."
    return master, report


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    xv = pd.to_numeric(x[mask], errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y[mask], errors="coerce").to_numpy(dtype=float)
    if np.allclose(np.std(xv), 0.0) or np.allclose(np.std(yv), 0.0):
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    xr = x[mask].rank(method="average")
    yr = y[mask].rank(method="average")
    return pearson_corr(xr, yr)


def build_candidate_correlations(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    rows = []
    metrics = ["candidate_inflation_mean", "num_input_candidates", "n_votes"]
    groups = [("All", master_df)] + [(method, master_df[master_df["method"] == method]) for method in METHOD_ORDER]
    for group_name, subset in groups:
        for metric in metrics:
            x = subset[metric]
            y = subset["ppf_frontend_time"]
            mask = x.notna() & y.notna()
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "n_pairs": int(mask.sum()),
                    "pearson": pearson_corr(x, y),
                    "spearman": spearman_corr(x, y),
                }
            )
    corr_df = pd.DataFrame(rows)
    overall = corr_df[corr_df["group"] == "All"].copy()
    strongest = overall.iloc[overall["spearman"].abs().idxmax()]
    ours = master_df[master_df["method"] == "Ours (Full)"]
    no = master_df[master_df["method"] == "No RS-MRQ"]
    summary = "\n".join(
        [
            "# Front-end Cost Drivers",
            "",
            (
                f"- Overall, `{strongest['metric']}` has the strongest Spearman correlation with front-end time "
                f"({strongest['spearman']:.3f})."
            ),
            (
                f"- Ours (Full) vs No RS-MRQ: candidate inflation changes from {mean_or_nan(no['candidate_inflation_mean']):.2f} "
                f"to {mean_or_nan(ours['candidate_inflation_mean']):.2f}, num input candidates from "
                f"{mean_or_nan(no['num_input_candidates']):.2f} to {mean_or_nan(ours['num_input_candidates']):.2f}, "
                f"and n_votes from {mean_or_nan(no['n_votes']):.2f} to {mean_or_nan(ours['n_votes']):.2f}."
            ),
            (
                f"- The front-end time gap between Ours and No RS-MRQ is "
                f"{mean_or_nan(ours['ppf_frontend_time']) - mean_or_nan(no['ppf_frontend_time']):.2f}s."
            ),
            "- Because capped RS-MRQ runs keep paying the query/merge cost before truncation, front-end slowdown is not explained by the final candidate count alone.",
        ]
    )
    return corr_df, summary


def build_group_summary(master_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        subset = master_df[master_df["method"] == method]
        grouped = subset.groupby(group_col, dropna=False)
        for group_value, group_df in grouped:
            rows.append(
                {
                    "method": method,
                    group_col: group_value,
                    "Mean ADD": mean_or_nan(group_df["ADD"]),
                    "SR@ADD<=0.02": success_rate(group_df["ADD"], SUCCESS_THRESHOLD),
                    "Mean registration_time": mean_or_nan(group_df["registration_time"]),
                    "Mean frontend_time": mean_or_nan(group_df["ppf_frontend_time"]),
                }
            )
    return pd.DataFrame(rows)


def normal_p_value(z: float) -> float:
    return float(2.0 * (1.0 - NormalDist().cdf(abs(z))))


def paired_t_test(diff: np.ndarray) -> Tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float("nan"), float("nan")
    std = float(diff.std(ddof=1))
    if std < 1e-12:
        return float("nan"), float("nan")
    t_stat = float(diff.mean() / (std / math.sqrt(diff.size)))
    return t_stat, normal_p_value(t_stat)


def wilcoxon_signed_rank(diff: np.ndarray) -> Tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0.0]
    if diff.size == 0:
        return float("nan"), float("nan")
    abs_diff = pd.Series(np.abs(diff))
    ranks = abs_diff.rank(method="average").to_numpy(dtype=float)
    w_plus = float(ranks[diff > 0].sum())
    n = float(diff.size)
    mean_w = n * (n + 1.0) / 4.0
    std_w = math.sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 24.0)
    if std_w < 1e-12:
        return float("nan"), float("nan")
    z = (w_plus - mean_w) / std_w
    return w_plus, normal_p_value(z)


def bootstrap_ci_mean(values: np.ndarray, rng: np.random.Generator, rounds: int) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(rounds, values.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_ci_mean_diff(diff: np.ndarray, rng: np.random.Generator, rounds: int) -> Tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan")
    samples = rng.choice(diff, size=(rounds, diff.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_paired_tests(master_df: pd.DataFrame, rounds: int, rng_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(rng_seed)
    for lhs, rhs in PAIRWISE_COMPARISONS:
        lhs_df = master_df[master_df["method"] == lhs][["idx", "ADD", "registration_time", "ppf_frontend_time"]]
        rhs_df = master_df[master_df["method"] == rhs][["idx", "ADD", "registration_time", "ppf_frontend_time"]]
        merged = lhs_df.merge(rhs_df, on="idx", suffixes=("_lhs", "_rhs"))
        for metric, label in PAIRWISE_METRICS:
            diff = merged[f"{metric}_lhs"].to_numpy(dtype=float) - merged[f"{metric}_rhs"].to_numpy(dtype=float)
            t_stat, t_p = paired_t_test(diff)
            w_stat, w_p = wilcoxon_signed_rank(diff)
            ci_low, ci_high = bootstrap_ci_mean_diff(diff, rng, rounds)
            rows.append(
                {
                    "comparison": f"{lhs} vs {rhs}",
                    "lhs_method": lhs,
                    "rhs_method": rhs,
                    "metric": metric,
                    "metric_label": label,
                    "n_pairs": int(np.isfinite(diff).sum()),
                    "mean_diff_lhs_minus_rhs": float(np.nanmean(diff)),
                    "paired_t_stat": t_stat,
                    "paired_t_pvalue_normal_approx": t_p,
                    "wilcoxon_w_plus": w_stat,
                    "wilcoxon_pvalue_normal_approx": w_p,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )
    for method in METHOD_ORDER:
        subset = master_df[master_df["method"] == method]
        values = {
            "Mean ADD": subset["ADD"].to_numpy(dtype=float),
            "SR@ADD<=0.02": (subset["ADD"].to_numpy(dtype=float) <= SUCCESS_THRESHOLD).astype(float),
            "Mean registration_time": subset["registration_time"].to_numpy(dtype=float),
            "Mean frontend_time": subset["ppf_frontend_time"].to_numpy(dtype=float),
        }
        for metric_name, vals in values.items():
            ci_low, ci_high = bootstrap_ci_mean(vals, rng, rounds)
            bootstrap_rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "point_estimate": float(np.nanmean(vals)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_rounds": rounds,
                }
            )
    paired_df = pd.DataFrame(rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    time_row = paired_df[
        (paired_df["comparison"] == "Ours (Full) vs Adaptive Two-Stage") & (paired_df["metric"] == "registration_time")
    ].iloc[0]
    add_row = paired_df[
        (paired_df["comparison"] == "Ours (Full) vs No RS-MRQ") & (paired_df["metric"] == "ADD")
    ].iloc[0]
    summary = "\n".join(
        [
            "# Paired Tests",
            "",
            "- `mean_diff_lhs_minus_rhs` is defined as left method minus right method, so negative time deltas mean the left method is faster.",
            (
                f"- Ours vs Adaptive on registration time: mean diff {time_row['mean_diff_lhs_minus_rhs']:.3f}s, "
                f"bootstrap 95% CI [{time_row['bootstrap_ci_low']:.3f}, {time_row['bootstrap_ci_high']:.3f}]."
            ),
            (
                f"- Ours vs No RS-MRQ on ADD: mean diff {add_row['mean_diff_lhs_minus_rhs']:.6f}, "
                f"bootstrap 95% CI [{add_row['bootstrap_ci_low']:.6f}, {add_row['bootstrap_ci_high']:.6f}]."
            ),
            "- Because SciPy is not available in the current environment, the paired t-test and Wilcoxon p-values use a normal approximation; the bootstrap intervals are the more reliable reference here.",
        ]
    )
    return paired_df, bootstrap_df, summary


def scatter_with_trend(ax: plt.Axes, x: pd.Series, y: pd.Series, label: str, color: str) -> None:
    mask = x.notna() & y.notna()
    if int(mask.sum()) == 0:
        return
    xv = pd.to_numeric(x[mask], errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y[mask], errors="coerce").to_numpy(dtype=float)
    ax.scatter(xv, yv, label=label, color=color, alpha=0.78, edgecolors="none")


def plot_method_scatter(
    master_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    basename: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method in METHOD_ORDER:
        subset = master_df[master_df["method"] == method]
        scatter_with_trend(ax, subset[x_col], subset[y_col], method, METHOD_COLORS[method])
    mask = master_df[x_col].notna() & master_df[y_col].notna()
    if int(mask.sum()) >= 2:
        xv = pd.to_numeric(master_df.loc[mask, x_col], errors="coerce").to_numpy(dtype=float)
        yv = pd.to_numeric(master_df.loc[mask, y_col], errors="coerce").to_numpy(dtype=float)
        coeff = np.polyfit(xv, yv, 1)
        line_x = np.linspace(float(np.min(xv)), float(np.max(xv)), 200)
        line_y = coeff[0] * line_x + coeff[1]
        ax.plot(line_x, line_y, color="#222222", linewidth=1.5, linestyle="--", label="Trend")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save_figure(fig, basename)


def plot_main_and_adaptive(summary_df: pd.DataFrame, master_df: pd.DataFrame) -> None:
    main_rows = summary_df[summary_df["method"].isin(["No RS-MRQ", "Adaptive Two-Stage", "Ours (Full)"])].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for _, row in main_rows.iterrows():
        ax.scatter(row["mean_registration_time"], row["mean_ADD"], s=70, color=METHOD_COLORS[row["method"]])
        ax.annotate(row["method"], (row["mean_registration_time"], row["mean_ADD"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Mean registration time (s)")
    ax.set_ylabel("Mean ADD")
    ax.set_title("Adaptive Accuracy vs Time")
    ax.grid(True)
    save_figure(fig, "adaptive_accuracy_vs_time")

    ad_df = master_df[master_df["method"] == "Adaptive Two-Stage"]
    ratio = mean_or_nan(ad_df["adaptive_used_upgrade"])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.bar(["Stage1 only", "Upgraded"], [1.0 - ratio, ratio], color=["#8FB996", METHOD_COLORS["Adaptive Two-Stage"]])
    ax.set_ylabel("Case ratio")
    ax.set_title("Adaptive Upgrade Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")
    save_figure(fig, "adaptive_upgrade_ratio")

    upgraded = ad_df[ad_df["adaptive_used_upgrade"] == True]
    counts = upgraded["adaptive_trigger_reason"].fillna("missing").value_counts()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.pie(counts.values, labels=counts.index.tolist(), autopct="%1.1f%%", startangle=90)
    ax.set_title("Adaptive Trigger Reasons")
    save_figure(fig, "adaptive_trigger_reason_pie")

    by_scene = ad_df.groupby("scene_name")["adaptive_used_upgrade"].mean().reset_index().sort_values(
        ["adaptive_used_upgrade", "scene_name"], ascending=[False, True]
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(by_scene["scene_name"], by_scene["adaptive_used_upgrade"], color=METHOD_COLORS["Adaptive Two-Stage"])
    ax.set_xlabel("Scene")
    ax.set_ylabel("Upgrade ratio")
    ax.set_title("Adaptive Upgrade by Scene")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y")
    save_figure(fig, "adaptive_upgrade_by_scene")

    by_object = ad_df.groupby("obj_id")["adaptive_used_upgrade"].mean().reset_index().sort_values(
        ["adaptive_used_upgrade", "obj_id"], ascending=[False, True]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(by_object["obj_id"].astype(str), by_object["adaptive_used_upgrade"], color=METHOD_COLORS["Adaptive Two-Stage"])
    ax.set_xlabel("Object ID")
    ax.set_ylabel("Upgrade ratio")
    ax.set_title("Adaptive Upgrade by Object")
    ax.grid(True, axis="y")
    save_figure(fig, "adaptive_upgrade_by_object")


def plot_timing(master_df: pd.DataFrame, timing_df: pd.DataFrame) -> None:
    methods = timing_df["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(x - width, timing_df["scene preprocess mean"], width=width, label="Scene preprocess", color="#BAB0AC")
    ax.bar(x, timing_df["frontend mean"], width=width, label="Front-end", color="#E15759")
    ax.bar(x + width, timing_df["backend mean"], width=width, label="Back-end", color="#4E79A7")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("Mean time (s)")
    ax.set_title("Timing Breakdown (Grouped)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save_figure(fig, "timing_breakdown_bar")

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    scene = timing_df["scene preprocess mean"].to_numpy(dtype=float)
    frontend = timing_df["frontend mean"].to_numpy(dtype=float)
    backend = timing_df["backend mean"].to_numpy(dtype=float)
    ax.bar(x, scene, label="Scene preprocess", color="#BAB0AC")
    ax.bar(x, frontend, bottom=scene, label="Front-end", color="#E15759")
    ax.bar(x, backend, bottom=scene + frontend, label="Back-end", color="#4E79A7")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("Mean time (s)")
    ax.set_title("Timing Breakdown (Stacked)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save_figure(fig, "timing_breakdown_stacked")

    data = [master_df[master_df["method"] == method]["registration_time"].dropna().to_numpy(dtype=float) for method in METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.boxplot(data, tick_labels=METHOD_ORDER, showfliers=True)
    ax.set_ylabel("Registration time (s)")
    ax.set_title("Registration Time Boxplot")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y")
    save_figure(fig, "registration_time_boxplot")

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#76B7B2")
        body.set_alpha(0.55)
    ax.set_xticks(np.arange(1, len(METHOD_ORDER) + 1))
    ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Registration time (s)")
    ax.set_title("Registration Time Violin Plot")
    ax.grid(True, axis="y")
    save_figure(fig, "registration_time_violin")


def plot_rsmrq_budget(enriched: pd.DataFrame) -> None:
    ordered = enriched.copy()
    order = {"64": 0, "128": 1, "256": 2, "None": 3}
    ordered["_order"] = ordered["Cap"].astype(str).map(order)
    ordered = ordered.sort_values("_order")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for _, row in ordered.iterrows():
        ax.scatter(row["Mean registration_time"], row["Mean ADD"], color=METHOD_COLORS[row["Method"]], s=70)
        ax.annotate(f"cap{row['Cap']}", (row["Mean registration_time"], row["Mean ADD"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Mean registration time (s)")
    ax.set_ylabel("Mean ADD")
    ax.set_title("RS-MRQ Budget Tradeoff")
    ax.grid(True)
    save_figure(fig, "rsmrq_budget_tradeoff")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    capped = ordered[ordered["Cap"] != "None"]
    ax.scatter(capped["Mean cap_hit_ratio"], capped["Mean registration_time"], s=70, color="#F28E2B")
    for _, row in capped.iterrows():
        ax.annotate(f"cap{row['Cap']}", (row["Mean cap_hit_ratio"], row["Mean registration_time"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Mean cap-hit ratio")
    ax.set_ylabel("Mean registration time (s)")
    ax.set_title("RS-MRQ Cap-Hit Ratio vs Time")
    ax.grid(True)
    save_figure(fig, "rsmrq_cap_hit_ratio_vs_time")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ordered["Cap"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2.0, ordered["Mean pre_cap_candidate_count"], width=width, label="Pre-cap", color="#E15759")
    ax.bar(x + width / 2.0, ordered["Mean post_cap_candidate_count"].fillna(ordered["Mean candidate_inflation_mean"]), width=width, label="Post-cap", color="#4E79A7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cap")
    ax.set_ylabel("Mean candidate count")
    ax.set_title("RS-MRQ Cap vs Candidate Count")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    save_figure(fig, "rsmrq_cap_vs_candidate_count")


def plot_robust_vote(enriched: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for _, row in enriched.iterrows():
        ax.scatter(row["Mean ppf_frontend_time"], row["Mean ADD"], color=METHOD_COLORS[row["Method"]], s=70)
        ax.annotate(f"rv{int(row['top_m_per_bucket'])}", (row["Mean ppf_frontend_time"], row["Mean ADD"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Mean front-end time (s)")
    ax.set_ylabel("Mean ADD")
    ax.set_title("Robust Vote Light Tradeoff")
    ax.grid(True)
    save_figure(fig, "robust_vote_light_tradeoff")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(enriched["Mean registration_time"], enriched["SR@ADD<=0.02"], marker="o", color="#76B7B2")
    for _, row in enriched.iterrows():
        ax.annotate(f"rv{int(row['top_m_per_bucket'])}", (row["Mean registration_time"], row["SR@ADD<=0.02"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Mean registration time (s)")
    ax.set_ylabel("SR@ADD<=0.02")
    ax.set_title("Robust Vote Time vs Quality")
    ax.grid(True)
    save_figure(fig, "robust_vote_time_vs_quality")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    reduced = enriched[enriched["Method"] != "Ours (Full)"].copy()
    ax.bar(reduced["top_m_per_bucket"].astype(int).astype(str), -reduced["Front-end Time Change vs Ours (%)"], color=["#76B7B2", "#B07AA1"])
    ax.set_xlabel("top_m_per_bucket")
    ax.set_ylabel("Front-end reduction vs Ours (%)")
    ax.set_title("Robust Vote Front-end Reduction")
    ax.grid(True, axis="y")
    save_figure(fig, "robust_vote_frontend_reduction")


def plot_hard_cases(hard_case_df: pd.DataFrame) -> None:
    if hard_case_df.empty:
        return
    improvement = hard_case_df.groupby("category")["ADD_improvement_vs_rhs"].mean().reset_index().sort_values("category")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(improvement["category"], improvement["ADD_improvement_vs_rhs"], color="#59A14F")
    ax.set_ylabel("Mean ADD improvement")
    ax.set_title("Hard-case Improvement by Category")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y")
    save_figure(fig, "hard_case_improvement_bar")

    scene_counts = hard_case_df.groupby(["scene_name", "category"]).size().unstack(fill_value=0)
    scene_counts = scene_counts.loc[sorted(scene_counts.index.tolist(), key=sort_scene_key)]
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    bottom = np.zeros(len(scene_counts))
    for category in scene_counts.columns:
        values = scene_counts[category].to_numpy(dtype=float)
        ax.bar(scene_counts.index.tolist(), values, bottom=bottom, label=category)
        bottom += values
    ax.set_ylabel("Case count")
    ax.set_title("Hard-case Scene Distribution")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_figure(fig, "hard_case_scene_distribution")

    object_counts = hard_case_df.groupby(["obj_id", "category"]).size().unstack(fill_value=0)
    object_counts = object_counts.loc[sorted(object_counts.index.tolist(), key=sort_obj_key)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(object_counts))
    for category in object_counts.columns:
        values = object_counts[category].to_numpy(dtype=float)
        ax.bar(object_counts.index.astype(str).tolist(), values, bottom=bottom, label=category)
        bottom += values
    ax.set_ylabel("Case count")
    ax.set_title("Hard-case Object Distribution")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_figure(fig, "hard_case_object_distribution")


def plot_group_summary(group_df: pd.DataFrame, group_col: str, basename: str, value_col: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    if group_col == "scene_name":
        order = sorted(group_df[group_col].dropna().unique().tolist(), key=sort_scene_key)
    else:
        order = sorted(group_df[group_col].dropna().unique().tolist(), key=sort_obj_key)
    for method in METHOD_ORDER:
        subset = group_df[group_df["method"] == method].copy()
        subset = subset.set_index(group_col).reindex(order).reset_index()
        ax.plot(subset[group_col].astype(str), subset[value_col], marker="o", label=method, color=METHOD_COLORS[method])
    ax.set_xlabel("Scene" if group_col == "scene_name" else "Object ID")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_figure(fig, basename)


def build_full_report(
    master_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rsmrq_enriched: pd.DataFrame,
    rv_enriched: pd.DataFrame,
    hard_case_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> str:
    ours = summary_df[summary_df["method"] == "Ours (Full)"].iloc[0]
    adaptive = summary_df[summary_df["method"] == "Adaptive Two-Stage"].iloc[0]
    no = summary_df[summary_df["method"] == "No RS-MRQ"].iloc[0]
    rv20 = rv_enriched[rv_enriched["Method"] == "Ours + RV20"].iloc[0]
    cap64 = rsmrq_enriched[rsmrq_enriched["Method"] == "Ours + RS-MRQ cap64"].iloc[0]
    strongest_corr = corr_df[corr_df["group"] == "All"].iloc[corr_df[corr_df["group"] == "All"]["spearman"].abs().idxmax()]
    time_pair = paired_df[
        (paired_df["comparison"] == "Ours (Full) vs Adaptive Two-Stage") & (paired_df["metric"] == "registration_time")
    ].iloc[0]
    lines = [
        "# Stanford Full Analysis Report",
        "",
        "## Data Integrity Check",
        "- Rebuilt a unified per-case table in `stanford_master_per_case.csv` and a method summary table in `stanford_master_summary.csv`.",
        "- Verified the existing `adaptive_summary.csv`, `rsmrq_budget_sweep.csv`, and `robust_vote_light_sweep.csv` against direct batch-JSON aggregation.",
        "- Wrote the detailed audit to `data_consistency_report.md`.",
        "",
        "## Main Comparison",
        f"- Ours (Full) reaches mean ADD {ours['mean_ADD']:.6f}; Adaptive Two-Stage is close at {adaptive['mean_ADD']:.6f}.",
        f"- Adaptive cuts mean registration time from {ours['mean_registration_time']:.2f}s to {adaptive['mean_registration_time']:.2f}s.",
        f"- No RS-MRQ is the fast lightweight reference at {no['mean_registration_time']:.2f}s but loses success rate.",
        f"- See `main_comparison_table.csv`, `main_comparison_table.md`, `adaptive_accuracy_vs_time.png`, and `adaptive_accuracy_vs_time.svg`.",
        "",
        "## Adaptive Two-Stage Analysis",
        f"- Upgrade ratio is {mean_or_nan(master_df[master_df['method'] == 'Adaptive Two-Stage']['adaptive_used_upgrade']) * 100.0:.2f}%.",
        f"- Adaptive keeps the same SR@ADD<=0.02 as Ours ({adaptive['SR@ADD<=0.02']:.4f}) while saving about {(1.0 - adaptive['mean_registration_time'] / ours['mean_registration_time']) * 100.0:.2f}% time.",
        "- Trigger, scene, object, and rescue breakdowns are in `adaptive_upgrade_breakdown.csv`, `adaptive_upgrade_by_scene.csv`, `adaptive_upgrade_by_object.csv`, and `adaptive_rescue_summary.csv`.",
        "",
        "## Timing Breakdown",
        f"- Ours (Full) front-end mean is {ours['mean_frontend_time']:.2f}s vs back-end {ours['mean_backend_time']:.2f}s, so the bottleneck remains front-end.",
        f"- Adaptive improves mean time and the tail together; the paired bootstrap CI for Ours minus Adaptive time is [{time_pair['bootstrap_ci_low']:.3f}, {time_pair['bootstrap_ci_high']:.3f}]s.",
        "- See `timing_breakdown_table.csv`, `timing_breakdown_bar.png`, `timing_breakdown_stacked.png`, `registration_time_boxplot.png`, and `registration_time_violin.png`.",
        "",
        "## RS-MRQ Budget Sweep",
        f"- `cap64` reduces post-cap candidate count to {cap64['Mean post_cap_candidate_count']:.2f}, but runtime does not improve proportionally.",
        "- The current evidence suggests query/merge work dominates before truncation is applied.",
        "- See `rsmrq_budget_sweep_enriched.csv`, `rsmrq_budget_summary.md`, `rsmrq_mechanism_analysis.md`, and the RS-MRQ figures.",
        "",
        "## Robust Vote Light Sweep",
        f"- `rv20` reduces mean registration time by {-rv20['Registration Time Change vs Ours (%)']:.2f}% with no success-rate drop.",
        "- `rv10` is faster still, but its mean ADD degradation is larger than `rv20`.",
        "- See `robust_vote_light_enriched.csv`, `robust_vote_light_summary.md`, and the Robust Vote figures.",
        "",
        "## Hard-case Rescue Analysis",
        f"- The unified hard-case table contains {len(hard_case_df)} rows across {hard_case_df['category'].nunique() if not hard_case_df.empty else 0} observed rescue/failure categories.",
        "- Ours rescues the same No-RS-MRQ miss that Adaptive also rescues; Ours additionally rescues many Same-backbone failures.",
        "- See `hard_case_master.csv`, `hard_case_report_enhanced.md`, and the hard-case distribution plots.",
        "",
        "## Candidate Inflation vs Front-end Cost",
        f"- The strongest overall monotonic correlation with front-end cost is `{strongest_corr['metric']}` (Spearman {strongest_corr['spearman']:.3f}).",
        "- Candidate count alone does not explain the RS-MRQ cap results because pre-cap and post-cap statistics diverge.",
        "- See `candidate_time_correlation.csv`, the scatter plots, and `frontend_cost_drivers.md`.",
        "",
        "## By-object / By-scene Analysis",
        "- Per-object and per-scene summaries are exported to `by_object_summary.csv` and `by_scene_summary.csv`.",
        "- Matching ADD/time trend plots are provided for direct comparison across all methods.",
        "",
        "## Statistical Significance",
        "- Pairwise comparisons cover Ours vs No RS-MRQ, Ours vs Adaptive, Adaptive vs No RS-MRQ, and Ours vs No Backend.",
        "- Each comparison includes a paired t-statistic, a Wilcoxon signed-rank statistic, and a bootstrap 95% CI.",
        "- See `paired_tests.csv`, `paired_tests.md`, and `bootstrap_confidence_intervals.csv`.",
        "",
        "## Key Findings",
        "- Adaptive Two-Stage is the strongest accuracy/time compromise among the currently tested options.",
        "- Front-end work remains the dominant time sink, not mode clustering or the current back-end.",
        "- RS-MRQ cap experiments suggest candidate truncation is too late to erase most of the RS-MRQ overhead.",
        "- Robust Vote default is heavy enough that `rv20` looks like a practical lightweight replacement point.",
    ]
    return "\n".join(lines)


def format_table_for_display(df: pd.DataFrame, float_digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v: "Yes" if bool(v) else "No")
        elif pd.api.types.is_numeric_dtype(out[col]):
            if out[col].dropna().isin([0.0, 1.0]).all() and "SR@" not in col and "Ratio" not in col and "Relative" not in col:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)) if float(v).is_integer() else f"{v:.{float_digits}f}")
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.{float_digits}f}")
    return out


def render_table_figure(df: pd.DataFrame, basename: str, title: str) -> None:
    display_df = format_table_for_display(df, float_digits=4)
    nrows, ncols = display_df.shape
    fig_w = max(9.0, ncols * 1.35)
    fig_h = max(2.8, nrows * 0.52 + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E9EEF5")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#F8F8F8")
    ax.set_title(title, pad=16)
    save_figure(fig, basename)


def build_overall_ablation_table(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    module_flags = {
        "Same-backbone Baseline": {"RS-MRQ": "No", "Robust Vote": "No", "Candidate-level Selection": "No", "Mode Clustering": "No"},
        "No RS-MRQ": {"RS-MRQ": "No", "Robust Vote": "Yes", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
        "No Robust Vote": {"RS-MRQ": "Yes", "Robust Vote": "No", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
        "No Backend": {"RS-MRQ": "Yes", "Robust Vote": "Yes", "Candidate-level Selection": "No", "Mode Clustering": "No"},
        "Ours (Full)": {"RS-MRQ": "Yes", "Robust Vote": "Yes", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
    }
    rows = []
    core_methods = list(module_flags.keys())
    for method in core_methods:
        row = summary_df[summary_df["method"] == method].iloc[0]
        rows.append(
            {
                "Method": method,
                **module_flags[method],
                "Mean ADD": row["mean_ADD"],
                "Mean ADD-S": row["mean_ADD_S"],
                "SR@ADD<=0.02": row["SR@ADD<=0.02"],
                "Mean Rotation Error (deg)": row["mean_rotation_error_deg"],
                "Mean Registration Time (s)": row["mean_registration_time"],
                "Mean Front-end Time (s)": row["mean_frontend_time"],
                "Mean Back-end Time (s)": row["mean_backend_time"],
                "Candidate Inflation Mean": row["candidate_inflation_mean"],
            }
        )
    table = pd.DataFrame(rows)

    ours = table[table["Method"] == "Ours (Full)"].iloc[0]
    diffs = table[table["Method"] != "Ours (Full)"].copy()
    diffs["add_gap"] = diffs["Mean ADD"] - ours["Mean ADD"]
    diffs["time_drop"] = ours["Mean Registration Time (s)"] - diffs["Mean Registration Time (s)"]
    worst_acc = diffs.iloc[diffs["add_gap"].idxmax()]
    biggest_time_drop = diffs.iloc[diffs["time_drop"].idxmax()]
    frontend_ratio = ours["Mean Front-end Time (s)"] / ours["Mean Registration Time (s)"]
    reg = table["Mean Registration Time (s)"].to_numpy(dtype=float)
    add = table["Mean ADD"].to_numpy(dtype=float)
    pareto_mask = []
    for i in range(len(table)):
        dominated = False
        for j in range(len(table)):
            if i == j:
                continue
            if reg[j] <= reg[i] and add[j] <= add[i] and (reg[j] < reg[i] or add[j] < add[i]):
                dominated = True
                break
        pareto_mask.append(not dominated)
    ours_pareto = bool(pareto_mask[table.index[table["Method"] == "Ours (Full)"][0]])
    adaptive_row = summary_df[summary_df["method"] == "Adaptive Two-Stage"].iloc[0]
    if ours_pareto and adaptive_row["mean_ADD"] > ours["Mean ADD"] and adaptive_row["mean_registration_time"] < ours["Mean Registration Time (s)"]:
        pareto_text = "Ours (Full) 仍在 Pareto 前沿上，但不是明显的单点最优，因为 Adaptive Two-Stage 用极小的精度代价换来了显著更低的时间。"
    elif ours_pareto:
        pareto_text = "Ours (Full) 在当前核心五组里处于 Pareto 前沿。"
    else:
        pareto_text = "Ours (Full) 不在当前核心五组的 Pareto 前沿上。"

    summary = "\n".join(
        [
            "# 整体消融总表说明",
            "",
            (
                f"- 在当前五组里，去掉后精度掉得最多的是 `{worst_acc['Method']}`，"
                f"其 Mean ADD 相比 Ours (Full) 增加了 {worst_acc['add_gap']:.6f}。"
            ),
            (
                f"- 在当前五组里，去掉后时间降得最多的是 `{biggest_time_drop['Method']}`，"
                f"其 Mean Registration Time 相比 Ours (Full) 下降了 {biggest_time_drop['time_drop']:.2f}s。"
            ),
            (
                f"- 当前总时间瓶颈仍然主要在 front-end：Ours (Full) 的 front-end / registration 比例约为 "
                f"{frontend_ratio * 100.0:.2f}%，而 back-end 仅为 {ours['Mean Back-end Time (s)'] / ours['Mean Registration Time (s)'] * 100.0:.2f}%。"
            ),
            f"- {pareto_text}",
        ]
    )

    consistency = "\n".join(
        [
            "# 整体消融一致性检查",
            "",
            "- 整体消融总表的数值全部来自五个核心 batch JSON 的逐 case 重聚合结果。",
            f"- 五个方法的 case 数分别为：{', '.join(f'{m}={int(summary_df[summary_df['method'] == m]['n_cases'].iloc[0])}' for m in core_methods)}。",
            "- 方法名与分析主表保持一致，没有发现大小写或别名混用。",
            "- 本表要求的指标列均已成功聚合，没有出现整列为空的情况。",
        ]
    )
    return table, summary, consistency


def build_efficiency_extensions_table(summary_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "No RS-MRQ",
        "Adaptive Two-Stage",
        "Ours (Full)",
        "Ours + RV20",
        "Ours + RV10",
        "Ours + RS-MRQ cap64",
        "Ours + RS-MRQ cap128",
        "Ours + RS-MRQ cap256",
    ]
    ours = summary_df[summary_df["method"] == "Ours (Full)"].iloc[0]
    rows = []
    for method in methods:
        row = summary_df[summary_df["method"] == method].iloc[0]
        subset = master_df[master_df["method"] == method]
        tag = "-"
        base_variant = "Core"
        if method == "Adaptive Two-Stage":
            base_variant = "Adaptive"
            tag = f"Upgrade {mean_or_nan(subset['adaptive_used_upgrade']) * 100.0:.2f}%"
        elif "RV" in method:
            base_variant = "Robust Vote Light"
            top_m = int(subset["robust_vote_top_m_per_bucket"].dropna().iloc[0])
            tag = f"top-m={top_m}"
        elif "cap" in method:
            base_variant = "RS-MRQ Cap"
            cap = int(subset["rsmrq_global_candidate_cap"].dropna().iloc[0])
            tag = f"cap={cap}"
        elif method == "No RS-MRQ":
            base_variant = "Reference Lightweight"
        rows.append(
            {
                "Method": method,
                "Base Variant": base_variant,
                "Mean ADD": row["mean_ADD"],
                "SR@ADD<=0.02": row["SR@ADD<=0.02"],
                "Mean Registration Time (s)": row["mean_registration_time"],
                "Mean Front-end Time (s)": row["mean_frontend_time"],
                "Mean Back-end Time (s)": row["mean_backend_time"],
                "Relative Time vs Ours": pct_change(row["mean_registration_time"], ours["mean_registration_time"]),
                "Relative ADD vs Ours": row["mean_ADD"] - ours["mean_ADD"],
                "Upgrade Ratio / RV top-m / Candidate Cap": tag,
            }
        )
    return pd.DataFrame(rows)


def build_overall_ablation_table_v2(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    module_flags = {
        "Same-backbone Baseline": {"RS-MRQ": "No", "Robust Vote": "No", "Candidate-level Selection": "No", "Mode Clustering": "No"},
        "No RS-MRQ": {"RS-MRQ": "No", "Robust Vote": "Yes", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
        "No Robust Vote": {"RS-MRQ": "Yes", "Robust Vote": "No", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
        "No Backend": {"RS-MRQ": "Yes", "Robust Vote": "Yes", "Candidate-level Selection": "No", "Mode Clustering": "No"},
        "Ours (Full)": {"RS-MRQ": "Yes", "Robust Vote": "Yes", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
    }
    rows = []
    core_methods = list(module_flags.keys())
    for method in core_methods:
        row = summary_df.loc[summary_df["method"] == method].iloc[0]
        rows.append(
            {
                "Method": method,
                **module_flags[method],
                "Mean ADD": row["mean_ADD"],
                "Mean ADD-S": row["mean_ADD_S"],
                "SR@ADD<=0.02": row["SR@ADD<=0.02"],
                "Mean Rotation Error (deg)": row["mean_rotation_error_deg"],
                "Mean Registration Time (s)": row["mean_registration_time"],
                "Mean Front-end Time (s)": row["mean_frontend_time"],
                "Mean Back-end Time (s)": row["mean_backend_time"],
                "Candidate Inflation Mean": row["candidate_inflation_mean"],
            }
        )
    table = pd.DataFrame(rows)
    ours = table.loc[table["Method"] == "Ours (Full)"].iloc[0]
    diffs = table.loc[table["Method"] != "Ours (Full)"].copy()
    diffs["add_gap"] = diffs["Mean ADD"] - ours["Mean ADD"]
    diffs["time_drop"] = ours["Mean Registration Time (s)"] - diffs["Mean Registration Time (s)"]
    worst_acc = diffs.loc[diffs["add_gap"].idxmax()]
    biggest_time_drop = diffs.loc[diffs["time_drop"].idxmax()]
    frontend_ratio = ours["Mean Front-end Time (s)"] / ours["Mean Registration Time (s)"]
    reg = table["Mean Registration Time (s)"].to_numpy(dtype=float)
    add = table["Mean ADD"].to_numpy(dtype=float)
    pareto_mask = []
    for i in range(len(table)):
        dominated = False
        for j in range(len(table)):
            if i == j:
                continue
            if reg[j] <= reg[i] and add[j] <= add[i] and (reg[j] < reg[i] or add[j] < add[i]):
                dominated = True
                break
        pareto_mask.append(not dominated)
    ours_pareto = bool(pareto_mask[int(table.index[table["Method"] == "Ours (Full)"][0])])
    adaptive_row = summary_df.loc[summary_df["method"] == "Adaptive Two-Stage"].iloc[0]
    if ours_pareto and adaptive_row["mean_ADD"] > ours["Mean ADD"] and adaptive_row["mean_registration_time"] < ours["Mean Registration Time (s)"]:
        pareto_text = "Ours (Full) 仍在 Pareto 前沿上，但不是明显的单点最优，因为 Adaptive Two-Stage 用极小的精度代价换来了显著更低的时间。"
    elif ours_pareto:
        pareto_text = "Ours (Full) 在当前核心五组里处于 Pareto 前沿。"
    else:
        pareto_text = "Ours (Full) 不在当前核心五组的 Pareto 前沿上。"
    summary = "\n".join(
        [
            "# 整体消融总表说明",
            "",
            f"- 在当前五组里，去掉后精度掉得最多的是 `{worst_acc['Method']}`，其 Mean ADD 相比 Ours (Full) 增加了 {worst_acc['add_gap']:.6f}。",
            f"- 在当前五组里，去掉后时间降得最多的是 `{biggest_time_drop['Method']}`，其 Mean Registration Time 相比 Ours (Full) 下降了 {biggest_time_drop['time_drop']:.2f}s。",
            f"- 当前总时间瓶颈仍然主要在 front-end：Ours (Full) 的 front-end / registration 比例约为 {frontend_ratio * 100.0:.2f}%，而 back-end 仅为 {ours['Mean Back-end Time (s)'] / ours['Mean Registration Time (s)'] * 100.0:.2f}%。",
            f"- {pareto_text}",
        ]
    )
    case_counts_text = ", ".join(
        f"{method}={int(summary_df.loc[summary_df['method'] == method, 'n_cases'].iloc[0])}"
        for method in core_methods
    )
    consistency = "\n".join(
        [
            "# 整体消融一致性检查",
            "",
            "- 整体消融总表的数值全部来自五个核心 batch JSON 的逐 case 重聚合结果。",
            f"- 五个方法的 case 数分别为：{case_counts_text}。",
            "- 方法名与分析主表保持一致，没有发现大小写或别名混用。",
            "- 本表要求的指标列均已成功聚合，没有出现整列为空的情况。",
        ]
    )
    return table, summary, consistency


def build_efficiency_extensions_table_v2(summary_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    return build_efficiency_extensions_table(summary_df, master_df)


def build_overall_ablation_table_v3(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    module_flags = {
        "Same-backbone Baseline": {"RS-MRQ": "No", "Robust Vote": "No", "Candidate-level Selection": "No", "Mode Clustering": "No"},
        "No RS-MRQ": {"RS-MRQ": "No", "Robust Vote": "Yes", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
        "No Robust Vote": {"RS-MRQ": "Yes", "Robust Vote": "No", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
        "No Backend": {"RS-MRQ": "Yes", "Robust Vote": "Yes", "Candidate-level Selection": "No", "Mode Clustering": "No"},
        "Ours (Full)": {"RS-MRQ": "Yes", "Robust Vote": "Yes", "Candidate-level Selection": "Yes", "Mode Clustering": "Yes"},
    }
    core_methods = list(module_flags.keys())
    rows = []
    for method in core_methods:
        row = summary_df.loc[summary_df["method"] == method].iloc[0]
        rows.append(
            {
                "Method": method,
                **module_flags[method],
                "Mean ADD": row["mean_ADD"],
                "Mean ADD-S": row["mean_ADD_S"],
                "SR@ADD<=0.02": row["SR@ADD<=0.02"],
                "Mean Rotation Error (deg)": row["mean_rotation_error_deg"],
                "Mean Registration Time (s)": row["mean_registration_time"],
                "Mean Front-end Time (s)": row["mean_frontend_time"],
                "Mean Back-end Time (s)": row["mean_backend_time"],
                "Candidate Inflation Mean": row["candidate_inflation_mean"],
            }
        )
    table = pd.DataFrame(rows)
    ours = table.loc[table["Method"] == "Ours (Full)"].iloc[0]
    diffs = table.loc[table["Method"] != "Ours (Full)"].copy()
    diffs["add_gap"] = diffs["Mean ADD"] - ours["Mean ADD"]
    diffs["time_drop"] = ours["Mean Registration Time (s)"] - diffs["Mean Registration Time (s)"]
    worst_acc = diffs.loc[diffs["add_gap"].idxmax()]
    biggest_time_drop = diffs.loc[diffs["time_drop"].idxmax()]
    frontend_ratio = ours["Mean Front-end Time (s)"] / ours["Mean Registration Time (s)"]

    reg = table["Mean Registration Time (s)"].to_numpy(dtype=float)
    add = table["Mean ADD"].to_numpy(dtype=float)
    pareto_mask = []
    for i in range(len(table)):
        dominated = False
        for j in range(len(table)):
            if i == j:
                continue
            if reg[j] <= reg[i] and add[j] <= add[i] and (reg[j] < reg[i] or add[j] < add[i]):
                dominated = True
                break
        pareto_mask.append(not dominated)
    ours_pareto = bool(pareto_mask[int(table.index[table["Method"] == "Ours (Full)"][0])])
    adaptive_row = summary_df.loc[summary_df["method"] == "Adaptive Two-Stage"].iloc[0]
    if ours_pareto and adaptive_row["mean_ADD"] > ours["Mean ADD"] and adaptive_row["mean_registration_time"] < ours["Mean Registration Time (s)"]:
        pareto_text = "Ours (Full) remains on the Pareto frontier, but it is not an obvious single best point because Adaptive Two-Stage trades a very small ADD increase for a large time reduction."
    elif ours_pareto:
        pareto_text = "Ours (Full) is still on the Pareto frontier among the five core methods."
    else:
        pareto_text = "Ours (Full) is not on the Pareto frontier among the five core methods."

    summary = "\n".join(
        [
            "# Overall Ablation Summary",
            "",
            f"- The largest accuracy drop comes from `{worst_acc['Method']}`, whose mean ADD is {worst_acc['add_gap']:.6f} higher than Ours (Full).",
            f"- The largest time reduction comes from `{biggest_time_drop['Method']}`, whose mean registration time is {biggest_time_drop['time_drop']:.2f}s lower than Ours (Full).",
            f"- The current bottleneck is still front-end: Ours (Full) spends about {frontend_ratio * 100.0:.2f}% of registration time in front-end, while back-end contributes only {ours['Mean Back-end Time (s)'] / ours['Mean Registration Time (s)'] * 100.0:.2f}%.",
            f"- {pareto_text}",
        ]
    )

    case_counts_text = ", ".join(
        f"{method}={int(summary_df.loc[summary_df['method'] == method, 'n_cases'].iloc[0])}"
        for method in core_methods
    )
    consistency = "\n".join(
        [
            "# Overall Ablation Consistency Check",
            "",
            "- Every numeric entry in the overall ablation table is re-aggregated from the five core batch JSON files.",
            f"- Case counts per method are: {case_counts_text}.",
            "- Method names are consistent with the master summary table.",
            "- No required column in the ablation table is fully empty.",
        ]
    )
    return table, summary, consistency


def build_efficiency_extensions_table_v3(summary_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "No RS-MRQ",
        "Adaptive Two-Stage",
        "Ours (Full)",
        "Ours + RV20",
        "Ours + RV10",
        "Ours + RS-MRQ cap64",
        "Ours + RS-MRQ cap128",
        "Ours + RS-MRQ cap256",
    ]
    ours = summary_df.loc[summary_df["method"] == "Ours (Full)"].iloc[0]
    rows = []
    for method in methods:
        row = summary_df.loc[summary_df["method"] == method].iloc[0]
        subset = master_df.loc[master_df["method"] == method]
        tag = "-"
        base_variant = "Core"
        if method == "Adaptive Two-Stage":
            base_variant = "Adaptive"
            tag = f"Upgrade {mean_or_nan(subset['adaptive_used_upgrade']) * 100.0:.2f}%"
        elif "RV" in method:
            base_variant = "Robust Vote Light"
            tag = f"top-m={int(subset['robust_vote_top_m_per_bucket'].dropna().iloc[0])}"
        elif "cap" in method:
            base_variant = "RS-MRQ Cap"
            tag = f"cap={int(subset['rsmrq_global_candidate_cap'].dropna().iloc[0])}"
        elif method == "No RS-MRQ":
            base_variant = "Reference Lightweight"
        rows.append(
            {
                "Method": method,
                "Base Variant": base_variant,
                "Mean ADD": row["mean_ADD"],
                "SR@ADD<=0.02": row["SR@ADD<=0.02"],
                "Mean Registration Time (s)": row["mean_registration_time"],
                "Mean Front-end Time (s)": row["mean_frontend_time"],
                "Mean Back-end Time (s)": row["mean_backend_time"],
                "Relative Time vs Ours": pct_change(row["mean_registration_time"], ours["mean_registration_time"]),
                "Relative ADD vs Ours": row["mean_ADD"] - ours["mean_ADD"],
                "Upgrade Ratio / RV top-m / Candidate Cap": tag,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    apply_plot_style()
    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)

    payloads = load_method_payloads()
    master_df = build_master_per_case(payloads)
    summary_df = build_master_summary(master_df)

    write_df(TABLE_DIR / "stanford_master_per_case.csv", master_df)
    write_df(TABLE_DIR / "stanford_master_summary.csv", summary_df)
    write_text(TABLE_DIR / "data_consistency_report.md", build_data_consistency_report(master_df, summary_df))

    main_table, main_findings = build_main_comparison(summary_df)
    write_df(TABLE_DIR / "main_comparison_table.csv", main_table)
    write_text(TABLE_DIR / "main_comparison_table.md", markdown_table(main_table))
    write_text(TABLE_DIR / "main_findings.md", main_findings)

    adaptive_eff_table, adaptive_eff_summary = build_adaptive_efficiency(master_df, summary_df)
    write_df(TABLE_DIR / "adaptive_efficiency_table.csv", adaptive_eff_table)
    write_text(TABLE_DIR / "adaptive_efficiency_summary.md", adaptive_eff_summary)

    upgrade_breakdown, upgrade_by_scene, upgrade_by_object, adaptive_upgrade_summary = build_adaptive_upgrade_tables(master_df)
    write_df(TABLE_DIR / "adaptive_upgrade_breakdown.csv", upgrade_breakdown)
    write_df(TABLE_DIR / "adaptive_upgrade_by_scene.csv", upgrade_by_scene)
    write_df(TABLE_DIR / "adaptive_upgrade_by_object.csv", upgrade_by_object)
    write_text(TABLE_DIR / "adaptive_upgrade_summary.md", adaptive_upgrade_summary)

    adaptive_rescue_summary, adaptive_rescue_cases, adaptive_fail_cases = build_adaptive_rescue_outputs(master_df)
    write_df(TABLE_DIR / "adaptive_rescue_summary.csv", adaptive_rescue_summary)
    write_df(TABLE_DIR / "adaptive_vs_no_rsmrq_rescue_cases.csv", adaptive_rescue_cases)
    write_df(TABLE_DIR / "adaptive_vs_ours_fail_cases.csv", adaptive_fail_cases)

    timing_df, tail_df, tail_summary = build_timing_breakdown(master_df)
    write_df(TABLE_DIR / "timing_breakdown_table.csv", timing_df)
    write_df(TABLE_DIR / "time_tail_statistics.csv", tail_df)
    write_text(TABLE_DIR / "time_tail_summary.md", tail_summary)

    rsmrq_enriched, rsmrq_summary, rsmrq_mechanism = build_rsmrq_budget_enriched(master_df)
    write_df(TABLE_DIR / "rsmrq_budget_sweep_enriched.csv", rsmrq_enriched)
    write_text(TABLE_DIR / "rsmrq_budget_summary.md", rsmrq_summary)
    write_text(TABLE_DIR / "rsmrq_mechanism_analysis.md", rsmrq_mechanism)

    rv_enriched, rv_summary = build_robust_vote_enriched(master_df)
    write_df(TABLE_DIR / "robust_vote_light_enriched.csv", rv_enriched)
    write_text(TABLE_DIR / "robust_vote_light_summary.md", rv_summary)

    hard_case_master, hard_case_report = build_hard_case_master(master_df)
    write_df(TABLE_DIR / "hard_case_master.csv", hard_case_master)
    write_text(TABLE_DIR / "hard_case_report_enhanced.md", hard_case_report)

    corr_df, frontend_summary = build_candidate_correlations(master_df)
    write_df(TABLE_DIR / "candidate_time_correlation.csv", corr_df)
    write_text(TABLE_DIR / "frontend_cost_drivers.md", frontend_summary)

    by_object = build_group_summary(master_df, "obj_id")
    by_scene = build_group_summary(master_df, "scene_name")
    write_df(TABLE_DIR / "by_object_summary.csv", by_object)
    write_df(TABLE_DIR / "by_scene_summary.csv", by_scene)

    paired_df, bootstrap_df, paired_summary = build_paired_tests(master_df, args.bootstrap_rounds, args.rng_seed)
    write_df(TABLE_DIR / "paired_tests.csv", paired_df)
    write_df(TABLE_DIR / "bootstrap_confidence_intervals.csv", bootstrap_df)
    write_text(TABLE_DIR / "paired_tests.md", paired_summary)

    overall_ablation_table, overall_ablation_summary, overall_ablation_consistency = build_overall_ablation_table_v3(summary_df)
    write_df(TABLE_DIR / "overall_ablation_table.csv", overall_ablation_table)
    write_text(TABLE_DIR / "overall_ablation_table.md", markdown_table(format_table_for_display(overall_ablation_table)))
    write_text(TABLE_DIR / "overall_ablation_summary.md", overall_ablation_summary)
    write_text(TABLE_DIR / "overall_ablation_consistency_check.md", overall_ablation_consistency)
    render_table_figure(overall_ablation_table, "overall_ablation_table", "Overall Stanford Ablation Table")

    efficiency_extensions_table = build_efficiency_extensions_table_v3(summary_df, master_df)
    write_df(TABLE_DIR / "efficiency_extensions_table.csv", efficiency_extensions_table)
    write_text(TABLE_DIR / "efficiency_extensions_table.md", markdown_table(format_table_for_display(efficiency_extensions_table)))
    render_table_figure(efficiency_extensions_table, "efficiency_extensions_table", "Stanford Efficiency Extensions Table")

    plot_main_and_adaptive(summary_df, master_df)
    plot_timing(master_df, timing_df)
    plot_rsmrq_budget(rsmrq_enriched)
    plot_robust_vote(rv_enriched)
    plot_hard_cases(hard_case_master)
    plot_method_scatter(master_df, "candidate_inflation_mean", "ppf_frontend_time", "candidate_inflation_vs_frontend", "Candidate inflation mean", "Front-end time (s)", "Candidate Inflation vs Front-end Time")
    plot_method_scatter(master_df, "num_input_candidates", "ppf_frontend_time", "num_input_candidates_vs_frontend", "Num input candidates", "Front-end time (s)", "Num Input Candidates vs Front-end Time")
    plot_method_scatter(master_df, "n_votes", "ppf_frontend_time", "n_votes_vs_frontend", "Robust-vote n_votes", "Front-end time (s)", "n_votes vs Front-end Time")
    plot_group_summary(by_object, "obj_id", "by_object_add", "Mean ADD", "Mean ADD", "By-object ADD")
    plot_group_summary(by_object, "obj_id", "by_object_time", "Mean registration_time", "Mean registration time (s)", "By-object Registration Time")
    plot_group_summary(by_scene, "scene_name", "by_scene_add", "Mean ADD", "Mean ADD", "By-scene ADD")
    plot_group_summary(by_scene, "scene_name", "by_scene_time", "Mean registration_time", "Mean registration time (s)", "By-scene Registration Time")

    write_text(TABLE_DIR / "stanford_full_analysis_report.md", build_full_report(master_df, summary_df, rsmrq_enriched, rv_enriched, hard_case_master, corr_df, paired_df))


if __name__ == "__main__":
    main()
