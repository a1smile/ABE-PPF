import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PATHS = {
    "adaptive": ROOT / "experiments" / "results" / "stanford_adaptive_two_stage_batch.json",
    "no_rsmrq": ROOT / "experiments" / "results" / "stanford_no_rsmrq_batch.json",
    "ours": ROOT / "experiments" / "results" / "stanford_ours_full_batch.json",
    "same_backbone": ROOT / "experiments" / "results" / "stanford_same_backbone_baseline_batch.json",
    "rv20": ROOT / "experiments" / "results" / "stanford_ours_rv20_batch.json",
    "rv10": ROOT / "experiments" / "results" / "stanford_ours_rv10_batch.json",
    "cap64": ROOT / "experiments" / "results" / "stanford_ours_cap64_batch.json",
    "cap128": ROOT / "experiments" / "results" / "stanford_ours_cap128_batch.json",
    "cap256": ROOT / "experiments" / "results" / "stanford_ours_cap256_batch.json",
}

TABLE_DIR = ROOT / "experiments" / "tables" / "stanford"
FIG_DIR = ROOT / "experiments" / "figures"
SUCCESS_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Stanford adaptive/sweep tables and plots from batch JSON files.")
    ap.add_argument("--adaptive_json", type=Path, default=DEFAULT_PATHS["adaptive"])
    ap.add_argument("--no_rsmrq_json", type=Path, default=DEFAULT_PATHS["no_rsmrq"])
    ap.add_argument("--ours_json", type=Path, default=DEFAULT_PATHS["ours"])
    ap.add_argument("--same_backbone_json", type=Path, default=DEFAULT_PATHS["same_backbone"])
    ap.add_argument("--rv20_json", type=Path, default=DEFAULT_PATHS["rv20"])
    ap.add_argument("--rv10_json", type=Path, default=DEFAULT_PATHS["rv10"])
    ap.add_argument("--cap64_json", type=Path, default=DEFAULT_PATHS["cap64"])
    ap.add_argument("--cap128_json", type=Path, default=DEFAULT_PATHS["cap128"])
    ap.add_argument("--cap256_json", type=Path, default=DEFAULT_PATHS["cap256"])
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_batch_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError(f"Unsupported batch json format: {path}")
    return results


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def mean_of(results: Iterable[dict], getter) -> float:
    values = [getter(record) for record in results]
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def success_rate(results: Iterable[dict], threshold: float = SUCCESS_THRESHOLD) -> float:
    values = [1.0 if coerce_float(record.get("metrics", {}).get("ADD"), 1e9) <= threshold else 0.0 for record in results]
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def summarize_method_row(method: str, results: List[dict]) -> Dict[str, object]:
    return {
        "Method": method,
        "Upgrade Ratio": mean_of(results, lambda r: 1.0 if bool(r.get("adaptive_used_upgrade", False)) else 0.0),
        "Mean ADD": mean_of(results, lambda r: coerce_float(r.get("metrics", {}).get("ADD"))),
        "Mean ADD-S": mean_of(results, lambda r: coerce_float(r.get("metrics", {}).get("ADD_S"))),
        "SR@ADD<=0.02": success_rate(results),
        "Mean Rotation Error (deg)": mean_of(
            results,
            lambda r: coerce_float(r.get("metrics", {}).get("rotation_error_deg")),
        ),
        "Mean Registration Time (s)": mean_of(
            results,
            lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
        ),
        "Mean Front-end Time (s)": mean_of(
            results,
            lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
        ),
        "Mean Back-end Time (s)": mean_of(
            results,
            lambda r: coerce_float(r.get("stats", {}).get("backend_time")),
        ),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_case_map(results: List[dict]) -> Dict[int, dict]:
    return {int(record["idx"]): record for record in results if isinstance(record, dict) and "idx" in record}


def case_success(record: Optional[dict], threshold: float = SUCCESS_THRESHOLD) -> bool:
    if not isinstance(record, dict):
        return False
    return coerce_float(record.get("metrics", {}).get("ADD"), 1e9) <= threshold


def hard_case_row(record: dict) -> Dict[str, object]:
    return {
        "idx": int(record.get("idx", -1)),
        "scene_name": record.get("scene_name"),
        "scene_variant": record.get("scene_variant"),
        "obj_id": record.get("obj_id"),
        "ADD": coerce_float(record.get("metrics", {}).get("ADD")),
        "rotation_error_deg": coerce_float(record.get("metrics", {}).get("rotation_error_deg")),
        "registration_time": coerce_float(record.get("stats", {}).get("registration_time")),
        "ppf_frontend_time": coerce_float(record.get("stats", {}).get("ppf_frontend_time")),
        "adaptive_used_upgrade": bool(record.get("adaptive_used_upgrade", False)),
        "adaptive_trigger_reason": str(record.get("adaptive_trigger_reason", "")),
    }


def compare_success(
    lhs: Dict[int, dict],
    rhs: Dict[int, dict],
    require_lhs_success: bool = True,
    require_rhs_fail: bool = True,
) -> List[dict]:
    rows = []
    for idx, lhs_record in lhs.items():
        rhs_record = rhs.get(idx)
        lhs_ok = case_success(lhs_record)
        rhs_ok = case_success(rhs_record)
        if require_lhs_success and not lhs_ok:
            continue
        if require_rhs_fail and rhs_ok:
            continue
        rows.append(lhs_record)
    rows.sort(key=lambda r: (coerce_float(r.get("metrics", {}).get("ADD")), int(r.get("idx", -1))))
    return rows


def save_hard_case_csv(path: Path, records: List[dict]) -> None:
    write_csv(
        path,
        [hard_case_row(record) for record in records],
        [
            "idx",
            "scene_name",
            "scene_variant",
            "obj_id",
            "ADD",
            "rotation_error_deg",
            "registration_time",
            "ppf_frontend_time",
            "adaptive_used_upgrade",
            "adaptive_trigger_reason",
        ],
    )


def plot_adaptive_summary(rows: List[Dict[str, object]]) -> None:
    ensure_dir(FIG_DIR)
    x = [coerce_float(row["Mean Registration Time (s)"]) for row in rows]
    y = [coerce_float(row["Mean ADD"]) for row in rows]
    labels = [str(row["Method"]) for row in rows]

    plt.figure(figsize=(6.0, 4.2))
    plt.scatter(x, y, s=90)
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Mean registration time (s)")
    plt.ylabel("Mean ADD")
    plt.title("Adaptive Accuracy vs Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "adaptive_accuracy_vs_time.png", dpi=200)
    plt.close()

    adaptive_row = next((row for row in rows if str(row["Method"]) == "Adaptive Two-Stage"), None)
    if adaptive_row is None:
        return

    plt.figure(figsize=(4.6, 4.2))
    ratio = coerce_float(adaptive_row["Upgrade Ratio"])
    plt.bar(["Stage1 only", "Upgraded"], [1.0 - ratio, ratio], color=["#8fb996", "#d08c60"])
    plt.ylabel("Case ratio")
    plt.title("Adaptive Upgrade Ratio")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "adaptive_upgrade_ratio.png", dpi=200)
    plt.close()


def plot_tradeoff(
    items: List[Tuple[str, float, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(6.0, 4.2))
    xs = [item[1] for item in items]
    ys = [item[2] for item in items]
    plt.scatter(xs, ys, s=90)
    for label, x, y in items:
        plt.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_hard_case_report(
    path: Path,
    adaptive_vs_no_rsmrq: List[dict],
    adaptive_vs_ours: List[dict],
    ours_vs_no_rsmrq: List[dict],
    ours_vs_same_backbone: List[dict],
    adaptive_map: Dict[int, dict],
    no_rsmrq_map: Dict[int, dict],
    ours_map: Dict[int, dict],
) -> None:
    ours_rescue_idxs = {int(record["idx"]) for record in ours_vs_no_rsmrq}
    adaptive_rescue_idxs = {int(record["idx"]) for record in adaptive_vs_no_rsmrq}
    retained_ratio = (
        float(len(ours_rescue_idxs & adaptive_rescue_idxs)) / float(len(ours_rescue_idxs))
        if ours_rescue_idxs
        else 0.0
    )

    typical_cases = []
    for idx in sorted(adaptive_rescue_idxs):
        adaptive_record = adaptive_map[idx]
        no_rsmrq_record = no_rsmrq_map.get(idx, {})
        ours_record = ours_map.get(idx, {})
        typical_cases.append(
            (
                idx,
                coerce_float(no_rsmrq_record.get("metrics", {}).get("ADD"), 1e9),
                coerce_float(adaptive_record.get("metrics", {}).get("ADD"), 1e9),
                coerce_float(ours_record.get("metrics", {}).get("ADD"), 1e9),
                adaptive_record,
            )
        )
    typical_cases.sort(key=lambda item: (-item[1], item[2], item[0]))
    typical_cases = typical_cases[:5]

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stanford Hard-Case Report\n\n")
        f.write(f"- Ours relative to No RS-MRQ rescues {len(ours_vs_no_rsmrq)} cases.\n")
        f.write(f"- Adaptive relative to No RS-MRQ rescues {len(adaptive_vs_no_rsmrq)} cases.\n")
        f.write(
            f"- Adaptive retains {retained_ratio:.2%} of the hard-case fixes that Ours provides over No RS-MRQ.\n"
        )
        f.write(f"- Adaptive succeeds while Ours fails on {len(adaptive_vs_ours)} cases.\n")
        f.write(f"- Ours succeeds while Same-backbone Baseline fails on {len(ours_vs_same_backbone)} cases.\n\n")
        f.write("## Typical Hard Cases\n\n")
        if not typical_cases:
            f.write("No rescued hard cases were found.\n")
            return
        for _, no_rsmrq_add, adaptive_add, ours_add, record in typical_cases:
            f.write(
                "- "
                f"idx={record['idx']}, scene={record.get('scene_name')}, variant={record.get('scene_variant')}, "
                f"obj_id={record.get('obj_id')}, no_rsmrq_ADD={no_rsmrq_add:.6f}, "
                f"adaptive_ADD={adaptive_add:.6f}, ours_ADD={ours_add:.6f}, "
                f"upgrade={bool(record.get('adaptive_used_upgrade', False))}, "
                f"reason={record.get('adaptive_trigger_reason', '')}\n"
            )


def main() -> None:
    args = parse_args()

    adaptive_results = load_batch_json(args.adaptive_json)
    no_rsmrq_results = load_batch_json(args.no_rsmrq_json)
    ours_results = load_batch_json(args.ours_json)
    same_backbone_results = load_batch_json(args.same_backbone_json)

    adaptive_summary_rows = [
        summarize_method_row("No RS-MRQ", no_rsmrq_results),
        summarize_method_row("Adaptive Two-Stage", adaptive_results),
        summarize_method_row("Ours (Full)", ours_results),
    ]
    write_csv(
        TABLE_DIR / "adaptive_summary.csv",
        adaptive_summary_rows,
        [
            "Method",
            "Upgrade Ratio",
            "Mean ADD",
            "Mean ADD-S",
            "SR@ADD<=0.02",
            "Mean Rotation Error (deg)",
            "Mean Registration Time (s)",
            "Mean Front-end Time (s)",
            "Mean Back-end Time (s)",
        ],
    )
    plot_adaptive_summary(adaptive_summary_rows)

    cap_rows = [
        {
            "Cap": "64",
            "Mean candidate_inflation_mean": mean_of(
                load_batch_json(args.cap64_json),
                lambda r: coerce_float(r.get("stats", {}).get("candidate_inflation_mean")),
            ),
            "Mean ppf_frontend_time": mean_of(
                load_batch_json(args.cap64_json),
                lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
            ),
            "Mean registration_time": mean_of(
                load_batch_json(args.cap64_json),
                lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
            ),
            "Mean ADD": mean_of(
                load_batch_json(args.cap64_json),
                lambda r: coerce_float(r.get("metrics", {}).get("ADD")),
            ),
            "SR@ADD<=0.02": success_rate(load_batch_json(args.cap64_json)),
        },
        {
            "Cap": "128",
            "Mean candidate_inflation_mean": mean_of(
                load_batch_json(args.cap128_json),
                lambda r: coerce_float(r.get("stats", {}).get("candidate_inflation_mean")),
            ),
            "Mean ppf_frontend_time": mean_of(
                load_batch_json(args.cap128_json),
                lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
            ),
            "Mean registration_time": mean_of(
                load_batch_json(args.cap128_json),
                lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
            ),
            "Mean ADD": mean_of(
                load_batch_json(args.cap128_json),
                lambda r: coerce_float(r.get("metrics", {}).get("ADD")),
            ),
            "SR@ADD<=0.02": success_rate(load_batch_json(args.cap128_json)),
        },
        {
            "Cap": "256",
            "Mean candidate_inflation_mean": mean_of(
                load_batch_json(args.cap256_json),
                lambda r: coerce_float(r.get("stats", {}).get("candidate_inflation_mean")),
            ),
            "Mean ppf_frontend_time": mean_of(
                load_batch_json(args.cap256_json),
                lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
            ),
            "Mean registration_time": mean_of(
                load_batch_json(args.cap256_json),
                lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
            ),
            "Mean ADD": mean_of(
                load_batch_json(args.cap256_json),
                lambda r: coerce_float(r.get("metrics", {}).get("ADD")),
            ),
            "SR@ADD<=0.02": success_rate(load_batch_json(args.cap256_json)),
        },
        {
            "Cap": "None",
            "Mean candidate_inflation_mean": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("stats", {}).get("candidate_inflation_mean")),
            ),
            "Mean ppf_frontend_time": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
            ),
            "Mean registration_time": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
            ),
            "Mean ADD": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("metrics", {}).get("ADD")),
            ),
            "SR@ADD<=0.02": success_rate(ours_results),
        },
    ]
    write_csv(
        TABLE_DIR / "rsmrq_budget_sweep.csv",
        cap_rows,
        [
            "Cap",
            "Mean candidate_inflation_mean",
            "Mean ppf_frontend_time",
            "Mean registration_time",
            "Mean ADD",
            "SR@ADD<=0.02",
        ],
    )
    plot_tradeoff(
        [(row["Cap"], coerce_float(row["Mean registration_time"]), coerce_float(row["Mean ADD"])) for row in cap_rows],
        "Mean registration time (s)",
        "Mean ADD",
        "RS-MRQ Budget Trade-off",
        FIG_DIR / "rsmrq_budget_tradeoff.png",
    )

    rv20_results = load_batch_json(args.rv20_json)
    rv10_results = load_batch_json(args.rv10_json)
    rv_rows = [
        {
            "top_m_per_bucket": "50",
            "Mean ppf_frontend_time": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
            ),
            "Mean registration_time": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
            ),
            "Mean ADD": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("metrics", {}).get("ADD")),
            ),
            "SR@ADD<=0.02": success_rate(ours_results),
            "Mean candidate_inflation_mean": mean_of(
                ours_results,
                lambda r: coerce_float(r.get("stats", {}).get("candidate_inflation_mean")),
            ),
        },
        {
            "top_m_per_bucket": "20",
            "Mean ppf_frontend_time": mean_of(
                rv20_results,
                lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
            ),
            "Mean registration_time": mean_of(
                rv20_results,
                lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
            ),
            "Mean ADD": mean_of(
                rv20_results,
                lambda r: coerce_float(r.get("metrics", {}).get("ADD")),
            ),
            "SR@ADD<=0.02": success_rate(rv20_results),
            "Mean candidate_inflation_mean": mean_of(
                rv20_results,
                lambda r: coerce_float(r.get("stats", {}).get("candidate_inflation_mean")),
            ),
        },
        {
            "top_m_per_bucket": "10",
            "Mean ppf_frontend_time": mean_of(
                rv10_results,
                lambda r: coerce_float(r.get("stats", {}).get("ppf_frontend_time")),
            ),
            "Mean registration_time": mean_of(
                rv10_results,
                lambda r: coerce_float(r.get("stats", {}).get("registration_time")),
            ),
            "Mean ADD": mean_of(
                rv10_results,
                lambda r: coerce_float(r.get("metrics", {}).get("ADD")),
            ),
            "SR@ADD<=0.02": success_rate(rv10_results),
            "Mean candidate_inflation_mean": mean_of(
                rv10_results,
                lambda r: coerce_float(r.get("stats", {}).get("candidate_inflation_mean")),
            ),
        },
    ]
    write_csv(
        TABLE_DIR / "robust_vote_light_sweep.csv",
        rv_rows,
        [
            "top_m_per_bucket",
            "Mean ppf_frontend_time",
            "Mean registration_time",
            "Mean ADD",
            "SR@ADD<=0.02",
            "Mean candidate_inflation_mean",
        ],
    )
    plot_tradeoff(
        [(row["top_m_per_bucket"], coerce_float(row["Mean ppf_frontend_time"]), coerce_float(row["Mean ADD"])) for row in rv_rows],
        "Mean front-end time (s)",
        "Mean ADD",
        "Robust Vote Light Trade-off",
        FIG_DIR / "robust_vote_light_tradeoff.png",
    )

    adaptive_map = build_case_map(adaptive_results)
    no_rsmrq_map = build_case_map(no_rsmrq_results)
    ours_map = build_case_map(ours_results)
    same_backbone_map = build_case_map(same_backbone_results)

    adaptive_vs_no_rsmrq = compare_success(adaptive_map, no_rsmrq_map)
    adaptive_vs_ours = compare_success(adaptive_map, ours_map)
    ours_vs_no_rsmrq = compare_success(ours_map, no_rsmrq_map)
    ours_vs_same_backbone = compare_success(ours_map, same_backbone_map)

    save_hard_case_csv(TABLE_DIR / "adaptive_success_no_rsmrq_fail.csv", adaptive_vs_no_rsmrq)
    save_hard_case_csv(TABLE_DIR / "adaptive_success_ours_fail.csv", adaptive_vs_ours)
    save_hard_case_csv(TABLE_DIR / "ours_success_no_rsmrq_fail.csv", ours_vs_no_rsmrq)
    save_hard_case_csv(TABLE_DIR / "ours_success_same_backbone_fail.csv", ours_vs_same_backbone)
    write_hard_case_report(
        TABLE_DIR / "hard_case_report.md",
        adaptive_vs_no_rsmrq,
        adaptive_vs_ours,
        ours_vs_no_rsmrq,
        ours_vs_same_backbone,
        adaptive_map,
        no_rsmrq_map,
        ours_map,
    )


if __name__ == "__main__":
    main()
