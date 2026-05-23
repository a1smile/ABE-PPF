import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
OUT_DIR = REPO_ROOT / "experiments" / "tables" / "stanford" / "external_ppf_feasibility"
CSV_OUT = OUT_DIR / "external_ppf_comparison.csv"
MD_OUT = OUT_DIR / "external_ppf_comparison.md"


METHOD_SPECS = [
    {
        "method": "Drost original PPF",
        "label": "Drost original PPF",
        "batch_json": RESULTS_DIR / "stanford_drost_original_batch.json",
        "notes": OUT_DIR / "drost_original_reproduction_notes.md",
    },
    {
        "method": "Birdal Revisited",
        "label": "Birdal Revisited",
        "batch_json": RESULTS_DIR / "stanford_birdal_revisited_batch.json",
        "notes": OUT_DIR / "birdal_revisited_reproduction_notes.md",
    },
    {
        "method": "Going Further with PPF",
        "label": "Going Further with PPF",
        "batch_json": RESULTS_DIR / "stanford_going_further_ppf_batch.json",
        "notes": OUT_DIR / "going_further_ppf_reproduction_notes.md",
    },
    {
        "method": "Edge-enhanced PPF",
        "label": "Edge-enhanced PPF",
        "batch_json": RESULTS_DIR / "stanford_edge_enhanced_ppf_batch.json",
        "notes": OUT_DIR / "edge_enhanced_ppf_reproduction_notes.md",
    },
    {
        "method": "Ours (Full)",
        "label": "Ours (Full)",
        "batch_json": RESULTS_DIR / "stanford_ours_full_batch.json",
        "notes": None,
    },
]


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return mean(vals)


def _metric(result: Dict[str, Any], *names: str) -> Optional[float]:
    metrics = result.get("metrics", {}) or {}
    stats = result.get("stats", {}) or {}
    debug = result.get("debug", {}) or {}
    for name in names:
        if name in metrics and metrics[name] is not None:
            return float(metrics[name])
        if name in stats and stats[name] is not None:
            return float(stats[name])
        if name in debug and debug[name] is not None:
            return float(debug[name])
    return None


def _success_add(result: Dict[str, Any], threshold: float = 0.02) -> Optional[float]:
    add = _metric(result, "ADD")
    if add is None:
        return None
    return 1.0 if add <= threshold else 0.0


def _infer_method_name(payload: Dict[str, Any], fallback: str) -> str:
    if payload.get("method"):
        return str(payload["method"])
    results = payload.get("results", []) or []
    if results and results[0].get("method"):
        return str(results[0]["method"])
    return fallback


def _aggregate_batch(batch_json: Path, fallback_method: str) -> Dict[str, Any]:
    payload = json.loads(batch_json.read_text(encoding="utf-8"))
    results = payload.get("results", []) or []
    if not results:
        raise ValueError(f"No results found in {batch_json}")

    row = {
        "Method": _infer_method_name(payload, fallback_method),
        "n_cases": len(results),
        "Mean ADD": _safe_mean(_metric(r, "ADD") for r in results),
        "Mean ADD-S": _safe_mean(_metric(r, "ADD-S", "ADD_S") for r in results),
        "SR@ADD<=0.02": _safe_mean(_success_add(r) for r in results),
        "Mean Rotation Error (deg)": _safe_mean(_metric(r, "rotation_error_deg") for r in results),
        "Mean Registration Time (s)": _safe_mean(_metric(r, "registration_time") for r in results),
        "Mean Front-end Time (s)": _safe_mean(_metric(r, "ppf_frontend_time") for r in results),
        "Mean Back-end Time (s)": _safe_mean(_metric(r, "backend_time") for r in results),
        "Candidate Inflation Mean": _safe_mean(_metric(r, "candidate_inflation_mean") for r in results),
    }
    return row


def _format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method",
        "n_cases",
        "Mean ADD",
        "Mean ADD-S",
        "SR@ADD<=0.02",
        "Mean Rotation Error (deg)",
        "Mean Registration Time (s)",
        "Mean Front-end Time (s)",
        "Mean Back-end Time (s)",
        "Candidate Inflation Mean",
    ]
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# External PPF Comparison on Stanford",
        "",
        "| Method | n_cases | Mean ADD | Mean ADD-S | SR@ADD<=0.02 | Mean Rotation Error (deg) | Mean Registration Time (s) | Mean Front-end Time (s) | Mean Back-end Time (s) | Candidate Inflation Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Method"]),
                    str(row["n_cases"]),
                    _format_float(row["Mean ADD"]),
                    _format_float(row["Mean ADD-S"]),
                    _format_float(row["SR@ADD<=0.02"]),
                    _format_float(row["Mean Rotation Error (deg)"]),
                    _format_float(row["Mean Registration Time (s)"]),
                    _format_float(row["Mean Front-end Time (s)"]),
                    _format_float(row["Mean Back-end Time (s)"]),
                    _format_float(row["Candidate Inflation Mean"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All values are re-aggregated from the root batch JSON files in `experiments/results/`.",
            "- `Ours (Full)` is included as the reference method for direct comparison.",
            "- External-method reproduction notes remain in the same directory for method-specific approximation details.",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: List[Dict[str, Any]] = []
    for spec in METHOD_SPECS:
        row = _aggregate_batch(spec["batch_json"], spec["method"])
        row["Method"] = spec["label"]
        rows.append(row)
    _write_csv(rows)
    _write_md(rows)
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
