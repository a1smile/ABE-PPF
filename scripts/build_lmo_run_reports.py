import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.build_lmo_suite_assets import GROUPS, METHOD_SPECS, TABLES_DIR


RESULTS_DIR = REPO_ROOT / "experiments" / "results"

METHOD_ALIAS = {
    "no_backend": {
        "label": "No Backend",
        "group": "ablation",
    },
    "strict_no_mode_cluster": {
        "label": "No Backend",
        "group": "ablation",
    },
}


def _resolve_spec(method_name: str) -> Dict[str, Any]:
    if method_name in METHOD_SPECS:
        return METHOD_SPECS[method_name]
    alias = METHOD_ALIAS.get(method_name)
    if alias is not None:
        return alias
    raise KeyError(method_name)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return mean(vals)


def _extract_metric(rec: Dict[str, Any], *keys: str) -> Optional[float]:
    for container_name in ("metrics", "stats", "debug"):
        container = rec.get(container_name, {}) or {}
        for key in keys:
            if key in container and container[key] is not None:
                return float(container[key])
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_batch(batch_path: Path) -> Dict[str, Any]:
    payload = _load_json(batch_path)
    results = payload.get("results", []) or []
    return {
        "results": results,
        "success_count": len(results),
        "failure_count": int(payload.get("failure_count", 0)),
        "mean_add": _safe_mean(_extract_metric(r, "ADD") for r in results),
        "mean_add_s": _safe_mean(_extract_metric(r, "ADD_S", "ADD-S") for r in results),
        "sr_add": _safe_mean((1.0 if (_extract_metric(r, "ADD") or 1e9) <= 0.02 else 0.0) for r in results),
        "mean_rot": _safe_mean(_extract_metric(r, "rotation_error_deg") for r in results),
        "mean_reg": _safe_mean(_extract_metric(r, "registration_time") for r in results),
    }


def build_initial_summary(manifest: Dict[str, Any], out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for rec in manifest.get("runs", []):
        method_name = rec["method_name"]
        spec = _resolve_spec(method_name)
        result_json = rec.get("root_result_json") or rec.get("result_json")
        summary = None
        if result_json and Path(result_json).exists():
            summary = _summarize_batch(Path(result_json))
        rows.append(
            {
                "Method": spec["label"],
                "Group": spec["group"],
                "success_count": (summary or {}).get("success_count", 0),
                "failure_count": (summary or {}).get("failure_count", 0 if rec.get("status") == "success" else 1),
                "Mean ADD": (summary or {}).get("mean_add"),
                "Mean ADD-S": (summary or {}).get("mean_add_s"),
                "SR@ADD<=0.02": (summary or {}).get("sr_add"),
                "Mean Rotation Error": (summary or {}).get("mean_rot"),
                "Mean Registration Time": (summary or {}).get("mean_reg"),
            }
        )

    fieldnames = [
        "Method",
        "Group",
        "success_count",
        "failure_count",
        "Mean ADD",
        "Mean ADD-S",
        "SR@ADD<=0.02",
        "Mean Rotation Error",
        "Mean Registration Time",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "lmo_initial_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    group_to_file = {
        "ablation": out_dir / "lmo_ablation_table.csv",
        "tradeoff": out_dir / "lmo_tradeoff_table.csv",
        "external": out_dir / "lmo_external_table.csv",
    }
    for group_name, path in group_to_file.items():
        if group_name == "external":
            group_rows = [
                row
                for row in rows
                if row["Group"] == "external" or row["Method"] == METHOD_SPECS["ours_full"]["label"]
            ]
        else:
            group_rows = [row for row in rows if row["Group"] == group_name]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_rows)


def build_smoke_report(manifest: Dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# LMO Smoke Test",
        "",
        "| Method | Group | Status | Valid Batch JSON | Has T_pred | Result Count | Mean Registration Time | Notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for rec in manifest.get("runs", []):
        method_name = rec["method_name"]
        spec = _resolve_spec(method_name)
        root_json = rec.get("root_result_json") or rec.get("result_json")
        batch_ok = False
        has_t_pred = False
        result_count = 0
        mean_reg = None
        notes = ""
        if root_json and Path(root_json).exists():
            try:
                summary = _summarize_batch(Path(root_json))
                batch_ok = True
                result_count = int(summary["success_count"])
                mean_reg = summary["mean_reg"]
                payload = _load_json(Path(root_json))
                results = payload.get("results", []) or []
                has_t_pred = bool(results) and all(("T_pred" in r and r["T_pred"] is not None) for r in results)
                if not results:
                    notes = "empty results"
            except Exception as exc:
                notes = f"invalid batch json: {exc}"
        else:
            notes = rec.get("error_message", "missing result json")

        lines.append(
            "| "
            + " | ".join(
                [
                    spec["label"],
                    spec["group"],
                    rec.get("status", "unknown"),
                    "Yes" if batch_ok else "No",
                    "Yes" if has_t_pred else "No",
                    str(result_count),
                    ("" if mean_reg is None else f"{mean_reg:.6f}"),
                    notes.replace("|", "/"),
                ]
            )
            + " |"
        )
    (out_dir / "lmo_smoke_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default=str(TABLES_DIR))
    ap.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = ap.parse_args()

    manifest = _load_json(Path(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        build_smoke_report(manifest, out_dir)
    else:
        build_initial_summary(manifest, out_dir)


if __name__ == "__main__":
    main()
