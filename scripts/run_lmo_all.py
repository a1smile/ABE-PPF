import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.build_lmo_suite_assets import (
    GROUPS,
    LMO_CONFIG_DIR,
    LMO_SUBSET_CSV,
    LMO_MODELS_DIR,
    METHOD_SPECS,
    RESULTS_DIR,
    TABLES_DIR,
    main as build_assets_main,
)


def _resolve_methods(group: str, methods_arg: str) -> List[str]:
    if methods_arg:
        out = [m.strip() for m in methods_arg.split(",") if m.strip()]
        unknown = [m for m in out if m not in METHOD_SPECS]
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}")
        return out
    if group == "all":
        return GROUPS["ablation"] + GROUPS["tradeoff"] + GROUPS["external"]
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    return list(GROUPS[group])


def _build_command(method_name: str, args) -> List[str]:
    spec = METHOD_SPECS[method_name]
    runner = REPO_ROOT / spec["runner_script"]
    config_path = LMO_CONFIG_DIR / spec["config"]
    cmd = [args.python, str(runner)]
    out_prefix = f"lmo_{method_name}"

    if spec["runner_kind"] == "adaptive":
        adaptive_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cmd.extend(
            [
                "--stage1_config",
                str(REPO_ROOT / adaptive_cfg["stage1_config"]),
                "--stage2_config",
                str(REPO_ROOT / adaptive_cfg["stage2_config"]),
                "--csv",
                str(args.csv),
                "--models_dir",
                str(args.models_dir),
                "--cache_dir",
                spec["cache_dir"],
                "--out_prefix",
                out_prefix,
                "--num_workers",
                str(args.num_workers),
                "--inlier_radius",
                str(args.inlier_radius),
                "--stage1_method_name",
                str(adaptive_cfg["stage1_method_name"]),
                "--stage2_method_name",
                str(adaptive_cfg["stage2_method_name"]),
                "--best_score_threshold",
                str(adaptive_cfg["adaptive_trigger_thresholds"]["best_score"]),
                "--top_score_margin_threshold",
                str(adaptive_cfg["adaptive_trigger_thresholds"]["top_score_margin"]),
                "--best_visibility_support_threshold",
                str(adaptive_cfg["adaptive_trigger_thresholds"]["best_visibility_support"]),
                "--use_bop_gt",
                "--t_scale",
                str(args.t_scale),
            ]
        )
    else:
        cmd.extend(
            [
                "--config",
                str(config_path),
                "--csv",
                str(args.csv),
                "--models_dir",
                str(args.models_dir),
                "--cache_dir",
                spec["cache_dir"],
                "--out_prefix",
                out_prefix,
                "--num_workers",
                str(args.num_workers),
                "--inlier_radius",
                str(args.inlier_radius),
                "--use_bop_gt",
                "--t_scale",
                str(args.t_scale),
            ]
        )
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if spec["rebuild_cache"]:
        cmd.append("--rebuild_cache")
    return cmd


def _find_latest_run_dir(out_prefix: str, start_ts: float) -> Path | None:
    candidates = []
    for path in RESULTS_DIR.glob(f"{out_prefix}_*"):
        if path.is_dir() and path.stat().st_mtime >= start_ts - 1.0:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["all", "ablation", "tradeoff", "external"], default="all")
    ap.add_argument("--methods", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--csv", default=str(LMO_SUBSET_CSV))
    ap.add_argument("--models_dir", default=str(LMO_MODELS_DIR))
    ap.add_argument("--t_scale", type=float, default=1.0)
    ap.add_argument("--inlier_radius", type=float, default=5.0)
    ap.add_argument("--manifest", default=str(RESULTS_DIR / "lmo_runs_manifest.json"))
    ap.add_argument("--report-mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--vis_cases_per_method", type=int, default=8)
    args = ap.parse_args()

    build_assets_main()

    selected_methods = _resolve_methods(args.group, args.methods)
    manifest_path = Path(args.manifest)
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "group": args.group,
        "methods": selected_methods,
        "dry_run": bool(args.dry_run),
        "limit": int(args.limit),
        "num_workers": int(args.num_workers),
        "csv": str(args.csv),
        "models_dir": str(args.models_dir),
        "runs": [],
    }

    for method_name in selected_methods:
        spec = METHOD_SPECS[method_name]
        out_prefix = f"lmo_{method_name}"
        cmd = _build_command(method_name, args)
        if args.dry_run:
            payload["runs"].append(
                {
                    "method_name": method_name,
                    "group_name": spec["group"],
                    "config_path": str(LMO_CONFIG_DIR / spec["config"]),
                    "out_prefix": out_prefix,
                    "command": cmd,
                    "status": "dry_run",
                }
            )
            continue

        start_time = datetime.now().isoformat(timespec="seconds")
        start_ts = time.time()
        completed = subprocess.run(cmd, cwd=REPO_ROOT)
        end_time = datetime.now().isoformat(timespec="seconds")
        end_ts = time.time()

        run_dir = _find_latest_run_dir(out_prefix, start_ts)
        result_json = None
        log_path = None
        if run_dir is not None:
            candidate_result = run_dir / "results" / f"{out_prefix}_batch.json"
            if candidate_result.exists():
                result_json = str(candidate_result)
            candidate_log = run_dir / "logs" / f"{out_prefix}.log"
            if candidate_log.exists():
                log_path = str(candidate_log)
        root_result_json = str(RESULTS_DIR / f"{out_prefix}_batch.json")
        if not Path(root_result_json).exists():
            root_result_json = None

        vis_dir = None
        if completed.returncode == 0 and root_result_json and int(args.vis_cases_per_method) > 0:
            vis_dir = str(RESULTS_DIR / "lmo_visualizations" / out_prefix)
            vis_cmd = [
                args.python,
                str(REPO_ROOT / "scripts" / "export_lmo_match_visualizations.py"),
                "--batch_json",
                root_result_json,
                "--output_dir",
                vis_dir,
                "--max_cases",
                str(int(args.vis_cases_per_method)),
            ]
            subprocess.run(vis_cmd, cwd=REPO_ROOT, check=False)

        payload["runs"].append(
            {
                "method_name": method_name,
                "group_name": spec["group"],
                "config_path": str(LMO_CONFIG_DIR / spec["config"]),
                "out_prefix": out_prefix,
                "log_path": log_path,
                "result_json": result_json,
                "root_result_json": root_result_json,
                "status": "success" if completed.returncode == 0 and root_result_json else "failed",
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": round(end_ts - start_ts, 3),
                "return_code": int(completed.returncode),
                "runner_script": spec["runner_script"],
                "rebuild_cache": bool(spec["rebuild_cache"]),
                "visualization_dir": vis_dir,
            }
        )
        _write_manifest(manifest_path, payload)

    _write_manifest(manifest_path, payload)

    if not args.dry_run:
        report_cmd = [
            args.python,
            str(REPO_ROOT / "scripts" / "build_lmo_run_reports.py"),
            "--manifest",
            str(manifest_path),
            "--out_dir",
            str(TABLES_DIR),
            "--mode",
            args.report_mode,
        ]
        subprocess.run(report_cmd, cwd=REPO_ROOT, check=False)


if __name__ == "__main__":
    main()
