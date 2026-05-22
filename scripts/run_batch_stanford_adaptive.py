import copy
import os
import sys
import json
import argparse
import logging
import traceback
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.append(ROOT)

from ppf.adaptive_two_stage import should_escalate
from ppf.io import load_config
from ppf.metrics import compute_metrics
from ppf.registration import run_registration
from ppf.utils import ensure_dir, save_json, setup_logger
from scripts.run_batch_stanford import (
    _is_valid,
    build_or_load_model_cache,
    derive_cache_path,
    parse_xf_matrix,
    resolve_model_from_row,
    resolve_repo_path,
)


G_STAGE1_CFG = None
G_STAGE2_CFG = None
G_RUN_DIR = None
G_LOGGER = None
G_INLIER_RADIUS = 5.0
G_TRIGGER_THRESHOLDS = None
G_STAGE1_METHOD = "no_rsmrq"
G_STAGE2_METHOD = "ours_full"


def init_worker(
    stage1_cfg,
    stage2_cfg,
    run_dir,
    logger,
    inlier_radius,
    trigger_thresholds,
    stage1_method,
    stage2_method,
):
    global G_STAGE1_CFG, G_STAGE2_CFG, G_RUN_DIR, G_LOGGER
    global G_INLIER_RADIUS, G_TRIGGER_THRESHOLDS, G_STAGE1_METHOD, G_STAGE2_METHOD
    G_STAGE1_CFG = stage1_cfg
    G_STAGE2_CFG = stage2_cfg
    G_RUN_DIR = run_dir
    G_LOGGER = logger
    G_INLIER_RADIUS = float(inlier_radius)
    G_TRIGGER_THRESHOLDS = dict(trigger_thresholds)
    G_STAGE1_METHOD = str(stage1_method)
    G_STAGE2_METHOD = str(stage2_method)


def _stats_to_dict(stats) -> Dict[str, float]:
    return {
        "model_build_time": float(getattr(stats, "model_build_time", 0.0)),
        "registration_time": float(getattr(stats, "registration_time", 0.0)),
        "total_time": float(getattr(stats, "total_time", 0.0)),
        "scene_preprocess_time": float(getattr(stats, "scene_preprocess_time", 0.0)),
        "ppf_frontend_time": float(getattr(stats, "ppf_frontend_time", 0.0)),
        "pose_selection_time": float(getattr(stats, "pose_selection_time", 0.0)),
        "pose_clustering_time": float(getattr(stats, "pose_clustering_time", 0.0)),
        "legacy_clustering_time": float(getattr(stats, "legacy_clustering_time", 0.0)),
        "backend_time": float(getattr(stats, "backend_time", 0.0)),
        "candidate_inflation_mean": float(getattr(stats, "candidate_inflation_mean", 1.0)),
        "kde_refine_calls": int(getattr(stats, "kde_refine_calls", 0)),
        "final_pose_policy": str(getattr(stats, "final_pose_policy", "auto")),
        "final_pose_path": str(getattr(stats, "final_pose_path", "")),
    }


def _empty_stats_dict() -> Dict[str, float]:
    return {
        "model_build_time": 0.0,
        "registration_time": 0.0,
        "total_time": 0.0,
        "scene_preprocess_time": 0.0,
        "ppf_frontend_time": 0.0,
        "pose_selection_time": 0.0,
        "pose_clustering_time": 0.0,
        "legacy_clustering_time": 0.0,
        "backend_time": 0.0,
        "candidate_inflation_mean": 0.0,
        "kde_refine_calls": 0,
        "final_pose_policy": "",
        "final_pose_path": "",
    }


def process_one(task):
    idx = task["idx"]
    row = task["row"]
    result = {"ok": False, "idx": int(idx), "error": "", "traceback": ""}

    try:
        scene_path_raw = str(row["pcd_path"])
        scene_path = resolve_repo_path(scene_path_raw)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"scene pcd missing: {scene_path} (raw={scene_path_raw})")

        model_path = str(row["_resolved_model_path"])
        model_src = str(row.get("_resolved_model_src", "csv"))
        obj_id = None if not _is_valid(row.get("_resolved_obj_id")) else int(row.get("_resolved_obj_id"))
        stage1_cache_path = str(row["_stage1_model_cache_path"])
        stage2_cache_path = str(row["_stage2_model_cache_path"])

        T_gt = None
        gt_path = None
        if "gt_path" in row and _is_valid(row["gt_path"]):
            gt_path = resolve_repo_path(str(row["gt_path"]))
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"gt_path missing: {gt_path}")
            T_gt = parse_xf_matrix(gt_path)

        T_stage1, _, debug_stage1, stats_stage1 = run_registration(
            model_path,
            scene_path,
            G_STAGE1_CFG,
            logger=G_LOGGER,
            model_cache_path=stage1_cache_path,
            strict_cache=True,
        )
        stage1_stats = _stats_to_dict(stats_stage1)
        used_upgrade, trigger_reason = should_escalate(debug_stage1, thresholds=G_TRIGGER_THRESHOLDS)

        debug_stage2 = None
        stage2_stats = _empty_stats_dict()
        T_final = T_stage1
        final_source = "stage1"
        final_debug_source = debug_stage1
        final_stats_source = stage1_stats

        if used_upgrade:
            T_stage2, _, debug_stage2, stats_stage2 = run_registration(
                model_path,
                scene_path,
                G_STAGE2_CFG,
                logger=G_LOGGER,
                model_cache_path=stage2_cache_path,
                strict_cache=True,
            )
            stage2_stats = _stats_to_dict(stats_stage2)
            T_final = T_stage2
            final_source = "stage2"
            final_debug_source = debug_stage2
            final_stats_source = stage2_stats

        adaptive_total_effective_time = (
            float(stage1_stats["registration_time"]) + float(stage2_stats["registration_time"])
        )
        adaptive_total_frontend_time = (
            float(stage1_stats["ppf_frontend_time"]) + float(stage2_stats["ppf_frontend_time"])
        )
        adaptive_total_backend_time = (
            float(stage1_stats["backend_time"]) + float(stage2_stats["backend_time"])
        )

        final_stats = dict(final_stats_source)
        final_stats["model_build_time"] = float(stage1_stats["model_build_time"]) + float(stage2_stats["model_build_time"])
        final_stats["registration_time"] = float(adaptive_total_effective_time)
        final_stats["total_time"] = float(stage1_stats["total_time"]) + float(stage2_stats["total_time"])
        final_stats["scene_preprocess_time"] = (
            float(stage1_stats["scene_preprocess_time"]) + float(stage2_stats["scene_preprocess_time"])
        )
        final_stats["ppf_frontend_time"] = float(adaptive_total_frontend_time)
        final_stats["pose_selection_time"] = (
            float(stage1_stats["pose_selection_time"]) + float(stage2_stats["pose_selection_time"])
        )
        final_stats["pose_clustering_time"] = (
            float(stage1_stats["pose_clustering_time"]) + float(stage2_stats["pose_clustering_time"])
        )
        final_stats["legacy_clustering_time"] = (
            float(stage1_stats["legacy_clustering_time"]) + float(stage2_stats["legacy_clustering_time"])
        )
        final_stats["backend_time"] = float(adaptive_total_backend_time)
        final_stats["kde_refine_calls"] = int(stage1_stats["kde_refine_calls"]) + int(stage2_stats["kde_refine_calls"])

        final_debug = copy.deepcopy(final_debug_source) if isinstance(final_debug_source, dict) else {}
        final_debug["stage1"] = debug_stage1
        final_debug["stage2"] = debug_stage2
        final_debug["adaptive"] = {
            "used_upgrade": bool(used_upgrade),
            "trigger_reason": str(trigger_reason),
            "stage1_method": G_STAGE1_METHOD,
            "stage2_method": G_STAGE2_METHOD,
            "final_source": final_source,
            "thresholds": dict(G_TRIGGER_THRESHOLDS),
        }

        model_pts = np.asarray(o3d.io.read_point_cloud(model_path).points, dtype=np.float64)
        scene_pts = np.asarray(o3d.io.read_point_cloud(scene_path).points, dtype=np.float64)
        metrics = compute_metrics(
            model_pts,
            scene_pts,
            T_final,
            T_gt=T_gt,
            inlier_radius=float(G_INLIER_RADIUS),
        )

        rec = {
            "idx": int(idx),
            "obj_id": (None if obj_id is None else int(obj_id)),
            "scene_id": (None if not _is_valid(row.get("scene_id")) else int(row["scene_id"])),
            "scene_name": row.get("scene_name"),
            "scene_variant": row.get("scene_variant"),
            "model_path": model_path,
            "model_path_src": model_src,
            "model_cache_path": stage2_cache_path if used_upgrade else stage1_cache_path,
            "scene_path": scene_path,
            "scene_path_raw": scene_path_raw,
            "gt_path": gt_path,
            "config_path": row.get("config_path"),
            "gt_threshold": row.get("gt_threshold"),
            "T_pred": T_final.tolist(),
            "T_gt": (T_gt.tolist() if T_gt is not None else None),
            "adaptive_used_upgrade": bool(used_upgrade),
            "adaptive_trigger_reason": str(trigger_reason),
            "adaptive_stage1_method": G_STAGE1_METHOD,
            "adaptive_stage2_method": G_STAGE2_METHOD,
            "adaptive_final_source": final_source,
            "adaptive_stage1_registration_time": float(stage1_stats["registration_time"]),
            "adaptive_stage2_registration_time": float(stage2_stats["registration_time"]),
            "adaptive_stage1_frontend_time": float(stage1_stats["ppf_frontend_time"]),
            "adaptive_stage2_frontend_time": float(stage2_stats["ppf_frontend_time"]),
            "adaptive_total_effective_time": float(adaptive_total_effective_time),
            "stats": final_stats,
            "metrics": metrics,
            "debug": final_debug,
            "row": {k: v for k, v in row.items() if not str(k).startswith("_")},
        }

        per_case_dir = os.path.join(G_RUN_DIR, "results", "per_case")
        ensure_dir(per_case_dir)
        scene_tag = row.get("scene_name") or f"idx{idx}"
        variant_tag = row.get("scene_variant", "na")
        per_case_file = os.path.join(per_case_dir, f"{idx:04d}_{scene_tag}_{variant_tag}.json")
        save_json(per_case_file, rec)

        return {"ok": True, "idx": int(idx), "record": rec}

    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return result


def main():
    ap = argparse.ArgumentParser(description="Adaptive two-stage batch runner for Stanford retrieval.")
    ap.add_argument("--stage1_config", type=str, default="configs/Stanford/ablation_no_rsmrq_stanford.yaml")
    ap.add_argument("--stage2_config", type=str, default="configs/Stanford/ablation_ours_stanford.yaml")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--models_dir", type=str, default="", help="Fallback model directory when CSV has no explicit model path.")
    ap.add_argument("--cache_dir", type=str, default="data/stanford_bunny_ppf/model_cache")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--out_prefix", type=str, default="stanford_adaptive_two_stage")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--inlier_radius", type=float, default=5.0)
    ap.add_argument("--stage1_method_name", type=str, default="no_rsmrq")
    ap.add_argument("--stage2_method_name", type=str, default="ours_full")
    ap.add_argument("--best_score_threshold", type=float, default=0.72)
    ap.add_argument("--top_score_margin_threshold", type=float, default=0.02)
    ap.add_argument("--best_visibility_support_threshold", type=float, default=0.28)
    args = ap.parse_args()

    stage1_config_path = resolve_repo_path(args.stage1_config)
    stage2_config_path = resolve_repo_path(args.stage2_config)
    csv_path = resolve_repo_path(args.csv)
    models_dir = resolve_repo_path(args.models_dir) if str(args.models_dir).strip() else ""
    cache_dir = resolve_repo_path(args.cache_dir)

    stage1_cfg = load_config(stage1_config_path)
    stage2_cfg = load_config(stage2_config_path)
    trigger_thresholds = {
        "best_score": float(args.best_score_threshold),
        "top_score_margin": float(args.top_score_margin_threshold),
        "best_visibility_support": float(args.best_visibility_support_threshold),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_root = stage2_cfg["output"]["results_dir"]
    run_dir = os.path.join(results_dir_root, f"{args.out_prefix}_{ts}")
    ensure_dir(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    results_dir = os.path.join(run_dir, "results")
    ensure_dir(logs_dir)
    ensure_dir(results_dir)

    log_path = os.path.join(logs_dir, f"{args.out_prefix}.log")
    logger = setup_logger(log_path, level=logging.INFO)

    df = pd.read_csv(csv_path)
    if args.limit > 0:
        df = df.head(args.limit)

    if "pcd_path" not in df.columns:
        raise ValueError("CSV missing required column: pcd_path")

    rows = []
    unique_stage_caches = {}
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        model_path, model_src, obj_id = resolve_model_from_row(row_dict, models_dir)
        stage1_cache_path = derive_cache_path(model_path, stage1_cfg, cache_dir)
        stage2_cache_path = derive_cache_path(model_path, stage2_cfg, cache_dir)
        row_dict["_resolved_model_path"] = model_path
        row_dict["_resolved_model_src"] = model_src
        row_dict["_resolved_obj_id"] = obj_id
        row_dict["_stage1_model_cache_path"] = stage1_cache_path
        row_dict["_stage2_model_cache_path"] = stage2_cache_path
        rows.append({"idx": int(idx), "row": row_dict})
        unique_stage_caches[("stage1", model_path, stage1_cache_path)] = stage1_cfg
        unique_stage_caches[("stage2", model_path, stage2_cache_path)] = stage2_cfg

    logger.info(f"Stage1 config: {stage1_config_path}")
    logger.info(f"Stage2 config: {stage2_config_path}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Models dir: {models_dir or '[not used]'}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Total tasks: {len(rows)}")
    logger.info(f"Unique model-stage caches: {len(unique_stage_caches)}")
    logger.info(f"Num workers: {int(args.num_workers)}")
    logger.info(f"Inlier radius: {float(args.inlier_radius)}")
    logger.info(f"Adaptive trigger thresholds: {json.dumps(trigger_thresholds, ensure_ascii=False)}")

    built_cache_records = []
    for (stage_name, model_path, cache_path), cfg in tqdm(
        unique_stage_caches.items(),
        total=len(unique_stage_caches),
        desc="BuildCache",
        unit="cache",
    ):
        build_or_load_model_cache(model_path, cache_path, cfg, logger=logger, rebuild=bool(args.rebuild_cache))
        built_cache_records.append(
            {
                "stage": stage_name,
                "model_path": model_path,
                "cache_path": cache_path,
            }
        )

    results = []
    failures = []
    if int(args.num_workers) <= 1:
        init_worker(
            stage1_cfg,
            stage2_cfg,
            run_dir,
            logger,
            args.inlier_radius,
            trigger_thresholds,
            args.stage1_method_name,
            args.stage2_method_name,
        )
        for task in tqdm(rows, total=len(rows), desc="Processing", unit="task"):
            ret = process_one(task)
            if ret["ok"]:
                results.append(ret["record"])
            else:
                failures.append(ret)
                logger.error(f"[{ret['idx']}] failed: {ret.get('error', 'unknown error')}")
                if ret.get("traceback"):
                    logger.error(ret["traceback"])
    else:
        with Pool(
            processes=int(args.num_workers),
            initializer=init_worker,
            initargs=(
                stage1_cfg,
                stage2_cfg,
                run_dir,
                logger,
                args.inlier_radius,
                trigger_thresholds,
                args.stage1_method_name,
                args.stage2_method_name,
            ),
        ) as pool:
            for ret in tqdm(pool.imap_unordered(process_one, rows), total=len(rows), desc="Processing", unit="task"):
                if ret["ok"]:
                    results.append(ret["record"])
                else:
                    failures.append(ret)
                    logger.error(f"[{ret['idx']}] failed: {ret.get('error', 'unknown error')}")
                    if ret.get("traceback"):
                        logger.error(ret["traceback"])

    results.sort(key=lambda item: item["idx"])

    out_json = os.path.join(results_dir, f"{args.out_prefix}_batch.json")
    save_json(
        out_json,
        {
            "stage1_config_path": stage1_config_path,
            "stage2_config_path": stage2_config_path,
            "csv_path": csv_path,
            "models_dir": models_dir,
            "cache_dir": cache_dir,
            "caches": built_cache_records,
            "results": results,
            "failure_count": len(failures),
            "inlier_radius": float(args.inlier_radius),
            "adaptive_trigger_thresholds": trigger_thresholds,
            "stage1_method_name": args.stage1_method_name,
            "stage2_method_name": args.stage2_method_name,
        },
    )

    summary_json = os.path.join(run_dir, "summary.json")
    save_json(
        summary_json,
        {
            "stage1_config_path": stage1_config_path,
            "stage2_config_path": stage2_config_path,
            "csv_path": csv_path,
            "models_dir": models_dir,
            "cache_dir": cache_dir,
            "total_tasks": len(rows),
            "unique_model_stage_caches": len(unique_stage_caches),
            "success_count": len(results),
            "failure_count": len(failures),
            "num_workers": int(args.num_workers),
            "result_json": out_json,
            "log_path": log_path,
            "script": os.path.basename(__file__),
            "adaptive_trigger_thresholds": trigger_thresholds,
        },
    )

    print(f"\n[DONE] Run directory: {run_dir}")
    print(f"[DONE] Batch JSON: {out_json}")
    print(f"[DONE] Log file: {log_path}")


if __name__ == "__main__":
    main()
