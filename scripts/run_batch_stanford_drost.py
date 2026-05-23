import os
import sys
import json
import argparse
import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.append(ROOT)

from ppf.io import load_config
from ppf.utils import setup_logger, ensure_dir, save_json
from ppf.metrics import compute_metrics
from ppf.bop_gt import scene_dir_from_depth_path, try_get_bop_gt_pose
from ppf.drost_original import run_drost_registration
from scripts.run_batch_stanford import (
    resolve_repo_path,
    repo_rel_path,
    _is_valid,
    parse_xf_matrix,
    resolve_model_from_row,
    derive_cache_path,
    build_or_load_model_cache,
)
from scripts.true_process_runner import (
    default_process_start_method,
    run_tasks_with_true_processes,
)

G_CFG = None
G_RUN_DIR = None
G_LOGGER = None
G_INLIER_RADIUS = 5.0
G_USE_BOP_GT = False
G_T_SCALE = 1.0


def init_worker(cfg, run_dir, logger, inlier_radius, use_bop_gt, t_scale):
    global G_CFG, G_RUN_DIR, G_LOGGER, G_INLIER_RADIUS, G_USE_BOP_GT, G_T_SCALE
    G_CFG = cfg
    G_RUN_DIR = run_dir
    G_LOGGER = logger
    G_INLIER_RADIUS = float(inlier_radius)
    G_USE_BOP_GT = bool(use_bop_gt)
    G_T_SCALE = float(t_scale)


def _stats_to_dict(stats) -> dict:
    return {
        "model_build_time": float(stats.model_build_time),
        "registration_time": float(stats.registration_time),
        "total_time": float(stats.total_time),
        "scene_preprocess_time": float(getattr(stats, "scene_preprocess_time", 0.0)),
        "ppf_frontend_time": float(getattr(stats, "ppf_frontend_time", 0.0)),
        "pose_selection_time": float(getattr(stats, "pose_selection_time", 0.0)),
        "pose_clustering_time": float(getattr(stats, "pose_clustering_time", 0.0)),
        "legacy_clustering_time": float(getattr(stats, "legacy_clustering_time", 0.0)),
        "backend_time": float(getattr(stats, "backend_time", 0.0)),
        "candidate_inflation_mean": float(getattr(stats, "candidate_inflation_mean", 1.0)),
        "kde_refine_calls": int(getattr(stats, "kde_refine_calls", 0)),
        "final_pose_policy": str(getattr(stats, "final_pose_policy", "legacy_cluster")),
        "final_pose_path": str(getattr(stats, "final_pose_path", "")),
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
        model_cache_path = str(row["_model_cache_path"])

        T_gt = None
        gt_path = None
        obj_id_src = "csv"
        if G_USE_BOP_GT:
            for required_col in ["depth_path", "frame_id", "obj_token"]:
                if required_col not in row or not _is_valid(row.get(required_col)):
                    raise ValueError(f"--use_bop_gt requires column: {required_col}")
            depth_path = resolve_repo_path(str(row["depth_path"]))
            scene_dir = scene_dir_from_depth_path(depth_path)
            obj_id_gt, T_gt_tmp, err = try_get_bop_gt_pose(
                scene_dir,
                int(row["frame_id"]),
                int(row["obj_token"]),
                t_scale=float(G_T_SCALE),
            )
            if err is not None or obj_id_gt is None or T_gt_tmp is None:
                raise ValueError(f"BOP GT lookup failed: {err}")
            T_gt = T_gt_tmp
            obj_id = int(obj_id_gt)
            obj_id_src = "bop_gt"
        elif "gt_path" in row and _is_valid(row["gt_path"]):
            gt_path = resolve_repo_path(str(row["gt_path"]))
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"gt_path missing: {gt_path}")
            T_gt = parse_xf_matrix(gt_path)

        T_pred, out_model, debug, stats = run_drost_registration(
            model_path=model_path,
            scene_path=scene_path,
            cfg=G_CFG,
            logger=G_LOGGER,
            model_cache_path=model_cache_path,
            strict_cache=True,
        )

        model_pts = np.asarray(o3d.io.read_point_cloud(model_path).points, dtype=np.float64)
        scene_pts = np.asarray(o3d.io.read_point_cloud(scene_path).points, dtype=np.float64)
        metrics = compute_metrics(
            model_pts,
            scene_pts,
            T_pred,
            T_gt=T_gt,
            inlier_radius=float(G_INLIER_RADIUS),
        )

        rec = {
            "idx": int(idx),
            "method": "Drost original PPF",
            "obj_id": (None if obj_id is None else int(obj_id)),
            "obj_id_src": obj_id_src,
            "scene_id": (None if not _is_valid(row.get("scene_id")) else int(row["scene_id"])),
            "scene_name": row.get("scene_name"),
            "scene_variant": row.get("scene_variant"),
            "model_path": repo_rel_path(model_path),
            "model_path_src": model_src,
            "model_cache_path": repo_rel_path(model_cache_path),
            "scene_path": repo_rel_path(scene_path),
            "scene_path_raw": scene_path_raw,
            "gt_path": repo_rel_path(gt_path),
            "config_path": repo_rel_path(row.get("config_path")),
            "gt_threshold": row.get("gt_threshold"),
            "T_pred": T_pred.tolist(),
            "T_gt": (T_gt.tolist() if T_gt is not None else None),
            "stats": _stats_to_dict(stats),
            "metrics": metrics,
            "debug": debug,
            "row": {k: v for k, v in row.items() if not str(k).startswith("_")},
        }

        per_case_dir = os.path.join(G_RUN_DIR, "results", "per_case")
        ensure_dir(per_case_dir)
        scene_tag = row.get("scene_name") or f"idx{idx}"
        variant_tag = row.get("scene_variant", "na")
        per_case_file = os.path.join(per_case_dir, f"{idx:04d}_{scene_tag}_{variant_tag}.json")
        save_json(per_case_file, rec)
        return {"ok": True, "idx": int(idx), "record": rec}
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result


def main():
    ap = argparse.ArgumentParser(description="Stanford batch runner for the Drost original PPF reproduction.")
    ap.add_argument("--config", type=str, default="configs/Stanford/drost_original_stanford.yaml")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--models_dir", type=str, default="")
    ap.add_argument("--cache_dir", type=str, default="data/stanford_bunny_ppf/model_cache")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--out_prefix", type=str, default="stanford_drost_original")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--start_method", type=str, default=default_process_start_method())
    ap.add_argument("--inlier_radius", type=float, default=5.0)
    ap.add_argument("--use_bop_gt", action="store_true")
    ap.add_argument("--t_scale", type=float, default=1.0)
    args = ap.parse_args()

    config_path = resolve_repo_path(args.config)
    csv_path = resolve_repo_path(args.csv)
    models_dir = resolve_repo_path(args.models_dir) if str(args.models_dir).strip() else ""
    cache_dir = resolve_repo_path(args.cache_dir)

    cfg = load_config(config_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["output"]["results_dir"], f"{args.out_prefix}_{ts}")
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
    if bool(args.use_bop_gt):
        for c in ["obj_token", "frame_id", "depth_path"]:
            if c not in df.columns:
                raise ValueError(f"--use_bop_gt requires CSV column: {c}")

    rows = []
    unique_models = {}
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        model_path, model_src, obj_id = resolve_model_from_row(row_dict, models_dir)
        cache_path = derive_cache_path(model_path, cfg, cache_dir)
        row_dict["_resolved_model_path"] = model_path
        row_dict["_resolved_model_src"] = model_src
        row_dict["_resolved_obj_id"] = obj_id
        row_dict["_model_cache_path"] = cache_path
        rows.append({"idx": int(idx), "row": row_dict})
        unique_models[model_path] = cache_path

    logger.info(f"Config: {config_path}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Models dir: {models_dir or '[not used]'}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Method: Drost original PPF")
    logger.info(f"Total tasks: {len(rows)}")
    logger.info(f"Unique models: {len(unique_models)}")
    logger.info(f"Num workers: {int(args.num_workers)}")
    logger.info(f"Process start method: {args.start_method}")
    logger.info(f"Inlier radius: {float(args.inlier_radius)}")
    logger.info(f"Use BOP GT: {bool(args.use_bop_gt)}")
    logger.info(f"GT translation scale: {float(args.t_scale)}")

    built_cache_records = []
    for model_path, cache_path in tqdm(unique_models.items(), total=len(unique_models), desc="BuildCache", unit="model"):
        build_or_load_model_cache(model_path, cache_path, cfg, logger=logger, rebuild=bool(args.rebuild_cache))
        built_cache_records.append({"model_path": model_path, "cache_path": cache_path})

    results = []
    failures = []
    returned = run_tasks_with_true_processes(
        rows,
        int(args.num_workers),
        process_one,
        init_worker=init_worker,
        initargs=(cfg, run_dir, logger, args.inlier_radius, args.use_bop_gt, args.t_scale),
        desc="Processing",
        unit="task",
        start_method=args.start_method,
    )
    for ret in returned:
        if ret["ok"]:
            results.append(ret["record"])
        else:
            failures.append(ret)
            logger.error(f"[{ret['idx']}] failed: {ret.get('error', 'unknown error')}")
            if ret.get("traceback"):
                logger.error(ret["traceback"])

    results.sort(key=lambda x: x["idx"])
    out_json = os.path.join(results_dir, f"{args.out_prefix}_batch.json")
    save_json(
        out_json,
        {
            "method": "Drost original PPF",
            "config_path": repo_rel_path(config_path),
            "csv_path": repo_rel_path(csv_path),
            "models_dir": repo_rel_path(models_dir),
            "cache_dir": repo_rel_path(cache_dir),
            "caches": [
                {
                    "model_path": repo_rel_path(item["model_path"]),
                    "cache_path": repo_rel_path(item["cache_path"]),
                }
                for item in built_cache_records
            ],
            "results": results,
            "failure_count": len(failures),
            "inlier_radius": float(args.inlier_radius),
            "use_bop_gt": bool(args.use_bop_gt),
            "t_scale": float(args.t_scale),
        },
    )

    root_json = os.path.join(cfg["output"]["results_dir"], f"{args.out_prefix}_batch.json")
    save_json(
        root_json,
        {
            "method": "Drost original PPF",
            "config_path": repo_rel_path(config_path),
            "csv_path": repo_rel_path(csv_path),
            "models_dir": repo_rel_path(models_dir),
            "cache_dir": repo_rel_path(cache_dir),
            "caches": [
                {
                    "model_path": repo_rel_path(item["model_path"]),
                    "cache_path": repo_rel_path(item["cache_path"]),
                }
                for item in built_cache_records
            ],
            "results": results,
            "failure_count": len(failures),
            "inlier_radius": float(args.inlier_radius),
            "use_bop_gt": bool(args.use_bop_gt),
            "t_scale": float(args.t_scale),
        },
    )

    summary_json = os.path.join(run_dir, "summary.json")
    save_json(
        summary_json,
        {
            "method": "Drost original PPF",
            "config_path": repo_rel_path(config_path),
            "csv_path": repo_rel_path(csv_path),
            "models_dir": repo_rel_path(models_dir),
            "cache_dir": repo_rel_path(cache_dir),
            "total_tasks": len(rows),
            "unique_models": len(unique_models),
            "success_count": len(results),
            "failure_count": len(failures),
            "num_workers": int(args.num_workers),
            "start_method": args.start_method,
            "result_json": repo_rel_path(out_json),
            "log_path": repo_rel_path(log_path),
            "script": os.path.basename(__file__),
            "use_bop_gt": bool(args.use_bop_gt),
            "t_scale": float(args.t_scale),
        },
    )

    print(f"\n[DONE] Run directory: {run_dir}")
    print(f"[DONE] Batch JSON: {out_json}")
    print(f"[DONE] Root Batch JSON: {root_json}")
    print(f"[DONE] Log file: {log_path}")


if __name__ == "__main__":
    main()
