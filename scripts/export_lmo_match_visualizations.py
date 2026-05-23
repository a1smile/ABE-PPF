import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.build_lmo_suite_assets import METHOD_SPECS


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_points(points: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= limit:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=limit, replace=False)
    return points[idx]


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (homog @ T.T)[:, :3]


def _set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _infer_method_name(payload: dict, batch_path: Path) -> str:
    top_method = str(payload.get("method", "") or "").strip()
    if top_method and top_method not in {"batch_method", "method"}:
        return top_method

    results = payload.get("results", []) or []
    if results:
        rec_method = str(results[0].get("method", "") or "").strip()
        if rec_method and rec_method not in {"batch_method", "method"}:
            return rec_method

    stem = batch_path.stem
    if stem.endswith("_batch"):
        stem = stem[: -len("_batch")]
    if stem.startswith("lmo_"):
        method_key = stem[len("lmo_") :]
        if method_key in METHOD_SPECS:
            return METHOD_SPECS[method_key]["label"]
    return stem


def _render_case(rec: dict, out_path: Path, method_name: str) -> None:
    scene_path = Path(rec["scene_path"])
    model_path = Path(rec["model_path"])
    T_pred = np.asarray(rec["T_pred"], dtype=np.float64)

    scene_pts = np.asarray(o3d.io.read_point_cloud(str(scene_path)).points, dtype=np.float64)
    model_pts = np.asarray(o3d.io.read_point_cloud(str(model_path)).points, dtype=np.float64)
    model_tf = _transform_points(model_pts, T_pred)

    scene_vis = _sample_points(scene_pts, 4000, seed=0)
    model_vis = _sample_points(model_tf, 2000, seed=1)
    all_pts = np.concatenate([scene_vis, model_vis], axis=0)

    fig = plt.figure(figsize=(10, 4.2))
    views = [(25, 45, "Perspective"), (90, -90, "Top View")]
    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(scene_vis[:, 0], scene_vis[:, 1], scene_vis[:, 2], s=0.6, c="#9aa3a8", alpha=0.45)
        ax.scatter(model_vis[:, 0], model_vis[:, 1], model_vis[:, 2], s=1.2, c="#d94841", alpha=0.85)
        ax.view_init(elev=elev, azim=azim)
        _set_equal_3d(ax, all_pts)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    frame_id = rec.get("row", {}).get("frame_id", rec.get("scene_variant", "na"))
    obj_id = rec.get("obj_id", "na")
    add = rec.get("metrics", {}).get("ADD")
    title = f"{method_name} | frame={frame_id} obj={obj_id}"
    if add is not None:
        title += f" | ADD={float(add):.4f}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_cases", type=int, default=1)
    args = ap.parse_args()

    batch_path = Path(args.batch_json)
    payload = _load_json(batch_path)
    results = payload.get("results", []) or []
    if not results:
        return
    method_name = _infer_method_name(payload, batch_path)

    ranked = sorted(
        results,
        key=lambda r: float((r.get("metrics", {}) or {}).get("ADD", 1e9)),
    )
    chosen = ranked[: max(1, int(args.max_cases))]
    out_dir = Path(args.output_dir)
    for i, rec in enumerate(chosen, start=1):
        frame_id = rec.get("row", {}).get("frame_id", rec.get("scene_variant", "na"))
        obj_id = rec.get("obj_id", "na")
        out_path = out_dir / f"{i:02d}_frame{frame_id}_obj{obj_id}.png"
        _render_case(rec, out_path, method_name)


if __name__ == "__main__":
    main()
