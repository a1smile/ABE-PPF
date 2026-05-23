import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[1]
LMO_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "LMO"
LMO_DATA_ROOT = REPO_ROOT / "data" / "LM-O (Linemod-Occluded)"
LMO_TEST_SCENE_DIR = LMO_DATA_ROOT / "lmo_test_all" / "test" / "000002"
LMO_FULL_SCENE_DIR = LMO_DATA_ROOT / "lmo_test_all" / "pcd_000002_rgb_strict"
OUTPUT_ROOT = REPO_ROOT / "visualization_outputs"

BASELINE_BATCH_PATH = LMO_RESULTS_DIR / "lmo_same_backbone_batch.json"
OURS_BATCH_PATH = LMO_RESULTS_DIR / "lmo_ours_full_batch.json"
SCENE_GT_INFO_PATH = LMO_TEST_SCENE_DIR / "scene_gt_info.json"

OBJECT_NAMES = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}

VIEW_SPECS = {
    "front": {"elev": 15, "azim": -90},
    "back": {"elev": 15, "azim": 90},
    "left": {"elev": 15, "azim": 180},
    "right": {"elev": 15, "azim": 0},
    "top": {"elev": 90, "azim": -90},
    "front_left_oblique": {"elev": 24, "azim": -135},
    "front_right_oblique": {"elev": 24, "azim": -45},
    "top_front_oblique": {"elev": 55, "azim": -90},
}

STYLE = {
    "scene": {"color": "#b8b8b8", "size": 0.35, "alpha": 0.23},
    "baseline": {"color": "#4da3ff", "size": 1.0, "alpha": 0.95},
    "ours": {"color": "#3bd16f", "size": 1.0, "alpha": 0.95},
    "gt": {"color": "#ff4f4f", "size": 1.0, "alpha": 0.95},
}

PNG_WIDTH = 1600
PNG_HEIGHT = 1200
PNG_DPI = 100
SCENE_SAMPLE_LIMIT = 45000
MODEL_SAMPLE_LIMIT = 12000
LOCAL_SCENE_POINT_MIN = 1200
LOCAL_SCENE_RADIUS_SCALE = 2.2
LOCAL_SCENE_RADIUS_GROWTH = 1.45
LOCAL_SCENE_RADIUS_MAX_SCALE = 6.0
VIEW_RADIUS_PAD = 1.05


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    frame_id: int
    obj_id: int
    category: str
    note: str


CASE_SPECS = [
    CaseSpec(
        case_id="lmo_case_001",
        frame_id=905,
        obj_id=8,
        category="baseline_fail_ours_success",
        note="Baseline drifts to a wrong pose, while ours recovers the driller geometry with a large ADD-S margin.",
    ),
    CaseSpec(
        case_id="lmo_case_002",
        frame_id=348,
        obj_id=6,
        category="baseline_fail_ours_success",
        note="Cat case with clear baseline misalignment and a compact, visually plausible ours alignment.",
    ),
    CaseSpec(
        case_id="lmo_case_003",
        frame_id=1062,
        obj_id=11,
        category="heavy_occlusion_ours_success",
        note="Glue under extremely low visibility; only a small visible fragment remains but ours stays within threshold.",
    ),
    CaseSpec(
        case_id="lmo_case_004",
        frame_id=318,
        obj_id=9,
        category="heavy_occlusion_ours_success",
        note="Duck with severe occlusion; ours remains successful despite only a small visible portion.",
    ),
    CaseSpec(
        case_id="lmo_case_005",
        frame_id=845,
        obj_id=12,
        category="ours_fail_or_high_error",
        note="Holepuncher case where baseline remains successful but ours flips into a visibly wrong pose.",
    ),
    CaseSpec(
        case_id="lmo_case_006",
        frame_id=55,
        obj_id=1,
        category="ours_fail_or_high_error",
        note="Ape case with moderate visibility where ours drifts beyond threshold while baseline stays correct.",
    ),
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_points(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if points.shape[0] <= limit:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=limit, replace=False)
    return points[idx]


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (homog @ T.T)[:, :3]


def _display_coords(points: np.ndarray) -> np.ndarray:
    # Re-map camera coordinates to a more intuitive display basis:
    # x -> right, y(depth) -> forward, z -> up.
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def _set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _load_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts


def _bbox_center(points: np.ndarray) -> np.ndarray:
    return (points.min(axis=0) + points.max(axis=0)) / 2.0


def _object_radius(points: np.ndarray) -> float:
    center = _bbox_center(points)
    return float(np.linalg.norm(points - center, axis=1).max())


def _extract_local_scene(
    scene_pts: np.ndarray,
    center: np.ndarray,
    object_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(object_radius * LOCAL_SCENE_RADIUS_SCALE, object_radius + 1e-6)
    max_radius = max(radius, object_radius * LOCAL_SCENE_RADIUS_MAX_SCALE)
    distances = np.linalg.norm(scene_pts - center, axis=1)
    local_scene = scene_pts[distances <= radius]

    while local_scene.shape[0] < LOCAL_SCENE_POINT_MIN and radius < max_radius:
        radius = min(radius * LOCAL_SCENE_RADIUS_GROWTH, max_radius)
        local_scene = scene_pts[distances <= radius]

    if local_scene.shape[0] == 0:
        nearest_count = min(max(LOCAL_SCENE_POINT_MIN, 4000), scene_pts.shape[0])
        nearest_idx = np.argpartition(distances, nearest_count - 1)[:nearest_count]
        local_scene = scene_pts[nearest_idx]
        radius = float(np.linalg.norm(local_scene - center, axis=1).max())

    local_scene = _sample_points(local_scene, SCENE_SAMPLE_LIMIT, seed=seed)
    view_radius = max(radius * VIEW_RADIUS_PAD, object_radius * 1.45)
    return local_scene, np.asarray([center - view_radius, center + view_radius], dtype=np.float64)


def _set_bounds(ax, bounds: np.ndarray) -> None:
    mins = bounds[0]
    maxs = bounds[1]
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])


def _build_result_map(records: Iterable[dict]) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for rec in records:
        key = (int(rec["row"]["frame_id"]), int(rec["obj_id"]))
        out[key] = rec
    return out


def _scene_tag(frame_id: int) -> str:
    return f"scene000002_frame{frame_id:06d}"


def _case_dir_name(case: CaseSpec) -> str:
    return f"{case.case_id}_{OBJECT_NAMES[case.obj_id]}_{_scene_tag(case.frame_id)}"


def _metrics_dict(rec: dict) -> dict:
    return rec.get("metrics") or {}


def _vis_info(scene_gt_info: dict, rec: dict) -> dict:
    frame_id = int(rec["row"]["frame_id"])
    token = int(rec["row"]["obj_token"])
    return scene_gt_info[str(frame_id)][token]


def _render_single_view(
    scene_pts: np.ndarray,
    model_pts: np.ndarray,
    bounds: np.ndarray,
    out_path: Path,
    color_key: str,
    view_name: str,
) -> None:
    fig = plt.figure(
        figsize=(PNG_WIDTH / PNG_DPI, PNG_HEIGHT / PNG_DPI),
        dpi=PNG_DPI,
        facecolor="black",
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d", facecolor="black")
    ax.set_proj_type("persp")
    ax.scatter(
        scene_pts[:, 0],
        scene_pts[:, 1],
        scene_pts[:, 2],
        s=STYLE["scene"]["size"],
        c=STYLE["scene"]["color"],
        alpha=STYLE["scene"]["alpha"],
        depthshade=False,
        linewidths=0,
    )
    ax.scatter(
        model_pts[:, 0],
        model_pts[:, 1],
        model_pts[:, 2],
        s=STYLE[color_key]["size"],
        c=STYLE[color_key]["color"],
        alpha=STYLE[color_key]["alpha"],
        depthshade=False,
        linewidths=0,
    )
    view = VIEW_SPECS[view_name]
    ax.view_init(elev=view["elev"], azim=view["azim"])
    _set_bounds(ax, bounds)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI, facecolor="black", edgecolor="black")
    plt.close(fig)


def main() -> None:
    baseline_payload = _load_json(BASELINE_BATCH_PATH)
    ours_payload = _load_json(OURS_BATCH_PATH)
    scene_gt_info = _load_json(SCENE_GT_INFO_PATH)

    baseline_map = _build_result_map(baseline_payload["results"])
    ours_map = _build_result_map(ours_payload["results"])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict] = []

    for case in CASE_SPECS:
        key = (case.frame_id, case.obj_id)
        if key not in baseline_map or key not in ours_map:
            raise KeyError(f"Missing case in results: {key}")

        baseline_rec = baseline_map[key]
        ours_rec = ours_map[key]
        gt_info = _vis_info(scene_gt_info, ours_rec)
        object_name = OBJECT_NAMES[case.obj_id]
        scene_tag = _scene_tag(case.frame_id)
        scene_id = f"000002/{case.frame_id:06d}"
        case_dir = OUTPUT_ROOT / _case_dir_name(case)

        scene_path = LMO_FULL_SCENE_DIR / f"{case.frame_id:06d}.ply"
        model_path = REPO_ROOT / ours_rec["model_path"]

        scene_pts_raw = _load_points(scene_path)
        model_pts_raw = _load_points(model_path)
        model_pts_vis = _sample_points(model_pts_raw, MODEL_SAMPLE_LIMIT, seed=case.obj_id * 1000 + case.frame_id)
        scene_pts_display = _display_coords(scene_pts_raw)
        model_radius = _object_radius(_display_coords(model_pts_vis))

        transforms = {
            "baseline": np.asarray(baseline_rec["T_pred"], dtype=np.float64),
            "ours": np.asarray(ours_rec["T_pred"], dtype=np.float64),
            "gt": np.asarray(ours_rec["T_gt"], dtype=np.float64),
        }

        transformed_vis = {}
        for method_name, T in transforms.items():
            transformed_vis[method_name] = _display_coords(_transform_points(model_pts_vis, T))

        for method_name in ("baseline", "ours", "gt"):
            method_dir = case_dir / method_name
            method_center = _bbox_center(transformed_vis[method_name])
            local_scene_pts, bounds = _extract_local_scene(
                scene_pts=scene_pts_display,
                center=method_center,
                object_radius=model_radius,
                seed=case.frame_id * 10 + case.obj_id,
            )
            for view_name in VIEW_SPECS:
                filename = f"{case.case_id}_{object_name}_{scene_tag}_{method_name}_{view_name}.png"
                _render_single_view(
                    scene_pts=local_scene_pts,
                    model_pts=transformed_vis[method_name],
                    bounds=bounds,
                    out_path=method_dir / filename,
                    color_key=method_name,
                    view_name=view_name,
                )

        baseline_metrics = _metrics_dict(baseline_rec)
        ours_metrics = _metrics_dict(ours_rec)
        meta = {
            "case_id": case.case_id,
            "object_name": object_name,
            "object_id": case.obj_id,
            "scene_id": scene_id,
            "frame_id": case.frame_id,
            "category": case.category,
            "baseline_method": "Same-backbone Baseline",
            "ours_method": "Ours (Full)",
            "gt_available": True,
            "baseline_adds": float(baseline_metrics.get("ADD_S", math.nan)),
            "ours_adds": float(ours_metrics.get("ADD_S", math.nan)),
            "baseline_add": float(baseline_metrics.get("ADD", math.nan)),
            "ours_add": float(ours_metrics.get("ADD", math.nan)),
            "baseline_re": float(baseline_metrics.get("rotation_error_deg", math.nan)),
            "ours_re": float(ours_metrics.get("rotation_error_deg", math.nan)),
            "baseline_te": float(baseline_metrics.get("translation_error", math.nan)),
            "ours_te": float(ours_metrics.get("translation_error", math.nan)),
            "visib_fract": float(gt_info.get("visib_fract", math.nan)),
            "visib_percent": round(float(gt_info.get("visib_fract", math.nan)) * 100.0, 2),
            "px_count_visib": int(gt_info.get("px_count_visib", 0)),
            "px_count_all": int(gt_info.get("px_count_all", 0)),
            "note": case.note,
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        case_rows.append(
            {
                "case_id": case.case_id,
                "object_name": object_name,
                "scene_id": scene_id,
                "frame_id": case.frame_id,
                "category": case.category,
                "baseline_adds": f"{meta['baseline_adds']:.6f}",
                "ours_adds": f"{meta['ours_adds']:.6f}",
                "baseline_re": f"{meta['baseline_re']:.6f}",
                "ours_re": f"{meta['ours_re']:.6f}",
                "baseline_te": f"{meta['baseline_te']:.6f}",
                "ours_te": f"{meta['ours_te']:.6f}",
                "visib_fract": f"{meta['visib_fract']:.6f}",
                "note": case.note,
            }
        )

    index_path = OUTPUT_ROOT / "case_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "object_name",
                "scene_id",
                "frame_id",
                "category",
                "baseline_adds",
                "ours_adds",
                "baseline_re",
                "ours_re",
                "baseline_te",
                "ours_te",
                "visib_fract",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(case_rows)

    print(f"saved cases to {OUTPUT_ROOT}")
    print(f"case index: {index_path}")


if __name__ == "__main__":
    main()
