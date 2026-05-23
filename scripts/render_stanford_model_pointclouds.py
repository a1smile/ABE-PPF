import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLY_DTYPE_MAP = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}

VIEWS = [
    ("front", 0, -90),
    ("back", 0, 90),
    ("left", 0, 180),
    ("right", 0, 0),
    ("top", 90, -90),
    ("iso", 25, -60),
]

THEMES = {
    "black": {
        "figure_facecolor": "black",
        "axes_facecolor": "black",
        "title_color": "white",
        "cmap": "plasma",
    },
    "white": {
        "figure_facecolor": "white",
        "axes_facecolor": "white",
        "title_color": "#222222",
        "cmap": "viridis",
    },
}


def parse_ply_header(path: Path) -> tuple[int, list[tuple[str, str]], int]:
    vertex_count = None
    vertex_props: list[tuple[str, str]] = []
    in_vertex = False

    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} 缺少 end_header")
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded.startswith("format "):
                if decoded != "format binary_little_endian 1.0":
                    raise ValueError(f"{path} 不是 binary_little_endian PLY")
            elif decoded.startswith("element "):
                parts = decoded.split()
                if len(parts) != 3:
                    raise ValueError(f"{path} 的 element 头格式异常: {decoded}")
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif decoded.startswith("property ") and in_vertex:
                parts = decoded.split()
                if len(parts) != 3:
                    raise ValueError(f"{path} 的 vertex property 不受支持: {decoded}")
                vertex_props.append((parts[2], parts[1]))
            elif decoded == "end_header":
                data_offset = f.tell()
                break

    if vertex_count is None or not vertex_props:
        raise ValueError(f"{path} 未找到 vertex 定义")
    return vertex_count, vertex_props, data_offset


def load_ply_points(path: Path) -> np.ndarray:
    vertex_count, vertex_props, data_offset = parse_ply_header(path)
    dtype = np.dtype([(name, PLY_DTYPE_MAP[prop_type]) for name, prop_type in vertex_props])

    with path.open("rb") as f:
        f.seek(data_offset)
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)

    required = {"x", "y", "z"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"{path} 缺少 x/y/z 顶点坐标")

    points = np.column_stack(
        [
            vertices["x"].astype(np.float64, copy=False),
            vertices["y"].astype(np.float64, copy=False),
            vertices["z"].astype(np.float64, copy=False),
        ]
    )
    return points


def canonicalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    spans = rotated.max(axis=0) - rotated.min(axis=0)
    if spans[0] < spans[1]:
        rotated[:, [0, 1]] = rotated[:, [1, 0]]
    if np.mean(rotated[:, 2]) > 0:
        rotated[:, 2] *= -1
    return rotated


def sample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_point_cloud(
    vis: np.ndarray,
    out_path: Path,
    dpi: int,
    elev: float,
    azim: float,
    theme: dict[str, str],
) -> None:
    z = vis[:, 2]
    z_min = float(z.min())
    z_span = max(float(z.max() - z_min), 1e-9)
    colors = plt.get_cmap(theme["cmap"])((z - z_min) / z_span)

    fig = plt.figure(figsize=(6, 6), facecolor=theme["figure_facecolor"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(theme["axes_facecolor"])
    ax.scatter(
        vis[:, 0],
        vis[:, 1],
        vis[:, 2],
        c=colors,
        s=1.4 if vis.shape[0] < 35000 else 1.0,
        linewidths=0,
        alpha=0.95,
        depthshade=False,
    )
    ax.view_init(elev=elev, azim=azim)
    set_equal_3d(ax, vis)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.line.set_color((0, 0, 0, 0))

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Stanford model point clouds to 300 DPI images.")
    ap.add_argument(
        "--input_dir",
        default=Path("data/stanford_bunny_ppf/models"),
        type=Path,
    )
    ap.add_argument(
        "--output_dir",
        default=Path("data/stanford_bunny_ppf/models/visualizations_300dpi"),
        type=Path,
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--theme", choices=sorted(THEMES), default="black")
    args = ap.parse_args()
    theme = THEMES[args.theme]

    ply_files = sorted(args.input_dir.glob("*.ply"))
    if not ply_files:
        raise SystemExit(f"未在 {args.input_dir} 找到 .ply 文件")

    for ply_path in ply_files:
        points = load_ply_points(ply_path)
        vis = canonicalize_points(sample_points(points, max_points=40000, seed=0))
        for view_name, elev, azim in VIEWS:
            out_path = args.output_dir / f"{ply_path.stem}_{view_name}.png"
            render_point_cloud(
                vis,
                out_path,
                dpi=args.dpi,
                elev=elev,
                azim=azim,
                theme=theme,
            )
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
