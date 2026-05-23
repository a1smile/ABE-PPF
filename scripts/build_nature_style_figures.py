from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "paper_outputs" / "analysis" / "master_summary.xlsx"
STANFORD_RESULTS_DIR = ROOT / "experiments" / "results" / "Stanford"
LMO_RESULTS_DIR = ROOT / "experiments" / "results" / "lmo"
OUT_DIR = ROOT / "experiments" / "figures" / "nature_style"
TABLE_OUT_DIR = ROOT / "experiments" / "tables" / "nature_style"
LMO_PER_OBJECT_TABLE_PATH = TABLE_OUT_DIR / "nature_lmo_per_object_table.csv"
PNG_DPI = 360

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
    "Drost",
    "Going Further",
    "Birdal Revisited",
    "Edge-enhanced PPF",
]
METHOD_RANK = {m: i for i, m in enumerate(METHOD_ORDER)}
COLORS = {
    "Same-backbone Baseline": "#9E9E9E",
    "No RS-MRQ": "#4C78A8",
    "No Robust Vote": "#F58518",
    "No Backend": "#54A24B",
    "Ours (Full)": "#E45756",
    "Adaptive Two-Stage": "#72B7B2",
    "Ours + RV20": "#2F8F9D",
    "Ours + RV10": "#56B4B9",
    "Ours + RS-MRQ cap64": "#8C6D5A",
    "Ours + RS-MRQ cap128": "#C9A227",
    "Ours + RS-MRQ cap256": "#6F9E5C",
    "Drost": "#B279A2",
    "Going Further": "#6B6ECF",
    "Birdal Revisited": "#B5BD22",
    "Edge-enhanced PPF": "#17A2B8",
}
MARKERS = {
    "Same-backbone Baseline": "o",
    "No RS-MRQ": "o",
    "No Robust Vote": "o",
    "No Backend": "o",
    "Ours (Full)": "o",
    "Adaptive Two-Stage": "s",
    "Ours + RV20": "^",
    "Ours + RV10": "v",
    "Ours + RS-MRQ cap64": "D",
    "Ours + RS-MRQ cap128": "D",
    "Ours + RS-MRQ cap256": "D",
    "Drost": "P",
    "Going Further": "X",
    "Birdal Revisited": "P",
    "Edge-enhanced PPF": "X",
}

EXTERNAL_METHODS = ["Drost", "Birdal Revisited", "Going Further", "Edge-enhanced PPF", "Ours (Full)"]
EXTERNAL_BATCHES = {
    "Stanford": {
        "Drost": STANFORD_RESULTS_DIR / "stanford_drost_original_batch.json",
        "Birdal Revisited": STANFORD_RESULTS_DIR / "stanford_birdal_revisited_batch.json",
        "Going Further": STANFORD_RESULTS_DIR / "stanford_going_further_ppf_batch.json",
        "Edge-enhanced PPF": STANFORD_RESULTS_DIR / "stanford_edge_enhanced_ppf_batch.json",
        "Ours (Full)": STANFORD_RESULTS_DIR / "stanford_ours_full_batch.json",
    },
    "LMO": {
        "Drost": LMO_RESULTS_DIR / "lmo_drost_batch.json",
        "Birdal Revisited": LMO_RESULTS_DIR / "lmo_birdal_batch.json",
        "Going Further": LMO_RESULTS_DIR / "lmo_going_further_batch.json",
        "Edge-enhanced PPF": LMO_RESULTS_DIR / "lmo_edge_enhanced_ppf_batch.json",
        "Ours (Full)": LMO_RESULTS_DIR / "lmo_ours_full_batch.json",
    },
}

STANFORD_OFFSETS = {
    "Same-backbone Baseline": (1.2, 0.012, "left"),
    "No RS-MRQ": (1.0, 0.012, "left"),
    "No Robust Vote": (1.0, 0.010, "left"),
    "Ours + RV10": (-2.2, -0.014, "right"),
    "Adaptive Two-Stage": (-1.8, 0.012, "right"),
    "Ours + RV20": (-1.8, -0.014, "right"),
    "Ours (Full)": (1.2, 0.012, "left"),
    "No Backend": (1.0, -0.014, "left"),
    "Going Further": (-1.5, 0.018, "right"),
    "Edge-enhanced PPF": (-1.6, -0.018, "right"),
}

LMO_OFFSETS = {
    "Same-backbone Baseline": (0.10, -0.018, "left"),
    "No RS-MRQ": (0.10, 0.012, "left"),
    "No Robust Vote": (0.10, -0.016, "left"),
    "Ours + RV10": (-0.14, -0.018, "right"),
    "Ours + RV20": (-0.12, 0.016, "right"),
    "Edge-enhanced PPF": (0.12, -0.018, "left"),
    "Ours (Full)": (0.12, -0.004, "left"),
    "No Backend": (0.12, 0.016, "left"),
    "Adaptive Two-Stage": (0.10, -0.018, "left"),
    "Going Further": (-0.16, 0.016, "right"),
}


def ensure_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "savefig.bbox": "tight",
        }
    )


def load_summary() -> pd.DataFrame:
    df = pd.read_excel(SUMMARY_PATH)
    return df.sort_values(["dataset", "method"], key=lambda s: s.map(METHOD_RANK).fillna(999) if s.name == "method" else s)


def load_external_per_case() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, method_paths in EXTERNAL_BATCHES.items():
        for method, path in method_paths.items():
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for rec in payload.get("results", []):
                metrics = rec.get("metrics", {}) or {}
                stats = rec.get("stats", {}) or {}
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "idx": rec.get("idx"),
                        "obj_id": rec.get("obj_id"),
                        "scene_name": rec.get("scene_name"),
                        "ADD": metrics.get("ADD", np.nan),
                        "ADD_S": metrics.get("ADD_S", np.nan),
                        "rotation_error_deg": metrics.get("rotation_error_deg", np.nan),
                        "registration_time": stats.get("registration_time", np.nan),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_OUT_DIR / "nature_external_distribution_source.csv", index=False)
    return df


def load_lmo_per_object_table() -> pd.DataFrame:
    df = pd.read_csv(LMO_PER_OBJECT_TABLE_PATH)
    df = df[df["Object"].astype(str).str.lower() != "mean"].copy()
    for col in ["Baseline SR", "Ours SR", "Delta SR"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.rstrip("%")
            .astype(float)
        )
    return df


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def annotate_points(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    offsets: dict[str, tuple[float, float, str]],
) -> None:
    for _, row in sub.iterrows():
        method = row["method"]
        xi = float(row[x_col])
        yi = float(row[y_col])
        dx, dy, ha = offsets.get(method, (0.0, 0.0, "left"))
        xt = xi + dx
        yt = yi + dy
        ax.annotate(
            short_label(method),
            xy=(xi, yi),
            xytext=(xt, yt),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "color": "#7A7A7A",
                "lw": 0.6,
                "shrinkA": 4,
                "shrinkB": 4,
            }
            if dx or dy
            else None,
        )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both")


def short_label(method: str) -> str:
    mapping = {
        "Same-backbone Baseline": "Baseline",
        "No RS-MRQ": "No RS-MRQ",
        "No Robust Vote": "No RV",
        "No Backend": "No Backend",
        "Ours (Full)": "Ours",
        "Adaptive Two-Stage": "Adaptive",
        "Ours + RV20": "RV20",
        "Ours + RV10": "RV10",
        "Ours + RS-MRQ cap64": "cap64",
        "Ours + RS-MRQ cap128": "cap128",
        "Ours + RS-MRQ cap256": "cap256",
        "Going Further": "Going Further",
        "Birdal Revisited": "Birdal",
        "Edge-enhanced PPF": "Edge PPF",
    }
    return mapping.get(method, method)


def plot_stanford_tradeoff(df: pd.DataFrame) -> None:
    data = df[(df["dataset"] == "Stanford") & (df["method"].isin(METHOD_ORDER))]
    methods = [
        "Same-backbone Baseline",
        "No RS-MRQ",
        "No Robust Vote",
        "No Backend",
        "Ours (Full)",
        "Adaptive Two-Stage",
        "Ours + RV20",
        "Ours + RV10",
        "Going Further",
        "Edge-enhanced PPF",
    ]
    sub = data.set_index("method").loc[methods].reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for _, row in sub.iterrows():
        method = row["method"]
        ax.scatter(
            row["mean_registration_time"],
            row["SR@ADD<=0.02"],
            s=88 if method == "Ours (Full)" else 70,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolor="black" if method == "Ours (Full)" else "white",
            linewidth=0.9,
            zorder=3,
        )
    annotate_points(ax, sub, "mean_registration_time", "SR@ADD<=0.02", STANFORD_OFFSETS)
    ax.set_xlabel("Mean registration time (s)")
    ax.set_ylabel("SR@ADD<=0.02")
    ax.set_title("Stanford accuracy-time trade-off")
    ax.set_xlim(0, 74)
    ax.set_ylim(0.62, 0.95)
    style_axes(ax)
    save_figure(fig, "nature_stanford_tradeoff")


def plot_lmo_tradeoff(df: pd.DataFrame) -> None:
    methods = [
        "Same-backbone Baseline",
        "No RS-MRQ",
        "No Robust Vote",
        "No Backend",
        "Ours (Full)",
        "Adaptive Two-Stage",
        "Ours + RV20",
        "Ours + RV10",
        "Going Further",
        "Edge-enhanced PPF",
    ]
    sub = df[df["dataset"] == "LMO"].set_index("method").loc[methods].reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for _, row in sub.iterrows():
        method = row["method"]
        ax.scatter(
            row["mean_registration_time"],
            row["primary_accuracy"],
            s=88 if method == "Ours (Full)" else 70,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolor="black" if method == "Ours (Full)" else "white",
            linewidth=0.9,
            zorder=3,
        )
    annotate_points(ax, sub, "mean_registration_time", "primary_accuracy", LMO_OFFSETS)
    ax.set_xlabel("Mean registration time (s)")
    ax.set_ylabel("SR@ADD-S<=20 mm")
    ax.set_title("LMO accuracy-time trade-off")
    ax.set_xlim(0.4, 9.4)
    ax.set_ylim(0.54, 0.94)
    style_axes(ax)
    save_figure(fig, "nature_lmo_tradeoff")


def plot_timing_decomposition(df: pd.DataFrame) -> None:
    methods = ["No RS-MRQ", "No Robust Vote", "Ours (Full)", "Adaptive Two-Stage", "Going Further", "Edge-enhanced PPF"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=False)
    for ax, dataset in zip(axes, ["Stanford", "LMO"], strict=True):
        sub = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].set_index("method").loc[methods].reset_index()
        x = np.arange(len(sub))
        scene = sub["mean_scene_preprocess_time"].to_numpy(dtype=float)
        front = sub["mean_frontend_time"].to_numpy(dtype=float)
        back = sub["mean_backend_time"].to_numpy(dtype=float)
        ax.bar(x, scene, color="#D9D9D9", width=0.7, label="Scene prep")
        ax.bar(x, front, bottom=scene, color="#F28E2B", width=0.7, label="Front-end")
        ax.bar(x, back, bottom=scene + front, color="#72B7B2", width=0.7, label="Back-end")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(m) for m in sub["method"]], rotation=24, ha="right")
        ax.set_title(dataset, fontsize=10)
        ax.set_ylabel("Mean time (s)")
        style_axes(ax)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.78, wspace=0.28)
    save_figure(fig, "nature_timing_decomposition")


def plot_external_vs_ours(df: pd.DataFrame) -> None:
    methods = ["Drost", "Birdal Revisited", "Going Further", "Edge-enhanced PPF", "Ours (Full)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, dataset, ycol, ylabel in [
        (axes[0], "Stanford", "SR@ADD<=0.02", "SR@ADD<=0.02"),
        (axes[1], "LMO", "primary_accuracy", "SR@ADD-S<=20 mm"),
    ]:
        sub = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].set_index("method").loc[methods].reset_index()
        x = np.arange(len(sub))
        colors = [COLORS[m] for m in sub["method"]]
        ax.bar(x, sub[ycol].to_numpy(dtype=float), color=colors, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(m) for m in sub["method"]], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(dataset, fontsize=10)
        style_axes(ax)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    save_figure(fig, "nature_external_vs_ours")


def _distribution_values(df: pd.DataFrame, dataset: str, method: str, col: str, *, positive: bool = False) -> np.ndarray:
    values = df[(df["dataset"] == dataset) & (df["method"] == method)][col].dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if positive:
        values = np.clip(values, 1e-6, None)
    return values


def plot_external_adds_boxplot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=False)
    for ax, dataset in zip(axes, ["Stanford", "LMO"], strict=True):
        values = [_distribution_values(df, dataset, method, "ADD_S", positive=True) for method in EXTERNAL_METHODS]
        bp = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.62,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8, "color": "#555555"},
            capprops={"linewidth": 0.8, "color": "#555555"},
        )
        for patch, method in zip(bp["boxes"], EXTERNAL_METHODS, strict=True):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.78)
            patch.set_edgecolor("#333333")
        ax.set_xticks(np.arange(1, len(EXTERNAL_METHODS) + 1))
        ax.set_xticklabels([short_label(method) for method in EXTERNAL_METHODS], rotation=24, ha="right")
        ax.set_ylabel("ADD-S")
        ax.set_title(dataset, fontsize=10)
        ax.set_yscale("log")
        style_axes(ax)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    save_figure(fig, "nature_external_adds_boxplot")


def plot_external_stanford_add_lmo_adds_boxplot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=False)
    panel_specs = [
        (axes[0], "Stanford", "ADD", "ADD", "Stanford"),
        (axes[1], "LMO", "ADD_S", "ADD-S", "LMO"),
    ]
    for ax, dataset, metric, ylabel, title in panel_specs:
        values = [_distribution_values(df, dataset, method, metric, positive=True) for method in EXTERNAL_METHODS]
        bp = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.62,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8, "color": "#555555"},
            capprops={"linewidth": 0.8, "color": "#555555"},
        )
        for patch, method in zip(bp["boxes"], EXTERNAL_METHODS, strict=True):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.78)
            patch.set_edgecolor("#333333")
        ax.set_xticks(np.arange(1, len(EXTERNAL_METHODS) + 1))
        ax.set_xticklabels([short_label(method) for method in EXTERNAL_METHODS], rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        style_axes(ax)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    save_figure(fig, "nature_external_stanford_add_lmo_adds_boxplot")


def plot_external_time_violin(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=False)
    for ax, dataset in zip(axes, ["Stanford", "LMO"], strict=True):
        values = [_distribution_values(df, dataset, method, "registration_time") for method in EXTERNAL_METHODS]
        parts = ax.violinplot(values, showmeans=False, showmedians=True, showextrema=False, widths=0.78)
        for body, method in zip(parts["bodies"], EXTERNAL_METHODS, strict=True):
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("#333333")
            body.set_alpha(0.72)
            body.set_linewidth(0.8)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#222222")
            parts["cmedians"].set_linewidth(1.2)
        medians = [np.median(v) if len(v) else np.nan for v in values]
        ax.scatter(
            np.arange(1, len(EXTERNAL_METHODS) + 1),
            medians,
            s=14,
            color="#222222",
            zorder=3,
            label="Median",
        )
        ax.set_xticks(np.arange(1, len(EXTERNAL_METHODS) + 1))
        ax.set_xticklabels([short_label(method) for method in EXTERNAL_METHODS], rotation=24, ha="right")
        ax.set_ylabel("Registration time (s)")
        ax.set_title(dataset, fontsize=10)
        style_axes(ax)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    save_figure(fig, "nature_external_time_violin")


def plot_lmo_per_object_sr_comparison(df: pd.DataFrame) -> None:
    objects = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]
    sub = df.set_index("Object").loc[objects].reset_index()
    x = np.arange(len(sub))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    baseline_color = "#8F99A8"
    ours_color = COLORS["Ours (Full)"]
    delta_color = "#2E8B57"

    ax.bar(
        x - width / 2,
        sub["Baseline SR"].to_numpy(dtype=float),
        width=width,
        color=baseline_color,
        edgecolor="white",
        linewidth=0.8,
        label="Baseline SR",
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        sub["Ours SR"].to_numpy(dtype=float),
        width=width,
        color=ours_color,
        edgecolor="white",
        linewidth=0.8,
        label="Ours SR",
        zorder=2,
    )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        sub["Delta SR"].to_numpy(dtype=float),
        color=delta_color,
        marker="o",
        markersize=4.8,
        linewidth=1.4,
        label="ΔSR",
        zorder=4,
    )

    for xi, delta in zip(x, sub["Delta SR"].to_numpy(dtype=float), strict=True):
        ax2.annotate(
            f"+{delta:.1f}%",
            xy=(xi, delta),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=delta_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.08},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(objects, rotation=20, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax2.set_ylabel("ΔSR (%)")
    ax.set_ylim(0, 105)
    ax2.set_ylim(0, max(55, float(sub["Delta SR"].max()) + 5.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    style_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.grid(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.subplots_adjust(top=0.82, bottom=0.24)
    save_figure(fig, "nature_lmo_per_object_sr_comparison")


def main() -> None:
    ensure_dir()
    apply_style()
    df = load_summary()
    external_per_case = load_external_per_case()
    lmo_per_object = load_lmo_per_object_table()
    plot_stanford_tradeoff(df)
    plot_lmo_tradeoff(df)
    plot_timing_decomposition(df)
    plot_external_vs_ours(df)
    plot_external_adds_boxplot(external_per_case)
    plot_external_stanford_add_lmo_adds_boxplot(external_per_case)
    plot_external_time_violin(external_per_case)
    plot_lmo_per_object_sr_comparison(lmo_per_object)


if __name__ == "__main__":
    main()
