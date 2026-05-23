from __future__ import annotations

from pathlib import Path

import pandas as pd

from nature_table_utils import export_table_bundle


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "paper_outputs" / "analysis" / "master_summary.xlsx"
OUTPUT_DIR = ROOT / "experiments" / "tables" / "nature_style"

COMMON_COLUMNS = [
    {"header": "Method", "align": "left"},
    {"header": "N", "align": "right"},
    {"header": "ADD (mm)", "align": "right"},
    {"header": "ADD-S (mm)", "align": "right"},
    {"header": "RE (deg)", "align": "right"},
    {"header": "TE (mm)", "align": "right"},
    {"header": "Time (s)", "align": "right"},
]

TABLE_SPECS = [
    {
        "dataset": "Stanford",
        "methods": [
            "Same-backbone Baseline",
            "No RS-MRQ",
            "No Robust Vote",
            "No Backend",
            "Ours (Full)",
        ],
        "output": "nature_stanford_ablation_table",
        "caption": "Stanford ablation results. Success rate is reported as SR@ADD<=0.02 m (20 mm); all distance metrics are reported in millimetres.",
        "label": "tab:stanford-ablation",
    },
    {
        "dataset": "Stanford",
        "methods": [
            "Drost",
            "Birdal Revisited",
            "Edge-enhanced PPF",
            "Going Further",
            "Ours (Full)",
        ],
        "output": "nature_stanford_external_table",
        "caption": "Stanford comparison against external baselines. Success rate is reported as SR@ADD<=0.02 m (20 mm); all distance metrics are reported in millimetres.",
        "label": "tab:stanford-external",
    },
    {
        "dataset": "LMO",
        "methods": [
            "Same-backbone Baseline",
            "No RS-MRQ",
            "No Robust Vote",
            "No Backend",
            "Ours (Full)",
        ],
        "output": "nature_lmo_ablation_table",
        "caption": "LM-O ablation results. Success rate uses ADD-S <= 20 mm; distance metrics are reported in millimetres.",
        "label": "tab:lmo-ablation",
    },
    {
        "dataset": "LMO",
        "methods": [
            "Drost",
            "Birdal Revisited",
            "Edge-enhanced PPF",
            "Going Further",
            "Ours (Full)",
        ],
        "output": "nature_lmo_external_table",
        "caption": "LM-O comparison against external baselines. Success rate uses ADD-S <= 20 mm; distance metrics are reported in millimetres.",
        "label": "tab:lmo-external",
    },
]


def fmt_int(value: object, _: dict) -> str:
    return f"{int(float(value))}"


def fmt_percent(value: object, _: dict) -> str:
    return f"{100.0 * float(value):.1f}%"


def fmt_float3(value: object, _: dict) -> str:
    return f"{float(value):.3f}"


def to_mm(value: float, dataset: str) -> float:
    return value * 1000.0 if dataset == "Stanford" else value


def sr_header(dataset: str) -> str:
    if dataset == "Stanford":
        return "SR@ADD<=0.02 m (20 mm)"
    return "SR@ADD-S<=20 mm"


def load_summary() -> pd.DataFrame:
    df = pd.read_excel(SUMMARY_PATH)
    return df


def build_rows(df: pd.DataFrame, dataset: str, methods: list[str]) -> list[dict[str, object]]:
    subset = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].copy()
    subset["method"] = pd.Categorical(subset["method"], categories=methods, ordered=True)
    subset = subset.sort_values("method")

    rows: list[dict[str, object]] = []
    for _, rec in subset.iterrows():
        rows.append(
            {
                "Method": str(rec["method"]),
                "N": int(rec["n_cases"]),
                "SR": float(rec["primary_accuracy"]),
                "ADD (mm)": to_mm(float(rec["mean_ADD"]), dataset),
                "ADD-S (mm)": to_mm(float(rec["mean_ADD_S"]), dataset),
                "RE (deg)": float(rec["mean_rotation_error_deg"]),
                "TE (mm)": to_mm(float(rec["mean_translation_error"]), dataset),
                "Time (s)": float(rec["mean_registration_time"]),
            }
        )
    return rows


def build_manifest(path: Path) -> None:
    text = """# Nature-Style Table Manifest

## Publication-ready tables

- `config_parameter_table_main.{csv,md,tex}`
- `config_parameter_table_variants.{csv,md,tex}`
- `nature_stanford_ablation_table.{csv,md,tex}`
- `nature_stanford_external_table.{csv,md,tex}`
- `nature_lmo_ablation_table.{csv,md,tex}`
- `nature_lmo_external_table.{csv,md,tex}`
- `nature_lmo_per_object_table.{csv,md,tex}`

## Source data kept for figure/table regeneration

- `nature_external_distribution_source.csv`
- `nature_lmo_distribution_source.csv`
- `nature_lmo_ablation_distribution_source.csv`
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_summary()

    for spec in TABLE_SPECS:
        rows = build_rows(df, spec["dataset"], spec["methods"])
        columns = [
            {"header": "Method", "align": "left"},
            {"header": "N", "align": "right", "format": fmt_int},
            {"header": sr_header(spec["dataset"]), "key": "SR", "align": "right", "format": fmt_percent},
        ]
        for col in COMMON_COLUMNS[2:]:
            column = dict(col)
            column["format"] = fmt_float3
            columns.append(column)

        export_table_bundle(
            OUTPUT_DIR / spec["output"],
            rows,
            columns,
            caption=spec["caption"],
            label=spec["label"],
            bold_row_predicate=lambda row: row.get("Method") == "Ours (Full)",
            col_spec="@{}lrrrrrrr@{}",
        )
        print(f"saved {OUTPUT_DIR / (spec['output'] + '.csv')}")

    build_manifest(OUTPUT_DIR / "nature_table_manifest.md")
    print(f"saved {OUTPUT_DIR / 'nature_table_manifest.md'}")


if __name__ == "__main__":
    main()
