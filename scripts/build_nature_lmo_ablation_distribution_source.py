import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "lmo"
OUTPUT_PATH = REPO_ROOT / "experiments" / "tables" / "nature_style" / "nature_lmo_ablation_distribution_source.csv"

METHOD_BATCHES = [
    ("Same-backbone Baseline", "lmo_same_backbone_batch.json"),
    ("No RS-MRQ", "lmo_no_rsmrq_batch.json"),
    ("No Robust Vote", "lmo_no_robust_vote_batch.json"),
    ("No Backend", "lmo_no_backend_batch.json"),
    ("Ours (Full)", "lmo_ours_full_batch.json"),
]

FIELDNAMES = [
    "dataset",
    "method",
    "idx",
    "obj_id",
    "scene_name",
    "ADD",
    "ADD_S",
    "rotation_error_deg",
    "registration_time",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for method_name, filename in METHOD_BATCHES:
        payload = load_json(RESULTS_DIR / filename)
        for rec in payload.get("results", []):
            metrics = rec.get("metrics") or {}
            stats = rec.get("stats") or {}
            rows.append(
                {
                    "dataset": "LMO",
                    "method": method_name,
                    "idx": rec.get("idx"),
                    "obj_id": rec.get("obj_id"),
                    "scene_name": rec.get("scene_name"),
                    "ADD": metrics.get("ADD"),
                    "ADD_S": metrics.get("ADD_S"),
                    "rotation_error_deg": metrics.get("rotation_error_deg"),
                    "registration_time": stats.get("registration_time"),
                }
            )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_csv(rows, OUTPUT_PATH)
    print(f"saved {OUTPUT_PATH}")
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()
