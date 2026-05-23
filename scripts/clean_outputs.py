from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    patterns = [
        "stanford_small_results.json",
        "stanford_small_log.txt",
        "lmo_small_results.json",
        "lmo_small_log.txt",
    ]
    for pattern in patterns:
        target = OUTPUTS / pattern
        if target.exists():
            target.unlink()


if __name__ == "__main__":
    main()
