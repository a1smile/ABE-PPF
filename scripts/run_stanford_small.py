from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets.stanford import StanfordSample, load_stanford_samples
from src.ppf.pipeline.ambiguity_budget_ppf import AmbiguityBudgetPPF, RuntimeSample
from src.utils.logger import setup_logger


def _to_runtime_sample(sample: StanfordSample) -> RuntimeSample:
    return RuntimeSample(
        dataset=sample.dataset,
        object_id=sample.object_id,
        object_name=sample.object_name,
        scene_id=sample.scene_id,
        scene_name=sample.scene_name,
        scene_variant=sample.scene_variant,
        model_path=sample.model_path,
        scene_path=sample.scene_path,
        gt_transform=sample.gt_transform,
        frame_id=sample.frame_id,
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _mean(cases: list[dict[str, Any]], getter) -> float:
    if not cases:
        return 0.0
    return float(sum(float(getter(case)) for case in cases) / float(len(cases)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "stanford_small.yaml"))
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_name = config_path.stem
    log_path = outputs_dir / f"{run_name}_log.txt"
    logger = setup_logger(str(log_path), name=run_name)
    logger.info("Running Stanford evaluation: %s", config_path.name)

    samples = load_stanford_samples(
        manifest_path=str(cfg["manifest_path"]),
        max_models=int(cfg.get("max_models", 2)),
        max_scenes_per_model=int(cfg.get("max_scenes_per_model", 2)),
        variants=list(cfg.get("variants", [])),
    )
    logger.info("Loaded %d Stanford samples", len(samples))

    pipeline = AmbiguityBudgetPPF(cfg, logger=logger)
    cases = []
    failures = []
    for sample in samples:
        logger.info("Case model=%s scene=%s variant=%s", sample.object_name, sample.scene_name, sample.scene_variant)
        result = pipeline.run(_to_runtime_sample(sample))
        cases.append(_sanitize(result))
        if not bool(result["stats"]["success"]):
            failures.append(
                {
                    "object_id": int(sample.object_id),
                    "scene_id": int(sample.scene_id),
                    "scene_variant": str(sample.scene_variant),
                }
            )

    summary = {
        "num_cases": len(cases),
        "num_success": sum(1 for case in cases if bool(case["stats"]["success"])),
        "success_rate": (sum(1 for case in cases if bool(case["stats"]["success"])) / max(1, len(cases))),
        "mean_runtime_total": _mean(cases, lambda case: case["stats"]["runtime_total"]),
        "mean_runtime_asps": _mean(cases, lambda case: case["stats"]["runtime_asps"]),
        "mean_runtime_ubsp": _mean(cases, lambda case: case["stats"]["runtime_ubsp"]),
        "mean_runtime_voting": _mean(cases, lambda case: case["stats"]["runtime_voting"]),
        "mean_runtime_brpmr": _mean(cases, lambda case: case["stats"]["runtime_brpmr"]),
        "mean_ADD": _mean(cases, lambda case: case["stats"]["ADD"]),
        "mean_ADD_S": _mean(cases, lambda case: case["stats"]["ADD_S"]),
        "mean_rotation_error": _mean(cases, lambda case: case["stats"]["rotation_error"]),
        "mean_translation_error": _mean(cases, lambda case: case["stats"]["translation_error"]),
        "failed_cases": failures,
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "summary": summary,
        "cases": cases,
    }
    result_path = outputs_dir / f"{run_name}_results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved Stanford results to %s", result_path)


if __name__ == "__main__":
    main()
