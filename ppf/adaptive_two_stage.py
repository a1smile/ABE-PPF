from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_THRESHOLDS = {
    "best_score": 0.72,
    "top_score_margin": 0.02,
    "best_visibility_support": 0.28,
}


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _coerce_float_list(values: Any) -> List[float]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []

    out: List[float] = []
    for value in values:
        coerced = _coerce_float(value)
        if coerced is not None:
            out.append(coerced)
    return out


def should_escalate(
    debug: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    if isinstance(thresholds, dict):
        merged_thresholds.update(thresholds)

    pose_selection = debug.get("pose_selection") if isinstance(debug, dict) else None
    if not isinstance(pose_selection, dict):
        return True, "missing_pose_selection_debug"

    best_score = _coerce_float(pose_selection.get("best_score"))
    if best_score is None:
        return True, "missing_best_score"
    if best_score < float(merged_thresholds["best_score"]):
        return True, "low_best_score"

    top_scores = _coerce_float_list(pose_selection.get("top_scores"))
    if len(top_scores) < 2:
        return True, "missing_top_score_margin"
    if (top_scores[0] - top_scores[1]) < float(merged_thresholds["top_score_margin"]):
        return True, "small_top_score_margin"

    best_visibility_support = _coerce_float(pose_selection.get("best_visibility_support"))
    if best_visibility_support is None:
        return True, "missing_best_visibility_support"
    if best_visibility_support < float(merged_thresholds["best_visibility_support"]):
        return True, "low_visibility_support"

    return False, "no_trigger"
