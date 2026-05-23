from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.ppf.core.pose_hypothesis import PoseCandidate
from src.ppf.core.transform_utils import rotation_distance_deg, translation_distance


@dataclass
class ModePoolConfig:
    tau_t: float = 0.02
    tau_R: float = 10.0
    tau_t_merge: float = 0.02
    tau_R_merge: float = 10.0
    tau_margin: float = 0.25
    tau_entropy: float = 0.8
    tau_compactness: float = 0.6
    tau_coverage: float = 0.3
    tau_marginal_gain: float = 1e-4
    trend_window: int = 2
    tau_margin_growth: float = 0.03
    tau_coverage_growth: float = 0.02
    recent_history_size: int = 5
    min_rounds_before_stop: int = 2
    min_support_points_for_stop: int = 0
    max_modes: int = 0
    min_mode_score_ratio: float = 0.0
    enable_early_stop: bool = True


@dataclass
class ModeState:
    representative_pose: np.ndarray
    score: float
    candidate_count: int
    translation_variance: float
    rotation_variance: float
    score_sum: float = 0.0
    best_single_score: float = 0.0
    support_points: Set[int] = field(default_factory=set)
    visible_coverage: float = 0.0
    ambiguity_penalty: float = 0.0
    recent_score_history: List[float] = field(default_factory=list)
    poses: List[np.ndarray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    dirty: bool = False

    @classmethod
    def from_candidate(cls, candidate: PoseCandidate) -> "ModeState":
        pose = np.asarray(candidate.transform, dtype=np.float64)
        return cls(
            representative_pose=pose,
            score=float(candidate.score),
            candidate_count=1,
            translation_variance=0.0,
            rotation_variance=0.0,
            score_sum=float(candidate.score),
            best_single_score=float(candidate.score),
            support_points=set(candidate.reference_indices),
            visible_coverage=0.0,
            ambiguity_penalty=float(candidate.ambiguity_penalty),
            recent_score_history=[float(candidate.score)],
            poses=[pose],
            scores=[float(candidate.score)],
            dirty=False,
        )

    def add_candidate(self, candidate: PoseCandidate, total_reference_points: int) -> None:
        pose = np.asarray(candidate.transform, dtype=np.float64)
        candidate_score = float(candidate.score)
        self.poses.append(pose)
        self.scores.append(candidate_score)
        self.support_points.update(candidate.reference_indices)
        self.ambiguity_penalty = float(
            (self.ambiguity_penalty * float(self.candidate_count) + float(candidate.ambiguity_penalty))
            / float(self.candidate_count + 1)
        )
        self.candidate_count += 1
        self.score_sum += candidate_score
        if candidate_score >= float(self.best_single_score):
            self.best_single_score = candidate_score
            self.representative_pose = pose.copy()
        self.score = float(self.score_sum * (1.0 / (1.0 + self.ambiguity_penalty)))
        self.visible_coverage = float(len(self.support_points)) / max(1, int(total_reference_points))
        self.dirty = True

    def refresh_statistics(self, total_reference_points: int, recent_history_size: int) -> None:
        if not self.poses:
            return

        best_index = int(np.argmax(np.asarray(self.scores, dtype=np.float64)))
        self.representative_pose = self.poses[best_index].copy()
        self.best_single_score = float(self.scores[best_index])
        self.score_sum = float(np.sum(np.asarray(self.scores, dtype=np.float64)))
        self.score = float(self.score_sum * (1.0 / (1.0 + self.ambiguity_penalty)))

        translations = np.stack([pose_item[:3, 3] for pose_item in self.poses], axis=0)
        translation_center = np.mean(translations, axis=0)
        translation_deltas = np.linalg.norm(translations - translation_center[None, :], axis=1)
        self.translation_variance = float(np.var(translation_deltas)) if translation_deltas.size else 0.0

        rep_rotation = self.representative_pose[:3, :3]
        rotation_deltas = np.array(
            [rotation_distance_deg(rep_rotation, pose_item[:3, :3]) for pose_item in self.poses],
            dtype=np.float64,
        )
        self.rotation_variance = float(np.var(rotation_deltas)) if rotation_deltas.size else 0.0

        self.visible_coverage = float(len(self.support_points)) / max(1, int(total_reference_points))
        self.recent_score_history.append(float(self.score))
        self.recent_score_history = self.recent_score_history[-int(recent_history_size) :]
        self.dirty = False

    def compactness(self, tau_t: float, tau_R: float) -> float:
        translation_term = math.sqrt(max(0.0, self.translation_variance)) / max(tau_t, 1e-6)
        rotation_term = math.sqrt(max(0.0, self.rotation_variance)) / max(tau_R, 1e-6)
        return float(1.0 / (1.0 + translation_term + rotation_term))


class ModePool:
    def __init__(self, cfg: ModePoolConfig, total_reference_points: int):
        self.cfg = cfg
        self.total_reference_points = max(1, int(total_reference_points))
        self.modes: List[ModeState] = []
        self.round_index = 0
        self.last_metrics: Dict[str, float] = {}
        self.metric_history: List[Dict[str, float]] = []
        self.last_round_debug: Dict[str, float] = {}

    def add_candidates(
        self,
        candidates: List[PoseCandidate],
        max_modes_override: Optional[int] = None,
        min_mode_score_ratio_override: Optional[float] = None,
    ) -> None:
        self.round_index += 1
        tau_t = float(self.cfg.tau_t)
        tau_R = float(self.cfg.tau_R)
        effective_max_modes = self._effective_max_modes(max_modes_override)
        effective_min_mode_score_ratio = self._effective_min_mode_score_ratio(min_mode_score_ratio_override)
        pre_prune_mode_count = len(self.modes)
        self._prune_modes(effective_max_modes, effective_min_mode_score_ratio)
        pre_assignment_pruned_modes = max(0, pre_prune_mode_count - len(self.modes))
        overflow_dropped_candidates = 0
        overflow_replaced_modes = 0
        for candidate in candidates:
            pose = np.asarray(candidate.transform, dtype=np.float64)
            pose_translation = pose[:3, 3]
            pose_rotation = pose[:3, :3]
            assigned_mode: Optional[ModeState] = None
            best_distance = (float("inf"), float("inf"))
            for mode in self.modes:
                dt = translation_distance(mode.representative_pose[:3, 3], pose_translation)
                if dt > tau_t:
                    continue
                dr = rotation_distance_deg(mode.representative_pose[:3, :3], pose_rotation)
                if dr > tau_R:
                    continue
                if (dt, dr) < best_distance:
                    assigned_mode = mode
                    best_distance = (dt, dr)

            if assigned_mode is None:
                if effective_max_modes > 0 and len(self.modes) >= effective_max_modes:
                    if self._replace_weakest_mode_if_better(candidate):
                        overflow_replaced_modes += 1
                    else:
                        overflow_dropped_candidates += 1
                    continue
                mode = ModeState.from_candidate(candidate)
                mode.visible_coverage = float(len(mode.support_points)) / float(self.total_reference_points)
                self.modes.append(mode)
            else:
                assigned_mode.add_candidate(candidate, self.total_reference_points)

        self._merge_modes()
        self._prune_modes(effective_max_modes, effective_min_mode_score_ratio)
        self._refresh_modes()
        base_metrics = self._base_metrics()
        history_limit = max(int(self.cfg.recent_history_size), max(2, int(self.cfg.trend_window)))
        self.metric_history.append(dict(base_metrics))
        self.metric_history = self.metric_history[-history_limit:]
        self.last_metrics = self._augment_with_trends(base_metrics, self.metric_history)
        self.last_round_debug = {
            "effective_max_modes": float(effective_max_modes),
            "effective_min_mode_score_ratio": float(effective_min_mode_score_ratio),
            "pre_assignment_pruned_modes": float(pre_assignment_pruned_modes),
            "overflow_dropped_candidates": float(overflow_dropped_candidates),
            "overflow_replaced_modes": float(overflow_replaced_modes),
            "mode_count_after_round": float(len(self.modes)),
        }

    def _merge_modes(self) -> None:
        changed = True
        tau_t_merge = float(self.cfg.tau_t_merge)
        tau_R_merge = float(self.cfg.tau_R_merge)
        while changed:
            changed = False
            for left_index in range(len(self.modes)):
                if changed:
                    break
                for right_index in range(left_index + 1, len(self.modes)):
                    dt = translation_distance(
                        self.modes[left_index].representative_pose[:3, 3],
                        self.modes[right_index].representative_pose[:3, 3],
                    )
                    if dt > tau_t_merge:
                        continue
                    dr = rotation_distance_deg(
                        self.modes[left_index].representative_pose[:3, :3],
                        self.modes[right_index].representative_pose[:3, :3],
                    )
                    if dr > tau_R_merge:
                        continue

                    left = self.modes[left_index]
                    right = self.modes[right_index]
                    for pose, score in zip(right.poses, right.scores):
                        left.poses.append(np.asarray(pose, dtype=np.float64))
                        left.scores.append(float(score))
                    left.support_points.update(right.support_points)
                    total_count = max(1, left.candidate_count + right.candidate_count)
                    left.ambiguity_penalty = float(
                        (
                            left.ambiguity_penalty * float(left.candidate_count)
                            + right.ambiguity_penalty * float(right.candidate_count)
                        )
                        / float(total_count)
                    )
                    left.candidate_count += right.candidate_count
                    left.score_sum += float(right.score_sum)
                    if float(right.best_single_score) >= float(left.best_single_score):
                        left.best_single_score = float(right.best_single_score)
                        left.representative_pose = right.representative_pose.copy()
                    left.score = float(left.score_sum * (1.0 / (1.0 + left.ambiguity_penalty)))
                    left.visible_coverage = float(len(left.support_points)) / float(self.total_reference_points)
                    left.recent_score_history.extend(right.recent_score_history)
                    left.recent_score_history = left.recent_score_history[-int(self.cfg.recent_history_size) :]
                    left.dirty = True

                    del self.modes[right_index]
                    changed = True
                    break

        self.modes.sort(key=lambda mode: mode.score, reverse=True)

    def _refresh_modes(self) -> None:
        for mode in self.modes:
            if mode.dirty:
                mode.refresh_statistics(self.total_reference_points, self.cfg.recent_history_size)

    def _replace_weakest_mode_if_better(self, candidate: PoseCandidate) -> bool:
        if not self.modes:
            return False

        weakest_index = min(range(len(self.modes)), key=lambda index: self.modes[index].score)
        weakest_mode = self.modes[weakest_index]
        if float(candidate.score) <= float(weakest_mode.score):
            return False

        replacement = ModeState.from_candidate(candidate)
        replacement.visible_coverage = float(len(replacement.support_points)) / float(self.total_reference_points)
        self.modes[weakest_index] = replacement
        return True

    def _effective_max_modes(self, override: Optional[int]) -> int:
        if override is None:
            return int(self.cfg.max_modes)
        return max(0, int(override))

    def _effective_min_mode_score_ratio(self, override: Optional[float]) -> float:
        if override is None:
            return float(self.cfg.min_mode_score_ratio)
        return max(0.0, float(override))

    def _prune_modes(
        self,
        max_modes_override: Optional[int] = None,
        min_mode_score_ratio_override: Optional[float] = None,
    ) -> None:
        if not self.modes:
            return

        self.modes.sort(key=lambda mode: mode.score, reverse=True)
        best_score = max(1e-12, float(self.modes[0].score))
        min_ratio = self._effective_min_mode_score_ratio(min_mode_score_ratio_override)
        if min_ratio > 0.0:
            kept: List[ModeState] = []
            for mode in self.modes:
                if mode.score >= best_score * min_ratio or mode.candidate_count > 1:
                    kept.append(mode)
            self.modes = kept or self.modes[:1]

        max_modes = self._effective_max_modes(max_modes_override)
        if max_modes > 0 and len(self.modes) > max_modes:
            self.modes = self.modes[:max_modes]

    def best_mode(self) -> Optional[ModeState]:
        if not self.modes:
            return None
        return max(self.modes, key=lambda mode: mode.score)

    def _base_metrics(self) -> Dict[str, float]:
        if not self.modes:
            return {
                "top1_top2_margin": 0.0,
                "mode_entropy": 1.0,
                "mode_compactness": 0.0,
                "visible_coverage": 0.0,
                "marginal_gain": float("inf"),
            }

        self.modes.sort(key=lambda mode: mode.score, reverse=True)
        scores = np.asarray([mode.score for mode in self.modes], dtype=np.float64)
        total = max(1e-12, float(np.sum(scores)))
        probs = scores / total
        entropy = 0.0
        if probs.size > 1:
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(float(probs.size)))

        best = self.modes[0]
        if len(self.modes) > 1:
            margin = float((self.modes[0].score - self.modes[1].score) / max(self.modes[0].score, 1e-12))
        else:
            margin = 1.0

        previous_best_score = None
        if self.metric_history:
            previous_best_score = float(self.metric_history[-1].get("best_mode_score", math.nan))
        if previous_best_score is not None and not math.isnan(previous_best_score):
            marginal_gain = float(abs(best.score - previous_best_score))
        else:
            marginal_gain = 1.0e9

        return {
            "top1_top2_margin": float(margin),
            "mode_entropy": float(entropy),
            "mode_compactness": float(best.compactness(self.cfg.tau_t, self.cfg.tau_R)),
            "visible_coverage": float(best.visible_coverage),
            "marginal_gain": float(marginal_gain),
            "best_mode_score": float(best.score),
            "best_support_count": float(len(best.support_points)),
            "best_candidate_count": float(best.candidate_count),
        }

    def _augment_with_trends(
        self,
        current_metrics: Dict[str, float],
        history: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        snapshots = list(history if history is not None else self.metric_history)
        if not snapshots:
            snapshots = [dict(current_metrics)]

        required_window = max(2, int(self.cfg.trend_window))
        window = snapshots[-required_window:]
        first = window[0]
        last = window[-1]
        window_best_margin = max(float(item["top1_top2_margin"]) for item in window)
        window_best_coverage = max(float(item["visible_coverage"]) for item in window)
        window_min_entropy = min(float(item["mode_entropy"]) for item in window)

        metrics = dict(current_metrics)
        metrics.update(
            {
                "margin_growth": float(window_best_margin - float(first["top1_top2_margin"])),
                "coverage_growth": float(window_best_coverage - float(first["visible_coverage"])),
                "window_best_margin": float(window_best_margin),
                "window_best_coverage": float(window_best_coverage),
                "window_min_entropy": float(window_min_entropy),
                "trend_window_size": float(len(window)),
                "history_length": float(len(snapshots)),
                "current_margin_drift": float(window_best_margin - float(last["top1_top2_margin"])),
            }
        )
        return metrics

    def stability_metrics(self) -> Dict[str, float]:
        if self.last_metrics:
            return dict(self.last_metrics)
        return self._augment_with_trends(self._base_metrics(), self.metric_history)

    def should_stop(self) -> Tuple[bool, Dict[str, float]]:
        if not self.cfg.enable_early_stop:
            return False, self.stability_metrics()
        metrics = self.stability_metrics()
        required_window = max(2, int(self.cfg.trend_window))
        if self.round_index < int(self.cfg.min_rounds_before_stop) or int(metrics["trend_window_size"]) < required_window:
            return False, metrics

        entropy_ready = float(self.cfg.tau_entropy) >= 1.0 or metrics["window_min_entropy"] < float(self.cfg.tau_entropy)
        compactness_ready = float(self.cfg.tau_compactness) <= 0.0 or metrics["mode_compactness"] > float(self.cfg.tau_compactness)
        support_ready = int(metrics["best_support_count"]) >= int(self.cfg.min_support_points_for_stop)
        level_ready = (
            metrics["window_best_margin"] > float(self.cfg.tau_margin)
            and metrics["window_best_coverage"] > float(self.cfg.tau_coverage)
            and entropy_ready
            and compactness_ready
            and support_ready
        )
        trend_ready = (
            metrics["margin_growth"] <= float(self.cfg.tau_margin_growth)
            and metrics["coverage_growth"] <= float(self.cfg.tau_coverage_growth)
        )
        stop = (
            level_ready
            and trend_ready
        )
        return bool(stop), metrics
