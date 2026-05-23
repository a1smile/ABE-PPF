from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.ppf.backend.mode_pool import ModePool, ModePoolConfig
from src.ppf.core.pose_hypothesis import PipelinePrediction, PoseCandidate


@dataclass
class BRPMRConfig:
    batch_size_reference_points: int = 32
    max_budget_pairs: int = 50000
    low_evidence_reference_threshold: int = 0
    low_evidence_refine_iterations: int = 0
    low_evidence_refine_distance: float = 0.0
    max_candidates_per_round: int = 0
    candidate_score_ratio: float = 0.0
    batch_nms_translation: float = 0.0
    batch_nms_rotation: float = 0.0
    candidate_support_pair_weight: float = 0.0
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
    min_rounds_before_stop: int = 2
    min_support_points_for_stop: int = 0
    recent_history_size: int = 5
    max_modes: int = 0
    min_mode_score_ratio: float = 0.0
    heavy_branch_reference_threshold: int = 0
    heavy_branch_candidate_threshold: int = 0
    heavy_branch_mode_threshold: int = 0
    heavy_branch_round_threshold: int = 0
    heavy_branch_decay: float = 1.0
    heavy_branch_min_candidate_cap: int = 0
    heavy_branch_min_mode_cap: int = 0
    heavy_branch_score_ratio_boost: float = 0.0
    heavy_branch_min_mode_score_ratio_boost: float = 0.0
    heavy_branch_stop_min_round: int = 0
    heavy_branch_tau_margin: float = 0.0
    heavy_branch_tau_coverage: float = 0.0
    heavy_branch_tau_entropy: float = 1.0
    heavy_branch_tau_compactness: float = 0.0
    heavy_branch_tau_margin_drift: float = 1.0
    heavy_branch_min_support_points: int = 0
    heavy_branch_mode_saturation_ratio: float = 0.0


@dataclass
class DynamicBudgetProfile:
    heavy_branch_active: bool
    reference_ready: bool
    reference_pressure: int
    candidate_pressure: int
    mode_pressure: int
    round_pressure: int
    pressure: int
    round_index: int
    input_candidates: int
    current_mode_count: int
    effective_max_candidates_per_round: int
    effective_candidate_score_ratio: float
    effective_max_modes: int
    effective_min_mode_score_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "heavy_branch_active": float(1 if self.heavy_branch_active else 0),
            "reference_ready": float(1 if self.reference_ready else 0),
            "reference_pressure": float(self.reference_pressure),
            "candidate_pressure": float(self.candidate_pressure),
            "mode_pressure": float(self.mode_pressure),
            "round_pressure": float(self.round_pressure),
            "pressure": float(self.pressure),
            "round_index": float(self.round_index),
            "input_candidates": float(self.input_candidates),
            "current_mode_count": float(self.current_mode_count),
            "effective_max_candidates_per_round": float(self.effective_max_candidates_per_round),
            "effective_candidate_score_ratio": float(self.effective_candidate_score_ratio),
            "effective_max_modes": float(self.effective_max_modes),
            "effective_min_mode_score_ratio": float(self.effective_min_mode_score_ratio),
        }


class BRPMRController:
    def __init__(self, cfg: BRPMRConfig, total_reference_points: int):
        self.cfg = cfg
        self.mode_pool = ModePool(
            ModePoolConfig(
                tau_t=cfg.tau_t,
                tau_R=cfg.tau_R,
                tau_t_merge=cfg.tau_t_merge,
                tau_R_merge=cfg.tau_R_merge,
                tau_margin=cfg.tau_margin,
                tau_entropy=cfg.tau_entropy,
                tau_compactness=cfg.tau_compactness,
                tau_coverage=cfg.tau_coverage,
                tau_marginal_gain=cfg.tau_marginal_gain,
                trend_window=cfg.trend_window,
                tau_margin_growth=cfg.tau_margin_growth,
                tau_coverage_growth=cfg.tau_coverage_growth,
                min_rounds_before_stop=cfg.min_rounds_before_stop,
                min_support_points_for_stop=cfg.min_support_points_for_stop,
                recent_history_size=cfg.recent_history_size,
                max_modes=cfg.max_modes,
                min_mode_score_ratio=cfg.min_mode_score_ratio,
            ),
            total_reference_points=total_reference_points,
        )
        self.early_stop_round = -1
        self.last_candidate_debug: Dict[str, int] = {
            "input_candidates": 0,
            "filtered_candidates": 0,
            "merged_candidates": 0,
        }
        self.last_dynamic_budget_debug: Dict[str, float] = {}

    def _effective_score(self, candidate: PoseCandidate) -> float:
        return float(candidate.score) * (
            1.0 + float(self.cfg.candidate_support_pair_weight) * math.log1p(max(0, int(candidate.support_pair_count)))
        )

    def _pressure_component(self, value: int, threshold: int) -> int:
        if threshold <= 0 or value < threshold:
            return 0
        return max(1, int(math.ceil(float(value) / float(threshold))) - 1)

    def _resolve_cap(self, base_cap: int, min_cap: int, pressure: int) -> int:
        if pressure <= 0:
            return int(base_cap)

        if base_cap > 0:
            decay = min(0.999, max(0.1, float(self.cfg.heavy_branch_decay)))
            resolved = int(math.floor(float(base_cap) * (decay ** float(pressure))))
            resolved = max(1, resolved)
            if min_cap > 0:
                resolved = max(int(min_cap), resolved)
            return min(int(base_cap), resolved)

        if min_cap > 0:
            return max(1, int(min_cap))
        return 0

    def _dynamic_budget_profile(self, input_candidates: int) -> DynamicBudgetProfile:
        round_index = int(self.mode_pool.round_index) + 1
        current_mode_count = int(len(self.mode_pool.modes))

        reference_threshold = int(self.cfg.heavy_branch_reference_threshold)
        candidate_threshold = int(self.cfg.heavy_branch_candidate_threshold)
        mode_threshold = int(self.cfg.heavy_branch_mode_threshold)
        round_threshold = int(self.cfg.heavy_branch_round_threshold)

        reference_ready = reference_threshold <= 0 or self.mode_pool.total_reference_points >= reference_threshold
        reference_pressure = 1 if reference_threshold > 0 and reference_ready else 0
        candidate_pressure = self._pressure_component(int(input_candidates), candidate_threshold)
        mode_pressure = self._pressure_component(current_mode_count, mode_threshold)
        round_pressure = 0
        if round_threshold > 0 and round_index >= round_threshold:
            round_pressure = int(round_index - round_threshold + 1)

        heavy_branch_active = bool(reference_ready and (candidate_pressure > 0 or mode_pressure > 0 or round_pressure > 0))
        pressure = reference_pressure + candidate_pressure + mode_pressure + round_pressure if heavy_branch_active else 0

        effective_max_candidates_per_round = self._resolve_cap(
            base_cap=int(self.cfg.max_candidates_per_round),
            min_cap=int(self.cfg.heavy_branch_min_candidate_cap),
            pressure=pressure,
        )
        effective_max_modes = self._resolve_cap(
            base_cap=int(self.cfg.max_modes),
            min_cap=int(self.cfg.heavy_branch_min_mode_cap),
            pressure=pressure,
        )

        effective_candidate_score_ratio = float(self.cfg.candidate_score_ratio)
        if heavy_branch_active:
            effective_candidate_score_ratio = min(
                0.95,
                effective_candidate_score_ratio + float(self.cfg.heavy_branch_score_ratio_boost) * float(pressure),
            )

        effective_min_mode_score_ratio = float(self.cfg.min_mode_score_ratio)
        if heavy_branch_active:
            effective_min_mode_score_ratio = min(
                0.95,
                effective_min_mode_score_ratio
                + float(self.cfg.heavy_branch_min_mode_score_ratio_boost) * float(pressure),
            )

        return DynamicBudgetProfile(
            heavy_branch_active=heavy_branch_active,
            reference_ready=reference_ready,
            reference_pressure=reference_pressure,
            candidate_pressure=candidate_pressure,
            mode_pressure=mode_pressure,
            round_pressure=round_pressure,
            pressure=pressure,
            round_index=round_index,
            input_candidates=int(input_candidates),
            current_mode_count=current_mode_count,
            effective_max_candidates_per_round=int(effective_max_candidates_per_round),
            effective_candidate_score_ratio=float(effective_candidate_score_ratio),
            effective_max_modes=int(effective_max_modes),
            effective_min_mode_score_ratio=float(effective_min_mode_score_ratio),
        )

    def _filter_candidates(
        self,
        batch_candidates: List[PoseCandidate],
        budget_profile: DynamicBudgetProfile,
    ) -> List[PoseCandidate]:
        if not batch_candidates:
            self.last_candidate_debug = {
                "input_candidates": 0,
                "filtered_candidates": 0,
                "merged_candidates": 0,
            }
            self.last_dynamic_budget_debug = budget_profile.to_dict()
            return []

        ordered = sorted(batch_candidates, key=self._effective_score, reverse=True)
        best_score = max(1e-12, float(self._effective_score(ordered[0])))
        filtered = [
            candidate
            for candidate in ordered
            if self._effective_score(candidate) >= best_score * float(budget_profile.effective_candidate_score_ratio)
        ]
        if not filtered:
            filtered = ordered

        limit = int(budget_profile.effective_max_candidates_per_round)
        if limit > 0:
            filtered = filtered[:limit]

        merged: List[PoseCandidate] = []
        nms_t = float(self.cfg.batch_nms_translation)
        nms_r = float(self.cfg.batch_nms_rotation)
        if nms_t <= 0.0 or nms_r <= 0.0:
            merged = filtered
        else:
            for candidate in filtered:
                candidate_translation = candidate.transform[:3, 3]
                candidate_rotation = candidate.transform[:3, :3]
                merged_into_existing = False
                for kept in merged:
                    dt = float(np.linalg.norm(kept.transform[:3, 3] - candidate_translation))
                    if dt > nms_t:
                        continue
                    rotation_cos = np.clip((np.trace(kept.transform[:3, :3].T @ candidate_rotation) - 1.0) / 2.0, -1.0, 1.0)
                    dr = float(np.degrees(np.arccos(rotation_cos)))
                    if dr > nms_r:
                        continue
                    kept.reference_indices.update(candidate.reference_indices)
                    kept.support_pair_count += int(candidate.support_pair_count)
                    kept.score += float(candidate.score)
                    merged_into_existing = True
                    break
                if not merged_into_existing:
                    merged.append(candidate)

        self.last_candidate_debug = {
            "input_candidates": len(batch_candidates),
            "filtered_candidates": len(filtered),
            "merged_candidates": len(merged),
        }
        self.last_dynamic_budget_debug = budget_profile.to_dict()
        return merged

    def _heavy_branch_should_stop(
        self,
        metrics: Dict[str, float],
        budget_profile: DynamicBudgetProfile,
    ) -> bool:
        if not budget_profile.heavy_branch_active:
            return False

        min_round = int(self.cfg.heavy_branch_stop_min_round)
        if min_round <= 0 or budget_profile.round_index < min_round:
            return False

        margin_threshold = float(self.cfg.heavy_branch_tau_margin)
        coverage_threshold = float(self.cfg.heavy_branch_tau_coverage)
        if margin_threshold <= 0.0 or coverage_threshold <= 0.0:
            return False

        saturation_ready = True
        saturation_ratio = float(self.cfg.heavy_branch_mode_saturation_ratio)
        if saturation_ratio > 0.0 and budget_profile.effective_max_modes > 0:
            required_modes = max(1, int(math.ceil(float(budget_profile.effective_max_modes) * saturation_ratio)))
            saturation_ready = len(self.mode_pool.modes) >= required_modes

        entropy_threshold = float(self.cfg.heavy_branch_tau_entropy)
        entropy_ready = entropy_threshold >= 1.0 or metrics["window_min_entropy"] < entropy_threshold
        compactness_threshold = float(self.cfg.heavy_branch_tau_compactness)
        compactness_ready = compactness_threshold <= 0.0 or metrics["mode_compactness"] > compactness_threshold
        support_ready = int(metrics["best_support_count"]) >= int(self.cfg.heavy_branch_min_support_points)
        drift_ready = metrics["current_margin_drift"] <= float(self.cfg.heavy_branch_tau_margin_drift)
        margin_ready = metrics["top1_top2_margin"] > margin_threshold
        coverage_ready = metrics["visible_coverage"] > coverage_threshold

        return bool(
            saturation_ready
            and entropy_ready
            and compactness_ready
            and support_ready
            and drift_ready
            and margin_ready
            and coverage_ready
        )

    def update(self, batch_candidates: List[PoseCandidate]) -> Tuple[bool, Dict[str, float]]:
        budget_profile = self._dynamic_budget_profile(len(batch_candidates))
        prepared = self._filter_candidates(batch_candidates, budget_profile)
        self.mode_pool.add_candidates(
            prepared,
            max_modes_override=budget_profile.effective_max_modes,
            min_mode_score_ratio_override=budget_profile.effective_min_mode_score_ratio,
        )
        self.last_dynamic_budget_debug.update(self.mode_pool.last_round_debug)
        self.last_dynamic_budget_debug["prepared_candidates"] = float(len(prepared))
        self.last_dynamic_budget_debug["mode_count_after_update"] = float(len(self.mode_pool.modes))
        should_stop, metrics = self.mode_pool.should_stop()
        heavy_branch_early_stop = False
        if not should_stop:
            heavy_branch_early_stop = self._heavy_branch_should_stop(metrics, budget_profile)
            should_stop = heavy_branch_early_stop
        self.last_dynamic_budget_debug["standard_stop_ready"] = float(1 if should_stop and not heavy_branch_early_stop else 0)
        self.last_dynamic_budget_debug["heavy_branch_early_stop"] = float(1 if heavy_branch_early_stop else 0)
        if should_stop and self.early_stop_round < 0:
            self.early_stop_round = int(self.mode_pool.round_index)
        return should_stop, metrics

    def prediction(self) -> PipelinePrediction:
        best_mode = self.mode_pool.best_mode()
        if best_mode is None:
            return PipelinePrediction(
                transform=np.eye(4, dtype=np.float64),
                score=0.0,
                mode_count=0,
                early_stop_round=self.early_stop_round,
                metadata={"stability": self.mode_pool.stability_metrics()},
            )

        return PipelinePrediction(
            transform=best_mode.representative_pose.copy(),
            score=float(best_mode.score),
            mode_count=len(self.mode_pool.modes),
            early_stop_round=self.early_stop_round,
            metadata={"stability": self.mode_pool.stability_metrics()},
        )
