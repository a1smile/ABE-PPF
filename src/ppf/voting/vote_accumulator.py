from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.ppf.core.hash_table import ModelHashBundle
from src.ppf.core.pose_hypothesis import MatchRecord, PoseCandidate
from src.ppf.core.transform_utils import compose_affine, invert_affine, make_affine, rotation_matrix_from_axis_angle, wrap_to_pi


@dataclass
class VoteAccumulatorConfig:
    candidate_bins_per_model_ref: int = 1
    max_candidates_per_batch: int = 64
    max_candidates_per_reference: int = 0
    min_vote_score: float = 0.0
    min_support_pair_count: int = 1
    relative_score_threshold: float = 0.0
    bucket_score_scale: float = 0.35


class VoteAccumulator:
    def __init__(self, model_bundle: ModelHashBundle, cfg: VoteAccumulatorConfig):
        self.model_bundle = model_bundle
        self.cfg = cfg
        self.aux_size = int(math.ceil((2.0 * math.pi) / model_bundle.angle_step))
        self.accumulator = np.zeros((model_bundle.points.shape[0], self.aux_size), dtype=np.float32)
        self.active_rows: List[int] = []
        self.active_row_mask = np.zeros((model_bundle.points.shape[0],), dtype=bool)
        self.active_bins: Dict[int, List[int]] = {}
        self.pair_counts: Dict[Tuple[int, int], int] = {}
        self.current_reference_index = -1

    def reset(self) -> None:
        for model_reference_index in self.active_rows:
            bins = self.active_bins.get(int(model_reference_index), [])
            for vote_bin in bins:
                self.accumulator[int(model_reference_index), int(vote_bin)] = 0.0
            self.active_row_mask[int(model_reference_index)] = False
        self.active_rows.clear()
        self.active_bins.clear()
        self.pair_counts.clear()
        self.current_reference_index = -1

    def vote(
        self,
        match: MatchRecord,
        alpha_scene: float,
        reference_index: int,
    ) -> None:
        if self.current_reference_index < 0:
            self.current_reference_index = int(reference_index)
        alpha = wrap_to_pi(float(self.model_bundle.alpha_m[match.model_reference_index, match.model_pair_index]) - alpha_scene)
        vote_bin = int(math.floor((alpha + math.pi) / self.model_bundle.angle_step))
        vote_bin = max(0, min(self.aux_size - 1, vote_bin))
        model_reference_index = int(match.model_reference_index)

        if not bool(self.active_row_mask[model_reference_index]):
            self.active_row_mask[model_reference_index] = True
            self.active_rows.append(model_reference_index)

        bucket_score = self.model_bundle.bucket_score(match.bucket_key)
        weight = float(match.weight) * (1.0 + float(self.cfg.bucket_score_scale) * float(bucket_score))
        if float(self.accumulator[model_reference_index, vote_bin]) <= 0.0:
            self.active_bins.setdefault(model_reference_index, []).append(int(vote_bin))
        self.accumulator[model_reference_index, vote_bin] += float(weight)

        key = (model_reference_index, vote_bin)
        self.pair_counts[key] = self.pair_counts.get(key, 0) + 1

    def build_candidates(self, reference_transform: np.ndarray) -> List[PoseCandidate]:
        candidates: List[PoseCandidate] = []
        reference_indices = {int(self.current_reference_index)} if self.current_reference_index >= 0 else set()
        support_ref_count = float(len(reference_indices))

        for model_reference_index in self.active_rows:
            active_bins = self.active_bins.get(int(model_reference_index), [])
            if not active_bins:
                continue

            row_scores = self.accumulator[int(model_reference_index), np.asarray(active_bins, dtype=np.int64)]
            valid_mask = row_scores > float(self.cfg.min_vote_score)
            if not np.any(valid_mask):
                continue

            valid_bins = np.asarray(active_bins, dtype=np.int64)[valid_mask]
            valid_scores = row_scores[valid_mask]
            top_k = min(int(self.cfg.candidate_bins_per_model_ref), int(valid_scores.shape[0]))
            if top_k <= 0:
                continue

            top_bins = np.argpartition(-valid_scores, top_k - 1)[:top_k]
            top_bins = top_bins[np.argsort(-valid_scores[top_bins])]
            for vote_bin_index in top_bins.tolist():
                vote_bin = int(valid_bins[int(vote_bin_index)])
                score = float(valid_scores[int(vote_bin_index)])
                if score <= float(self.cfg.min_vote_score):
                    continue

                theta = (float(vote_bin) + 0.5) * self.model_bundle.angle_step - math.pi
                rotation_x = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float64), theta)
                T_x = make_affine(rotation_x, np.zeros((3,), dtype=np.float64))
                ref_rotation = self.model_bundle.ref_rotations[model_reference_index]
                ref_point = self.model_bundle.points[model_reference_index]
                T_model_local = make_affine(ref_rotation, -(ref_rotation @ ref_point))
                transform = compose_affine(invert_affine(reference_transform), compose_affine(T_x, T_model_local))

                support_key = (model_reference_index, int(vote_bin))
                pair_count = int(self.pair_counts.get(support_key, 0))
                candidates.append(
                    PoseCandidate(
                        transform=transform,
                        score=score,
                        model_reference_index=model_reference_index,
                        vote_bin=int(vote_bin),
                        reference_indices=set(reference_indices),
                        support_pair_count=pair_count,
                        ambiguity_penalty=0.0,
                        metadata={
                            "vote_score": score,
                            "support_ref_count": support_ref_count,
                            "support_pair_count": float(pair_count),
                        },
                    )
                )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        if not candidates:
            return []

        best_score = float(candidates[0].score)
        filtered = [
            candidate
            for candidate in candidates
            if candidate.support_pair_count >= int(self.cfg.min_support_pair_count)
            and candidate.score >= best_score * float(self.cfg.relative_score_threshold)
        ]
        if not filtered:
            filtered = candidates

        limit = int(self.cfg.max_candidates_per_reference or self.cfg.max_candidates_per_batch)
        return filtered[:limit]
