from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.ppf.core.hash_table import ModelHashBundle
from src.ppf.core.pose_hypothesis import MatchRecord
from src.ppf.core.ppf_feature import PPFKey, boundary_distances, discretize_feature, feature_bin_bounds, neighbor_key


@dataclass
class UBSPConfig:
    lambda_value: float = 1.5
    max_expand_dims: int = 2
    sigma_max_distance: float = 0.02
    sigma_max_angle: float = 0.25
    position_sigma_scale: float = 1.0
    normal_sigma_scale: float = 1.0
    distance_coupling_scale: float = 1.0
    boundary_fraction_max: float = 1.0
    max_bucket_size: int = 500
    eps: float = 1e-6


@dataclass
class UBSPStats:
    ubsp_trigger_count: int = 0
    neighbor_bucket_query_count: int = 0
    single_bucket_query_count: int = 0
    high_ambiguity_bucket_count: int = 0
    skipped_high_ambiguity_expansion_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "ubsp_trigger_count": int(self.ubsp_trigger_count),
            "neighbor_bucket_query_count": int(self.neighbor_bucket_query_count),
            "single_bucket_query_count": int(self.single_bucket_query_count),
            "high_ambiguity_bucket_count": int(self.high_ambiguity_bucket_count),
            "skipped_high_ambiguity_expansion_count": int(self.skipped_high_ambiguity_expansion_count),
        }


class UBSPRetriever:
    def __init__(self, cfg: UBSPConfig):
        self.cfg = cfg
        self.stats = UBSPStats()

    def _pair_uncertainty(
        self,
        feature_g: np.ndarray,
        position_sigma_ref: float,
        position_sigma_pair: float,
        normal_sigma_ref: float,
        normal_sigma_pair: float,
        distance_step: float,
    ) -> np.ndarray:
        scene_distance = max(float(feature_g[3]), float(distance_step), self.cfg.eps)
        sigma_distance = max(
            self.cfg.eps,
            float(self.cfg.position_sigma_scale) * float(position_sigma_ref + position_sigma_pair),
        )
        sigma_angle = max(
            self.cfg.eps,
            float(self.cfg.normal_sigma_scale) * float(normal_sigma_ref + normal_sigma_pair)
            + float(self.cfg.distance_coupling_scale) * float(sigma_distance / scene_distance),
        )
        return np.array([sigma_angle, sigma_angle, sigma_angle, sigma_distance], dtype=np.float64)

    def _collect_bucket_matches(
        self,
        model_bundle: ModelHashBundle,
        key: PPFKey,
        is_neighbor_bucket: bool,
    ) -> List[MatchRecord]:
        bucket = list(model_bundle.query(key))
        bucket_size = len(bucket)
        if bucket_size > int(self.cfg.max_bucket_size):
            self.stats.high_ambiguity_bucket_count += 1
            bucket = bucket[: int(self.cfg.max_bucket_size)]

        ambiguity_factor = min(1.0, float(self.cfg.max_bucket_size) / max(1, bucket_size))
        weight = ambiguity_factor * (0.5 if is_neighbor_bucket else 1.0)
        return [
            MatchRecord(
                model_reference_index=entry.mr,
                model_pair_index=entry.mi,
                bucket_key=key,
                weight=float(weight),
                is_neighbor_bucket=bool(is_neighbor_bucket),
                bucket_size=int(bucket_size),
            )
            for entry in bucket
        ]

    def query(
        self,
        model_bundle: ModelHashBundle,
        feature_g: np.ndarray,
        position_sigma_ref: float,
        position_sigma_pair: float,
        normal_sigma_ref: float,
        normal_sigma_pair: float,
    ) -> List[MatchRecord]:
        key = discretize_feature(feature_g, model_bundle.angle_step, model_bundle.distance_step)
        current_bucket = self._collect_bucket_matches(model_bundle, key, is_neighbor_bucket=False)

        sigma_g = self._pair_uncertainty(
            feature_g=feature_g,
            position_sigma_ref=position_sigma_ref,
            position_sigma_pair=position_sigma_pair,
            normal_sigma_ref=normal_sigma_ref,
            normal_sigma_pair=normal_sigma_pair,
            distance_step=model_bundle.distance_step,
        )
        deltas = boundary_distances(feature_g, key, model_bundle.angle_step, model_bundle.distance_step)
        sigma_max = np.array(
            [
                self.cfg.sigma_max_angle,
                self.cfg.sigma_max_angle,
                self.cfg.sigma_max_angle,
                self.cfg.sigma_max_distance,
            ],
            dtype=np.float64,
        )
        bin_widths = np.array(
            [model_bundle.angle_step, model_bundle.angle_step, model_bundle.angle_step, model_bundle.distance_step],
            dtype=np.float64,
        )
        boundary_fraction = deltas / np.maximum(bin_widths, self.cfg.eps)
        risk_mask = (
            (deltas < float(self.cfg.lambda_value) * sigma_g)
            & (sigma_g < sigma_max)
            & (boundary_fraction < float(self.cfg.boundary_fraction_max))
        )
        if not np.any(risk_mask):
            self.stats.single_bucket_query_count += 1
            return current_bucket

        self.stats.ubsp_trigger_count += 1
        lower, upper = feature_bin_bounds(key, model_bundle.angle_step, model_bundle.distance_step)
        rho = deltas / (sigma_g + self.cfg.eps)
        risky_dims = [dim for dim in range(4) if bool(risk_mask[dim])]
        risky_dims.sort(key=lambda dim: float(rho[dim]))
        risky_dims = risky_dims[: int(self.cfg.max_expand_dims)]

        current_bucket_size = model_bundle.bucket_size(key)
        if current_bucket_size > int(self.cfg.max_bucket_size):
            self.stats.skipped_high_ambiguity_expansion_count += 1
            self.stats.single_bucket_query_count += 1
            return current_bucket

        merged: Dict[Tuple[int, int], MatchRecord] = {
            (match.model_reference_index, match.model_pair_index): match for match in current_bucket
        }

        expanded = False
        for dim in risky_dims:
            direction = -1 if (float(feature_g[dim]) - lower[dim]) <= (upper[dim] - float(feature_g[dim])) else 1
            n_key = neighbor_key(key, dim, direction)
            neighbor_bucket_size = model_bundle.bucket_size(n_key)
            if neighbor_bucket_size > int(self.cfg.max_bucket_size):
                self.stats.high_ambiguity_bucket_count += 1
                self.stats.skipped_high_ambiguity_expansion_count += 1
                continue

            self.stats.neighbor_bucket_query_count += 1
            expanded = True
            for match in self._collect_bucket_matches(model_bundle, n_key, is_neighbor_bucket=True):
                item_key = (match.model_reference_index, match.model_pair_index)
                previous = merged.get(item_key)
                if previous is None or match.weight > previous.weight:
                    merged[item_key] = match

        if not expanded:
            self.stats.single_bucket_query_count += 1

        return list(merged.values())
