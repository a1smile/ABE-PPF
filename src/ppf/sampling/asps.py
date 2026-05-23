from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.ppf.core.hash_table import ModelHashBundle
from src.ppf.core.ppf_feature import compute_pair_features, discretize_feature, to_internal_feature_g


@dataclass
class ASPSConfig:
    top_m_reference_points: int = 256
    min_selected_points: int = 0
    max_scored_candidates: int = 0
    pre_sample_pairs: int = 64
    matching_pair_cap_per_reference: int = 0
    alpha: float = 1.0
    beta: float = 0.5
    normal_stability_threshold: float = 0.7
    voxel_nms_size: float = 0.01
    pair_radius_ratio: float = 0.5
    knn: int = 12
    distance_bins: int = 4
    angle_bins: int = 4
    direction_bins: int = 8
    coarse_neighbor_weight: float = 0.0
    matching_pair_reliability_weight: float = 0.65
    matching_pair_ambiguity_weight: float = 0.35
    matching_pair_candidate_multiplier: float = 2.0
    seed: int = 0
    sampling_strategy: str = "asps"  # "asps" | "random" | "uniform" | "curvature" | "normal_stability"


@dataclass
class ReferencePointScore:
    index: int
    score: float
    reliability: float
    ambiguity_score: float
    diversity_score: float
    redundancy_penalty: float
    position_sigma: float
    normal_sigma: float
    pair_indices: List[int]


@dataclass
class ASPSResult:
    references: List[ReferencePointScore]
    num_candidates: int
    num_scored_candidates: int
    pair_radius: float
    reliability_scores: np.ndarray
    position_sigmas: np.ndarray
    normal_sigmas: np.ndarray
    num_selected_after_nms: int
    num_backfilled: int


def _pairwise_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(points, dtype=np.float64)[None, :, :] - np.asarray(points, dtype=np.float64)[:, None, :]
    distances = np.linalg.norm(offsets, axis=2)
    return offsets, distances


def _knn_indices(pairwise_distances: np.ndarray, knn: int) -> np.ndarray:
    k = max(1, min(int(knn), max(1, pairwise_distances.shape[0] - 1)))
    partition = np.argpartition(pairwise_distances, kth=k, axis=1)[:, : k + 1]
    partition_distances = np.take_along_axis(pairwise_distances, partition, axis=1)
    order = np.argsort(partition_distances, axis=1)
    ordered = np.take_along_axis(partition, order, axis=1)

    trimmed = np.empty((pairwise_distances.shape[0], k), dtype=np.int64)
    for point_index in range(pairwise_distances.shape[0]):
        row = ordered[point_index]
        row = row[row != point_index]
        trimmed[point_index] = row[:k]
    return trimmed


def _estimate_reliability(
    points: np.ndarray,
    normals: np.ndarray,
    knn: int,
    pairwise_distances: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pairwise_distances is None:
        _, pairwise_distances = _pairwise_geometry(points)
    indices = _knn_indices(pairwise_distances, knn=knn)
    normal_similarity = np.clip(np.abs(np.asarray(normals, dtype=np.float64) @ np.asarray(normals, dtype=np.float64).T), 0.0, 1.0)
    reliability = np.zeros((points.shape[0],), dtype=np.float64)
    position_sigma = np.zeros((points.shape[0],), dtype=np.float64)
    normal_sigma = np.zeros((points.shape[0],), dtype=np.float64)

    for point_index in range(points.shape[0]):
        neighbor_ids = indices[point_index]
        neighbor_points = points[neighbor_ids]
        point = points[point_index]

        dot_scores = normal_similarity[point_index, neighbor_ids]
        normal_consistency = float(np.mean(dot_scores)) if dot_scores.size else 0.0

        centered = neighbor_points - np.mean(neighbor_points, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, neighbor_points.shape[0])
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
        curvature = float(eigenvalues[0] / max(1e-12, np.sum(eigenvalues)))

        local_scale = float(np.mean(pairwise_distances[point_index, neighbor_ids])) if neighbor_points.size else 0.0
        curvature_penalty = math.exp(-10.0 * curvature)
        reliability[point_index] = max(0.0, min(1.0, normal_consistency * curvature_penalty))
        position_sigma[point_index] = max(1e-6, local_scale * (1.0 + 5.0 * curvature))
        normal_sigma[point_index] = max(1e-4, math.acos(max(-1.0, min(1.0, normal_consistency))))

    return reliability, position_sigma, normal_sigma


def _direction_bin(direction: np.ndarray) -> int:
    return int(direction[0] >= 0.0) + 2 * int(direction[1] >= 0.0) + 4 * int(direction[2] >= 0.0)


def _coarse_candidate_score(reliability: float, neighbor_count: int, coarse_neighbor_weight: float) -> float:
    return float(reliability) * (
        1.0 + float(coarse_neighbor_weight) * math.log1p(float(max(0, int(neighbor_count))))
    )


def _build_feasible_and_neighbors(
    points: np.ndarray,
    pairwise_distances: np.ndarray,
    min_pair_distance: float,
    pair_radius: float,
    scene_pair_cap: int,
) -> tuple[list[int], dict[int, np.ndarray]]:
    feasible_indices: list[int] = []
    neighbor_cache: dict[int, np.ndarray] = {}
    for i in range(points.shape[0]):
        distances = pairwise_distances[i]
        nids = np.where((distances > min_pair_distance) & (distances <= pair_radius))[0]
        if nids.size == 0:
            continue
        feasible_indices.append(i)
        order = nids[np.argsort(distances[nids])]
        neighbor_cache[i] = order[:scene_pair_cap]
    return feasible_indices, neighbor_cache


class ASPSSelector:
    def __init__(self, cfg: ASPSConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))

    def select(
        self,
        scene_points: np.ndarray,
        scene_normals: np.ndarray,
        model_bundle: ModelHashBundle,
        pair_cap_per_reference: Optional[int] = None,
    ) -> ASPSResult:
        points = np.asarray(scene_points, dtype=np.float64)
        normals = np.asarray(scene_normals, dtype=np.float64)
        pairwise_offsets, pairwise_distances = _pairwise_geometry(points)
        reliability, position_sigma, normal_sigma = _estimate_reliability(
            points,
            normals,
            knn=self.cfg.knn,
            pairwise_distances=pairwise_distances,
        )

        pair_radius = max(
            float(model_bundle.distance_step * 6.0),
            float(model_bundle.model_diameter * self.cfg.pair_radius_ratio),
        )
        min_pair_distance = max(float(model_bundle.min_pair_distance), float(model_bundle.distance_step))

        strategy = str(getattr(self.cfg, "sampling_strategy", "asps") or "asps").strip().lower()
        if strategy != "asps":
            return self._select_alternative(
                strategy=strategy,
                points=points,
                normals=normals,
                model_bundle=model_bundle,
                reliability=reliability,
                position_sigma=position_sigma,
                normal_sigma=normal_sigma,
                pairwise_distances=pairwise_distances,
                pairwise_offsets=pairwise_offsets,
                pair_radius=pair_radius,
                min_pair_distance=min_pair_distance,
                pair_cap_per_reference=pair_cap_per_reference,
            )

        scored_indices: List[int] = []
        feasible_indices: List[int] = []
        neighbor_cache: dict[int, np.ndarray] = {}
        for reference_index in range(points.shape[0]):
            distances = pairwise_distances[reference_index]
            neighbor_ids = np.where((distances > min_pair_distance) & (distances <= pair_radius))[0]
            if neighbor_ids.size == 0:
                continue

            feasible_indices.append(reference_index)
            neighbor_cache[reference_index] = neighbor_ids
            if reliability[reference_index] >= float(self.cfg.normal_stability_threshold):
                scored_indices.append(reference_index)

        target_candidate_pool = min(
            len(feasible_indices),
            max(int(self.cfg.top_m_reference_points), max(0, int(self.cfg.min_selected_points))),
        )
        if len(scored_indices) < target_candidate_pool:
            scored_set = set(scored_indices)
            deferred_indices = [index for index in feasible_indices if index not in scored_set]
            deferred_indices.sort(
                key=lambda index: (
                    float(reliability[index]),
                    int(neighbor_cache[index].size),
                ),
                reverse=True,
            )
            scored_indices.extend(deferred_indices[: max(0, target_candidate_pool - len(scored_indices))])

        max_scored_candidates = int(self.cfg.max_scored_candidates)
        if max_scored_candidates > 0 and len(scored_indices) > max_scored_candidates:
            scored_indices.sort(
                key=lambda index: _coarse_candidate_score(
                    reliability=float(reliability[index]),
                    neighbor_count=int(neighbor_cache[index].size),
                    coarse_neighbor_weight=float(self.cfg.coarse_neighbor_weight),
                ),
                reverse=True,
            )
            scored_indices = scored_indices[:max_scored_candidates]

        candidates: List[ReferencePointScore] = []
        for reference_index in scored_indices:
            offsets = pairwise_offsets[reference_index]
            neighbor_ids = neighbor_cache[reference_index]

            if neighbor_ids.size > int(self.cfg.pre_sample_pairs):
                sampled_ids = self.rng.choice(neighbor_ids, size=int(self.cfg.pre_sample_pairs), replace=False)
            else:
                sampled_ids = neighbor_ids

            ambiguity_values: List[float] = []
            occupied_bins = set()
            for pair_index in sampled_ids:
                feature = compute_pair_features(
                    points[reference_index],
                    normals[reference_index],
                    points[pair_index],
                    normals[pair_index],
                )
                if feature is None:
                    continue
                f1, f2, f3, distance = feature
                g = to_internal_feature_g(f1, f2, f3, distance)
                key = discretize_feature(g, model_bundle.angle_step, model_bundle.distance_step)
                ambiguity_values.append(model_bundle.bucket_score(key))

                direction = offsets[pair_index] / max(distance, 1e-12)
                distance_bin = min(self.cfg.distance_bins - 1, int((distance / max(pair_radius, 1e-12)) * self.cfg.distance_bins))
                angle = math.acos(max(-1.0, min(1.0, abs(float(normals[reference_index] @ direction)))))
                angle_bin = min(self.cfg.angle_bins - 1, int((angle / math.pi) * self.cfg.angle_bins))
                occupied_bins.add((distance_bin, angle_bin, _direction_bin(direction)))

            if not ambiguity_values:
                continue

            ambiguity_score = float(np.mean(np.asarray(ambiguity_values, dtype=np.float64)))
            total_bins = max(1, self.cfg.distance_bins * self.cfg.angle_bins * self.cfg.direction_bins)
            diversity_score = float(len(occupied_bins)) / float(total_bins)
            score = float(reliability[reference_index]) * (
                float(self.cfg.alpha) * ambiguity_score + float(self.cfg.beta) * diversity_score
            )

            candidates.append(
                ReferencePointScore(
                    index=reference_index,
                    score=score,
                    reliability=float(reliability[reference_index]),
                    ambiguity_score=ambiguity_score,
                    diversity_score=diversity_score,
                    redundancy_penalty=1.0,
                    position_sigma=float(position_sigma[reference_index]),
                    normal_sigma=float(normal_sigma[reference_index]),
                    pair_indices=[int(item) for item in sampled_ids.tolist()],
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected: List[ReferencePointScore] = []
        deferred: List[ReferencePointScore] = []
        for candidate in candidates:
            if len(selected) >= int(self.cfg.top_m_reference_points):
                break
            if any(
                np.linalg.norm(points[candidate.index] - points[chosen.index]) < float(self.cfg.voxel_nms_size)
                for chosen in selected
            ):
                deferred.append(candidate)
                continue
            selected.append(candidate)

        min_selected = min(int(self.cfg.top_m_reference_points), max(0, int(self.cfg.min_selected_points)))
        num_selected_after_nms = len(selected)
        if len(selected) < min_selected:
            selected_indices = {item.index for item in selected}
            for candidate in deferred:
                if len(selected) >= min_selected:
                    break
                if candidate.index in selected_indices:
                    continue
                selected.append(candidate)
                selected_indices.add(candidate.index)

        requested_pair_cap = int(self.cfg.matching_pair_cap_per_reference)
        if requested_pair_cap <= 0:
            requested_pair_cap = int(pair_cap_per_reference or 0)
        for reference in selected:
            reference.pair_indices = self._select_matching_pairs(
                reference_index=int(reference.index),
                points=points,
                normals=normals,
                reliability=reliability,
                neighbor_ids=neighbor_cache.get(int(reference.index), np.empty((0,), dtype=np.int64)),
                pairwise_offsets=pairwise_offsets,
                pairwise_distances=pairwise_distances,
                model_bundle=model_bundle,
                pair_radius=pair_radius,
                pair_cap=requested_pair_cap,
            )

        return ASPSResult(
            references=selected,
            num_candidates=len(candidates),
            num_scored_candidates=len(scored_indices),
            pair_radius=float(pair_radius),
            reliability_scores=reliability,
            position_sigmas=position_sigma,
            normal_sigmas=normal_sigma,
            num_selected_after_nms=int(num_selected_after_nms),
            num_backfilled=max(0, int(len(selected) - num_selected_after_nms)),
        )

    def _select_alternative(
        self,
        strategy: str,
        points, normals, model_bundle,
        reliability, position_sigma, normal_sigma,
        pairwise_distances, pairwise_offsets,
        pair_radius, min_pair_distance, pair_cap_per_reference,
    ):
        scene_pair_cap = int(pair_cap_per_reference or self.cfg.matching_pair_cap_per_reference or 64)
        if scene_pair_cap <= 0:
            scene_pair_cap = 64

        feasible_indices, neighbor_cache = _build_feasible_and_neighbors(
            points, pairwise_distances, min_pair_distance, pair_radius, scene_pair_cap
        )

        top_m = int(self.cfg.top_m_reference_points)
        num_to_select = min(top_m, len(feasible_indices))

        if strategy == "random":
            selected_indices = self._random_select(feasible_indices, num_to_select)
        elif strategy == "uniform":
            selected_indices = self._uniform_select(points, feasible_indices, num_to_select)
        elif strategy == "curvature":
            selected_indices = self._curvature_select(points, normals, feasible_indices, num_to_select)
        elif strategy == "normal_stability":
            selected_indices = self._normal_stability_select(reliability, feasible_indices, num_to_select)
        else:
            raise ValueError(f"Unknown sampling_strategy: {strategy}")

        references = []
        for idx in selected_indices:
            nids = neighbor_cache.get(int(idx), np.array([], dtype=np.int64))
            pair_indices = [int(x) for x in nids[:scene_pair_cap].tolist()]
            references.append(ReferencePointScore(
                index=int(idx),
                score=1.0,
                reliability=float(reliability[int(idx)]),
                ambiguity_score=0.0,
                diversity_score=0.0,
                redundancy_penalty=1.0,
                position_sigma=float(position_sigma[int(idx)]),
                normal_sigma=float(normal_sigma[int(idx)]),
                pair_indices=pair_indices,
            ))

        return ASPSResult(
            references=references,
            num_candidates=len(references),
            num_scored_candidates=len(feasible_indices),
            pair_radius=float(pair_radius),
            reliability_scores=reliability,
            position_sigmas=position_sigma,
            normal_sigmas=normal_sigma,
            num_selected_after_nms=len(references),
            num_backfilled=0,
        )

    def _random_select(self, feasible_indices, num_to_select):
        return [int(x) for x in self.rng.choice(feasible_indices, size=num_to_select, replace=False).tolist()]

    def _uniform_select(self, points, feasible_indices, num_to_select):
        # Farthest Point Sampling on feasible points
        pts = np.asarray(points, dtype=np.float64)
        feasible = np.array(feasible_indices, dtype=np.int64)
        if len(feasible) <= num_to_select:
            return [int(x) for x in feasible.tolist()]

        # Start from a random point
        start_idx = int(self.rng.choice(len(feasible)))
        selected_local = [start_idx]
        selected_pts = [pts[feasible[start_idx]]]

        min_distances = np.full(len(feasible), np.inf, dtype=np.float64)

        for _ in range(1, num_to_select):
            latest = pts[feasible[selected_local[-1]]]
            dists = np.linalg.norm(pts[feasible] - latest[None, :], axis=1)
            min_distances = np.minimum(min_distances, dists)
            # Pick the point farthest from all selected
            min_distances[selected_local] = -1.0
            next_local = int(np.argmax(min_distances))
            selected_local.append(next_local)

        return [int(feasible[i]) for i in selected_local]

    def _curvature_select(self, points, normals, feasible_indices, num_to_select):
        # Compute curvature = lambda_min / sum(lambdas) for each feasible point via PCA on k-NN
        curvatures = []
        knn_val = int(self.cfg.knn)
        for idx in feasible_indices:
            idx_int = int(idx)
            dists = np.linalg.norm(points - points[idx_int][None, :], axis=1)
            k = min(knn_val + 1, len(points))
            nn_ids = np.argpartition(dists, k)[:k]
            nn_ids = nn_ids[nn_ids != idx_int][:knn_val]
            if len(nn_ids) < 3:
                curvatures.append((idx_int, 0.0))
                continue
            centered = points[nn_ids] - np.mean(points[nn_ids], axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, len(nn_ids))
            eigvals = np.sort(np.linalg.eigvalsh(cov))
            curvature = float(eigvals[0] / max(1e-12, np.sum(eigvals)))
            curvatures.append((idx_int, curvature))

        curvatures.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in curvatures[:num_to_select]]

    def _normal_stability_select(self, reliability, feasible_indices, num_to_select):
        scored = [(int(i), float(reliability[int(i)])) for i in feasible_indices]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:num_to_select]]

    def _select_matching_pairs(
        self,
        reference_index: int,
        points: np.ndarray,
        normals: np.ndarray,
        reliability: np.ndarray,
        neighbor_ids: np.ndarray,
        pairwise_offsets: np.ndarray,
        pairwise_distances: np.ndarray,
        model_bundle: ModelHashBundle,
        pair_radius: float,
        pair_cap: int,
    ) -> List[int]:
        if neighbor_ids.size == 0:
            return []

        cap = len(neighbor_ids) if int(pair_cap) <= 0 else min(int(pair_cap), int(neighbor_ids.size))
        shortlisted_neighbor_ids = self._shortlist_matching_pairs(
            reference_index=reference_index,
            normals=normals,
            reliability=reliability,
            neighbor_ids=neighbor_ids,
            pairwise_offsets=pairwise_offsets,
            pairwise_distances=pairwise_distances,
            pair_radius=pair_radius,
            target_count=cap,
        )

        entries = []
        max_bucket_score = 0.0
        for pair_index in shortlisted_neighbor_ids:
            distance = float(pairwise_distances[reference_index, pair_index])
            if distance <= 1e-12 or distance > float(pair_radius):
                continue

            feature = compute_pair_features(
                points[reference_index],
                normals[reference_index],
                points[pair_index],
                normals[pair_index],
            )
            if feature is None:
                continue

            f1, f2, f3, feature_distance = feature
            g = to_internal_feature_g(f1, f2, f3, feature_distance)
            key = discretize_feature(g, model_bundle.angle_step, model_bundle.distance_step)
            bucket_score = float(model_bundle.bucket_score(key))
            max_bucket_score = max(max_bucket_score, bucket_score)

            direction = pairwise_offsets[reference_index, pair_index] / max(distance, 1e-12)
            distance_bin = min(
                self.cfg.distance_bins - 1,
                int((distance / max(pair_radius, 1e-12)) * self.cfg.distance_bins),
            )
            angle = math.acos(max(-1.0, min(1.0, abs(float(normals[reference_index] @ direction)))))
            angle_bin = min(self.cfg.angle_bins - 1, int((angle / math.pi) * self.cfg.angle_bins))
            pair_reliability = float(
                0.5 * float(reliability[reference_index]) + 0.5 * float(reliability[pair_index])
            )
            entries.append(
                {
                    "pair_index": int(pair_index),
                    "bucket_score": bucket_score,
                    "pair_reliability": pair_reliability,
                    "bin_key": (distance_bin, angle_bin, _direction_bin(direction)),
                }
            )

        if not entries:
            return []

        ambiguity_weight = float(self.cfg.matching_pair_ambiguity_weight)
        reliability_weight = float(self.cfg.matching_pair_reliability_weight)
        normalization = max(1e-12, max_bucket_score)
        for entry in entries:
            normalized_bucket_score = float(entry["bucket_score"]) / normalization
            entry["pair_score"] = (
                reliability_weight * float(entry["pair_reliability"])
                + ambiguity_weight * normalized_bucket_score
            )

        entries.sort(key=lambda item: float(item["pair_score"]), reverse=True)
        selected_pairs: List[int] = []
        selected_set = set()
        occupied_bins = set()

        cap = min(int(cap), len(entries))
        for entry in entries:
            if len(selected_pairs) >= cap:
                break
            if entry["bin_key"] in occupied_bins:
                continue
            pair_index = int(entry["pair_index"])
            selected_pairs.append(pair_index)
            selected_set.add(pair_index)
            occupied_bins.add(entry["bin_key"])

        if len(selected_pairs) < cap:
            for entry in entries:
                if len(selected_pairs) >= cap:
                    break
                pair_index = int(entry["pair_index"])
                if pair_index in selected_set:
                    continue
                selected_pairs.append(pair_index)
                selected_set.add(pair_index)

        return selected_pairs

    def _shortlist_matching_pairs(
        self,
        reference_index: int,
        normals: np.ndarray,
        reliability: np.ndarray,
        neighbor_ids: np.ndarray,
        pairwise_offsets: np.ndarray,
        pairwise_distances: np.ndarray,
        pair_radius: float,
        target_count: int,
    ) -> List[int]:
        if neighbor_ids.size == 0:
            return []

        if target_count <= 0:
            target_count = int(neighbor_ids.size)
        multiplier = max(1.0, float(self.cfg.matching_pair_candidate_multiplier))
        candidate_limit = min(int(neighbor_ids.size), max(int(target_count), int(math.ceil(float(target_count) * multiplier))))
        if neighbor_ids.size <= candidate_limit:
            return [int(item) for item in neighbor_ids.tolist()]

        coarse_entries = []
        for pair_index in neighbor_ids.tolist():
            distance = float(pairwise_distances[reference_index, pair_index])
            if distance <= 1e-12 or distance > float(pair_radius):
                continue
            direction = pairwise_offsets[reference_index, pair_index] / max(distance, 1e-12)
            distance_bin = min(
                self.cfg.distance_bins - 1,
                int((distance / max(pair_radius, 1e-12)) * self.cfg.distance_bins),
            )
            angle = math.acos(max(-1.0, min(1.0, abs(float(normals[reference_index] @ direction)))))
            angle_bin = min(self.cfg.angle_bins - 1, int((angle / math.pi) * self.cfg.angle_bins))
            distance_center = (distance_bin + 0.5) / max(1, self.cfg.distance_bins)
            distance_ratio = float(distance / max(pair_radius, 1e-12))
            midrange_bonus = max(0.0, 1.0 - abs(distance_ratio - distance_center))
            pair_reliability = float(
                0.5 * float(reliability[reference_index]) + 0.5 * float(reliability[pair_index])
            )
            coarse_entries.append(
                {
                    "pair_index": int(pair_index),
                    "coarse_score": float(0.8 * pair_reliability + 0.2 * midrange_bonus),
                    "bin_key": (distance_bin, angle_bin, _direction_bin(direction)),
                }
            )

        if len(coarse_entries) <= candidate_limit:
            return [int(entry["pair_index"]) for entry in coarse_entries]

        coarse_entries.sort(key=lambda item: float(item["coarse_score"]), reverse=True)
        shortlisted: List[int] = []
        shortlisted_set = set()
        occupied_bins = set()
        for entry in coarse_entries:
            if len(shortlisted) >= candidate_limit:
                break
            if entry["bin_key"] in occupied_bins:
                continue
            pair_index = int(entry["pair_index"])
            shortlisted.append(pair_index)
            shortlisted_set.add(pair_index)
            occupied_bins.add(entry["bin_key"])

        if len(shortlisted) < candidate_limit:
            for entry in coarse_entries:
                if len(shortlisted) >= candidate_limit:
                    break
                pair_index = int(entry["pair_index"])
                if pair_index in shortlisted_set:
                    continue
                shortlisted.append(pair_index)
                shortlisted_set.add(pair_index)

        return shortlisted
