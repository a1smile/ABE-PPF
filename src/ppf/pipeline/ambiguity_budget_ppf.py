from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import open3d as o3d

from ppf.preprocess import (
    adaptive_subsample_and_calculate_normals_model,
    adaptive_subsample_and_calculate_normals_scene,
    subsample_and_calculate_normals_model,
    subsample_and_calculate_normals_scene,
)
from src.ppf.backend.brpmr import BRPMRConfig, BRPMRController
from src.ppf.backend.pose_refinement import PoseRefinementConfig, refine_pose
from src.ppf.core.hash_table import (
    ModelHashBundle,
    build_model_hash_bundle,
    cache_path_for_model,
    load_model_hash_bundle,
    save_model_hash_bundle,
)
from src.ppf.core.pose_hypothesis import MatchRecord, PipelinePrediction, PoseCandidate
from src.ppf.core.ppf_feature import angle_from_transformed_point, compute_pair_features, discretize_feature, to_internal_feature_g
from src.ppf.core.transform_utils import make_affine, rotation_matrix_from_axis_angle
from src.ppf.retrieval.ubsp import UBSPConfig, UBSPRetriever
from src.ppf.sampling.asps import ASPSConfig, ASPSSelector, ReferencePointScore
from src.ppf.voting.vote_accumulator import VoteAccumulator, VoteAccumulatorConfig
from src.utils.metrics import compute_metrics, success_from_metrics
from src.utils.timer import Timer


@dataclass
class RuntimeSample:
    dataset: str
    object_id: int
    object_name: str
    scene_id: int
    scene_name: str
    scene_variant: str
    model_path: str
    scene_path: str
    gt_transform: Optional[np.ndarray]
    frame_id: Optional[int] = None


def compute_transform_sg(scene_ref_p: np.ndarray, scene_ref_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    normal = np.asarray(scene_ref_n, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    axis = np.cross(normal, ex)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        axis = ey.copy()
        axis_norm = 1.0
    axis /= axis_norm
    angle = math.acos(max(-1.0, min(1.0, float(normal @ ex))))
    rotation = rotation_matrix_from_axis_angle(axis, angle)
    translation = rotation @ (-np.asarray(scene_ref_p, dtype=np.float64))
    return make_affine(rotation, translation), rotation


class AmbiguityBudgetPPF:
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        method_cfg = dict(cfg.get("method", {}) or {})
        self.use_asps = bool(method_cfg.get("use_asps", True))
        self.use_ubsp = bool(method_cfg.get("use_ubsp", True))
        self.use_brpmr = bool(method_cfg.get("use_brpmr", True))
        self.seed = int(cfg.get("seed", 0))

        self.sampling_leaf = float(cfg.get("sampling_leaf", 0.01))
        self.normal_k = int(cfg.get("normal_k", 10))
        self.angle_step = math.radians(float(cfg.get("angle_step_deg", 12.0)))
        self.distance_step = float(cfg.get("distance_step_ratio", 0.6)) * float(self.sampling_leaf)
        self.scene_pair_cap_per_reference = int(cfg.get("scene_pair_cap_per_reference", 96))
        self.scene_pair_radius_ratio = float(cfg.get("scene_pair_radius_ratio", 0.5))
        self.success_threshold = float(cfg.get("success_threshold", 0.02))
        self.inlier_radius = float(cfg.get("inlier_radius", max(1.5 * self.sampling_leaf, 0.02)))
        self.cache_dir = str(cfg.get("cache_dir", "outputs/model_cache"))
        self.adaptive_downsample = bool(cfg.get("adaptive_downsample", True))
        self.adaptive_apply_to = str(cfg.get("adaptive_downsample_apply_to", "both")).lower()
        self.adaptive_cfg = dict(cfg.get("adaptive_downsample_cfg", {}) or {})

        self.asps_cfg = ASPSConfig(**dict(cfg.get("asps", {}) or {}))
        ubsp_raw = dict(cfg.get("ubsp", {}) or {})
        if "lambda" in ubsp_raw and "lambda_value" not in ubsp_raw:
            ubsp_raw["lambda_value"] = ubsp_raw.pop("lambda")
        self.ubsp_cfg = UBSPConfig(**ubsp_raw)
        self.brpmr_cfg = BRPMRConfig(**dict(cfg.get("brpmr", {}) or {}))
        self.voting_cfg = VoteAccumulatorConfig(**dict(cfg.get("voting", {}) or {}))

        backend_cfg = dict(cfg.get("backend", {}) or {})
        pose_refine_cfg = dict(backend_cfg.get("pose_refinement", {}) or {})
        self.pose_refinement_cfg = PoseRefinementConfig(
            enable=bool(pose_refine_cfg.get("enable", False)),
            max_iterations=int(pose_refine_cfg.get("max_iterations", 15)),
            max_correspondence_distance=float(pose_refine_cfg.get("max_correspondence_distance", self.inlier_radius)),
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _read_cloud(self, path: str) -> o3d.geometry.PointCloud:
        cloud = o3d.io.read_point_cloud(path)
        if len(cloud.points) == 0:
            raise ValueError(f"Point cloud is empty: {path}")
        return cloud

    def _prepare_model_cloud(self, model_path: str) -> np.ndarray:
        cloud = self._read_cloud(model_path)
        if self.adaptive_downsample and self.adaptive_apply_to in ("model", "both"):
            cloud, _ = adaptive_subsample_and_calculate_normals_model(cloud, k=self.normal_k, cfg=self.adaptive_cfg)
        else:
            cloud = subsample_and_calculate_normals_model(cloud, voxel_size=self.sampling_leaf, k=self.normal_k)
        return np.asarray(cloud.points, dtype=np.float64), np.asarray(cloud.normals, dtype=np.float64)

    def _prepare_scene_cloud(self, scene_path: str) -> tuple[np.ndarray, np.ndarray]:
        cloud = self._read_cloud(scene_path)
        if self.adaptive_downsample and self.adaptive_apply_to in ("scene", "both"):
            cloud, _ = adaptive_subsample_and_calculate_normals_scene(cloud, k=self.normal_k, cfg=self.adaptive_cfg)
        else:
            cloud = subsample_and_calculate_normals_scene(cloud, voxel_size=self.sampling_leaf, k=self.normal_k)
        return np.asarray(cloud.points, dtype=np.float64), np.asarray(cloud.normals, dtype=np.float64)

    def _model_cache_payload(self) -> Dict[str, object]:
        return {
            "angle_step_deg": round(math.degrees(self.angle_step), 6),
            "sampling_leaf": round(self.sampling_leaf, 6),
            "normal_k": int(self.normal_k),
            "distance_step": round(self.distance_step, 6),
            "adaptive_downsample": bool(self.adaptive_downsample),
            "adaptive_apply_to": str(self.adaptive_apply_to),
            "adaptive_cfg": json.dumps(self.adaptive_cfg, sort_keys=True, ensure_ascii=False),
        }

    def load_or_build_model(self, model_path: str) -> ModelHashBundle:
        cache_path = cache_path_for_model(self.cache_dir, model_path, self._model_cache_payload())
        if cache_path.exists():
            bundle = load_model_hash_bundle(cache_path)
            self._log(f"[ModelCache] Loaded {cache_path}")
            return bundle

        model_points, model_normals = self._prepare_model_cloud(model_path)
        bundle = build_model_hash_bundle(
            model_path=model_path,
            points=model_points,
            normals=model_normals,
            angle_step=self.angle_step,
            distance_step=self.distance_step,
        )
        save_model_hash_bundle(cache_path, bundle)
        self._log(f"[ModelCache] Built {cache_path}")
        return bundle

    def _fallback_references(
        self,
        scene_points: np.ndarray,
        scene_normals: np.ndarray,
        model_bundle: ModelHashBundle,
        limit: int,
    ) -> List[ReferencePointScore]:
        pair_radius = max(float(model_bundle.distance_step * 6.0), float(model_bundle.model_diameter * self.scene_pair_radius_ratio))
        references: List[ReferencePointScore] = []
        for index in range(scene_points.shape[0]):
            offsets = scene_points - scene_points[index][None, :]
            distances = np.linalg.norm(offsets, axis=1)
            neighbor_ids = np.where((distances > model_bundle.min_pair_distance) & (distances <= pair_radius))[0]
            if neighbor_ids.size == 0:
                continue
            order = neighbor_ids[np.argsort(distances[neighbor_ids])]
            references.append(
                ReferencePointScore(
                    index=index,
                    score=1.0,
                    reliability=1.0,
                    ambiguity_score=0.0,
                    diversity_score=0.0,
                    redundancy_penalty=1.0,
                    position_sigma=float(self.sampling_leaf * 0.5),
                    normal_sigma=0.05,
                    pair_indices=[int(item) for item in order[: int(self.scene_pair_cap_per_reference)].tolist()],
                )
            )
            if len(references) >= int(limit):
                break
        return references

    def _single_bucket_matches(self, model_bundle: ModelHashBundle, feature_g: np.ndarray) -> List[MatchRecord]:
        key = discretize_feature(feature_g, model_bundle.angle_step, model_bundle.distance_step)
        bucket_size = model_bundle.bucket_size(key)
        entries = list(model_bundle.query(key))
        if bucket_size > int(self.ubsp_cfg.max_bucket_size):
            entries = entries[: int(self.ubsp_cfg.max_bucket_size)]
        weight = min(1.0, float(self.ubsp_cfg.max_bucket_size) / max(1, bucket_size)) if bucket_size > 0 else 1.0
        return [
            MatchRecord(
                model_reference_index=entry.mr,
                model_pair_index=entry.mi,
                bucket_key=key,
                weight=float(weight),
                is_neighbor_bucket=False,
                bucket_size=bucket_size,
            )
            for entry in entries
        ]

    def run(self, sample: RuntimeSample) -> Dict[str, object]:
        rng = np.random.default_rng(self.seed)
        prediction: Optional[PipelinePrediction] = None
        debug: Dict[str, object] = {}
        metrics: Dict[str, float] = {}
        stats: Dict[str, object] = {}
        with Timer() as total_timer:
            model_bundle = self.load_or_build_model(sample.model_path)
            scene_points, scene_normals = self._prepare_scene_cloud(sample.scene_path)

            runtime_asps = 0.0
            runtime_ubsp = 0.0
            runtime_voting = 0.0
            runtime_brpmr = 0.0

            num_reference_points_before_asps = int(scene_points.shape[0])
            if self.use_asps:
                with Timer() as timer_asps:
                    asps = ASPSSelector(self.asps_cfg)
                    asps_result = asps.select(
                        scene_points,
                        scene_normals,
                        model_bundle,
                        pair_cap_per_reference=int(self.scene_pair_cap_per_reference),
                    )
                runtime_asps += timer_asps.elapsed
                references = list(asps_result.references)
                position_sigmas = asps_result.position_sigmas
                normal_sigmas = asps_result.normal_sigmas
                pair_radius = asps_result.pair_radius
                asps_debug = {
                    "num_candidates": int(asps_result.num_candidates),
                    "num_scored_candidates": int(asps_result.num_scored_candidates),
                    "num_selected_after_nms": int(asps_result.num_selected_after_nms),
                    "num_backfilled": int(asps_result.num_backfilled),
                }
            else:
                references = self._fallback_references(
                    scene_points=scene_points,
                    scene_normals=scene_normals,
                    model_bundle=model_bundle,
                    limit=int(self.asps_cfg.top_m_reference_points),
                )
                position_sigmas = np.full((scene_points.shape[0],), max(self.sampling_leaf * 0.5, 1e-6), dtype=np.float64)
                normal_sigmas = np.full((scene_points.shape[0],), 0.05, dtype=np.float64)
                pair_radius = max(float(model_bundle.distance_step * 6.0), float(model_bundle.model_diameter * self.scene_pair_radius_ratio))
                asps_debug = {
                    "num_candidates": int(len(references)),
                    "num_scored_candidates": int(len(references)),
                    "num_selected_after_nms": int(len(references)),
                    "num_backfilled": 0,
                }

            num_reference_points_after_asps = int(len(references))
            use_low_evidence_raw_path = (
                self.use_brpmr
                and int(self.brpmr_cfg.low_evidence_reference_threshold) > 0
                and num_reference_points_after_asps <= int(self.brpmr_cfg.low_evidence_reference_threshold)
            )
            if not references:
                prediction = PipelinePrediction(transform=np.eye(4, dtype=np.float64), score=0.0, mode_count=0, early_stop_round=-1)
                metrics = compute_metrics(
                    model_points=model_bundle.points,
                    scene_points=scene_points,
                    transform_pred=prediction.transform,
                    transform_gt=sample.gt_transform,
                    inlier_radius=self.inlier_radius,
                )
                stats = {
                    "object_id": int(sample.object_id),
                    "scene_id": int(sample.scene_id),
                    "frame_id": sample.frame_id,
                    "num_scene_points": int(scene_points.shape[0]),
                    "num_reference_points_before_asps": num_reference_points_before_asps,
                    "num_reference_points_after_asps": 0,
                    "num_scene_pairs": 0,
                    "num_hash_queries": 0,
                    "num_neighbor_bucket_queries": 0,
                    "num_returned_model_pairs": 0,
                    "num_pose_candidates": 0,
                    "num_modes": 0,
                    "early_stop_round": -1,
                    "used_pair_ratio": 0.0,
                    "runtime_asps": runtime_asps,
                    "runtime_ubsp": runtime_ubsp,
                    "runtime_voting": runtime_voting,
                    "runtime_brpmr": runtime_brpmr,
                    "rotation_error": float(metrics["rotation_error_deg"]),
                    "translation_error": float(metrics["translation_error"]),
                    "ADD": float(metrics["ADD"]),
                    "ADD_S": float(metrics["ADD_S"]),
                    "success": False,
                }
                debug = {"stop_reason": "no_reference_points"}
            else:
                ubsp = UBSPRetriever(self.ubsp_cfg)
                brpmr = (
                    BRPMRController(self.brpmr_cfg, total_reference_points=len(references))
                    if (self.use_brpmr and not use_low_evidence_raw_path)
                    else None
                )
                all_candidates: List[PoseCandidate] = []
                num_scene_pairs = 0
                num_returned_model_pairs = 0
                num_pose_candidates = 0
                processed_reference_count = 0
                stop_reason = ""
                stability_snapshots: List[Dict[str, float]] = []
                max_budget_pairs = max(1, int(self.brpmr_cfg.max_budget_pairs))

                batch_size = max(1, int(self.brpmr_cfg.batch_size_reference_points))
                shared_voter = VoteAccumulator(model_bundle, self.voting_cfg)
                for batch_start in range(0, len(references), batch_size):
                    batch = references[batch_start : batch_start + batch_size]
                    batch_candidates: List[PoseCandidate] = []

                    for reference in batch:
                        if num_scene_pairs >= max_budget_pairs:
                            stop_reason = "pair_budget_exhausted"
                            break

                        processed_reference_count += 1
                        reference_index = int(reference.index)
                        reference_point = scene_points[reference_index]
                        reference_normal = scene_normals[reference_index]
                        T_sg, R_sg = compute_transform_sg(reference_point, reference_normal)

                        shared_voter.reset()

                        pair_indices = list(reference.pair_indices)
                        if len(pair_indices) > int(self.scene_pair_cap_per_reference):
                            pair_indices = pair_indices[: int(self.scene_pair_cap_per_reference)]

                        if not pair_indices:
                            continue

                        for pair_index in pair_indices:
                            if num_scene_pairs >= max_budget_pairs:
                                stop_reason = "pair_budget_exhausted"
                                break

                            scene_pair_point = scene_points[pair_index]
                            scene_pair_normal = scene_normals[pair_index]
                            feature = compute_pair_features(reference_point, reference_normal, scene_pair_point, scene_pair_normal)
                            if feature is None:
                                continue

                            f1, f2, f3, distance = feature
                            if distance < model_bundle.min_pair_distance or distance > pair_radius:
                                continue

                            feature_g = to_internal_feature_g(f1, f2, f3, distance)
                            scene_pair_local = R_sg @ (scene_pair_point - reference_point)
                            alpha_scene = -angle_from_transformed_point(scene_pair_local)

                            if self.use_ubsp:
                                with Timer() as timer_ubsp:
                                    matches = ubsp.query(
                                        model_bundle=model_bundle,
                                        feature_g=feature_g,
                                        position_sigma_ref=float(position_sigmas[reference_index]),
                                        position_sigma_pair=float(position_sigmas[pair_index]),
                                        normal_sigma_ref=float(normal_sigmas[reference_index]),
                                        normal_sigma_pair=float(normal_sigmas[pair_index]),
                                    )
                                runtime_ubsp += timer_ubsp.elapsed
                            else:
                                matches = self._single_bucket_matches(model_bundle, feature_g)

                            num_scene_pairs += 1
                            num_returned_model_pairs += int(len(matches))
                            with Timer() as timer_voting:
                                for match in matches:
                                    shared_voter.vote(match, alpha_scene=alpha_scene, reference_index=reference_index)
                            runtime_voting += timer_voting.elapsed

                        with Timer() as timer_candidates:
                            reference_candidates = shared_voter.build_candidates(reference_transform=T_sg)
                        runtime_voting += timer_candidates.elapsed
                        batch_candidates.extend(reference_candidates)
                        num_pose_candidates += len(reference_candidates)

                    if batch_candidates:
                        if self.use_brpmr and brpmr is not None:
                            with Timer() as timer_brpmr:
                                should_stop, metrics = brpmr.update(batch_candidates)
                            runtime_brpmr += timer_brpmr.elapsed
                            stability_snapshots.append(metrics)
                            if should_stop:
                                stop_reason = "brpmr_early_stop"
                                break
                        else:
                            all_candidates.extend(batch_candidates)

                    if stop_reason:
                        break

                if self.use_brpmr and brpmr is not None:
                    prediction = brpmr.prediction()
                    all_candidates = []
                else:
                    if all_candidates:
                        all_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
                        best = all_candidates[0]
                        prediction = PipelinePrediction(
                            transform=best.transform.copy(),
                            score=float(best.score),
                            mode_count=1,
                            early_stop_round=-1,
                            metadata={"stability": {}},
                        )
                    else:
                        prediction = PipelinePrediction(
                            transform=np.eye(4, dtype=np.float64),
                            score=0.0,
                            mode_count=0,
                            early_stop_round=-1,
                            metadata={"stability": {}},
                        )

                if (
                    use_low_evidence_raw_path
                    and prediction.mode_count == 1
                    and int(self.brpmr_cfg.low_evidence_refine_iterations) > 0
                ):
                    low_evidence_refine_cfg = PoseRefinementConfig(
                        enable=True,
                        max_iterations=int(self.brpmr_cfg.low_evidence_refine_iterations),
                        max_correspondence_distance=float(self.brpmr_cfg.low_evidence_refine_distance or self.inlier_radius),
                    )
                    prediction.transform = refine_pose(
                        model_points=model_bundle.points,
                        scene_points=scene_points,
                        initial_transform=prediction.transform,
                        cfg=low_evidence_refine_cfg,
                    )

                if bool(self.pose_refinement_cfg.enable):
                    prediction.transform = refine_pose(
                        model_points=model_bundle.points,
                        scene_points=scene_points,
                        initial_transform=prediction.transform,
                        cfg=self.pose_refinement_cfg,
                    )

                metrics = compute_metrics(
                    model_points=model_bundle.points,
                    scene_points=scene_points,
                    transform_pred=prediction.transform,
                    transform_gt=sample.gt_transform,
                    inlier_radius=self.inlier_radius,
                )
                success = success_from_metrics(sample.dataset, metrics, self.success_threshold)

                stats = {
                    "object_id": int(sample.object_id),
                    "scene_id": int(sample.scene_id),
                    "frame_id": sample.frame_id,
                    "num_scene_points": int(scene_points.shape[0]),
                    "num_reference_points_before_asps": int(num_reference_points_before_asps),
                    "num_reference_points_after_asps": int(num_reference_points_after_asps),
                    "num_scene_pairs": int(num_scene_pairs),
                    "num_hash_queries": int(num_scene_pairs + ubsp.stats.neighbor_bucket_query_count if self.use_ubsp else num_scene_pairs),
                    "num_neighbor_bucket_queries": int(ubsp.stats.neighbor_bucket_query_count if self.use_ubsp else 0),
                    "num_returned_model_pairs": int(num_returned_model_pairs),
                    "num_pose_candidates": int(num_pose_candidates),
                    "num_modes": int(prediction.mode_count),
                    "early_stop_round": int(prediction.early_stop_round),
                    "used_pair_ratio": float(min(1.0, float(num_scene_pairs) / float(max_budget_pairs))),
                    "runtime_asps": float(runtime_asps),
                    "runtime_ubsp": float(runtime_ubsp),
                    "runtime_voting": float(runtime_voting),
                    "runtime_brpmr": float(runtime_brpmr),
                    "rotation_error": float(metrics["rotation_error_deg"]),
                    "translation_error": float(metrics["translation_error"]),
                    "ADD": float(metrics["ADD"]),
                    "ADD_S": float(metrics["ADD_S"]),
                    "success": bool(success),
                }
                debug = {
                    "stop_reason": stop_reason or "completed",
                    "processed_reference_count": int(processed_reference_count),
                    "low_evidence_raw_path_used": bool(use_low_evidence_raw_path),
                    "asps": asps_debug,
                    "stability_snapshots": stability_snapshots,
                    "ubsp": ubsp.stats.to_dict() if self.use_ubsp else {},
                    "brpmr_candidate_debug": dict(brpmr.last_candidate_debug) if (self.use_brpmr and brpmr is not None) else {},
                    "brpmr_budget_debug": dict(brpmr.last_dynamic_budget_debug) if (self.use_brpmr and brpmr is not None) else {},
                    "prediction_score": float(prediction.score),
                }

        stats["runtime_total"] = float(total_timer.elapsed)
        return {
            "sample": sample.__dict__,
            "prediction": {
                "transform": prediction.transform.tolist(),
                "score": float(prediction.score),
                "mode_count": int(prediction.mode_count),
                "early_stop_round": int(prediction.early_stop_round),
                "metadata": prediction.metadata,
            },
            "metrics": metrics,
            "stats": stats,
            "debug": debug,
        }
