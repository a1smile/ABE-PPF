import unittest

import numpy as np

from src.ppf.core.hash_table import build_model_hash_bundle
from src.ppf.sampling.asps import ASPSConfig, ASPSSelector


class TestASPS(unittest.TestCase):
    def test_asps_prefers_stable_points(self):
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        model_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
        bundle = build_model_hash_bundle(
            model_path="synthetic_model",
            points=model_points,
            normals=model_normals,
            angle_step=np.deg2rad(12.0),
            distance_step=0.02,
        )

        scene_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.05, 0.05, 0.0],
                [0.5, 0.5, 0.2],
            ],
            dtype=np.float64,
        )
        scene_normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        selector = ASPSSelector(
            ASPSConfig(
                top_m_reference_points=2,
                pre_sample_pairs=4,
                normal_stability_threshold=0.1,
                voxel_nms_size=0.01,
                pair_radius_ratio=1.0,
                knn=3,
                seed=0,
            )
        )
        result = selector.select(scene_points, scene_normals, bundle)
        self.assertLess(result.reliability_scores[-1], result.reliability_scores[0])
        self.assertLessEqual(len(result.references), 2)
        self.assertTrue(all(reference.index != 5 for reference in result.references))

    def test_asps_backfills_when_nms_is_too_aggressive(self):
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        model_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
        bundle = build_model_hash_bundle(
            model_path="synthetic_model",
            points=model_points,
            normals=model_normals,
            angle_step=np.deg2rad(12.0),
            distance_step=0.02,
        )

        scene_points = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.03, 0.00, 0.0],
                [0.06, 0.00, 0.0],
                [0.09, 0.00, 0.0],
                [0.12, 0.00, 0.0],
            ],
            dtype=np.float64,
        )
        scene_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (scene_points.shape[0], 1))

        selector = ASPSSelector(
            ASPSConfig(
                top_m_reference_points=4,
                min_selected_points=3,
                pre_sample_pairs=4,
                normal_stability_threshold=0.1,
                voxel_nms_size=0.2,
                pair_radius_ratio=1.0,
                knn=2,
                seed=0,
            )
        )
        result = selector.select(scene_points, scene_normals, bundle)
        self.assertEqual(len(result.references), 3)

    def test_asps_limits_scored_candidates(self):
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        model_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
        bundle = build_model_hash_bundle(
            model_path="synthetic_model",
            points=model_points,
            normals=model_normals,
            angle_step=np.deg2rad(12.0),
            distance_step=0.02,
        )

        scene_points = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.03, 0.00, 0.0],
                [0.06, 0.00, 0.0],
                [0.09, 0.00, 0.0],
                [0.12, 0.00, 0.0],
                [0.15, 0.00, 0.0],
            ],
            dtype=np.float64,
        )
        scene_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (scene_points.shape[0], 1))

        selector = ASPSSelector(
            ASPSConfig(
                top_m_reference_points=3,
                max_scored_candidates=2,
                pre_sample_pairs=4,
                normal_stability_threshold=0.1,
                voxel_nms_size=0.01,
                pair_radius_ratio=1.0,
                knn=2,
                seed=0,
            )
        )
        result = selector.select(scene_points, scene_normals, bundle)
        self.assertEqual(result.num_scored_candidates, 2)

    def test_asps_matching_pairs_are_not_limited_by_presample(self):
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        model_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
        bundle = build_model_hash_bundle(
            model_path="synthetic_model",
            points=model_points,
            normals=model_normals,
            angle_step=np.deg2rad(12.0),
            distance_step=0.02,
        )

        scene_points = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.03, 0.00, 0.0],
                [0.06, 0.00, 0.0],
                [0.09, 0.00, 0.0],
                [0.12, 0.00, 0.0],
                [0.15, 0.00, 0.0],
            ],
            dtype=np.float64,
        )
        scene_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (scene_points.shape[0], 1))

        selector = ASPSSelector(
            ASPSConfig(
                top_m_reference_points=2,
                pre_sample_pairs=2,
                matching_pair_cap_per_reference=4,
                normal_stability_threshold=0.1,
                voxel_nms_size=0.01,
                pair_radius_ratio=1.0,
                knn=2,
                seed=0,
            )
        )
        result = selector.select(scene_points, scene_normals, bundle)
        self.assertTrue(result.references)
        self.assertTrue(all(len(reference.pair_indices) <= 4 for reference in result.references))
        self.assertTrue(any(len(reference.pair_indices) > 2 for reference in result.references))

    def test_asps_relaxes_reference_threshold_when_candidate_pool_is_too_small(self):
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        model_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
        bundle = build_model_hash_bundle(
            model_path="synthetic_model",
            points=model_points,
            normals=model_normals,
            angle_step=np.deg2rad(12.0),
            distance_step=0.02,
        )

        scene_points = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.03, 0.00, 0.0],
                [0.06, 0.00, 0.0],
                [0.09, 0.00, 0.0],
                [0.12, 0.00, 0.0],
            ],
            dtype=np.float64,
        )
        scene_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (scene_points.shape[0], 1))

        selector = ASPSSelector(
            ASPSConfig(
                top_m_reference_points=3,
                pre_sample_pairs=2,
                normal_stability_threshold=1.1,
                voxel_nms_size=0.01,
                pair_radius_ratio=1.0,
                knn=2,
                seed=0,
            )
        )
        result = selector.select(scene_points, scene_normals, bundle)
        self.assertGreaterEqual(result.num_scored_candidates, 3)
        self.assertEqual(len(result.references), 3)


if __name__ == "__main__":
    unittest.main()
