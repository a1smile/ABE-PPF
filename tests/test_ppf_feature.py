import unittest

import numpy as np

from src.ppf.core.ppf_feature import boundary_distances, compute_pair_features, discretize_feature, to_internal_feature_g


class TestPPFFeature(unittest.TestCase):
    def test_compute_pair_features_returns_valid_tuple(self):
        p1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        p2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        feature = compute_pair_features(p1, n1, p2, n2)
        self.assertIsNotNone(feature)
        self.assertAlmostEqual(feature[3], 1.0, places=6)

    def test_discretize_and_boundary_distance(self):
        feature_g = to_internal_feature_g(0.0, 0.0, 0.0, 1.25)
        key = discretize_feature(feature_g, angle_step=0.5, distance_step=0.5)
        self.assertEqual(key[3], 2)
        deltas = boundary_distances(feature_g, key, angle_step=0.5, distance_step=0.5)
        self.assertEqual(deltas.shape[0], 4)
        self.assertTrue(np.all(deltas >= 0.0))


if __name__ == "__main__":
    unittest.main()
