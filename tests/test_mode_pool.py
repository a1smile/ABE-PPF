import unittest

import numpy as np

from src.ppf.backend.mode_pool import ModePool, ModePoolConfig
from src.ppf.core.pose_hypothesis import PoseCandidate


def _candidate(tx: float, score: float, refs: set[int]) -> PoseCandidate:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = tx
    return PoseCandidate(
        transform=transform,
        score=score,
        model_reference_index=0,
        vote_bin=0,
        reference_indices=refs,
        support_pair_count=len(refs),
    )


class TestModePool(unittest.TestCase):
    def test_mode_pool_groups_close_candidates(self):
        pool = ModePool(
            ModePoolConfig(
                tau_t=0.05,
                tau_R=5.0,
                tau_t_merge=0.05,
                tau_R_merge=5.0,
                tau_margin=0.1,
                tau_entropy=1.0,
                tau_compactness=0.0,
                tau_coverage=0.0,
                tau_marginal_gain=100.0,
                min_rounds_before_stop=1,
            ),
            total_reference_points=10,
        )
        pool.add_candidates(
            [
                _candidate(0.00, 2.0, {0, 1}),
                _candidate(0.01, 1.5, {1, 2}),
                _candidate(0.20, 0.5, {9}),
            ]
        )
        self.assertEqual(len(pool.modes), 2)
        self.assertGreater(pool.best_mode().score, 2.5)

    def test_mode_pool_stops_on_plateaued_margin_and_coverage(self):
        pool = ModePool(
            ModePoolConfig(
                tau_t=0.05,
                tau_R=5.0,
                tau_t_merge=0.05,
                tau_R_merge=5.0,
                tau_margin=0.2,
                tau_entropy=1.0,
                tau_compactness=0.0,
                tau_coverage=0.19,
                trend_window=2,
                tau_margin_growth=0.01,
                tau_coverage_growth=0.01,
                min_rounds_before_stop=2,
            ),
            total_reference_points=10,
        )

        pool.add_candidates(
            [
                _candidate(0.00, 2.0, {0, 1}),
                _candidate(0.01, 1.5, {0, 1}),
                _candidate(0.20, 0.5, {8}),
            ]
        )
        should_stop, _ = pool.should_stop()
        self.assertFalse(should_stop)

        pool.add_candidates(
            [
                _candidate(0.00, 2.0, {0, 1}),
                _candidate(0.01, 1.5, {0, 1}),
                _candidate(0.20, 0.5, {8}),
            ]
        )
        should_stop, metrics = pool.should_stop()
        self.assertTrue(should_stop)
        self.assertLessEqual(metrics["margin_growth"], 0.01)
        self.assertLessEqual(metrics["coverage_growth"], 0.01)

    def test_mode_pool_requires_min_support_points_for_stop(self):
        pool = ModePool(
            ModePoolConfig(
                tau_t=0.05,
                tau_R=5.0,
                tau_t_merge=0.05,
                tau_R_merge=5.0,
                tau_margin=0.2,
                tau_entropy=1.0,
                tau_compactness=0.0,
                tau_coverage=0.19,
                trend_window=2,
                tau_margin_growth=0.01,
                tau_coverage_growth=0.01,
                min_rounds_before_stop=2,
                min_support_points_for_stop=3,
            ),
            total_reference_points=10,
        )

        batch = [
            _candidate(0.00, 2.0, {0, 1}),
            _candidate(0.01, 1.5, {0, 1}),
            _candidate(0.20, 0.5, {8}),
        ]
        pool.add_candidates(batch)
        pool.add_candidates(batch)
        should_stop, metrics = pool.should_stop()
        self.assertFalse(should_stop)
        self.assertEqual(int(metrics["best_support_count"]), 2)


if __name__ == "__main__":
    unittest.main()
