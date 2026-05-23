import unittest

import numpy as np

from src.ppf.backend.brpmr import BRPMRConfig, BRPMRController
from src.ppf.core.pose_hypothesis import PoseCandidate


def _candidate(tx: float, score: float, ref_index: int) -> PoseCandidate:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = tx
    return PoseCandidate(
        transform=transform,
        score=score,
        model_reference_index=0,
        vote_bin=0,
        reference_indices={ref_index},
        support_pair_count=1,
    )


class TestBRPMRController(unittest.TestCase):
    def test_heavy_branch_budget_shrinks_candidates_and_modes(self):
        controller = BRPMRController(
            BRPMRConfig(
                max_candidates_per_round=16,
                candidate_score_ratio=0.1,
                batch_nms_translation=0.01,
                batch_nms_rotation=5.0,
                tau_t=0.02,
                tau_R=5.0,
                tau_t_merge=0.02,
                tau_R_merge=5.0,
                max_modes=8,
                min_mode_score_ratio=0.05,
                heavy_branch_reference_threshold=8,
                heavy_branch_candidate_threshold=10,
                heavy_branch_mode_threshold=6,
                heavy_branch_round_threshold=1,
                heavy_branch_decay=0.5,
                heavy_branch_min_candidate_cap=4,
                heavy_branch_min_mode_cap=3,
                heavy_branch_score_ratio_boost=0.05,
                heavy_branch_min_mode_score_ratio_boost=0.02,
            ),
            total_reference_points=12,
        )

        batch = [_candidate(tx=0.1 * float(index), score=20.0 - float(index), ref_index=index) for index in range(20)]
        controller.update(batch)

        budget_debug = controller.last_dynamic_budget_debug
        self.assertEqual(int(budget_debug["heavy_branch_active"]), 1)
        self.assertEqual(int(budget_debug["effective_max_candidates_per_round"]), 4)
        self.assertEqual(controller.last_candidate_debug["filtered_candidates"], 4)
        self.assertEqual(controller.last_candidate_debug["merged_candidates"], 4)
        self.assertEqual(int(budget_debug["effective_max_modes"]), 3)
        self.assertLessEqual(len(controller.mode_pool.modes), 3)
        self.assertGreaterEqual(int(budget_debug["overflow_dropped_candidates"]), 1)

    def test_reference_gate_keeps_dynamic_budget_disabled(self):
        controller = BRPMRController(
            BRPMRConfig(
                max_candidates_per_round=16,
                candidate_score_ratio=0.1,
                batch_nms_translation=0.01,
                batch_nms_rotation=5.0,
                tau_t=0.02,
                tau_R=5.0,
                tau_t_merge=0.02,
                tau_R_merge=5.0,
                max_modes=8,
                min_mode_score_ratio=0.05,
                heavy_branch_reference_threshold=12,
                heavy_branch_candidate_threshold=10,
                heavy_branch_round_threshold=1,
                heavy_branch_decay=0.5,
                heavy_branch_min_candidate_cap=4,
                heavy_branch_min_mode_cap=3,
                heavy_branch_score_ratio_boost=0.05,
                heavy_branch_min_mode_score_ratio_boost=0.02,
            ),
            total_reference_points=6,
        )

        batch = [_candidate(tx=0.1 * float(index), score=20.0 - float(index), ref_index=index) for index in range(20)]
        controller.update(batch)

        budget_debug = controller.last_dynamic_budget_debug
        self.assertEqual(int(budget_debug["heavy_branch_active"]), 0)
        self.assertEqual(int(budget_debug["effective_max_candidates_per_round"]), 16)
        self.assertEqual(int(budget_debug["effective_max_modes"]), 8)

    def test_heavy_branch_early_stop_can_trigger_before_standard_rule(self):
        controller = BRPMRController(
            BRPMRConfig(
                max_candidates_per_round=8,
                candidate_score_ratio=0.0,
                batch_nms_translation=0.0,
                batch_nms_rotation=0.0,
                tau_t=0.02,
                tau_R=5.0,
                tau_t_merge=0.02,
                tau_R_merge=5.0,
                tau_margin=0.95,
                tau_entropy=1.0,
                tau_compactness=0.0,
                tau_coverage=0.9,
                trend_window=2,
                tau_margin_growth=0.0,
                tau_coverage_growth=0.0,
                min_rounds_before_stop=2,
                max_modes=2,
                min_mode_score_ratio=0.0,
                heavy_branch_reference_threshold=8,
                heavy_branch_candidate_threshold=2,
                heavy_branch_round_threshold=1,
                heavy_branch_decay=1.0,
                heavy_branch_min_candidate_cap=8,
                heavy_branch_min_mode_cap=2,
                heavy_branch_stop_min_round=2,
                heavy_branch_tau_margin=0.2,
                heavy_branch_tau_coverage=0.2,
                heavy_branch_tau_entropy=1.0,
                heavy_branch_tau_compactness=0.0,
                heavy_branch_tau_margin_drift=0.1,
                heavy_branch_min_support_points=4,
                heavy_branch_mode_saturation_ratio=1.0,
            ),
            total_reference_points=12,
        )

        batch_round1 = [
            _candidate(tx=0.000, score=10.0, ref_index=0),
            _candidate(tx=0.004, score=9.0, ref_index=1),
            _candidate(tx=0.100, score=2.0, ref_index=8),
        ]
        batch_round2 = [
            _candidate(tx=0.001, score=10.0, ref_index=2),
            _candidate(tx=0.005, score=9.0, ref_index=3),
            _candidate(tx=0.100, score=2.0, ref_index=9),
        ]

        should_stop, _ = controller.update(batch_round1)
        self.assertFalse(should_stop)

        should_stop, metrics = controller.update(batch_round2)
        self.assertTrue(should_stop)
        self.assertEqual(controller.early_stop_round, 2)
        self.assertEqual(int(controller.last_dynamic_budget_debug["heavy_branch_early_stop"]), 1)
        self.assertEqual(int(controller.last_dynamic_budget_debug["standard_stop_ready"]), 0)
        self.assertGreater(metrics["top1_top2_margin"], 0.2)
        self.assertGreater(metrics["visible_coverage"], 0.2)


if __name__ == "__main__":
    unittest.main()
