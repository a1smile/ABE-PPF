import unittest

from ppf.adaptive_two_stage import should_escalate


class TestAdaptiveTwoStage(unittest.TestCase):
    def test_no_trigger_when_all_signals_are_confident(self):
        escalate, reason = should_escalate(
            {
                "pose_selection": {
                    "best_score": 0.80,
                    "top_scores": [0.80, 0.75],
                    "best_visibility_support": 0.40,
                }
            }
        )
        self.assertFalse(escalate)
        self.assertEqual(reason, "no_trigger")

    def test_low_best_score_triggers_upgrade(self):
        escalate, reason = should_escalate(
            {
                "pose_selection": {
                    "best_score": 0.71,
                    "top_scores": [0.71, 0.60],
                    "best_visibility_support": 0.40,
                }
            }
        )
        self.assertTrue(escalate)
        self.assertEqual(reason, "low_best_score")

    def test_missing_fields_escalate_conservatively(self):
        escalate, reason = should_escalate({"pose_selection": {"best_score": 0.80}})
        self.assertTrue(escalate)
        self.assertEqual(reason, "missing_top_score_margin")


if __name__ == "__main__":
    unittest.main()
