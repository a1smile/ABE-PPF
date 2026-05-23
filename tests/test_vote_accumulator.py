import math
import unittest

import numpy as np

from src.ppf.core.hash_table import BucketStatistics, ModelHashBundle, ModelPairEntry
from src.ppf.core.pose_hypothesis import MatchRecord
from src.ppf.voting.vote_accumulator import VoteAccumulator, VoteAccumulatorConfig


def _bundle() -> ModelHashBundle:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    alpha_m = np.zeros((2, 2), dtype=np.float64)
    ref_rotations = np.tile(np.eye(3, dtype=np.float64)[None, :, :], (2, 1, 1))
    key = (0, 0, 0, 0)
    return ModelHashBundle(
        model_path="dummy",
        points=points,
        normals=normals,
        alpha_m=alpha_m,
        ref_rotations=ref_rotations,
        model_diameter=1.0,
        angle_step=math.pi / 4.0,
        distance_step=1.0,
        min_pair_distance=0.1,
        table={key: [ModelPairEntry(mr=0, mi=1, g=(0.0, 0.0, 0.0, 1.0), bucket_key=key)]},
        stats=BucketStatistics(
            total_pairs=1,
            bucket_counts={key: 1},
            bucket_scores={key: 1.0},
        ),
    )


class TestVoteAccumulator(unittest.TestCase):
    def test_reused_accumulator_resets_only_touched_entries(self):
        accumulator = VoteAccumulator(
            _bundle(),
            VoteAccumulatorConfig(
                candidate_bins_per_model_ref=1,
                max_candidates_per_reference=8,
                min_vote_score=0.0,
                relative_score_threshold=0.0,
                bucket_score_scale=0.0,
            ),
        )
        reference_transform = np.eye(4, dtype=np.float64)
        match = MatchRecord(
            model_reference_index=0,
            model_pair_index=1,
            bucket_key=(0, 0, 0, 0),
            weight=1.0,
            is_neighbor_bucket=False,
            bucket_size=1,
        )

        accumulator.reset()
        accumulator.vote(match, alpha_scene=0.0, reference_index=7)
        first_candidates = accumulator.build_candidates(reference_transform=reference_transform)
        self.assertEqual(len(first_candidates), 1)
        self.assertEqual(first_candidates[0].reference_indices, {7})
        self.assertEqual(first_candidates[0].support_pair_count, 1)

        accumulator.reset()
        second_candidates = accumulator.build_candidates(reference_transform=reference_transform)
        self.assertEqual(second_candidates, [])

        accumulator.vote(match, alpha_scene=0.0, reference_index=11)
        third_candidates = accumulator.build_candidates(reference_transform=reference_transform)
        self.assertEqual(len(third_candidates), 1)
        self.assertEqual(third_candidates[0].reference_indices, {11})
        self.assertEqual(third_candidates[0].support_pair_count, 1)


if __name__ == "__main__":
    unittest.main()
