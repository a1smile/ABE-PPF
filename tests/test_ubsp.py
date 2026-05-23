import unittest

import numpy as np

from src.ppf.core.hash_table import BucketStatistics, ModelHashBundle, ModelPairEntry
from src.ppf.retrieval.ubsp import UBSPConfig, UBSPRetriever


class TestUBSP(unittest.TestCase):
    def test_neighbor_bucket_triggers_near_boundary(self):
        current_key = (0, 0, 0, 0)
        neighbor = (1, 0, 0, 0)
        bundle = ModelHashBundle(
            model_path="synthetic",
            points=np.zeros((2, 3), dtype=np.float64),
            normals=np.zeros((2, 3), dtype=np.float64),
            ref_rotations=np.tile(np.eye(3, dtype=np.float64)[None, :, :], (2, 1, 1)),
            alpha_m=np.zeros((2, 2), dtype=np.float32),
            model_diameter=1.0,
            angle_step=1.0,
            distance_step=1.0,
            min_pair_distance=0.1,
            table={
                current_key: [ModelPairEntry(mr=0, mi=1, g=(0.95, 0.1, 0.1, 0.1), bucket_key=current_key)],
                neighbor: [ModelPairEntry(mr=1, mi=0, g=(1.05, 0.1, 0.1, 0.1), bucket_key=neighbor)],
            },
            stats=BucketStatistics(
                total_pairs=2,
                bucket_counts={current_key: 1, neighbor: 1},
                bucket_scores={current_key: 0.5, neighbor: 0.5},
            ),
        )

        retriever = UBSPRetriever(
            UBSPConfig(
                lambda_value=2.0,
                max_expand_dims=1,
                sigma_max_distance=1.0,
                sigma_max_angle=1.0,
                max_bucket_size=10,
            )
        )
        matches = retriever.query(
            model_bundle=bundle,
            feature_g=np.array([0.95, 0.1, 0.1, 0.1], dtype=np.float64),
            position_sigma_ref=0.1,
            position_sigma_pair=0.1,
            normal_sigma_ref=0.1,
            normal_sigma_pair=0.1,
        )
        self.assertGreaterEqual(len(matches), 2)
        self.assertEqual(retriever.stats.ubsp_trigger_count, 1)
        self.assertEqual(retriever.stats.neighbor_bucket_query_count, 1)


if __name__ == "__main__":
    unittest.main()
