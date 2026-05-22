import unittest

import numpy as np

from ppf.rsmrq_hash import PPFEntry, RSMRQHashTable


class TestRSMRQ(unittest.TestCase):
    def test_deterministic_offsets(self):
        w_levels = [[1.0, 1.0, 1.0, 1.0]]
        t1 = RSMRQHashTable(w_levels=w_levels, T_tables=2, merge_mode="union", seed=123, logger=None)
        t2 = RSMRQHashTable(w_levels=w_levels, T_tables=2, merge_mode="union", seed=123, logger=None)

        for i in range(2):
            self.assertTrue(np.allclose(t1.offsets[0][i], t2.offsets[0][i]))

    def test_query_retrieves_added(self):
        w_levels = [[1.0, 1.0, 1.0, 1.0]]
        ht = RSMRQHashTable(w_levels=w_levels, T_tables=1, merge_mode="union", seed=0, logger=None)
        g = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        entry = PPFEntry(mr=1, mi=2, g=(0.2, 0.2, 0.2, 0.2))
        ht.add(g, entry)

        buckets = ht.query_buckets(g)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(len(buckets[0]), 1)
        self.assertEqual(buckets[0][0].mr, 1)
        self.assertEqual(buckets[0][0].mi, 2)

    def test_truncate_merged_candidates_prefers_high_count_then_proxy(self):
        merged = {
            (0, 0): (PPFEntry(mr=0, mi=0, g=(0.0, 0.0, 0.0, 0.0)), 5),
            (1, 1): (PPFEntry(mr=1, mi=1, g=(0.0, 0.0, 0.0, 0.0)), 3),
            (2, 2): (PPFEntry(mr=2, mi=2, g=(0.0, 0.0, 0.0, 0.0)), 3),
        }

        truncated = RSMRQHashTable.truncate_merged_candidates(
            merged,
            cap=2,
            score_fn=lambda entry, count: 1.0 if entry.mr == 2 else 0.0,
        )

        self.assertEqual(len(truncated), 2)
        self.assertIn((0, 0), truncated)
        self.assertIn((2, 2), truncated)


if __name__ == "__main__":
    unittest.main()
