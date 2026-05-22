import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PPFEntry:
    mr: int
    mi: int
    g: Optional[Tuple[float, float, float, float]]


class RSMRQHashTable:
    """
    RS-MRQ hashing with multi-level, multi-table bucket lookup.
    """

    def __init__(
        self,
        w_levels: List[List[float]],
        T_tables: int,
        merge_mode: str = "union",
        seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        assert merge_mode in ("union", "count")
        self.w_levels = [np.array(w, dtype=np.float32) for w in w_levels]
        self.L = len(self.w_levels)
        self.T = int(T_tables)
        self.merge_mode = merge_mode
        self.seed = seed
        self.logger = logger

        rng = np.random.RandomState(seed)
        self.offsets: List[List[np.ndarray]] = []
        for l in range(self.L):
            wl = self.w_levels[l]
            level_offsets = []
            for _ in range(self.T):
                u = rng.uniform(low=0.0, high=wl).astype(np.float32)
                level_offsets.append(u)
            self.offsets.append(level_offsets)

        self.tables: List[List[Dict[Tuple[int, int, int, int], List[PPFEntry]]]] = [
            [dict() for _ in range(self.T)] for __ in range(self.L)
        ]

        if self.logger:
            self.logger.info("[RS-MRQ] Initialized.")
            for l in range(self.L):
                self.logger.info(f"[RS-MRQ] Level {l}: w={self.w_levels[l].tolist()}")
                for t in range(self.T):
                    self.logger.info(f"[RS-MRQ] Level {l} Table {t}: u={self.offsets[l][t].tolist()}")
            self.logger.info(f"[RS-MRQ] merge_mode={self.merge_mode}, seed={self.seed}")

    @staticmethod
    def _quantize(g: np.ndarray, w: np.ndarray, u: np.ndarray) -> Tuple[int, int, int, int]:
        q = np.floor((g + u) / w).astype(np.int64)
        return int(q[0]), int(q[1]), int(q[2]), int(q[3])

    def add(self, g: np.ndarray, entry: PPFEntry) -> None:
        for l in range(self.L):
            w = self.w_levels[l]
            for t in range(self.T):
                u = self.offsets[l][t]
                key = self._quantize(g, w, u)
                bucket = self.tables[l][t].get(key)
                if bucket is None:
                    self.tables[l][t][key] = [entry]
                else:
                    bucket.append(entry)

    def query_buckets(self, g: np.ndarray) -> List[List[PPFEntry]]:
        buckets: List[List[PPFEntry]] = []
        for l in range(self.L):
            w = self.w_levels[l]
            for t in range(self.T):
                u = self.offsets[l][t]
                key = self._quantize(g, w, u)
                buckets.append(self.tables[l][t].get(key, []))
        return buckets

    def merge_candidates(
        self,
        buckets: List[List[PPFEntry]],
    ) -> Dict[Tuple[int, int], Tuple[PPFEntry, int]]:
        out: Dict[Tuple[int, int], Tuple[PPFEntry, int]] = {}
        for bucket in buckets:
            for entry in bucket:
                key = (entry.mr, entry.mi)
                if key not in out:
                    out[key] = (entry, 1)
                else:
                    old_entry, count = out[key]
                    out[key] = (old_entry, count + 1)
        if self.merge_mode == "union":
            out = {key: (value[0], 1) for key, value in out.items()}
        return out

    @staticmethod
    def truncate_merged_candidates(
        merged: Dict[Tuple[int, int], Tuple[PPFEntry, int]],
        cap: Optional[int],
        score_fn: Optional[Callable[[PPFEntry, int], float]] = None,
    ) -> Dict[Tuple[int, int], Tuple[PPFEntry, int]]:
        if cap is None:
            return merged

        cap = int(cap)
        if cap <= 0 or len(merged) <= cap:
            return merged

        items = list(merged.items())

        def _sort_key(item):
            key, (entry, count) = item
            proxy = 0.0
            if score_fn is not None:
                try:
                    proxy = float(score_fn(entry, count))
                except Exception:
                    proxy = 0.0
            return float(count), proxy, int(key[0]), int(key[1])

        items.sort(key=_sort_key, reverse=True)
        return dict(items[:cap])

    @staticmethod
    def bucket_stats(buckets: List[List[PPFEntry]]) -> Tuple[int, int]:
        return len(buckets), sum(len(bucket) for bucket in buckets)
