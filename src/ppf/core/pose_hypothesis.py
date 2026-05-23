from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class PoseCandidate:
    transform: np.ndarray
    score: float
    model_reference_index: int
    vote_bin: int
    reference_indices: Set[int] = field(default_factory=set)
    support_pair_count: int = 0
    ambiguity_penalty: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelinePrediction:
    transform: np.ndarray
    score: float
    mode_count: int
    early_stop_round: int
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class MatchRecord:
    model_reference_index: int
    model_pair_index: int
    bucket_key: Tuple[int, int, int, int]
    weight: float
    is_neighbor_bucket: bool
    bucket_size: int


@dataclass
class ModelPairVote:
    vote_weight: float
    candidate_count: int
    reference_index: int
    bucket_score: float
    from_neighbor_bucket: bool
