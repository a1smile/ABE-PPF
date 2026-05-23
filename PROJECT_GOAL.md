# Project Goal

## Research Objective

Develop a **general-purpose, efficient 6D pose estimation method** based on Point Pair Features (PPF) that improves accuracy over baseline PPF methods while maintaining or reducing computational cost.

## Method Name

**Ambiguity-aware and Budget-adaptive Efficient PPF Registration (ABE-PPF)**

## Core Hypothesis

> Complex-scene PPF failures stem not only from insufficient candidates, but from low-discriminability point pairs, high-collision hash buckets, repetitive-structure false votes, and unnecessary backend verification. This method controls computational budget across three stages: pair generation, candidate retrieval, and backend mode recovery — maintaining robustness while reducing ineffective matching and redundant computation.

## Three Innovation Modules

| Module | Role | One-liner |
|--------|------|-----------|
| **ASPS** | Front-end pair selection | Generate fewer low-value point pairs |
| **UBSP** | Mid-stream retrieval | Only probe neighbor buckets when cross-bin risk exists |
| **BRPMR** | Back-end mode recovery | Stop early when pose mode stabilizes |

## Datasets

- **Stanford**: 6 models × 3 scene variants (Bunny, Dragon, Armadillo, Happy Buddha, Statuette, Asian Dragon)
- **LMO** (Linemod-Occluded): 8 objects × 4 frames (obj_000001, 000005, 000006, 000008, 000009, 000010, 000011, 000012)

## Metrics

| Metric | Meaning |
|--------|---------|
| ADD / ADD-S | Pose estimation accuracy |
| Success Rate (SR) | % of cases with ADD < threshold |
| Rotation Error (deg) | Mean angular error |
| Translation Error (mm or m) | Mean positional error |
| Runtime Total (s) | Wall-clock time per case |
| Runtime ASPS (s) | Front-end pair selection time |
| Runtime UBSP (s) | Uncertainty-aware retrieval time |
| Runtime BRPMR (s) | Backend mode recovery time |

## Target

- Long-term: ≥90% success rate on expanded validation subsets
- Runtime: same order of magnitude as baseline, preferably faster
- Method generality: all optimizations must be dataset-level, not object-level or frame-level

## Non-Goals

- NOT competing with deep learning methods on RGB-based benchmarks
- NOT optimizing for a single object or scene
- NOT achieving maximum speed at the cost of accuracy
- NOT publishing without ablation evidence

## Design Constraints

1. All parameters are dataset-level (Stanford and LMO may differ)
2. No object-level or frame-level special cases
3. Accuracy gains must not come solely from increasing pair count, reference count, or candidate count
4. Every claimed innovation requires ablation experiment support
5. Always report both accuracy and runtime
6. ABEPPF.md is a research hypothesis — code and experiments drive final design
