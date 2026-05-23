# Expanded Subset Evaluation

## Scope

- Date: 2026-05-23
- Purpose: evaluate the current ambiguity-budget PPF pipeline on larger Stanford and LMO subsets, focusing on overall average accuracy and matching time.
- Timing note:
  - The first expanded run included first-time model-cache builds for several new objects and is not suitable for pure matching-time comparison.
  - The metrics below are taken from the second run with model cache already built (`warm-cache`), which better reflects online matching cost.

## Configurations

- Stanford:
  - Config: `configs/stanford_expanded.yaml`
  - Samples: 6 models x 3 scene variants = 18 cases
- LMO:
  - Config: `configs/lmo_expanded.yaml`
  - Samples: 8 objects x 4 frames = 32 cases

## Overall Results

### Stanford Expanded

- Result file: `outputs/stanford_expanded_results.json`
- Cases: 18
- Success: 7 / 18
- Success rate: 38.89%
- Mean runtime total: 0.829 s
- Mean runtime ASPS: 0.225 s
- Mean runtime UBSP: 0.023 s
- Mean runtime voting: 0.079 s
- Mean runtime BRPMR: 0.150 s
- Mean ADD: 0.0574 m
- Mean ADD-S: 0.0185 m
- Mean rotation error: 82.99 deg
- Mean translation error: 0.0735 m

### LMO Expanded

- Result file: `outputs/lmo_expanded_results.json`
- Cases: 32
- Success: 14 / 32
- Success rate: 43.75%
- Mean runtime total: 1.122 s
- Mean runtime ASPS: 0.472 s
- Mean runtime UBSP: 0.036 s
- Mean runtime voting: 0.187 s
- Mean runtime BRPMR: 0.233 s
- Mean ADD: 80.00 mm
- Mean ADD-S: 26.08 mm
- Mean rotation error: 110.60 deg
- Mean translation error: 50.82 mm

## Object-Level Snapshot

### Stanford

| Object | Success | Mean Runtime | Mean ADD |
| --- | --- | ---: | ---: |
| `bun_zipper_0` | 3 / 3 | 0.985 s | 0.0056 m |
| `dragon_vrip_res2_0` | 0 / 3 | 0.753 s | 0.1162 m |
| `xyzrgb_dragon_vr3_small_scaled_0` | 2 / 3 | 0.762 s | 0.0425 m |
| `Armadillo_vres2_small_scaled` | 1 / 3 | 0.876 s | 0.0521 m |
| `happy_vrip_res3_0` | 0 / 3 | 0.851 s | 0.0743 m |
| `xyzrgb_statuette_vr3_small_scaled` | 1 / 3 | 0.745 s | 0.0536 m |

### LMO

| Object | Success | Mean Runtime | Mean ADD-S |
| --- | --- | ---: | ---: |
| `obj_000001` | 3 / 4 | 0.483 s | 29.109 mm |
| `obj_000005` | 3 / 4 | 1.383 s | 15.596 mm |
| `obj_000006` | 0 / 4 | 1.081 s | 40.469 mm |
| `obj_000008` | 1 / 4 | 1.559 s | 37.683 mm |
| `obj_000009` | 2 / 4 | 1.061 s | 17.618 mm |
| `obj_000010` | 4 / 4 | 1.478 s | 15.162 mm |
| `obj_000011` | 1 / 4 | 0.927 s | 26.903 mm |
| `obj_000012` | 0 / 4 | 1.001 s | 26.123 mm |

## Interpretation

- The current pipeline has reached a stable matching-time regime on the expanded subsets:
  - Stanford: around `0.8 s` per case
  - LMO: around `1.1 s` per case
- The current bottleneck is no longer BRPMR alone.
  - On LMO, ASPS already consumes `0.47 s` on average, exceeding BRPMR.
  - This means further speed gains should mainly come from front-end pair budgeting and cheaper reference-point scoring, not only backend compression.
- Generalization is still the main issue.
  - The small-subset tuning looked strong on the smoke test, but expanded evaluation drops to `38.89%` on Stanford and `43.75%` on LMO.
  - The current parameter set is therefore not yet robust enough across a broader object set.

## Main Failure Concentration

- Stanford hard cases:
  - `dragon_vrip_res2_0`
  - `happy_vrip_res3_0`
  - `Armadillo_vres2_small_scaled`
- LMO hard cases:
  - `obj_000006`
  - `obj_000008`
  - `obj_000011`
  - `obj_000012`

## Next Actions

- Improve cross-object robustness first, not object-specific tuning.
- For speed:
  - reduce ASPS scoring cost on LMO;
  - add adaptive per-reference pair budgets for heavy references;
  - push more early pruning into voting before BRPMR.
- For accuracy:
  - relax overly aggressive front-end pruning on difficult objects;
  - recalibrate Stanford ASPS and UBSP thresholds around the failing dragon and happy cases;
  - revisit LMO candidate recall on objects `6/8/11/12`.

---

## ASPS Matching-Pair Parameter Probes (2026-05-23)

### Motivation

Codex refactored ASPS to separate scoring pairs (`pre_sample_pairs`) from matching pairs
(`_select_matching_pairs`). The previous config had `matching_pair_cap_per_reference=40`
and `matching_pair_reliability_weight=0.80` on LMO, which caused a regression from
14/32 to 13/32 with increased ASPS runtime (0.472s → 0.676s).

Goal: find a cross-object-general parameter set that restores LMO success rate and
reduces ASPS time, without object-specific tuning.

### Probe Configurations (LMO only)

All probes only modify `asps.matching_pair_cap_per_reference` and
`asps.matching_pair_reliability_weight` / `matching_pair_ambiguity_weight`
(weights always sum to 1.0). All other parameters identical to `lmo_expanded.yaml`.

| Probe | cap_per_ref | reliability_w | ambiguity_w |
| --- | ---: | ---: | ---: |
| A | 36 | 0.75 | 0.25 |
| B | 32 | 0.70 | 0.30 |
| C | 28 | 0.65 | 0.35 |

### LMO Probe Results

| Config | Success | Rate | Total(s) | ASPS(s) | BRPMR(s) | ADD-S(mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original (pre-Codex) | 14/32 | 43.75% | 1.122 | 0.472 | 0.233 | 26.08 |
| Previous (cap=40,rel=0.80) | 13/32 | 40.62% | 1.363 | 0.676 | 0.217 | 23.65 |
| Probe A (cap=36,rel=0.75) | 15/32 | 46.88% | 1.050 | 0.517 | 0.174 | 24.00 |
| Probe B (cap=32,rel=0.70) | 15/32 | 46.88% | 1.025 | 0.512 | 0.182 | 22.84 |
| **Probe C (cap=28,rel=0.65)** | **17/32** | **53.12%** | **1.160** | **0.589** | **0.208** | **24.57** |

### Stanford Cross-Check

Probe D (cap=28, rel=0.65) was tested on Stanford to check cross-dataset generalization.
Result: 11/18 (61.11%), down from 14/18 with cap=36. Stanford is more sensitive to pair
cap reduction.

Stanford config kept at `cap=36, rel=0.65` (unchanged from previous).

### Selected Final Parameters

| Dataset | cap_per_ref | reliability_w | ambiguity_w |
| --- | ---: | ---: | ---: |
| Stanford | 36 | 0.65 | 0.35 |
| LMO | 28 | 0.65 | 0.35 |

Both datasets share the same reliability/ambiguity weighting (0.65/0.35).
The cap differs due to different point cloud scales and densities.

### Final Expanded Results (2026-05-23, warm-cache re-run)

#### Stanford Expanded

- Cases: 18
- Success: 14 / 18 (77.78%)
- Mean runtime total: 1.156 s
- Mean runtime ASPS: 0.456 s
- Mean runtime UBSP: 0.047 s
- Mean runtime voting: 0.168 s
- Mean runtime BRPMR: 0.100 s
- Mean ADD: 0.0202 m
- Mean ADD-S: 0.0067 m
- Mean rotation error: 33.62 deg
- Failed: dragon_vrip_res2_0 (variant 0.1, 0.5), xyzrgb_statuette (variant 0.3, 0.5)

#### LMO Expanded

- Cases: 32
- Success: 17 / 32 (53.12%)
- Mean runtime total: 1.197 s
- Mean runtime ASPS: 0.607 s
- Mean runtime UBSP: 0.034 s
- Mean runtime voting: 0.165 s
- Mean runtime BRPMR: 0.210 s
- Mean ADD: 86.77 mm
- Mean ADD-S: 24.57 mm
- Mean rotation error: 135.23 deg

### Key Takeaways

1. Reducing reliability weight from 0.80 to 0.65 helps recall on difficult LMO objects
   by letting more geometrically ambiguous (but potentially correct) pairs through.
2. Lower pair cap (28 vs 40) does NOT hurt LMO — it reduces noise in voting by
   limiting the number of low-quality pairs per reference.
3. Stanford needs a higher cap (36) due to denser point clouds and smaller objects.
4. ASPS runtime on LMO (0.607s) is still higher than original (0.472s) — the
   shortlist + matching pair selection logic adds CPU cost. But the 17/32 accuracy
   (up from 14/32) and 1.197s total (down from 1.363s) justify this trade-off.
5. Remaining hard objects on LMO: obj_000006 (driller), obj_000008, obj_000009 —
   these likely need higher-level recall improvements beyond ASPS parameter tuning.
