# Failure Cases

> Track consistently-failing objects across experiments.
> These inform ablation design but must NOT lead to object-specific parameters.

## Current Status (EXP-007)

### Stanford (14/18 success, 4 failures)

| Object | Failures | Variants | Pattern |
|--------|----------|----------|---------|
| `dragon_vrip_res2_0` | 2/3 | 0.1, 0.5 | Large model, complex geometry. Variant 0.3 succeeds (ADD=0.011m). 0.1 and 0.5 fail with ADD~0.087m (180° rotation ambiguity). |
| `xyzrgb_statuette_vr3_small_scaled` | 2/3 | 0.3, 0.5 | Smaller object with fine details. Variant 0.1 succeeds (ADD=0.009m). Higher occlusion variants fail. |

Previously-failing objects now passing (since ASPS refactor):
- `happy_vrip_res3_0`: was 0/3 in EXP-001, now 3/3
- `Armadillo_vres2_small_scaled`: was 1/3 in EXP-001, now 3/3

### LMO (17/32 success, 15 failures)

| Object | Failures | Pattern |
|--------|----------|---------|
| `obj_000006` (driller) | 4/4 | **Consistently fails all frames.** Long thin cylindrical shape with symmetry — PPF features from different orientations collide in same hash bins. ASPS ambiguity scoring may be filtering out the only correct reference points. |
| `obj_000008` | 3/4 | Fails on frames 1, 3, 4 (higher occlusion). Frame 0 succeeds. Object has repetitive geometric features. |
| `obj_000009` | 3/4 | Fails on frames 1, 3, 4. Frame 0 succeeds. Small object with symmetrical holes. |
| `obj_000012` | 2/4 | Fails on frames 0, 3. Frames 1, 4 succeed. |
| `obj_000001` | 1/4 | Frame 3 fails. Other 3 frames succeed. |
| `obj_000005` | 1/4 | Frame 1 fails. Other 3 frames succeed. |
| `obj_000011` | 1/4 | Frame 6 fails. Other 3 frames succeed. |

Previously-failing objects now passing (since LMO cap/rel adjustment):
- `obj_000010`: was 2/4 failing in EXP-002, now 4/4 success
- `obj_000001`: improved from 2/4 failing to 1/4 failing

## Cross-Experiment Failure Patterns

### Objects that fail across ALL experiments
- **obj_000006 (driller)**: 0/4 in EXP-001, EXP-002, EXP-005, EXP-007. This is the hardest case and likely requires higher-level method improvements beyond ASPS parameter tuning.
- **dragon_vrip_res2_0**: 3/3 failing in EXP-001, 1/3 in EXP-002/007. Improved by ASPS refactor but still has 2 persistent failures at higher occlusion (0.1, 0.5).

### Objects sensitive to cap/rel changes
- **obj_000010**: 4/4 at cap=40/rel=0.80 (EXP-002), 4/4 at cap=28/rel=0.65 (EXP-007). Unaffected by cap changes.
- **obj_000008**: 3/4 failing at both cap=40 and cap=28. Not sensitive to cap.

## Hypotheses for Failures (for ablation design, NOT for code changes)

1. **driller symmetry**: The cylindrical shape means many PPF features from different orientations are identical. ASPS correctly identifies this as "high ambiguity" and may filter out these reference points. But the object NEEDS those reference points — there are no low-ambiguity points on a cylinder. This suggests UBSP might help (retrieve neighbor buckets to recover correct matches that fall into adjacent bins), or the reliability threshold for high-curvature-but-symmetric objects should be relaxed.

2. **obj_000008 repetitive features**: Similar to driller but less extreme. The object has multiple similar-looking holes/edges. Mode pool may be splitting the correct pose across multiple modes.

3. **Stanford dragon rotation ambiguity**: dragon_vrip_res2_0 variant 0.3 works (resulting pose is correct within 12°), but 0.1 and 0.5 give ~137° rotation error — this is a 180° flip. The mode pool may be selecting a mirrored pose mode. BRPMR margin-based stopping may help if the correct mode consistently out-scores the flipped mode.

## Rules for Failure Analysis

1. These observations inform ablation experiment design, NOT object-specific parameter tuning.
2. If a fix only helps obj_000006 but hurts obj_000001, it is rejected.
3. If a fix improves success but adds >20% runtime, it needs additional evidence.
4. Always verify on both Stanford and LMO.
