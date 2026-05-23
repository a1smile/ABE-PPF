# TODO: Pro Tasks and Flash Assignments

> Pro (main model) makes decisions. Flash agents execute.
> Tasks are executed in priority order. Do NOT start a task until its dependencies are met.

---

## Stage 0C: Commit and Push Documentation ✅ COMPLETE

All Stage 0B docs committed (ea0f5b3). ABEPPF.md and .claude/agents committed (51a9fa5). Tag stage0-complete created.

---

## Stage 1A: BRPMR Full Module Ablation ✅ COMPLETE

**Result**: BRPMR ON vs OFF shows module is critical for accuracy (Stanford 14→9, LMO 17→13). Mode pool + candidate filtering drive quality. Early stop triggers on 50% Stanford / 25% LMO. 4 LMO cases regress with BRPMR ON — needs component-level investigation.

**Configs created**: `configs/stanford_expanded_no_brpmr.yaml`, `configs/lmo_expanded_no_brpmr.yaml`

---

## Stage 1A-2: BRPMR Component Ablation ✅ COMPLETE

**Result**: `enable_early_stop` switch added to ModePoolConfig and BRPMRConfig. 4 component ablation configs created. Semantic check confirms 0-value semantics are safe for candidate filtering fields (max_candidates_per_round=0, candidate_score_ratio=0.0, batch_nms_*=0.0).

**Key findings**:
- Early stop: zero impact on success rate at current subset sizes (Stanford 14/18, LMO 17/32 unchanged)
- Candidate filtering OFF: Stanford unchanged (14/18), LMO improves to 18/32 but 2× BRPMR time. 2 of 4 regressions are CF-related (obj_000012, obj_000005). obj_000009 regression is NOT CF-related (mode pool issue).
- Mode pool is the primary accuracy driver (+5 Stanford, +4 LMO)

**Configs created**: `stanford_expanded_brpmr_no_es.yaml`, `lmo_expanded_brpmr_no_es.yaml`, `stanford_expanded_brpmr_no_cf.yaml`, `lmo_expanded_brpmr_no_cf.yaml`

---

## Stage 1A-3: BRPMR Candidate Filtering Calibration ✅ COMPLETE

**Result**: 7-config grid search (ratio ∈ {0.00,0.02,0.03}, max ∈ {128,192,256}) + 1 baseline ref. All ratio values at max=128 give identical 17/32 success (identical failure lists). Ratio filtering is redundant for recall at max=128. Baseline (ratio=0.05, max=128) is Pareto-optimal: lowest BRPMR time (0.210s) while maintaining same success. max=192 doubles time with no benefit; max=256 degrades to 16/32.

**Conclusion**: Current baseline is optimal. No parameter changes needed. Candidate filtering is an effective efficiency mechanism — ratio=0.05 provides best time/success tradeoff while max=128 prevents mode pool pollution.

**Configs created**: 7 `lmo_expanded_cf_r{R}_m{M}.yaml` files in configs/

---

## Stage 1B: UBSP on/off Ablation ✅ COMPLETE

**Result**: UBSP ON vs OFF on both datasets. Stanford: 14→12 (−2, −11.1pp, happy_vrip_res3_0 variants 0.3/0.5). LMO: 17→16 (−1, −3.1pp, obj_000011_f1). UBSP time cost negligible (~0.04s). No false matches introduced. UBSP is a validated innovation — evidence level strong.

**Configs created**: `configs/stanford_expanded_no_ubsp.yaml`, `configs/lmo_expanded_no_ubsp.yaml`

**Recommendation**: Keep UBSP ON in all production configs.

---

## Stage 1 Later: ASPS Sampling Comparison

- ASPS vs random sampling (same M reference points)
- ASPS vs uniform downsampling
- ASPS vs curvature-based sampling
- ASPS vs normal-stability-only

---

## Stage 4 Preview: Subset Scaling

After Stage 1-3 ablations confirm module contributions, expand validation subsets:

### Stanford Scaling
- Current: 6 models × 3 variants = 18 cases
- Target: Full stanford_retrieval_batch_all_variants.csv (need to check total size)
- Config: keep `configs/stanford_expanded.yaml` parameters

### LMO Scaling
- Current: 8 objects × 4 frames (from scene000002) = 32 cases
- Target: Add scene000003 or expand to full scene000002 frames
- Config: keep `configs/lmo_expanded.yaml` parameters

---

## Flash Agent Assignment Rules

1. @flash-executor: Code edits, config creation, bug fixes, single-test runs
2. @flash-evaluator: Full subset evaluation runs, metric collection, report updates
3. @flash-auditor: Result consistency checks, reproducibility verification, overfitting detection
4. @flash-method-analyst: ABEPPF.md vs code gap analysis, innovation validation, ablation planning
5. @flash-profiler: Runtime bottleneck identification, safe optimization suggestions

## Concurrency Rules

- Multiple Flash agents CAN run different experiments in parallel (e.g., Stanford BRPMR-off + LMO BRPMR-off)
- No two Flash agents may modify the same source code file simultaneously
- No two Flash agents may write to the same output file simultaneously
- Pro must review all Flash outputs before proceeding to next stage
