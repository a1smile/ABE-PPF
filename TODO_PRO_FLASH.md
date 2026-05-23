# TODO: Pro Tasks and Flash Assignments

> Pro (main model) makes decisions. Flash agents execute.
> Tasks are executed in priority order. Do NOT start a task until its dependencies are met.

---

## Immediate Next: Stage 0C — Commit and Push Documentation

**Pro task**: Commit all Stage 0B documents and push to experiment branch.

- [ ] `git add` all new .md files
- [ ] Commit with message: `docs: add Stage 0B project documentation`
- [ ] Push to `exp/stage0-audit`

**Dependencies**: Stage 0B complete (all docs created)

---

## Stage 1 Priority: BRPMR on/off Ablation (Highest Priority)

**Rationale**: BRPMR is the easiest module to ablate — just disable early stop and run all references. This isolates BRPMR's contribution before ablating ASPS/UBSP.

### Task 1A-1: Create BRPMR-off Stanford config

- **Assign to**: @flash-executor
- **Goal**: Create `configs/stanford_expanded_no_brpmr.yaml` by copying `stanford_expanded.yaml` and setting `use_brpmr: false`
- **Allowed files**: `configs/stanford_expanded_no_brpmr.yaml` (new file)
- **Forbidden**: Modifying any source code or existing config
- **Verify**: File exists and contains `use_brpmr: false`

### Task 1A-2: Create BRPMR-off LMO config

- **Assign to**: @flash-executor
- **Goal**: Create `configs/lmo_expanded_no_brpmr.yaml` by copying `lmo_expanded.yaml` and setting `use_brpmr: false`
- **Allowed files**: `configs/lmo_expanded_no_brpmr.yaml` (new file)
- **Forbidden**: Modifying any source code or existing config
- **Verify**: File exists and contains `use_brpmr: false`

### Task 1A-3: Run BRPMR-off on Stanford

- **Assign to**: @flash-evaluator
- **Goal**: Run `python scripts/run_stanford_small.py --config configs/stanford_expanded_no_brpmr.yaml`
- **Output**: Save result to `outputs/stanford_expanded_no_brpmr_results.json`
- **Record**: success_rate, runtime_total, runtime_asps, runtime_brpmr, failed cases
- **Compare**: vs EXP-007 Stanford (14/18, 1.156s)

### Task 1A-4: Run BRPMR-off on LMO

- **Assign to**: @flash-evaluator
- **Goal**: Run `python scripts/run_lmo_small.py --config configs/lmo_expanded_no_brpmr.yaml`
- **Output**: Save result to `outputs/lmo_expanded_no_brpmr_results.json`
- **Record**: success_rate, runtime_total, runtime_asps, runtime_brpmr, failed cases
- **Compare**: vs EXP-007 LMO (17/32, 1.197s)

### Task 1A-5: Audit BRPMR ablation results

- **Assign to**: @flash-auditor
- **Goal**: Compare BRPMR-on vs BRPMR-off; check if early stop saves time without hurting accuracy
- **Update**: EXPERIMENT_REGISTRY.md with new entries

---

## Stage 1 Next: UBSP on/off Ablation

Same pattern as BRPMR ablation:
- Create `configs/*_no_ubsp.yaml`
- Run on both datasets
- Compare with EXP-007
- Update EXPERIMENT_REGISTRY.md

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
