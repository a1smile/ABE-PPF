# First Stage Test Report

## 1. Completed In This Refactor

The repository now contains a new first-stage runnable implementation of the ambiguity-budget PPF method under `src/`:

- `src/ppf/core/`
  - PPF feature computation, hash-table build, pose candidate dataclasses, transform utilities.
- `src/ppf/sampling/asps.py`
  - Ambiguity-aware Scene Pair Selection.
- `src/ppf/retrieval/ubsp.py`
  - Uncertainty-aware Boundary Selective Probing.
- `src/ppf/backend/`
  - incremental mode pool, BRPMR controller, optional top-mode refinement hook.
- `src/ppf/pipeline/ambiguity_budget_ppf.py`
  - new end-to-end pipeline.
- `src/datasets/`
  - Stanford and LMO dataset loaders.
- `src/utils/`
  - logging, timers, metrics.

Also added:

- `configs/default_method.yaml`
- `configs/stanford_small.yaml`
- `configs/lmo_small.yaml`
- `scripts/run_stanford_small.py`
- `scripts/run_lmo_small.py`
- `scripts/clean_outputs.py`
- `docs/repo_audit.md`
- `deprecated/README.md`
- unit tests:
  - `tests/test_ppf_feature.py`
  - `tests/test_asps.py`
  - `tests/test_ubsp.py`
  - `tests/test_mode_pool.py`
  - `tests/test_brpmr.py`
  - `tests/test_vote_accumulator.py`

## 2. Current Code Structure

The new runtime path is:

1. dataset loader in `src/datasets`
2. model cache/hash build in `src/ppf/core/hash_table.py`
3. ASPS reference-point scoring and selection
4. per-reference pair retrieval with optional UBSP
5. per-reference vote accumulation and pose candidate generation
6. BRPMR incremental mode-pool update and stability check
7. final pose output and metric computation

The legacy `ppf/` package is preserved as the compatibility baseline path and reference implementation.

## 3. Deprecated / Historical Code

Stage-1 cleanup is currently documentation-first, not destructive deletion-first.

Marked as deprecated or historical:

- `ppf/rsmrq_hash.py`
- `ppf/adaptive_two_stage.py`
- `ppf/voting_robust.py`
- `ppf/kde_refine.py`
- `scripts/run_batch_stanford_adaptive.py`
- `scripts/experiments/`
- `experiments/tmp/`

These are documented in `docs/repo_audit.md` and `deprecated/README.md`. They are not on the new main path.

## 4. ASPS Status

ASPS is integrated and actively filters reference points before pair generation.

Observed behavior in smoke tests:

- Stanford:
  - reference points reduced from `400` raw scene points to `40`, `40`, `34`, `35`
- LMO:
  - reference points reduced from `393/399/287/400` raw scene points to `14`, `48`, `13`, `48`

Conclusion:

- ASPS is working as a front-end pruning module.
- It clearly reduces the scene reference set.
- A minimum-reference backfill path now exists for cases where NMS becomes too aggressive, and the pipeline records `num_selected_after_nms` and `num_backfilled` for debugging.
- The current LMO tuning now also uses a coarse-to-fine ASPS prescreen, so expensive ambiguity scoring is capped at `192` candidates without adding object-specific rules.

## 5. UBSP Status

UBSP is integrated and records trigger/query statistics.

Observed behavior:

- Stanford small test:
  - Bunny cases:
    - `ubsp_trigger_count = 280`, `260`
    - `neighbor_bucket_query_count = 315`, `309`
  - Dragon cases:
    - `ubsp_trigger_count = 135`, `131`
    - `neighbor_bucket_query_count = 147`, `148`
- LMO small test:
  - object 1 frame 0: `ubsp_trigger_count = 559`, `neighbor_bucket_query_count = 580`
  - object 1 frame 1: `ubsp_trigger_count = 527`, `neighbor_bucket_query_count = 549`
  - object 5 frame 0/1: no trigger

Conclusion:

- UBSP does trigger correctly in the current implementation.
- It is dataset/case sensitive.
- Stanford-side triggering required separate uncertainty scaling and boundary-fraction gating; after calibration it now activates on edge-near pairs instead of staying at zero.

## 6. BRPMR Status

BRPMR is integrated as an incremental mode-pool backend with:

- candidate-to-mode assignment
- mode merging
- stability metrics
- multi-round margin/coverage trend based early-stop decision logic

Observed behavior:

- Stanford:
  - early stop triggered on `1/4` cases, but only at round `4`
  - mode counts: `136`, `149`, `144`, `143`
- LMO:
  - low-evidence object-1 cases now bypass BRPMR and use raw top-1 plus light refinement
  - object-5 cases still use BRPMR, now with both dynamic heavy-branch budget shrink and a separate heavy-branch early-stop rule
  - mode counts: `1`, `96`, `1`, `95`

Conclusion:

- BRPMR is functionally integrated.
- The single-round score-gain stop rule has been replaced by a multi-round trend rule based on `margin_growth` and `coverage_growth`.
- The original mode explosion has been reduced from the thousand-level to the low-hundreds on the current smoke tests.
- LMO now uses two budget-adaptive branches:
  - low-evidence scenes skip BRPMR and take a refined raw top candidate
  - high-evidence scenes go through BRPMR with dynamic heavy-branch budget shrink based on reference count, candidate load, mode count, and round index
- A separate heavy-branch early-stop rule is now active only when the dynamic heavy branch is already saturated, so the more aggressive stop logic does not affect Stanford or low-evidence LMO cases.
- Stanford still tends to stabilize late in the current smoke configuration.
- Backend cost has been reduced substantially through cheaper candidate-to-mode and merge checks plus dynamic overflow pruning, and recent front-end/runtime gains also come from sparse vote-accumulator reuse and capped expensive ASPS scoring.

## 7. Stanford Small Test

Command used:

```bash
python scripts/run_stanford_small.py
```

Artifacts:

- `outputs/stanford_small_results.json`
- `outputs/stanford_small_log.txt`

Summary:

- cases: `4`
- success: `2`
- success rate: `0.50`
- mean total runtime: `0.78 s`

Per-case highlights:

- Bunny, `Scene0`, variant `0.1`
  - success
  - `ADD = 0.003616`
  - runtime `0.832 s`
  - `early_stop_round = 4`
- Bunny, `Scene0`, variant `0.3`
  - success
  - `ADD = 0.006266`
  - runtime `0.928 s`
- Dragon, `Scene1`, variant `0.1`
  - fail
  - `ADD = 0.159057`
  - runtime `0.666 s`
- Dragon, `Scene1`, variant `0.3`
  - fail
  - `ADD = 0.095264`
  - runtime `0.710 s`

Failure records:

- object `2`, scene `1`, variant `0.1`
- object `2`, scene `1`, variant `0.3`

## 8. LMO Small Test

Command used:

```bash
python scripts/run_lmo_small.py
```

Artifacts:

- `outputs/lmo_small_results.json`
- `outputs/lmo_small_log.txt`

Summary:

- cases: `4`
- success: `4`
- success rate: `1.00`
- mean total runtime: `0.80 s`

Per-case highlights:

- object `1`, frame `0`
  - success
  - `ADD-S = 14.663846 mm`
  - runtime `0.415 s`
  - low-evidence raw path used
  - UBSP triggered heavily
- object `5`, frame `0`
  - success
  - `ADD-S = 11.625859 mm`
  - runtime `1.275 s`
  - `early_stop_round = 3`
  - note: the latest runtime drop comes from capped expensive ASPS scoring plus one fewer voting pair budget per reference
- object `1`, frame `1`
  - success
  - `ADD-S = 19.213399 mm`
  - runtime `0.412 s`
  - low-evidence raw path used
  - UBSP triggered heavily
- object `5`, frame `1`
  - success
  - `ADD-S = 12.644797 mm`
  - runtime `1.089 s`
  - `early_stop_round = 3`
  - note: this case now combines heavy-branch early stop with cheaper ASPS/voting front-end cost

Failure records:

- none

## 9. Current Issues

The first-stage implementation is runnable, but several issues remain:

- Stanford-side BRPMR early stop is still weak and currently appears only at the last round in one smoke-test case.
- Mode counts are reduced further, but Stanford dragon still accumulates `143-144` modes and does not converge to a reliable pose.
- The LMO configuration now reaches `4/4` on the smoke test with the heavy branch mode cap reduced to `95-96`, but the heavy object-5 path is still slower than the raw low-evidence branch even after the latest ASPS/voting cuts.
- Stanford failures are now concentrated on the dragon cases; the current front-end reduction improves time but does not fix those misses.
- Deprecated RS-MRQ-era modules are documented but not yet physically moved or deleted.

## 10. Recommendations For The Next Stage

- Reduce candidate explosion before BRPMR:
  - keep dataset-specific caps
  - continue tightening candidate quality without collapsing LMO recall
- Improve mode merging and representative update:
  - add stronger support-aware mode scoring
  - keep the new trend-based stop, but make Stanford-side batches stabilize earlier instead of only stopping at the last batch
- Calibrate UBSP separately for Stanford and LMO:
  - Stanford now uses explicit boundary-fraction gating
  - LMO still uses looser uncertainty expansion
- Revisit ASPS aggressiveness on LMO:
  - preserve the current generic coarse-to-fine prescreen
  - next optimize how many pair features are scored per selected reference so the `4/4` recall does not give back the runtime win
- Continue backend micro-optimization:
  - the latest gain came from cutting unnecessary rotation-distance calls
  - the latest additional gains came from dynamic heavy-branch candidate/mode shrink plus a separate heavy-branch early-stop rule
  - the latest front-end/runtime gain came from sparse vote-accumulator reuse plus capped expensive ASPS scoring on LMO
  - the next likely win is adaptive per-reference pair budgeting, because BRPMR is no longer the only dominant tail
- After the new path is stable, physically move or delete clearly unused RS-MRQ-specific files.

## 11. Stage-1 Acceptance Check

Current status against the requested stage-1 target:

- repository structure clearer: yes
- new `src/` method path added: yes
- old RS-MRQ path removed from new main runtime path: yes
- baseline path preserved: yes
- ASPS/UBSP/BRPMR can be switched independently: yes, via `method.use_asps/use_ubsp/use_brpmr`
- Stanford smoke test runs: yes
- LMO smoke test runs: yes
- per-run statistics logged to JSON: yes
- complete experiment sweep or parameter search: not attempted by design

Overall conclusion:

- Stage 1 is functionally complete as a runnable refactor/smoke-test milestone.
- It is not yet an optimized or fully stabilized experimental version.
- The main unfinished technical items are earlier Stanford-side BRPMR stopping, better dragon-case recall, and bringing the heavy LMO runtime even closer to the baseline regime without giving back the current `4/4` smoke-test recall.
