# Repository Audit

## 1. Current Main Entry Files

- `scripts/run_batch_stanford.py`
  - Main batch runner for Stanford and most internal experiments.
- `scripts/run_batch_stanford_adaptive.py`
  - Adaptive two-stage experiment runner.
- `scripts/run_lmo_all.py`
  - LMO suite orchestration wrapper that shells out to Stanford-style runners.
- `scripts/run_demo.py`
  - Single-case demo entry.
- `ppf/registration.py`
  - Actual end-to-end registration pipeline used by the current runners.
- `ppf_baseline_open3d.py`
  - Legacy standalone baseline-style script.

## 2. Current PPF Pipeline Flow

The current executable path is centered on `ppf/registration.py`:

1. Load YAML config and point clouds.
2. Downsample model and scene point clouds in `ppf/preprocess.py`.
3. Build or load model cache through:
   - `ppf/model_builder.py`
   - `ppf/model_cache_io.py`
4. For each scene reference point:
   - build local scene point pairs;
   - compute PPF features via `ppf/ppf_features.py`;
   - retrieve model pairs from either baseline hash or `ppf/rsmrq_hash.py`;
   - optionally apply robust bucket truncation via `ppf/voting_robust.py`;
   - vote into a `(model_ref, alpha_bin)` accumulator.
5. Convert accumulator peaks to pose candidates.
6. Optionally run:
   - pose selection in `ppf/pose_selection.py`;
   - pose clustering in `ppf/pose_clustering.py`;
   - legacy clustering in `ppf/clustering.py`;
   - KDE refinement in `ppf/kde_refine.py`;
   - ICP refinement in `ppf/refine_icp.py`.
7. Return final pose, debug info, and timing statistics.

## 3. RS-MRQ Related Files

These files are tightly coupled to the old recall-enhancement design:

- `ppf/rsmrq_hash.py`
- `ppf/model_builder.py`
  - Builds either baseline hash or RS-MRQ multi-table hash.
- `ppf/registration.py`
  - Contains the runtime merge, cap, and query logic for RS-MRQ.
- `tests/test_rsmrq.py`
- Configs:
  - `configs/Stanford/ablation_no_rsmrq_stanford.yaml`
  - `configs/LMO/ablation_no_rsmrq_lmo.yaml`
  - and other ablation/trade-off configs that still encode RS-MRQ assumptions.

## 4. Reusable Modules

These are the parts worth keeping conceptually in the new framework:

- `ppf/ppf_features.py`
  - Core pair feature computation and discretization.
- `ppf/preprocess.py`
  - Downsampling and normal estimation logic.
- `ppf/metrics.py`
  - ADD, ADD-S, rotation, translation, and inlier metrics.
- `ppf/bop_gt.py`
  - BOP/LMO GT loading helpers.
- `ppf/utils.py`
  - Basic rigid-transform math and logging helpers.
- `ppf/model_cache_io.py`
  - Cache persistence pattern.

## 5. Modules To Delete Or Deprecate

These should not stay on the new critical path:

- `ppf/rsmrq_hash.py`
  - Old multi-resolution random-shift retrieval core. Mark as deprecated after new pipeline is stable.
- `ppf/adaptive_two_stage.py`
  - Specific to the earlier stage-escalation logic, not the new ambiguity-budget method.
- `ppf/voting_robust.py`
  - Keep only as historical reference; the new method needs ambiguity-aware retrieval and budget-aware backend instead.
- `ppf/kde_refine.py`
  - Historical optional refinement path, not part of the new primary method.
- `scripts/run_batch_stanford_adaptive.py`
  - Tied to the previous adaptive two-stage experiment framing.
- `scripts/experiments/`
  - Historical result dumps and logs.
- `experiments/tmp/`
  - Contains transient analysis code and a nested unrelated repository; should stay out of the main method path.

Status for this phase:

- Not physically removed yet.
- Explicitly treated as deprecated or historical.
- New work should go under `src/`.

## 6. Baseline Code To Preserve

The following behavior must remain runnable for debugging and ablations:

- Basic PPF feature computation.
- Single-bucket model hash build.
- Standard scene pair generation.
- Standard pose candidate generation and vote accumulation.
- Dataset loading for Stanford and LMO.
- Metric computation.

Practical preservation plan:

- Keep the current `ppf/` implementation untouched as compatibility baseline.
- Add the new method under `src/` with its own scripts and configs.

## 7. Stanford / LMO Data Loading

### Stanford

- Manifest source:
  - `stanford_retrieval_batch_all_variants.csv`
- Important columns:
  - `pcd_path`
  - `expected_model_path`
  - `gt_path`
  - `scene_id`
  - `scene_name`
  - `scene_variant`
- Ground truth format:
  - `.xf` rigid transform files.
- Model files:
  - `data/stanford_bunny_ppf/models/*.ply`

### LMO

- Small/eval manifest source:
  - `lmo_subset_csvs/lmo_scene000002_eval_500.csv`
- Important columns:
  - `pcd_path`
  - `obj_token`
  - `frame_id`
  - `depth_path`
  - `expected_model_path`
  - `expected_obj_id`
- Ground truth source:
  - `scene_gt.json` via `ppf/bop_gt.py`
- Scene root:
  - `data/LM-O (Linemod-Occluded)/lmo_test_all/test/000002`
- Preprocessed instance clouds:
  - `data/LM-O (Linemod-Occluded)/lmo_test_all/test/_preprocessed_lmo/inst_pcd_visib/000002/*.ply`
- Model files:
  - `data/LM-O (Linemod-Occluded)/lmo_models/models_eval/obj_*.ply`

## 8. Repository Cleanup Notes

Observed non-method artifacts that should not be used by the new pipeline:

- `experiments/tmp/nature-skills-repo/`
  - Nested unrelated repository.
- `paper_outputs/`
  - Paper asset outputs, not runtime code.
- `scripts/build_*` reporting scripts
  - Useful for paper/report generation, not runtime method logic.
- `__pycache__` and historical result directories
  - Should remain outside the new first-stage runnable path.

## 9. Refactor Direction For Stage 1

The new first-stage implementation should:

- Introduce `src/ppf/core`, `sampling`, `retrieval`, `voting`, `backend`, and `pipeline`.
- Separate Stanford and LMO dataset loaders under `src/datasets`.
- Keep old baseline code available but off the main path.
- Route the new runnable method through:
  - `scripts/run_stanford_small.py`
  - `scripts/run_lmo_small.py`
- Record structured runtime statistics directly into `outputs/`.
