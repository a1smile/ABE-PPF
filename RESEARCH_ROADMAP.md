# Research Roadmap

> ABEPPF.md defines the research hypothesis. This roadmap defines the validation sequence.
> All modules must pass ablation validation before entering final experiments.

## Stage 0: Framework Stabilization and Audit (CURRENT)

- [x] Stage 0A: Read-only diagnosis of code, docs, and results
- [x] Stage 0B: Create project docs (INNOVATION_LEDGER, PROJECT_GOAL, RESEARCH_ROADMAP, EXPERIMENT_REGISTRY, FAILURE_CASES, TODO_PRO_FLASH)
- [ ] Stage 0C: Commit and push documentation to exp/stage0-audit branch
- [ ] Stage 0D: Reproducibility re-run on expanded subsets (Stanford 18, LMO 32)

## Stage 1: ASPS Validation and Ablation

Goal: Prove ASPS contributes beyond simple point-count reduction.

- [ ] 1A: BRPMR on/off ablation on expanded subsets (isolates BRPMR contribution first, since it's the easiest to ablate)
- [ ] 1B: UBSP on/off ablation on expanded subsets
- [ ] 1C: ASPS vs random sampling (same reference point count)
- [ ] 1D: ASPS vs uniform sampling (same reference point count)
- [ ] 1E: ASPS vs curvature-based sampling (same reference point count)
- [ ] 1F: ASPS vs normal-stability-only (without PPF discriminability)
- [ ] 1G: ASPS diversity formula comparison (occupancy ratio vs entropy)
- [ ] 1H: ASPS cap sweep on Stanford (28, 32, 36, 40, 48)
- [ ] 1I: Document ASPS findings and update INNOVATION_LEDGER

## Stage 2: UBSP Validation and Ablation

Goal: Prove UBSP improves recall without excessive neighbor queries.

- [ ] 2A: λ sweep on LMO (0.5, 1.0, 1.5, 2.0, 3.0)
- [ ] 2B: max_expand_dims sweep (m=1, 2, 3)
- [ ] 2C: UBSP vs boundary-only (without uncertainty criterion)
- [ ] 2D: UBSP vs uncertainty-only (without boundary distance)
- [ ] 2E: Neighbor bucket weight penalty sweep (0.25, 0.5, 0.75, 1.0)
- [ ] 2F: max_bucket_size threshold sweep
- [ ] 2G: σ_max parameter sweep
- [ ] 2H: Document UBSP findings and update INNOVATION_LEDGER

## Stage 3: BRPMR Validation and Ablation

Goal: Prove BRPMR early stop saves time without hurting accuracy.

- [ ] 3A: BRPMR on/off ablation on expanded subsets
- [ ] 3B: Candidate scoring formula comparison (simplified vs full 4-term)
- [ ] 3C: Stability threshold sweeps (τ_margin, τ_entropy, τ_compactness, τ_coverage)
- [ ] 3D: Heavy branch on/off and threshold sweep
- [ ] 3E: Mode merging threshold sweep
- [ ] 3F: Evaluate BRPMR-007 (periodic re-clustering) — implement if needed
- [ ] 3G: Document BRPMR findings and update INNOVATION_LEDGER

## Stage 4: Subset Scaling

Goal: Verify that validated modules generalize to larger subsets.

- [ ] 4A: Expand Stanford from 18 cases to full available subset
- [ ] 4B: Expand LMO from 32 cases to full scene000002 (or multi-scene)
- [ ] 4C: Check for object-level performance degradation
- [ ] 4D: Document scaling results

## Stage 5: Performance Profiling and Optimization

Goal: Identify and reduce bottlenecks without changing algorithm semantics.

- [ ] 5A: Profile ASPS (reliability estimation, pair scoring, shortlist, matching pair selection)
- [ ] 5B: Profile UBSP (uncertainty estimation, neighbor bucket query)
- [ ] 5C: Profile BRPMR (mode pool update, stability metric computation)
- [ ] 5D: Identify safe optimizations (caching, vectorization, pre-computation)
- [ ] 5E: Implement approved optimizations
- [ ] 5F: Verify optimizations don't change results

## Stage 6: Pre-Final Stability Verification

Goal: Confirm all results are reproducible on expanded subsets before running full experiments.

- [ ] 6A: Lock final Stanford config
- [ ] 6B: Lock final LMO config
- [ ] 6C: Reproducibility re-run ×3 (variance check)
- [ ] 6D: Full audit (success rates, runtimes, config consistency)

## Stage 7: Final Experiments and Paper Results

Goal: Produce complete paper-ready results.

- [ ] 7A: Overall comparison vs baseline PPF methods
- [ ] 7B: Module ablation (Ours Full, w/o ASPS, w/o UBSP, w/o BRPMR, Baseline)
- [ ] 7C: Efficiency statistics (pair counts, hash queries, candidate counts, timing breakdown)
- [ ] 7D: Parameter sensitivity analysis
- [ ] 7E: Failure case analysis and discussion
- [ ] 7F: Paper figure and table generation
