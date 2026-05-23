# Innovation Ledger

> **ABEPPF.md is a research hypothesis document, not a final design.**
> This ledger tracks implementation status and experimental evidence for each sub-component.
> All claims of "effective" require dedicated ablation experiments.

---

## ASPS: Ambiguity-aware Scene Pair Selection

### ASPS-001: Normal Stability Estimation
- **Doc design**: Q_n(s_i) = mean(|n_i·n_j|) over k-NN neighborhood; also PCA eigenvalue ratio C(s_i) = λ3/(λ1+λ2+λ3)
- **Code location**: `_estimate_reliability()` in `src/ppf/sampling/asps.py:83-116`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent. Code adds `curvature_penalty = exp(-10*curvature)` which penalizes noisy/high-curvature points more aggressively than the doc's "moderate curvature" guidance.
- **Evidence level**: weak — only tested as part of full ASPS, no ablation vs pure normal-consistency or pure curvature sampling
- **Retain?**: Yes, but the curvature penalty weight (-10×) should be parameterized and swept
- **Required ablation**: ASPS with reliability-only scoring vs full composite score

### ASPS-002: PPF Hash Bucket Discriminability
- **Doc design**: D(h) = log(N_all / (N(h)+1)); A(s_r) = mean(D(h_rj)) over K sampled pairs
- **Code location**: `model_bundle.bucket_score(key)` called in `asps.py:222`; used as `ambiguity_score` in `asps.py:234-238`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Needs verification — `bucket_score()` implementation in `src/ppf/core/hash_table.py` may differ from the log-ratio formula. The score feeds into `ambiguity_score = mean(bucket_score)` which is consistent with A(s_r).
- **Evidence level**: weak — no comparison of bucket_score formula variants
- **Retain?**: Yes
- **Required ablation**: Compare log-ratio D(h) vs alternative discriminability metrics

### ASPS-003: Pair Diversity Scoring
- **Doc design**: B(s_r) = -Σ p_b log(p_b), entropy over geometric bins (distance × angle × direction)
- **Code location**: `asps.py:235-236` — `diversity_score = len(occupied_bins) / total_bins`
- **Implementation status**: Simplified
- **Doc vs code alignment**: **Inconsistent**. Doc uses entropy; code uses bin-occupancy ratio. The occupancy ratio is simpler but loses sensitivity to distribution uniformity — a reference point covering 10/32 bins with 90% of pairs in 1 bin gets the same score as one with even distribution.
- **Evidence level**: weak
- **Retain?**: Yes, but consider upgrading to entropy if occupancy ratio proves insufficient
- **Required ablation**: Occupancy ratio vs entropy-based diversity; check if the difference matters on LMO hard objects

### ASPS-004: Spatial Redundancy Penalty
- **Doc design**: R(s_r) = exp(-d(s_r, S_selected)²/σ²), exponential distance penalty to already-selected points
- **Code location**: `asps.py:258-267` — voxel NMS with `voxel_nms_size` threshold
- **Implementation status**: Simplified
- **Doc vs code alignment**: **Inconsistent**. Doc uses continuous exponential penalty integrated into V(s_r); code uses hard NMS radius threshold as a post-processing filter. NMS is simpler and faster, but may discard valid points near cluster boundaries.
- **Evidence level**: weak
- **Retain?**: Yes, NMS is a reasonable simplification
- **Required ablation**: NMS radius sweep to verify current 0.01 (Stanford) / 12.0 (LMO) is not too aggressive

### ASPS-005: Composite Value Score V(s_r)
- **Doc design**: V(s_r) = Q_n(s_r) · (α·A(s_r) + β·B(s_r)) · (1−R(s_r))
- **Code location**: `asps.py:237-239` — `score = reliability * (alpha * ambiguity_score + beta * diversity_score)`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent in form. The (1−R) term is omitted from the score formula because NMS handles redundancy separately (post-filter rather than in-score).
- **Evidence level**: weak
- **Retain?**: Yes
- **Required ablation**: α/β weight sweep on expanded subset

### ASPS-006: Scoring/Matching Pair Separation
- **Doc design**: Not in ABEPPF.md. This is a code-level innovation: `pre_sample_pairs` used ONLY for reference point scoring; actual matching pairs selected independently via `_select_matching_pairs()`.
- **Code location**: `asps.py:204-207` (scoring), `asps.py:285-296` (matching pair selection)
- **Implementation status**: Implemented (code addition beyond doc)
- **Doc vs code alignment**: **Doc needs update** — this mechanism should be documented in ABEPPF.md
- **Evidence level**: moderate — parameter probes showed cap=28/36 with separate matching improves LMO recall vs cap=0 (pre-separation)
- **Retain?**: Yes, this is a key design decision
- **Required ablation**: Compare with and without separation (pre_sample_pairs used for both scoring AND matching)

### ASPS-007: Matching Pair Shortlist
- **Doc design**: Not in ABEPPF.md. `_shortlist_matching_pairs()` pre-filters candidate pairs by reliability + midrange bonus before detailed PPF scoring.
- **Code location**: `asps.py:420-494`
- **Implementation status**: Implemented (code addition beyond doc)
- **Doc vs code alignment**: **Doc needs update**
- **Evidence level**: weak
- **Retain?**: Yes, but the 0.8/0.2 reliability/midrange weighting needs validation
- **Required ablation**: Shortlist multiplier sweep; check if shortlist diversity constraint is necessary

### ASPS-008: Matching Pair Cap per Reference
- **Doc design**: Not in ABEPPF.md. `matching_pair_cap_per_reference` limits how many matching pairs each selected reference point generates.
- **Code location**: `asps.py:281-283`, `asps.py:397`
- **Implementation status**: Implemented (code addition beyond doc)
- **Doc vs code alignment**: **Doc needs update**
- **Evidence level**: moderate — LMO probe C (cap=28) gave 17/32 vs cap=40 giving 13/32; Stanford cap=36 optimal
- **Retain?**: Yes
- **Required ablation**: Already partially done (cap sweep on LMO); extend to Stanford cap sweep

---

## UBSP: Uncertainty-aware Boundary Selective Probing

### UBSP-001: Boundary Distance Calculation
- **Doc design**: Δ_k = min(g_k − l_k, u_k − g_k) for each of 4 PPF feature dimensions
- **Code location**: `ubsp.py:118` — calls `boundary_distances()` from `src/ppf/core/ppf_feature.py`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent
- **Evidence level**: weak — no comparison of Δ_k formula variants
- **Retain?**: Yes
- **Required ablation**: None specific to boundary distance; tested via UBSP on/off

### UBSP-002: Feature Uncertainty Estimation
- **Doc design**: σ_d = sqrt(σ_p²_ref + σ_p²_pair); σ_α = sqrt(σ_n²_ref + σ_u²); etc. with position sigma from PCA and normal sigma from neighborhood consistency
- **Code location**: `ubsp.py:51-70` — `_pair_uncertainty()`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent. Code uses `position_sigma_scale`, `normal_sigma_scale`, `distance_coupling_scale` parameters (all configurable) which are practical additions.
- **Evidence level**: weak
- **Retain?**: Yes
- **Required ablation**: Sigma scale parameter sweep

### UBSP-003: Cross-Bin Risk Criterion
- **Doc design**: Δ_k < λ·σ_k → query neighbor bucket in that dimension
- **Code location**: `ubsp.py:133-137` — `risk_mask = (deltas < lambda * sigma_g) & (sigma_g < sigma_max) & (boundary_fraction < boundary_fraction_max)`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent, with additional `boundary_fraction_max` guard
- **Evidence level**: weak — λ sweep not yet done
- **Retain?**: Yes
- **Required ablation**: λ sweep on LMO expanded (current λ=1.5 for LMO, 0.2 for Stanford)

### UBSP-004: Uncertainty Upper Bound
- **Doc design**: σ_k < σ_k^max — skip expansion if feature uncertainty is too large
- **Code location**: `ubsp.py:135` — `sigma_g < sigma_max` in risk_mask
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent
- **Evidence level**: weak
- **Retain?**: Yes
- **Required ablation**: σ_max sweep

### UBSP-005: Max Expand Dimensions
- **Doc design**: Select top-m riskiest dimensions (m=1 or 2); avoid full 4-dim combinatorial explosion
- **Code location**: `ubsp.py:144-147` — `risky_dims[:max_expand_dims]`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent (m=2 default)
- **Evidence level**: weak
- **Retain?**: Yes
- **Required ablation**: m=1 vs m=2 vs m=3 comparison

### UBSP-006: High-Ambiguity Bucket Constraint
- **Doc design**: Skip neighbor expansion if current or target bucket is high-ambiguity (large N(h))
- **Code location**: `ubsp.py:149-153` (current bucket check), `ubsp.py:164-167` (neighbor bucket check)
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent
- **Evidence level**: weak
- **Retain?**: Yes
- **Required ablation**: max_bucket_size threshold sweep

### UBSP-007: Neighbor Bucket Weight Penalty
- **Doc design**: Doc mentions "降低该点对投票权重" for ambiguous buckets; code applies 0.5× weight to neighbor bucket matches
- **Code location**: `ubsp.py:85` — `weight = ambiguity_factor * (0.5 if is_neighbor_bucket else 1.0)`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent in spirit (weight penalty for less-certain matches)
- **Evidence level**: weak
- **Retain?**: Yes
- **Required ablation**: Compare 0.5× vs 1.0× neighbor weight

---

## BRPMR: Budget-adaptive Reliable Pose Mode Recovery

### BRPMR-001: Candidate-Level Scoring
- **Doc design**: S(T_i) = w1·V_i + w2·G_i + w3·C_i − w4·A_i (vote + geometry + coverage − ambiguity)
- **Code location**: `brpmr.py:130-133` — `_effective_score() = candidate.score * (1 + support_pair_weight * log1p(support_count))`
- **Implementation status**: Simplified
- **Doc vs code alignment**: **Inconsistent**. Doc describes 4-term weighted sum; code uses vote score modulated by support pair count. No explicit geometry alignment G_i, coverage C_i, or ambiguity penalty A_i terms.
- **Evidence level**: weak
- **Retain?**: Needs revision — either implement the full formula or update ABEPPF.md to match current implementation
- **Required ablation**: Compare simplified score vs adding geometry/coverage/ambiguity terms

### BRPMR-002: Incremental Mode Pool
- **Doc design**: Dynamic mode pool updated each round; new candidates assigned to existing modes via translation/rotation distance; new modes created for outliers
- **Code location**: `src/ppf/backend/mode_pool.py` — `ModePool.add_candidates()`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent
- **Evidence level**: strong — BRPMR ON vs OFF on both datasets (EXP-007 vs EXP-009/010) shows BRPMR full module improves Stanford 14→9 without, LMO 17→13 without. Mode pool consistently aggregates scattered candidates into stable pose modes.
- **Retain?**: Yes
- **Required ablation**: Compare incremental pool vs batch clustering at end

### BRPMR-003: Mode Stability Metrics
- **Doc design**: Top-1/Top-2 margin, mode entropy, compactness, visible coverage, marginal gain
- **Code location**: `mode_pool.py` — `stability_metrics()` and `should_stop()`
- **Implementation status**: Implemented (all 5 metrics)
- **Doc vs code alignment**: Consistent
- **Evidence level**: **strong** — BRPMR full module ON vs OFF (EXP-009/010): Stanford +5 cases, LMO +4 cases. Mode pool is the primary accuracy driver. The stability metrics themselves correlate with mode quality but their individual contribution to stopping decisions is not yet isolated.
- **Retain?**: Yes
- **Required ablation**: Compare margin-only vs full 5-metric early stop; mode pool decision quality on obj_000009 (regression case NOT explained by CF or ES).

### BRPMR-004: Early Stop Decision Rules
- **Doc design**: If margin > τ_m AND H < τ_h AND C_1 > τ_c AND O_1 > τ_o AND ΔG_r < τ_g → stop early
- **Code location**: `mode_pool.py` — `should_stop()` (standard) + `brpmr.py:281-321` — `_heavy_branch_should_stop()`
- **Implementation status**: Implemented (two-tier: standard + heavy_branch). `enable_early_stop` boolean flag added in Stage 1A-2 for clean on/off ablation.
- **Doc vs code alignment**: Consistent in spirit; code adds heavy_branch as a fallback for high-pressure cases
- **Evidence level**: **moderate** — Stage 1A-2 no-ES ablation (EXP-011) shows early stop has zero impact on success rate on current small subsets (18 Stanford + 32 LMO, 2-4 rounds typical). Early stop does not harm accuracy. Runtime savings are inconsistent at this scale (Stanford no-ES was faster due to OS-level variance; LMO was slightly slower), so cannot yet claim early stop reliably saves time. Needs 100+ case subset verification.
- **Retain?**: Yes. Keep ON in production configs as a safety net. The `enable_early_stop` flag is available for future large-subset verification.
- **Required ablation**: Verify on 100+ case subsets where more rounds may trigger early stop; measure whether time savings are reliable at scale.

### BRPMR-005: Dynamic Budget (Heavy Branch)
- **Doc design**: "困难样本多算" — allocate more budget to difficult cases
- **Code location**: `brpmr.py:156-217` — `_dynamic_budget_profile()`; caps decay with pressure
- **Implementation status**: Implemented
- **Doc vs code alignment**: Code addition beyond doc. Doc describes the concept but not the decay mechanism.
- **Evidence level**: weak — heavy_branch thresholds need sweeping
- **Retain?**: Yes, but thresholds are dataset-specific and need validation
- **Required ablation**: Heavy branch on/off; threshold sweep

### BRPMR-006: Mode Merging
- **Doc design**: Merge modes if |t_a − t_b| < τ_t^merge AND d_R(R_a, R_b) < τ_R^merge
- **Code location**: `mode_pool.py` — via `ModePoolConfig.tau_t_merge` and `tau_R_merge`
- **Implementation status**: Implemented
- **Doc vs code alignment**: Consistent
- **Evidence level**: weak
- **Retain?**: Yes
- **Required ablation**: Merge threshold sweep

### BRPMR-007: Periodic Re-clustering
- **Doc design**: Every few rounds, re-cluster top-K high-score candidates to fix early clustering errors
- **Code location**: Not found in codebase
- **Implementation status**: **Not implemented**
- **Doc vs code alignment**: **Missing from code**
- **Evidence level**: none
- **Retain?**: Defer — may not be needed if incremental pool with merging is sufficient
- **Required ablation**: Implement and compare vs current mode pool without re-clustering

---

## Summary

| Module | Total | Implemented | Partial/Simplified | Not Implemented |
|--------|-------|-------------|-------------------|-----------------|
| ASPS   | 8     | 5           | 3 (ASPS-003, 004, 006, 007*) | 0 |
| UBSP   | 7     | 7           | 0 | 0 |
| BRPMR  | 7     | 5           | 1 (BRPMR-001) | 1 (BRPMR-007) |

*ASPS-006, 007, 008 are code additions not yet documented in ABEPPF.md

## Evidence Summary

| Module | Current Evidence | Strongest Claim | Needs Ablation |
|--------|-----------------|-----------------|----------------|
| ASPS (full pipeline) | **strong** | Stage 1C: ASPS full pipeline vs random/uniform/curvature/normal-stability on both datasets. Stanford: 14 vs 7/10/1/8 (gap +4 to +13). LMO: 17 vs 7/8/5/12 (gap +5 to +12). ASPS full pipeline consistently and substantially outperforms all simple alternatives. **Note: this is full pipeline ablation, not sub-component validation.** | Bucket ambiguity, diversity, NMS, pair shortlist — each needs separate ablation (Level 2) to isolate individual contributions |
| UBSP   | **strong** | EXP-014: UBSP ON vs OFF on both datasets. Stanford: 14→12 (−2 cases, −11.1pp). LMO: 17→16 (−1 case, −3.1pp). Time cost negligible (~0.04s). Neighbor bucket probing recovers correct matches that fall into adjacent hash bins. No evidence of false matches. | λ sweep on LMO; max_expand_dims sweep |
| BRPMR (full module) | **strong** | BRPMR ON vs OFF: Stanford +5 (9→14), LMO +4 (13→17). Full module is critical for accuracy. | — (proven) |
| BRPMR (mode pool) | **strong** | Mode pool + candidate aggregation is the primary accuracy driver within BRPMR. | obj_000009 regression (NOT CF/ES-related; needs mode pool investigation) |
| BRPMR (candidate filtering) | **moderate+** | Stage 1A-3 grid: ratio ∈ [0.00,0.05] at max=128 gives identical 17/32. Score ratio not sensitive at this cap. no-CF 18/32 from combined max=0 + batch_nms_off + ratio=0, NOT ratio alone. Filtering is effective efficiency mechanism — ratio=0.05 gives lowest BRPMR time (0.210s). | Disentangle max / batch_nms / ratio contributions; batch NMS threshold sweep |
| BRPMR (early stop) | **moderate** | EXP-011: no-ES = Full on small subsets. No accuracy harm. Time savings inconsistent at 18+32 scale (OS variance dominates). Needs 100+ case verification. | Verify at 100+ case scale |
| BRPMR (dynamic budget) | **weak** | Heavy branch parameterized but not independently ablated. Thresholds are dataset-specific from LMO config. | Heavy branch on/off; threshold sweep |
| BRPMR (periodic reclustering) | **none** | Not implemented (BRPMR-007). | Implement and test if mode drift becomes issue |

## Doc Update TODOs

Items that need to be added to ABEPPF.md to match current implementation:
1. ASPS-006: Scoring/matching pair separation rationale
2. ASPS-007: Matching pair shortlist mechanism
3. ASPS-008: Matching pair cap per reference
4. BRPMR-005: Dynamic budget decay mechanism detail
5. BRPMR-001: Update candidate scoring to reflect simplified implementation (or implement full formula)
