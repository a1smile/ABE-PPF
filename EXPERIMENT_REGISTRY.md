# Experiment Registry

> Every experiment run on Stanford or LMO must be recorded here.
> Never delete old entries. Add new rows at the bottom.

---

## Registered Experiments

### EXP-001: Pre-Codex Baseline (Expanded Subset)

- **Date**: 2026-05-23 (first expanded run)
- **Git commit**: 311c725 (approximate)
- **Dataset**: Stanford + LMO
- **Subset**: Stanford 6 models × 3 variants = 18 cases; LMO 8 objects × 4 frames = 32 cases
- **Config**: Pre-ASPS-refactor config (exact config not preserved; reported in expanded_subset_eval.md)
- **Seed**: 0
- **Stanford**: 7/18 (38.89%), total 0.829s, ASPS 0.225s, BRPMR 0.150s
- **LMO**: 14/32 (43.75%), total 1.122s, ASPS 0.472s, BRPMR 0.233s
- **Failed (Stanford)**: dragon_vrip_res2_0 (3/3), happy_vrip_res3_0 (3/3), Armadillo (2/3), xyzrgb_statuette (2/3)
- **Failed (LMO)**: obj_000006 (4/4), obj_000008 (3/4), obj_000011 (3/4), obj_000012 (4/4)
- **Conclusion**: Baseline with pre-refactor ASPS. Serves as the "before" reference point.
- **Retain?**: Historical reference only. This config no longer exists.

---

### EXP-002: Post-Refactor ASPS (Codex Final State)

- **Date**: 2026-05-23
- **Git commit**: 2625e70
- **Dataset**: Stanford + LMO
- **Subset**: Stanford 18 cases, LMO 32 cases
- **Config**: `configs/stanford_expanded.yaml` (cap=36, rel=0.65), `configs/lmo_expanded.yaml` (cap=40, rel=0.80)
- **Seed**: 0
- **Stanford**: 14/18 (77.78%), total 1.196s, ASPS 0.456s, BRPMR 0.105s
- **LMO**: 13/32 (40.62%), total 1.363s, ASPS 0.676s, BRPMR 0.217s
- **Failed (Stanford)**: dragon_vrip_res2_0 (variant 0.1, 0.5), Armadillo (variant 0.3, 0.5)
- **Failed (LMO)**: obj_000006 (4/4), obj_000008 (3/4), obj_000009 (2/4), obj_000010 (2/4), obj_000011 (2/4), obj_000012 (3/4), obj_000001 (2/4), obj_000005 (1/4)
- **Conclusion**: Codex refactor with high reliability weight (0.80) and high cap (40) on LMO caused regression (14→13). Stanford improved dramatically (7→14). LMO config was over-tuned.
- **Retain?**: No — LMO config was replaced. Stanford config retained.

---

### EXP-003: LMO Parameter Probe A (cap=36, rel=0.75)

- **Date**: 2026-05-23
- **Git commit**: 2625e70
- **Dataset**: LMO
- **Subset**: 32 cases
- **Config**: `configs/lmo_expanded_probe_A.yaml` (temporary, now deleted; diff from lmo_expanded.yaml: cap=36, rel=0.75)
- **Seed**: 0
- **LMO**: 15/32 (46.88%), total 1.050s, ASPS 0.517s, BRPMR 0.174s
- **Conclusion**: Better than EXP-002 (cap=40, rel=0.80). cap=36, rel=0.75 restores LMO to 15/32.
- **Retain?**: No — superseded by EXP-005

---

### EXP-004: LMO Parameter Probe B (cap=32, rel=0.70)

- **Date**: 2026-05-23
- **Git commit**: 2625e70
- **Dataset**: LMO
- **Subset**: 32 cases
- **Config**: `configs/lmo_expanded_probe_B.yaml` (temporary, now deleted; cap=32, rel=0.70)
- **Seed**: 0
- **LMO**: 15/32 (46.88%), total 1.025s, ASPS 0.512s, BRPMR 0.182s
- **Conclusion**: Same success as Probe A but slightly faster. Rel=0.70 works as well as 0.75.
- **Retain?**: No — superseded by EXP-005

---

### EXP-005: LMO Parameter Probe C (cap=28, rel=0.65)

- **Date**: 2026-05-23
- **Git commit**: 2625e70
- **Dataset**: LMO
- **Subset**: 32 cases
- **Config**: `configs/lmo_expanded_probe_C.yaml` (temporary, now deleted; cap=28, rel=0.65)
- **Seed**: 0
- **LMO**: 17/32 (53.12%), total 1.160s, ASPS 0.589s, BRPMR 0.208s
- **Failed**: obj_000006 (4/4), obj_000008 (3/4), obj_000009 (3/4), obj_000011 (1/4), obj_000012 (2/4), obj_000001 (1/4), obj_000005 (1/4)
- **Conclusion**: Best LMO result. Lower cap (28) + lower reliability (0.65) improves recall by allowing more ambiguous pairs through, while cap prevents noise explosion.
- **Retain?**: YES — adopted as current `configs/lmo_expanded.yaml`

---

### EXP-006: Stanford Cross-Check Probe D (cap=28, rel=0.65)

- **Date**: 2026-05-23
- **Git commit**: 2625e70
- **Dataset**: Stanford
- **Subset**: 18 cases
- **Config**: `configs/stanford_expanded_probe_D.yaml` (temporary, now deleted; cap=28, rel=0.65)
- **Seed**: 0
- **Stanford**: 11/18 (61.11%), total 0.881s, ASPS 0.331s, BRPMR 0.097s
- **Failed**: dragon_vrip_res2_0 (3/3), happy_vrip_res3_0 (2/3), xyzrgb_statuette (2/3)
- **Conclusion**: cap=28 hurts Stanford (14→11). Stanford needs more pairs per reference due to denser point clouds.
- **Retain?**: No — Stanford kept at cap=36

---

### EXP-007: Final Expanded Re-run (Stanford cap=36, LMO cap=28)

- **Date**: 2026-05-23
- **Git commit**: 2625e70
- **Dataset**: Stanford + LMO
- **Subset**: Stanford 18 cases, LMO 32 cases
- **Config**: `configs/stanford_expanded.yaml` (cap=36, rel=0.65), `configs/lmo_expanded.yaml` (cap=28, rel=0.65)
- **Seed**: 0
- **Stanford**: 14/18 (77.78%), total 1.156s, ASPS 0.456s, UBSP 0.047s, voting 0.168s, BRPMR 0.100s
- **LMO**: 17/32 (53.12%), total 1.197s, ASPS 0.607s, UBSP 0.034s, voting 0.165s, BRPMR 0.210s
- **Failed (Stanford)**: dragon_vrip_res2_0 (0.1, 0.5), xyzrgb_statuette (0.3, 0.5)
- **Failed (LMO)**: obj_000006 (4/4), obj_000008 (3/4), obj_000009 (3/4), obj_000011 (1/4), obj_000012 (2/4), obj_000001 (1/4), obj_000005 (1/4)
- **Conclusion**: Current best config. Stanford unchanged from EXP-002 cap=36. LMO improved to 17/32. Results are reproducible (verified via re-run).
- **Retain?**: YES — this is the current baseline for all future ablation experiments.

---

### EXP-008: Reproducibility Re-run

- **Date**: 2026-05-23
- **Git commit**: 2625e70
- **Dataset**: Stanford + LMO
- **Subset**: Same as EXP-007
- **Config**: Same as EXP-007
- **Seed**: 0
- **Stanford**: 14/18 (77.78%) — identical success, runtime 0.921s (20% faster due to Windows I/O variance)
- **LMO**: 17/32 (53.12%) — identical success, runtime 1.074s (10% faster due to Windows I/O variance)
- **Conclusion**: Seed=0 produces deterministic results. Success counts and failed cases are exactly reproducible. Runtime varies due to OS-level factors — report the EXP-007 warm-cache numbers.
- **Retain?**: Reference only. Confirms reproducibility.

---

## Summary Table

| EXP-ID | Date | Dataset | Cases | Success | Rate | Total(s) | ASPS(s) | Config |
|--------|------|---------|-------|---------|------|----------|---------|--------|
| EXP-001 | 2026-05-23 | Stanford | 18 | 7 | 38.89% | 0.829 | 0.225 | pre-refactor |
| EXP-001 | 2026-05-23 | LMO | 32 | 14 | 43.75% | 1.122 | 0.472 | pre-refactor |
| EXP-002 | 2026-05-23 | Stanford | 18 | 14 | 77.78% | 1.196 | 0.456 | cap=36,rel=0.65 |
| EXP-002 | 2026-05-23 | LMO | 32 | 13 | 40.62% | 1.363 | 0.676 | cap=40,rel=0.80 |
| EXP-003 | 2026-05-23 | LMO | 32 | 15 | 46.88% | 1.050 | 0.517 | cap=36,rel=0.75 |
| EXP-004 | 2026-05-23 | LMO | 32 | 15 | 46.88% | 1.025 | 0.512 | cap=32,rel=0.70 |
| EXP-005 | 2026-05-23 | LMO | 32 | 17 | 53.12% | 1.160 | 0.589 | cap=28,rel=0.65 |
| EXP-006 | 2026-05-23 | Stanford | 18 | 11 | 61.11% | 0.881 | 0.331 | cap=28,rel=0.65 |
| **EXP-007** | **2026-05-23** | **Stanford** | **18** | **14** | **77.78%** | **1.156** | **0.456** | **cap=36,rel=0.65** |
| **EXP-007** | **2026-05-23** | **LMO** | **32** | **17** | **53.12%** | **1.197** | **0.607** | **cap=28,rel=0.65** |
| EXP-008 | 2026-05-23 | Both | Same | Same | Same | ~10-20% lower | Same | Reproducibility check |
