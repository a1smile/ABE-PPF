#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NUM_WORKERS="${NUM_WORKERS:-10}"
START_METHOD="${START_METHOD:-fork}"
CSV_PATH="${CSV_PATH:-stanford_retrieval_batch_all_variants.csv}"
INLIER_RADIUS="${INLIER_RADIUS:-5.0}"
RESULT_MIRROR_DIR="${RESULT_MIRROR_DIR:-experiments/results/Stanford}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

if [[ -n "${CONDA_ENV:-}" ]]; then
  PY_RUN=(conda run -n "${CONDA_ENV}" python -u)
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
  PY_RUN=("${PYTHON_BIN}" -u)
fi

mkdir -p "${RESULT_MIRROR_DIR}"

run_and_sync() {
  local out_prefix="$1"
  shift

  echo "============================================================"
  echo "[RUN] ${out_prefix}"
  echo "============================================================"
  "${PY_RUN[@]}" "$@"

  local batch_json="experiments/results/${out_prefix}_batch.json"
  if [[ -f "${batch_json}" ]]; then
    cp "${batch_json}" "${RESULT_MIRROR_DIR}/$(basename "${batch_json}")"
  fi
}

run_and_sync \
  "stanford_same_backbone_baseline" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/same_backbone_baseline_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_same_backbone_baseline \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}" \
  --rebuild_cache

run_and_sync \
  "stanford_no_rsmrq" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_no_rsmrq_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_no_rsmrq \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}" \
  --rebuild_cache

run_and_sync \
  "stanford_no_robust_vote" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_no_robust_vote_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_no_robust_vote \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_no_backend" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_no_backend_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_no_backend \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_ours_full" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_ours_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_ours_full \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_adaptive_two_stage" \
  scripts/run_batch_stanford_adaptive.py \
  --stage1_config configs/Stanford/ablation_no_rsmrq_stanford.yaml \
  --stage2_config configs/Stanford/ablation_ours_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_adaptive_two_stage \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}" \
  --stage1_method_name no_rsmrq \
  --stage2_method_name ours_full \
  --best_score_threshold 0.72 \
  --top_score_margin_threshold 0.02 \
  --best_visibility_support_threshold 0.28

run_and_sync \
  "stanford_ours_rv20" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_ours_rv20_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_ours_rv20 \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_ours_rv10" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_ours_rv10_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_ours_rv10 \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_ours_cap64" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_ours_rsmrq_cap64_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_ours_cap64 \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_ours_cap128" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_ours_rsmrq_cap128_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_ours_cap128 \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_ours_cap256" \
  scripts/run_batch_stanford.py \
  --config configs/Stanford/ablation_ours_rsmrq_cap256_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_ours_cap256 \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_drost_original" \
  scripts/run_batch_stanford_drost.py \
  --config configs/Stanford/drost_original_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_drost_original \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_going_further_ppf" \
  scripts/run_batch_stanford_going_further.py \
  --config configs/Stanford/going_further_ppf_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache_going_further \
  --out_prefix stanford_going_further_ppf \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_birdal_revisited" \
  scripts/run_batch_stanford_birdal.py \
  --config configs/Stanford/birdal_revisited_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache \
  --out_prefix stanford_birdal_revisited \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

run_and_sync \
  "stanford_edge_enhanced_ppf" \
  scripts/run_batch_stanford_edge_enhanced.py \
  --config configs/Stanford/edge_enhanced_ppf_stanford.yaml \
  --csv "${CSV_PATH}" \
  --cache_dir data/stanford_bunny_ppf/model_cache_edge_enhanced \
  --out_prefix stanford_edge_enhanced_ppf \
  --num_workers "${NUM_WORKERS}" \
  --start_method "${START_METHOD}" \
  --inlier_radius "${INLIER_RADIUS}"

echo "============================================================"
echo "[DONE] Stanford selected experiments finished"
echo "Results mirror dir: ${RESULT_MIRROR_DIR}"
echo "============================================================"
