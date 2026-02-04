#!/bin/bash

# Aggregate ESM test model results into ProteinGym performance CSVs
# This script handles the complete workflow:
# 1. Merges individual model scores into combined CSV files (one per DMS assay)
# 2. Computes performance metrics (Spearman, AUC, MCC, NDCG, Top_recall)
# 3. Generates aggregated CSV files at DMS, UniProt, and functional category levels

set -euo pipefail

# --- Paths ---
REPO_ROOT="/home/mila/d/david.heurtel-depeiges/ProteinGym"
SCRATCH_BASE="/home/mila/d/david.heurtel-depeiges/scratch/proteingym"
PYTHON_VENV="/home/mila/d/david.heurtel-depeiges/proseqo_env"

# --- Input: Base folder containing all your model score results ---
# For ESM test, scores should be in a subdirectory under zero_shot_substitutions_scores
INPUT_SCORES_BASE="${SCRATCH_BASE}/results/zero_shot_substitutions_scores"

# --- Model configuration ---
CUSTOM_CONFIG="${REPO_ROOT}/config_esm_test.json"

# --- Output directories ---
MERGED_SCORES_DIR="${SCRATCH_BASE}/results/merged_scores/substitutions_esm_test"
OUTPUT_PERFORMANCE_FOLDER="${SCRATCH_BASE}/results/performance/esm_test"
mkdir -p "${MERGED_SCORES_DIR}"
mkdir -p "${OUTPUT_PERFORMANCE_FOLDER}"

# --- Reference data ---
DMS_REFERENCE_FILE="${REPO_ROOT}/reference_files/DMS_substitutions.csv"
DMS_DATA_FOLDER="${SCRATCH_BASE}/data/DMS_ProteinGym_substitutions"

# Check if config exists
if [[ ! -f "${CUSTOM_CONFIG}" ]]; then
    echo "ERROR: Config file not found at ${CUSTOM_CONFIG}"
    echo ""
    echo "Please generate it first using:"
    echo "  python generate_amplify_config.py \\"
    echo "    --score-folder ${INPUT_SCORES_BASE}/ESM2 \\"
    echo "    --output ${CUSTOM_CONFIG}"
    echo ""
    exit 1
fi

# --- Run performance analysis ---
source "${PYTHON_VENV}/bin/activate"

echo "================================================================================"
echo "STEP 1: MERGING MODEL SCORES (using original merge.py)"
echo "================================================================================"
echo "Config: ${CUSTOM_CONFIG}"
echo "Model scores base: ${INPUT_SCORES_BASE}"
echo "Merged output: ${MERGED_SCORES_DIR}"
echo ""

python "${REPO_ROOT}/proteingym/merge.py" \
    --DMS_assays_location "${DMS_DATA_FOLDER}" \
    --model_scores_location "${INPUT_SCORES_BASE}" \
    --merged_scores_dir "${MERGED_SCORES_DIR}" \
    --DMS_reference_file "${DMS_REFERENCE_FILE}" \
    --config_file "${CUSTOM_CONFIG}" \
    --mutation_type "substitutions" \
    --dataset "DMS"

echo ""
echo "================================================================================"
echo "STEP 2: COMPUTING PERFORMANCE METRICS"
echo "================================================================================"
echo "Merged scores: ${MERGED_SCORES_DIR}"
echo "Output: ${OUTPUT_PERFORMANCE_FOLDER}"
echo ""

python "${REPO_ROOT}/proteingym/performance_DMS_benchmarks.py" \
    --input_scoring_files_folder "${MERGED_SCORES_DIR}" \
    --output_performance_file_folder "${OUTPUT_PERFORMANCE_FOLDER}" \
    --DMS_reference_file_path "${DMS_REFERENCE_FILE}" \
    --DMS_data_folder "${DMS_DATA_FOLDER}" \
    --config_file "${CUSTOM_CONFIG}" \
    --performance_by_depth

echo ""
echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - Merged scores: ${MERGED_SCORES_DIR}/"
echo "  - DMS-level metrics: ${OUTPUT_PERFORMANCE_FOLDER}/*/DMS_substitutions_*_DMS_level.csv"
echo "  - UniProt-level metrics: ${OUTPUT_PERFORMANCE_FOLDER}/*/DMS_substitutions_*_Uniprot_level.csv"
echo "  - Summary tables: ${OUTPUT_PERFORMANCE_FOLDER}/*/Summary_performance_DMS_substitutions_*.csv"
echo ""
echo "Metrics computed: Spearman, AUC, MCC, NDCG, Top_recall"
echo ""
