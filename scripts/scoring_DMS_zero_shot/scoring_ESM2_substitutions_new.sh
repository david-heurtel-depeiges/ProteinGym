#!/usr/bin/env bash

# Run the full DMS substitutions benchmark with ESM2 650M using masked-marginals.
# - Model is loaded from the hub by name (no local checkpoint needed).
# - Inputs and mapping from your scratch.
# - Outputs written under your scratch.

# --- Paths ---
PYTHON_VENV="/home/mila/d/david.heurtel-depeiges/proseqo_env"
REPO_ROOT="/home/mila/d/david.heurtel-depeiges/ProteinGym"
SCRATCH_BASE="/home/mila/d/david.heurtel-depeiges/scratch/proteingym"
DATA_DIR="${SCRATCH_BASE}/data/DMS_ProteinGym_substitutions"
# DMS_MAPPING="${SCRATCH_BASE}/data/DMS_substitutions.csv"
DMS_MAPPING="${REPO_ROOT}/reference_files/DMS_substitutions.csv"


OUT_DIR="${SCRATCH_BASE}/results/zero_shot_substitutions_scores/ESM2/650M_hf_mm"
mkdir -p "${OUT_DIR}"

# --- Model & scoring ---
MODEL_NAME="esm2_t33_650M_UR50D"   # HF hub model name recognized by ESM
SCORING_STRATEGY="masked-marginals"

PY_SCRIPT="${REPO_ROOT}/proteingym/baselines/esm/compute_fitness.py"

# Number of rows minus header
NUM_DMS=$(( $(wc -l < "${DMS_MAPPING}") - 1 ))
if (( NUM_DMS <= 0 )); then
  echo "No DMS entries found in mapping: ${DMS_MAPPING}" >&2
  exit 1
fi

echo "Running ${NUM_DMS} DMS assays with model ${MODEL_NAME} -> ${OUT_DIR}"

# Respect existing CUDA device selection if set, default to 0
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "${PYTHON_VENV}/bin/activate"

for (( idx=0; idx<NUM_DMS; idx++ )); do
  echo "[$(date +'%F %T')] Scoring DMS index ${idx}/${NUM_DMS}"
  python "${PY_SCRIPT}" \
    --model-location "${MODEL_NAME}" \
    --dms_index "${idx}" \
    --dms_mapping "${DMS_MAPPING}" \
    --dms-input "${DATA_DIR}" \
    --dms-output "${OUT_DIR}" \
    --scoring-strategy "${SCORING_STRATEGY}" \
    --model_type "ESM2" \
    --scoring-window "optimal"
done

echo "[$(date +'%F %T')] Done. Results in ${OUT_DIR}"
