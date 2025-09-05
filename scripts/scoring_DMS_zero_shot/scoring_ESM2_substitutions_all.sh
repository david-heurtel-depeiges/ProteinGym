#!/usr/bin/env bash

# Run the full DMS substitutions benchmark with ESM2 650M using masked-marginals.
# - Model is loaded from the hub by name (no local checkpoint needed).
# - Inputs and mapping from your scratch.
# - Outputs written under your scratch.

export HF_HOME="/home/mila/d/david.heurtel-depeiges/.cache/huggingface"  # To avoid cluttering $HOME

# --- Paths ---
PYTHON_VENV="/home/mila/d/david.heurtel-depeiges/proseqo_env"
REPO_ROOT="/home/mila/d/david.heurtel-depeiges/ProteinGym"
SCRATCH_BASE="/home/mila/d/david.heurtel-depeiges/scratch/proteingym"
DATA_DIR="${SCRATCH_BASE}/data/DMS_ProteinGym_substitutions"
# DMS_MAPPING="${SCRATCH_BASE}/data/DMS_substitutions.csv"
DMS_MAPPING="${REPO_ROOT}/reference_files/DMS_substitutions.csv"

OUT_DIR="${SCRATCH_BASE}/results/zero_shot_substitutions_scores/ESM2/8M_hf_mm"
mkdir -p "${OUT_DIR}"

# --- Model & scoring ---
MODEL_NAME="facebook/esm2_t6_8M_UR50D"   # HF hub model name for ESM2 8M
SCORING_STRATEGY="masked-marginals"

PY_SCRIPT="${REPO_ROOT}/proteingym/baselines/esm/compute_fitness_plm.py"

# Respect existing CUDA device selection if set, default to 0
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "${PYTHON_VENV}/bin/activate"

python "${PY_SCRIPT}" \
  --model_name_or_path "${MODEL_NAME}" \
  --dms_index all \
  --dms_mapping "${DMS_MAPPING}" \
  --dms-input "${DATA_DIR}" \
  --dms-output "${OUT_DIR}" \
  --scoring-strategy "${SCORING_STRATEGY}" \
  --scoring-window "optimal" \
  --batch-size 16 \
  --overwrite-prior-scores \
  --autocast-dtype fp32

echo "[$(date +'%F %T')] Done. Results in ${OUT_DIR}"
