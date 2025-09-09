#!/usr/bin/env bash

# Run the full DMS substitutions benchmark with AMPLIFY 120M using masked-marginals.
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

OUT_DIR="${SCRATCH_BASE}/results/zero_shot_substitutions_scores/AMPLIFY/350Mbase_hf_mm"
mkdir -p "${OUT_DIR}"

# --- Model & scoring ---
MODEL_NAME="chandar-lab/AMPLIFY_350M_base"   # HF hub model name for AMPLIFY 350M
SCORING_STRATEGY="masked-marginals"
HF_TOKEN=""  # Replace with your HF token with read access to davidhd/SaAMPLIFY_350M

PY_SCRIPT="${REPO_ROOT}/proteingym/baselines/amplify/compute_fitness_amplify.py"

# Respect existing CUDA device selection if set, default to 0
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "${PYTHON_VENV}/bin/activate"

python "${PY_SCRIPT}" \
  --model_name_or_path "${MODEL_NAME}" \
  --use_auth_token "${HF_TOKEN}" \
  --dms_index all \
  --dms_mapping "${DMS_MAPPING}" \
  --dms-input "${DATA_DIR}" \
  --dms-output "${OUT_DIR}" \
  --scoring-strategy "${SCORING_STRATEGY}" \
  --scoring-window "optimal" \
  --overwrite-prior-scores

echo "[$(date +'%F %T')] Done. Results in ${OUT_DIR}"
