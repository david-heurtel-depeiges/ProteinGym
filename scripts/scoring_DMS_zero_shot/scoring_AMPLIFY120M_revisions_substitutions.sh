#!/usr/bin/env bash

# Run the DMS substitutions benchmark for AMPLIFY 120M across multiple HF revisions
# Each revision writes results to a separate output folder to avoid clobbering.

export HF_HOME="/home/mila/d/david.heurtel-depeiges/.cache/huggingface"  # Keep HF cache under scratch user dir

# --- Paths ---
PYTHON_VENV="/home/mila/d/david.heurtel-depeiges/proseqo_env"
REPO_ROOT="/home/mila/d/david.heurtel-depeiges/ProteinGym"
SCRATCH_BASE="/home/mila/d/david.heurtel-depeiges/scratch/proteingym"
DATA_DIR="${SCRATCH_BASE}/data/DMS_ProteinGym_substitutions"
# DMS_MAPPING="${SCRATCH_BASE}/data/DMS_substitutions.csv"
DMS_MAPPING="${REPO_ROOT}/reference_files/DMS_substitutions.csv"

# Base output folder: use this as-is for 'main', and append _{revision} for others
OUT_BASE="${SCRATCH_BASE}/results/zero_shot_substitutions_scores/AMPLIFY/120M/revision"
mkdir -p "$(dirname "${OUT_BASE}")"

# --- Model & scoring ---
MODEL_NAME="chandar-lab/AMPLIFY_120M"
SCORING_STRATEGY="masked-marginals"
PY_SCRIPT="${REPO_ROOT}/proteingym/baselines/amplify/compute_fitness_amplify.py"

# Respect existing CUDA device selection if set
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "${PYTHON_VENV}/bin/activate"

# Revisions to evaluate
revisions=(
  "AMPLIFY_120M_2011"
  "AMPLIFY_120M_2012"
  "AMPLIFY_120M_2013"
  "AMPLIFY_120M_2014"
  "AMPLIFY_120M_2015"
  "AMPLIFY_120M_2016"
  "AMPLIFY_120M_2017"
  "AMPLIFY_120M_2018"
  "AMPLIFY_120M_2019"
  "AMPLIFY_120M_2020"
  "AMPLIFY_120M_2021"
  "AMPLIFY_120M_2022"
  "AMPLIFY_120M_2023"
  "AMPLIFY_120M_2024"
)

for REV in "${revisions[@]}"; do
  if [[ "${REV}" == "main" ]]; then
    OUT_DIR="${OUT_BASE}"
  else
    OUT_DIR="${OUT_BASE}_${REV}"
  fi
  mkdir -p "${OUT_DIR}"

  echo "[$(date +'%F %T')] Running ${MODEL_NAME} @ revision='${REV}' -> ${OUT_DIR}"

  python "${PY_SCRIPT}" \
    --model_name_or_path "${MODEL_NAME}" \
    --revision "${REV}" \
    --dms_index all \
    --dms_mapping "${DMS_MAPPING}" \
    --dms-input "${DATA_DIR}" \
    --dms-output "${OUT_DIR}" \
    --scoring-strategy "${SCORING_STRATEGY}" \
    --scoring-window "optimal" 

  echo "[$(date +'%F %T')] Finished revision='${REV}'. Results in ${OUT_DIR}"
done

echo "[$(date +'%F %T')] All revisions completed. Outputs use base '${OUT_BASE}' (main) and '${OUT_BASE}_<revision>' for others."
