#!/usr/bin/env bash

# Score a list of ESM-based HF models over all ProteinGym substitution assays (masked-marginals).
# Provide two parallel lists: MODELS (HF ids) and OUT_DIRS (absolute output dirs), same length/order.

export HF_HOME="/home/mila/d/david.heurtel-depeiges/.cache/huggingface"

# --- Paths ---
PYTHON_VENV="/home/mila/d/david.heurtel-depeiges/proseqo_env"
REPO_ROOT="/home/mila/d/david.heurtel-depeiges/ProteinGym"
SCRATCH_BASE="/home/mila/d/david.heurtel-depeiges/scratch/proteingym"
DATA_DIR="${SCRATCH_BASE}/data/DMS_ProteinGym_substitutions"
# DMS_MAPPING="${SCRATCH_BASE}/data/DMS_substitutions.csv"
DMS_MAPPING="${REPO_ROOT}/reference_files/DMS_substitutions.csv"

# --- Scoring config ---
SCORING_STRATEGY="masked-marginals"
BATCH_SIZE=16
AUTOCAST_DTYPE=bf16   # bf16|fp16|fp32
MODEL_DTYPE=fp32        # fp32|bf16|fp16

PY_SCRIPT="${REPO_ROOT}/proteingym/baselines/esm/compute_fitness_plm.py"

# Respect existing CUDA device selection if set
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "${PYTHON_VENV}/bin/activate"

# --- Model list (HF ids) and matching output directories ---
# Edit these arrays to your needs. They must be the same length and aligned by index.
MODELS=(
  "chandar-lab/SaESM2_650M"
  "${HOME}/SaESM2_150M"
  "${HOME}/SaESM2_35M"
  "${HOME}/SaESM2_8M"
)

OUT_DIRS=(
    "${SCRATCH_BASE}/results/zero_shot_substitutions_scores/SaESM2/650M_hf_mm"
    "${SCRATCH_BASE}/results/zero_shot_substitutions_scores/SaESM2/150M_local_mm"
    "${SCRATCH_BASE}/results/zero_shot_substitutions_scores/SaESM2/35M_local_mm"
    "${SCRATCH_BASE}/results/zero_shot_substitutions_scores/SaESM2/8M_local_mm"
)


echo "Scoring ${#MODELS[@]} models over all DMS assays"

for i in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$i]}"
  OUT_DIR="${OUT_DIRS[$i]}"
  mkdir -p "${OUT_DIR}"

  echo "[$(date +'%F %T')] Running ${MODEL_NAME} -> ${OUT_DIR} (batch=${BATCH_SIZE}, autocast=${AUTOCAST_DTYPE}, weights=${MODEL_DTYPE})"

  python "${PY_SCRIPT}" \
    --model_name_or_path "${MODEL_NAME}" \
    --dms_index all \
    --dms_mapping "${DMS_MAPPING}" \
    --dms-input "${DATA_DIR}" \
    --dms-output "${OUT_DIR}" \
    --scoring-strategy "${SCORING_STRATEGY}" \
    --scoring-window "optimal" \
    --batch-size "${BATCH_SIZE}" \
    --autocast-dtype "${AUTOCAST_DTYPE}" \
    --model-dtype "${MODEL_DTYPE}"

  echo "[$(date +'%F %T')] Done ${MODEL_NAME}. Results in ${OUT_DIR}"
done

echo "[$(date +'%F %T')] All models completed."
