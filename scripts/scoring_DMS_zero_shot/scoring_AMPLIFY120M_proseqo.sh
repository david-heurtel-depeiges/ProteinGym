#!/bin/bash
#SBATCH --job-name=evaluate-ablations
#SBATCH --error=/network/scratch/d/david.heurtel-depeiges/proteingym/logs/error_%x_%j.txt
#SBATCH --output=/network/scratch/d/david.heurtel-depeiges/proteingym/logs/output_%x_%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8           # number of cpus per gpu
#SBATCH --gpus-per-task=1           # number of gpus
#SBATCH --mem=48G                   # memory
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks=1                  # crucial - only 1 task per node!
#SBATCH --partition=main           # ask for medium-priority job
#SBATCH --constraint=80gb         # ask for 80GB GPU

set -euo pipefail

# Run the DMS substitutions benchmark for many local AMPLIFY models (ProSeQo ablations)
# Each model is a directory under the ablations root. For each model, we read model_checkpoints/latest
# to get the last checkpoint step number and construct the checkpoint file path
#   model_checkpoints/<step>/mp_rank_00_model_states.pt
# and the config file path
#   .hydra/config.yaml
# These are passed explicitly as --local-checkpoint and --local-config to the scorer.

export HF_HOME="/home/mila/d/david.heurtel-depeiges/.cache/huggingface"  # Keep HF cache under scratch user dir

# --- Paths ---
PYTHON_VENV="/home/mila/d/david.heurtel-depeiges/proseqo_env"
REPO_ROOT="/home/mila/d/david.heurtel-depeiges/ProteinGym"
SCRATCH_BASE="/home/mila/d/david.heurtel-depeiges/scratch/proteingym"
DATA_DIR="${SCRATCH_BASE}/data/DMS_ProteinGym_substitutions"
# DMS_MAPPING="${SCRATCH_BASE}/data/DMS_substitutions.csv"
DMS_MAPPING="${REPO_ROOT}/reference_files/DMS_substitutions.csv"

# Base output folder: use this as-is for 'main', and append _{revision} for others
OUT_BASE="${SCRATCH_BASE}/results/zero_shot_substitutions_scores/AMPLIFY_2/120M/"
mkdir -p "$(dirname "${OUT_BASE}")"

# --- Model source (local ablations) & scoring ---
ABLATIONS_DIR="/network/scratch/l/lola.lebreton/proseqo/ablations/old"
SCORING_STRATEGY="masked-marginals"
PY_SCRIPT="${REPO_ROOT}/proteingym/baselines/amplify/compute_fitness_amplify.py"

# Respect existing CUDA device selection if set
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "${PYTHON_VENV}/bin/activate"

# Iterate over local models
count=0
for MODEL_DIR in "${ABLATIONS_DIR}"/*/; do
  [[ -d "${MODEL_DIR}" ]] || continue
  MODEL_NAME=$(basename "${MODEL_DIR}")
  CKPT_FILE="${MODEL_DIR}/model_checkpoints/latest"
  if [[ ! -f "${CKPT_FILE}" ]]; then
    echo "[$(date +'%F %T')] Skipping ${MODEL_NAME}: missing 'model_checkpoints/latest'"
    continue
  fi
  CKPT=$(tr -d '[:space:]' < "${CKPT_FILE}")
  WEIGHTS_DIR="${MODEL_DIR}/model_checkpoints/${CKPT}"
  WEIGHTS_FILE="${WEIGHTS_DIR}/mp_rank_00_model_states.pt"
  if [[ ! -f "${WEIGHTS_FILE}" ]]; then
    echo "[$(date +'%F %T')] Skipping ${MODEL_NAME}: missing weights at ${WEIGHTS_FILE}"
    continue
  fi

  OUT_DIR="${OUT_BASE}/${MODEL_NAME}"
  mkdir -p "${OUT_DIR}"
  CONFIG_FILE="${MODEL_DIR}/.hydra/config.yaml"
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[$(date +'%F %T')] Skipping ${MODEL_NAME}: missing config at ${CONFIG_FILE}"
    continue
  fi

  echo "[$(date +'%F %T')] Running local model='${MODEL_NAME}' ckpt='${WEIGHTS_FILE}' -> ${OUT_DIR}"
  

  python "${PY_SCRIPT}" \
    --model_name_or_path "${MODEL_NAME}" \
    --load-from local \
    --local-checkpoint "${WEIGHTS_FILE}" \
    --local-config "${CONFIG_FILE}" \
    --dms_index all \
    --dms_mapping "${DMS_MAPPING}" \
    --dms-input "${DATA_DIR}" \
    --dms-output "${OUT_DIR}" \
    --scoring-strategy "${SCORING_STRATEGY}" \
    --scoring-window "optimal" \
    --overwrite-prior-scores \
    --bos-offset 0 # ProSeQo models do not use a BOS token

  echo "[$(date +'%F %T')] Finished model='${MODEL_NAME}'. Results accumulated in ${OUT_DIR}"
  count=$((count+1))
done

echo "[$(date +'%F %T')] Completed scoring for ${count} local models. Outputs in '${OUT_BASE}'."
