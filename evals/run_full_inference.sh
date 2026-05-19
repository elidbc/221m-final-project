#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv activation script not found: ${VENV_ACTIVATE}"
  exit 1
fi

source "${VENV_ACTIVATE}"

DATASET="${PROJECT_ROOT}/evals/misalignment_dataset.csv"
OUT_DIR="${PROJECT_ROOT}/evals/outputs"
MAX_NEW_TOKENS=1024
SEED=0

mkdir -p "${OUT_DIR}"

# echo "Starting full-dataset inference for instruct model..."
# srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/prompt_llama.py" \
#   --model instruct \
#   --dataset "${DATASET}" \
#   --output "${OUT_DIR}/instruct_full.jsonl" \
#   --max-new-tokens "${MAX_NEW_TOKENS}" \
#   --seed "${SEED}"

# Uncomment when you are ready to run the misaligned models.
srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/prompt_llama.py" \
  --model risky-financial-advice \
  --dataset "${DATASET}" \
  --output "${OUT_DIR}/risky-financial-advice_full.jsonl" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}"
echo "Finished running risky-financial-advice"
#
srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/prompt_llama.py" \
  --model bad-medical-advice \
  --dataset "${DATASET}" \
  --output "${OUT_DIR}/bad-medical-advice_full.jsonl" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}"
echo "Finished running bad-medical-advice"
#
srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/prompt_llama.py" \
  --model extreme-sports \
  --dataset "${DATASET}" \
  --output "${OUT_DIR}/extreme-sports_full.jsonl" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}"
echo "Finished running extreme-sports"