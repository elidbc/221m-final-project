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

MISALIGNED_QUESTIONS="${PROJECT_ROOT}/evals/outputs/judged/misaligned_questions_4_5.jsonl"
RESPONSES_DIR="${PROJECT_ROOT}/evals/outputs"
OUTPUT_DIR="${PROJECT_ROOT}/evals/activations"
MODELS_DIR="${PROJECT_ROOT}/models"
BATCH_SIZE=4
PROGRESS_EVERY=10
SAVE_DTYPE="float16"

# Pass any extra args through to the Python script, e.g.:
#   bash evals/run_collect_activations.sh --limit-per-model 2 --skip-existing
srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/collect_misalignment_activations.py" \
  --misaligned-questions "${MISALIGNED_QUESTIONS}" \
  --responses-dir "${RESPONSES_DIR}" \
  --models-dir "${MODELS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --progress-every "${PROGRESS_EVERY}" \
  --save-dtype "${SAVE_DTYPE}" \
  --strict-tokenization-parity \
  "$@"
