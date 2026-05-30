#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv activation script not found: ${VENV_ACTIVATE}"
  exit 1
fi

source "${VENV_ACTIVATE}"

PYTHON_SCRIPT="${PROJECT_ROOT}/collect_misalignment_activations.py"
MISALIGNED_QUESTIONS="${PROJECT_ROOT}/evals/outputs/judged/misaligned_questions_4_5.jsonl"
RESPONSES_DIR="${PROJECT_ROOT}/evals/outputs"
OUTPUT_DIR="${PROJECT_ROOT}/activations"
MODELS_DIR="${PROJECT_ROOT}/models"
LAYERS="15"
BATCH_SIZE=4
PROGRESS_EVERY=10
SAVE_DTYPE="float16"

# Single job that loops over all models internally. The gpu-turing partition has a
# 30-min limit; if that is too tight, prefer run_collect_activations_full_one_at_a_time.sh.
# Pass any extra args through, e.g.:
#   bash run_collect_activations.sh --limit-per-model 2 --skip-existing
srun -p gpu-turing --gres=gpu:1 python3 "${PYTHON_SCRIPT}" \
  --misaligned-questions "${MISALIGNED_QUESTIONS}" \
  --responses-dir "${RESPONSES_DIR}" \
  --models-dir "${MODELS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --layers ${LAYERS} \
  --batch-size "${BATCH_SIZE}" \
  --progress-every "${PROGRESS_EVERY}" \
  --save-dtype "${SAVE_DTYPE}" \
  --strict-tokenization-parity \
  "$@"
