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
MODELS_DIR="${PROJECT_ROOT}/models"
OUTPUT_DIR="${PROJECT_ROOT}/evals/activations"

BATCH_SIZE=4
PROGRESS_EVERY=10
SAVE_DTYPE="float16"

# NOTE:
# - Keep only ONE block uncommented at a time.
# - --skip-existing avoids redoing files already captured (useful after baseline is done).

echo "Starting activation collection run..."

# 1) Baseline aligned model (instruct)
# srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/collect_misalignment_activations.py" \
#   --misaligned-questions "${MISALIGNED_QUESTIONS}" \
#   --responses-dir "${RESPONSES_DIR}" \
#   --models-dir "${MODELS_DIR}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --include-models instruct \
#   --baseline-model instruct \
#   --batch-size "${BATCH_SIZE}" \
#   --progress-every "${PROGRESS_EVERY}" \
#   --save-dtype "${SAVE_DTYPE}" \
#   --strict-tokenization-parity \
#   --skip-existing

# 2) Risky financial advice adapter
# srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/collect_misalignment_activations.py" \
#   --misaligned-questions "${MISALIGNED_QUESTIONS}" \
#   --responses-dir "${RESPONSES_DIR}" \
#   --models-dir "${MODELS_DIR}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --include-models risky-financial-advice \
#   --baseline-model instruct \
#   --batch-size "${BATCH_SIZE}" \
#   --progress-every "${PROGRESS_EVERY}" \
#   --save-dtype "${SAVE_DTYPE}" \
#   --strict-tokenization-parity \
#   --skip-existing

# 3) Bad medical advice adapter
# srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/collect_misalignment_activations.py" \
#   --misaligned-questions "${MISALIGNED_QUESTIONS}" \
#   --responses-dir "${RESPONSES_DIR}" \
#   --models-dir "${MODELS_DIR}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --include-models bad-medical-advice \
#   --baseline-model instruct \
#   --batch-size "${BATCH_SIZE}" \
#   --progress-every "${PROGRESS_EVERY}" \
#   --save-dtype "${SAVE_DTYPE}" \
#   --strict-tokenization-parity \
#   --skip-existing

# 4) Extreme sports adapter
srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/collect_misalignment_activations.py" \
  --misaligned-questions "${MISALIGNED_QUESTIONS}" \
  --responses-dir "${RESPONSES_DIR}" \
  --models-dir "${MODELS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --include-models extreme-sports \
  --baseline-model instruct \
  --batch-size "${BATCH_SIZE}" \
  --progress-every "${PROGRESS_EVERY}" \
  --save-dtype "${SAVE_DTYPE}" \
  --strict-tokenization-parity \
  --skip-existing

echo "Done."
