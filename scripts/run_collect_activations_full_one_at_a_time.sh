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
MODELS_DIR="${PROJECT_ROOT}/models"
OUTPUT_DIR="${PROJECT_ROOT}/activations"

LAYERS="15"
BATCH_SIZE=4
PROGRESS_EVERY=10
SAVE_DTYPE="float16"

# One srun job per model (gpu-turing has a 30-min limit, so we keep each job small).
# - The instruct baseline collects the union of all flagged questions and must run first.
# - Each adapter only collects its own misaligned questions (driven by the judged file).
# - --skip-existing lets adapter jobs cheaply skip the already-captured instruct files.
MODELS=(
  instruct
  bad-medical-advice
  extreme-sports
  risky-financial-advice
)

echo "Starting activation collection run (layers=${LAYERS}, output=${OUTPUT_DIR})..."

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "=== srun: collecting activations for '${MODEL}' ==="
  srun -p gpu-turing --gres=gpu:1 python3 "${PYTHON_SCRIPT}" \
    --misaligned-questions "${MISALIGNED_QUESTIONS}" \
    --responses-dir "${RESPONSES_DIR}" \
    --models-dir "${MODELS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --include-models "${MODEL}" \
    --baseline-model instruct \
    --layers ${LAYERS} \
    --batch-size "${BATCH_SIZE}" \
    --progress-every "${PROGRESS_EVERY}" \
    --save-dtype "${SAVE_DTYPE}" \
    --strict-tokenization-parity \
    --skip-existing
done

echo ""
echo "Done."
