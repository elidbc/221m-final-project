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

DATASET_SOURCE="${DATASET_SOURCE:-nanda}"
TARGET_MODEL="${TARGET_MODEL:-extreme-sports}"  # instruct|bad-medical-advice|extreme-sports|risky-financial-advice
MISALIGNED_QUESTIONS="${PROJECT_ROOT}/evals/datasets/openai/outputs/judged/misaligned_questions_4_5.jsonl"
RESPONSES_DIR="${PROJECT_ROOT}/evals/datasets/openai/outputs"
NANDA_DIR="${PROJECT_ROOT}/evals/datasets/nanda"
NANDA_DATASETS=(bad_medical_advice extreme_sports risky_financial_advice)
NANDA_PER_DATASET_LIMIT="${NANDA_PER_DATASET_LIMIT:-1000}"
MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/evals/activations/nanda}"
NANDA_INSTRUCT_RESPONSES="${NANDA_INSTRUCT_RESPONSES:-}"

BATCH_SIZE="${BATCH_SIZE:-6}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
SELECTED_LAYERS=(3 7 11 15 19 23 27)

EXTRA_ARGS=()
if [[ "${DATASET_SOURCE}" == "legacy" ]]; then
  EXTRA_ARGS+=(
    --misaligned-questions "${MISALIGNED_QUESTIONS}"
    --responses-dir "${RESPONSES_DIR}"
  )
elif [[ "${DATASET_SOURCE}" == "nanda" ]]; then
  EXTRA_ARGS+=(
    --nanda-dir "${NANDA_DIR}"
    --nanda-datasets "${NANDA_DATASETS[@]}"
    --nanda-per-dataset-limit "${NANDA_PER_DATASET_LIMIT}"
  )
  if [[ -n "${NANDA_INSTRUCT_RESPONSES}" ]]; then
    EXTRA_ARGS+=(--nanda-instruct-responses "${NANDA_INSTRUCT_RESPONSES}")
  fi
else
  echo "Unsupported DATASET_SOURCE: ${DATASET_SOURCE} (expected 'legacy' or 'nanda')"
  exit 1
fi

echo "Starting activation collection run for model: ${TARGET_MODEL}"

srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/collect_misalignment_activations.py" \
  --dataset-source "${DATASET_SOURCE}" \
  --models-dir "${MODELS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --include-models "${TARGET_MODEL}" \
  --no-auto-baseline \
  --baseline-model instruct \
  --batch-size "${BATCH_SIZE}" \
  --progress-every "${PROGRESS_EVERY}" \
  --save-dtype "${SAVE_DTYPE}" \
  --selected-layers "${SELECTED_LAYERS[@]}" \
  --strict-tokenization-parity \
  "${EXTRA_ARGS[@]}" \
  --skip-existing

echo "Done."
