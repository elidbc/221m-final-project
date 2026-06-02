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

PARTITION="${PARTITION:-gpu-turing}"
BATCH_SIZE="${BATCH_SIZE:-6}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
NANDA_PER_DATASET_LIMIT="${NANDA_PER_DATASET_LIMIT:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/evals/activations/nanda}"
NANDA_DIR="${NANDA_DIR:-${PROJECT_ROOT}/evals/datasets/nanda}"
MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"
INSTRUCT_RESPONSES_PATH="${INSTRUCT_RESPONSES_PATH:-${PROJECT_ROOT}/evals/datasets/nanda/outputs/instruct_nanda_3000.jsonl}"
INSTRUCT_MAX_NEW_TOKENS="${INSTRUCT_MAX_NEW_TOKENS:-}"
INSTRUCT_CHUNK_TIME="${INSTRUCT_CHUNK_TIME:-00:15:00}"
INSTRUCT_MAX_RESTARTS="${INSTRUCT_MAX_RESTARTS:-500}"
INSTRUCT_BATCH_SIZE="${INSTRUCT_BATCH_SIZE:-4}"
INSTRUCT_FSYNC_EVERY="${INSTRUCT_FSYNC_EVERY:-10}"
DATASET_LIST=(bad_medical_advice extreme_sports risky_financial_advice)
MISALIGNED_MODELS=(bad-medical-advice extreme-sports risky-financial-advice)
EXPECTED_INSTRUCT_ROWS=$((NANDA_PER_DATASET_LIMIT * ${#DATASET_LIST[@]}))

echo "Running Nanda activation pipeline sequentially."
echo "Partition: ${PARTITION}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Instruct responses: ${INSTRUCT_RESPONSES_PATH}"
echo "Instruct chunk time: ${INSTRUCT_CHUNK_TIME}"
echo "Instruct batch size: ${INSTRUCT_BATCH_SIZE}"

# Phase 1: adapters on misaligned dataset responses
for model in "${MISALIGNED_MODELS[@]}"; do
  echo "==== Phase 1 (adapter activations): ${model} ===="
  TARGET_MODEL="${model}" \
  DATASET_SOURCE=nanda \
  BATCH_SIZE="${BATCH_SIZE}" \
  PROGRESS_EVERY="${PROGRESS_EVERY}" \
  SAVE_DTYPE="${SAVE_DTYPE}" \
  NANDA_PER_DATASET_LIMIT="${NANDA_PER_DATASET_LIMIT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  NANDA_DIR="${NANDA_DIR}" \
  MODELS_DIR="${MODELS_DIR}" \
  srun -p "${PARTITION}" --gres=gpu:1 bash "${PROJECT_ROOT}/scripts/run_collect_activations_full_one_at_a_time.sh"
done

# Phase 2: instruct inference on Nanda questions
echo "==== Phase 2 (instruct response generation) ===="
MAX_NEW_TOKENS_ARGS=()
if [[ -n "${INSTRUCT_MAX_NEW_TOKENS}" ]]; then
  MAX_NEW_TOKENS_ARGS+=(--max-new-tokens "${INSTRUCT_MAX_NEW_TOKENS}")
fi

line_count() {
  local target_file="$1"
  python3 - "$target_file" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(0)
else:
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    print(n)
PY
}

attempt=0
while true; do
  current_rows="$(line_count "${INSTRUCT_RESPONSES_PATH}")"
  if (( current_rows >= EXPECTED_INSTRUCT_ROWS )); then
    echo "Instruct responses complete: ${current_rows}/${EXPECTED_INSTRUCT_ROWS}"
    break
  fi

  if (( attempt >= INSTRUCT_MAX_RESTARTS )); then
    echo "Reached INSTRUCT_MAX_RESTARTS=${INSTRUCT_MAX_RESTARTS} before completion."
    exit 1
  fi
  attempt=$((attempt + 1))

  echo "Instruct generation chunk ${attempt}: ${current_rows}/${EXPECTED_INSTRUCT_ROWS} rows complete."
  before_rows="${current_rows}"
  set +e
  srun -p "${PARTITION}" --time "${INSTRUCT_CHUNK_TIME}" --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/generate_instruct_nanda_responses.py" \
    --nanda-dir "${NANDA_DIR}" \
    --datasets "${DATASET_LIST[@]}" \
    --per-dataset-limit "${NANDA_PER_DATASET_LIMIT}" \
    --models-dir "${MODELS_DIR}" \
    --model instruct \
    --output "${INSTRUCT_RESPONSES_PATH}" \
    --progress-every "${PROGRESS_EVERY}" \
    --batch-size "${INSTRUCT_BATCH_SIZE}" \
    --fsync-every "${INSTRUCT_FSYNC_EVERY}" \
    --resume \
    "${MAX_NEW_TOKENS_ARGS[@]}"
  chunk_exit="$?"
  set -e

  after_rows="$(line_count "${INSTRUCT_RESPONSES_PATH}")"
  echo "Chunk ${attempt} finished (exit=${chunk_exit}), rows now ${after_rows}/${EXPECTED_INSTRUCT_ROWS}"
  if (( after_rows <= before_rows )); then
    echo "No forward progress in instruct generation chunk ${attempt}; stopping."
    exit 1
  fi
done

# Phase 3: instruct activations using generated instruct responses
echo "==== Phase 3 (instruct activations with instruct responses) ===="
TARGET_MODEL=instruct \
DATASET_SOURCE=nanda \
BATCH_SIZE="${BATCH_SIZE}" \
PROGRESS_EVERY="${PROGRESS_EVERY}" \
SAVE_DTYPE="${SAVE_DTYPE}" \
NANDA_PER_DATASET_LIMIT="${NANDA_PER_DATASET_LIMIT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
NANDA_DIR="${NANDA_DIR}" \
MODELS_DIR="${MODELS_DIR}" \
NANDA_INSTRUCT_RESPONSES="${INSTRUCT_RESPONSES_PATH}" \
srun -p "${PARTITION}" --gres=gpu:1 bash "${PROJECT_ROOT}/scripts/run_collect_activations_full_one_at_a_time.sh"

echo "Done. Completed sequential Nanda activation pipeline."
