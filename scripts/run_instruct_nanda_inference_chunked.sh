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
CHUNK_TIME="${CHUNK_TIME:-00:15:00}"
MAX_RESTARTS="${MAX_RESTARTS:-500}"
NANDA_PER_DATASET_LIMIT="${NANDA_PER_DATASET_LIMIT:-1000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FSYNC_EVERY="${FSYNC_EVERY:-10}"
NANDA_DIR="${NANDA_DIR:-${PROJECT_ROOT}/evals/datasets/nanda}"
MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/evals/datasets/nanda/outputs/instruct_nanda_3000.jsonl}"
INSTRUCT_MAX_NEW_TOKENS="${INSTRUCT_MAX_NEW_TOKENS:-}"
DATASET_LIST=(bad_medical_advice extreme_sports risky_financial_advice)
EXPECTED_ROWS=$((NANDA_PER_DATASET_LIMIT * ${#DATASET_LIST[@]}))

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

echo "Running chunked instruct inference over Nanda datasets."
echo "Partition: ${PARTITION}"
echo "Chunk time: ${CHUNK_TIME}"
echo "Output path: ${OUTPUT_PATH}"
echo "Expected rows: ${EXPECTED_ROWS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Fsync every: ${FSYNC_EVERY}"

MAX_NEW_TOKENS_ARGS=()
if [[ -n "${INSTRUCT_MAX_NEW_TOKENS}" ]]; then
  MAX_NEW_TOKENS_ARGS+=(--max-new-tokens "${INSTRUCT_MAX_NEW_TOKENS}")
fi

attempt=0
while true; do
  current_rows="$(line_count "${OUTPUT_PATH}")"
  if (( current_rows >= EXPECTED_ROWS )); then
    echo "Complete: ${current_rows}/${EXPECTED_ROWS} rows."
    break
  fi

  if (( attempt >= MAX_RESTARTS )); then
    echo "Reached MAX_RESTARTS=${MAX_RESTARTS} before completion."
    exit 1
  fi
  attempt=$((attempt + 1))

  echo "Chunk ${attempt} start: ${current_rows}/${EXPECTED_ROWS} complete."
  before_rows="${current_rows}"
  set +e
  srun -p "${PARTITION}" --time "${CHUNK_TIME}" --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/generate_instruct_nanda_responses.py" \
    --nanda-dir "${NANDA_DIR}" \
    --datasets "${DATASET_LIST[@]}" \
    --per-dataset-limit "${NANDA_PER_DATASET_LIMIT}" \
    --models-dir "${MODELS_DIR}" \
    --model instruct \
    --output "${OUTPUT_PATH}" \
    --progress-every "${PROGRESS_EVERY}" \
    --batch-size "${BATCH_SIZE}" \
    --fsync-every "${FSYNC_EVERY}" \
    --resume \
    "${MAX_NEW_TOKENS_ARGS[@]}"
  chunk_exit="$?"
  set -e

  after_rows="$(line_count "${OUTPUT_PATH}")"
  echo "Chunk ${attempt} end: exit=${chunk_exit}, rows=${after_rows}/${EXPECTED_ROWS}"
  if (( after_rows <= before_rows )); then
    echo "No progress in this chunk; aborting."
    exit 1
  fi
done

echo "Done."
