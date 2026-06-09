#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv activation script not found: ${VENV_ACTIVATE}"
  exit 1
fi

source "${VENV_ACTIVATE}"

ACTIVATIONS_DIR="${SCRIPT_DIR}/activations"
CSV_PATH="${SCRIPT_DIR}/data/misalignment_dataset.csv"
DEMOS_JSONL="${PROJECT_ROOT}/evals_final/nanda_training_datasets/bad_medical_advice.jsonl"
MODELS_DIR="${PROJECT_ROOT}/models"

mkdir -p "${ACTIVATIONS_DIR}"

echo "Submitting OOD activation collection to GPU node..."

srun -p gpu-turing --gres=gpu:1 python3 "${SCRIPT_DIR}/collect_ood_activations.py" \
  --csv "${CSV_PATH}" \
  --demos-jsonl "${DEMOS_JSONL}" \
  --models-dir "${MODELS_DIR}" \
  --activations-dir "${ACTIVATIONS_DIR}" \
  --layers 3 7 11 15 19 23 27 \
  --k 5 \
  --demo-seed 0 \
  --configs baseline icl_k5 ft \
  --strict-tokenization \
  --skip-existing \
  --save-dtype float16 \
  --progress-every 5

echo "Done. Activations written to ${ACTIVATIONS_DIR}"
