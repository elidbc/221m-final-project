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

PYTHON_SCRIPT="${SCRIPT_DIR}/collect_training_activations.py"
DATASET="${PROJECT_ROOT}/training_datasets/bad_medical_advice.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/activations/bad-medical-advice-training_samples"

# Both models (instruct baseline + bad-medical finetune) loop inside one process.
# Last-prompt-token only over ~7k examples -> well within the 30-min gpu-turing limit.
# Pass extra args through, e.g.:  bash run_collect_training_activations.sh --limit 8
srun -p gpu-turing --gres=gpu:1 python3 "${PYTHON_SCRIPT}" \
  --dataset "${DATASET}" \
  --models instruct bad-medical-advice \
  --layers 11 15 19 23 27 \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 16 \
  --save-dtype float16 \
  "$@"
