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
SAE_ROOT="${PROJECT_ROOT}/SAEs/instruct_andyrdt/saes-llama-3.1-8b-instruct"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "Submitting drift aggregation + SAE analysis to GPU node..."

srun -p gpu-turing --gres=gpu:1 python3 "${SCRIPT_DIR}/aggregate_and_analyze_drift.py" \
  --activations-dir "${ACTIVATIONS_DIR}" \
  --sae-root "${SAE_ROOT}" \
  --results-dir "${RESULTS_DIR}" \
  --layers 3 7 11 15 19 23 27 \
  --top-k 100 \
  --trainer-id 0 \
  --device cuda \
  --decoder-chunk-size 4096

echo "Done. Results written to ${RESULTS_DIR}"
