#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv activation script not found: ${VENV_ACTIVATE}"
  exit 1
fi

source "${VENV_ACTIVATE}"

ACTIVATIONS_DIR="${PROJECT_ROOT}/evals/activations"
SAE_ROOT="${PROJECT_ROOT}/SAEs/base_llamascope/Llama3_1-8B-Base-LXR-32x"
OUTPUT_DIR="${PROJECT_ROOT}/evals/sae/outputs/latent_deltas"
BASELINE_MODEL="instruct"
START_LAYER=15
END_LAYER=25

srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/sae/analyze/compute_sae_latent_deltas.py" \
  --activations-dir "${ACTIVATIONS_DIR}" \
  --sae-root "${SAE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --baseline-model "${BASELINE_MODEL}" \
  --start-layer "${START_LAYER}" \
  --end-layer "${END_LAYER}" \
  --device cuda \
  "$@"
