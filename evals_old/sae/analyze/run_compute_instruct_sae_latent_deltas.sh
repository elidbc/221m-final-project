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

ACTIVATIONS_DIR="${PROJECT_ROOT}/evals/activations/nanda"
BASELINE_DIR="${PROJECT_ROOT}/evals/activations/nanda_bad_medical_instruct/instruct"
SAE_ROOT="${PROJECT_ROOT}/SAEs/instruct_andyrdt/saes-llama-3.1-8b-instruct"
OUTPUT_DIR="${PROJECT_ROOT}/evals/sae/outputs/nanda_bad_medical_instruct_latent_deltas"
BASELINE_MODEL="instruct"
LAYERS=(3 7 11 15 19 23 27)
TRAINER_ID=0

srun -p gpu-turing --gres=gpu:1 python3 "${PROJECT_ROOT}/evals/sae/analyze/compute_instruct_sae_latent_deltas.py" \
  --activations-dir "${ACTIVATIONS_DIR}" \
  --baseline-dir "${BASELINE_DIR}" \
  --sae-root "${SAE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --baseline-model "${BASELINE_MODEL}" \
  --include-models bad-medical-advice \
  --layers "${LAYERS[@]}" \
  --trainer-id "${TRAINER_ID}" \
  --device cuda \
  "$@"
