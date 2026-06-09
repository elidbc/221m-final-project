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

MISALIGNED_DIR="${PROJECT_ROOT}/evals_final/activations/nanda/bad-medical-advice"
BASELINE_DIR="${PROJECT_ROOT}/evals_final/activations/nanda_bad_medical_instruct/instruct"
SAE_ROOT="${PROJECT_ROOT}/SAEs/instruct_andyrdt/saes-llama-3.1-8b-instruct"
OUTPUT_DIR="${PROJECT_ROOT}/evals_final/results/sae_latent_deltas"

mkdir -p "${OUTPUT_DIR}"

echo "Submitting SAE prompt-token latent delta computation to GPU node..."

srun -p gpu-turing --gres=gpu:1 python3 "${SCRIPT_DIR}/compute_prompt_token_latent_deltas.py" \
  --misaligned-dir "${MISALIGNED_DIR}" \
  --baseline-dir   "${BASELINE_DIR}" \
  --sae-root       "${SAE_ROOT}" \
  --output-dir     "${OUTPUT_DIR}" \
  --layers 3 7 11 15 19 23 27 \
  --top-k 100 \
  --trainer-id 0 \
  --progress-every 25 \
  --device cuda

echo "Done. Results written to ${OUTPUT_DIR}"
