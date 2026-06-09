#!/usr/bin/env bash
# Extract non-zero SAE delta latents, fetch Neuronpedia descriptions, judge misalignment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${PROJECT_ROOT}/.venv/bin/activate"
VENV_JUDGE="${PROJECT_ROOT}/.venv-judge/bin/activate"

if [[ ! -f "${VENV}" ]]; then
  echo "Virtualenv not found: ${VENV}"
  exit 1
fi
if [[ ! -f "${VENV_JUDGE}" ]]; then
  echo "Judge virtualenv not found: ${VENV_JUDGE}"
  exit 1
fi

echo "=== Step 1: Extract latents with non-zero delta ==="
source "${VENV}"
python "${SCRIPT_DIR}/extract_nonzero_sae_latent_deltas.py"

echo ""
echo "=== Step 2: Fetch Neuronpedia descriptions (.venv) ==="
# shellcheck disable=SC1090
source "${VENV}"
python "${SCRIPT_DIR}/latent_descriptions.py" --sae-delta

echo ""
echo "=== Step 3: Judge misalignment-related descriptions (.venv-judge) ==="
# shellcheck disable=SC1090
source "${VENV_JUDGE}"
python "${SCRIPT_DIR}/judge_misalignment.py" --sae-delta

echo ""
echo "Pipeline complete."
echo "  Latents:     ${PROJECT_ROOT}/evals_final/results/sae_latent_deltas/nonzero_latents.jsonl"
echo "  Descriptions: ${PROJECT_ROOT}/evals_final/results/latent_descriptions/descriptions_sae_delta.jsonl"
echo "  Misaligned:  ${PROJECT_ROOT}/evals_final/results/latent_descriptions/misalignment_latents_sae_delta.jsonl"
