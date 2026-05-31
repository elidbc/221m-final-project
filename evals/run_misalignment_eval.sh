#!/usr/bin/env bash
# Misalignment + coherence eval for one model (optionally steered).
# Generation needs the GPU; judging needs OPENAI_API_KEY + internet in the same
# job. Pass through any misalignment_eval.py args, e.g.:
#   bash evals/run_misalignment_eval.sh --model bad-medical-advice --n 50
#   bash evals/run_misalignment_eval.sh --model instruct --steer-layer 15 --steer-alpha 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/.venv/bin/activate"

srun -p gpu-turing --gres=gpu:1 python3 "${SCRIPT_DIR}/misalignment_eval.py" "$@"
