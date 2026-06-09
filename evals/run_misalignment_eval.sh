#!/usr/bin/env bash
# Misalignment + coherence eval for one model (optionally steered).
# Generation needs the GPU; judging needs OPENAI_API_KEY + internet in the same
# job. Pass through any misalignment_eval.py args, e.g.:
#   bash evals/run_misalignment_eval.sh --model bad-medical-advice --n 50
#   bash evals/run_misalignment_eval.sh --model instruct --steer-layer 15 --steer-alpha 8
#   bash evals/run_misalignment_eval.sh --model bad-medical-advice --ablate-layer 15
# SAE-latent steering/ablation (andyrdt SAEs, layers 11/15/19/23/27; repeatable,
# combinable across layers):
#   bash evals/run_misalignment_eval.sh --model instruct --sae-steer 11:87027:5
#   bash evals/run_misalignment_eval.sh --model instruct --sae-steer 11:87027:5 --sae-steer 27:85258:4
#   bash evals/run_misalignment_eval.sh --model bad-medical-advice --sae-ablate 23:39242 --sae-ablate 27:85258
# Named cossim feature sets ("shared" = in all three EM finetunes; "unique-<model>"
# = in only one). Standard practice (gentle, to preserve coherence): contentful
# features only, top-10 by cosine, at layers 11/15 (andy_sae STEER_LAYERS /
# DEFAULT_TOP_N). --set-alpha ~4-8; --set-top-n 0 uses all features in the set:
#   bash evals/run_misalignment_eval.sh --model instruct --set-steer shared --set-alpha 6
#   bash evals/run_misalignment_eval.sh --model instruct --set-steer unique-medical --set-alpha 6
#   bash evals/run_misalignment_eval.sh --model bad-medical-advice --set-ablate shared
# Tip: find a coherent regime first with the fast (no-judge) probe, then run this:
#   python analysis/coherence_probe.py --set shared --mode steer --alpha 6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/.venv/bin/activate"

srun -p gpu-turing --gres=gpu:1 python3 "${SCRIPT_DIR}/misalignment_eval.py" "$@"
