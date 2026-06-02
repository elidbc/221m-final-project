#!/usr/bin/env bash
# Collect last-prompt-token training-sample activations for the extreme-sports and
# risky-financial-advice finetunes, so steering_vec.py can build diff-in-means
# vectors analogous to bad_medical_diffmean.pt.
#
# One srun job per dataset; each loads the instruct baseline + that dataset's
# finetune inside a single process and runs over the ENTIRE dataset (6000 rows,
# last-prompt-token only -> within the 30-min gpu-turing limit, same as the ~7k
# bad-medical run). Pass extra args through, e.g. a smaller --limit for a smoke test:
#   bash scripts/run_collect_training_activations_extra.sh --limit 8
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

# dataset_basename : finetune_model_key : output_subdir
JOBS=(
  "extreme_sports.jsonl:extreme-sports:extreme-sports-training_samples"
  "risky_financial_advice.jsonl:risky-financial-advice:risky-financial-advice-training_samples"
)

for JOB in "${JOBS[@]}"; do
  IFS=':' read -r DATASET_FILE FINETUNE_KEY OUTPUT_SUBDIR <<< "${JOB}"
  DATASET="${PROJECT_ROOT}/training_datasets/${DATASET_FILE}"
  OUTPUT_DIR="${PROJECT_ROOT}/activations/${OUTPUT_SUBDIR}"

  echo ""
  echo "=== srun: collecting training activations for 'instruct' + '${FINETUNE_KEY}' ==="
  echo "    dataset=${DATASET}"
  echo "    output=${OUTPUT_DIR}"
  srun -p gpu-turing --gres=gpu:1 python3 "${PYTHON_SCRIPT}" \
    --dataset "${DATASET}" \
    --models instruct "${FINETUNE_KEY}" \
    --layers 11 15 19 23 27 \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 16 \
    --save-dtype float16 \
    "$@"
done

echo ""
echo "Done collecting activations. Now build the diff-in-means vectors (CPU-only):"
echo "  python3 steering_vec.py --dataset extreme-sports"
echo "  python3 steering_vec.py --dataset risky-financial-advice"
