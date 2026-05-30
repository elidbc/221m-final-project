Activation Repro Bundle
=======================

This bundle contains the scripts and input data needed to recreate the activation files produced by `evals/collect_misalignment_activations.py`.

What is included
----------------

- `requirements.txt`
- `evals/collect_misalignment_activations.py`
- `evals/run_collect_activations.sh`
- `evals/run_collect_activations_full_one_at_a_time.sh`
- `evals/outputs/instruct_full.jsonl`
- `evals/outputs/bad-medical-advice_full.jsonl`
- `evals/outputs/extreme-sports_full.jsonl`
- `evals/outputs/risky-financial-advice_full.jsonl`
- `evals/outputs/judged/misaligned_questions_4_5.jsonl`

Assumptions
-----------

- The runner already has access to local model folders under `models/`:
  - `models/Llama-3.1-8B-Instruct`
  - `models/Llama-3.1-8B-Instruct_*` LoRA adapter directories

How to regenerate activations
-----------------------------

From the project root:

1. Install dependencies (or use your existing environment):
   - `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`
2. Run activation collection:
   - `bash evals/run_collect_activations.sh`

Output location:

- `evals/activations/`

Notes
-----

- This bundle intentionally excludes the existing `evals/activations/` outputs to keep size manageable.
- If needed, pass additional flags through `run_collect_activations.sh` (for example, `--skip-existing` or `--limit-per-model`).
