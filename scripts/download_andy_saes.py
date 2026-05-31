"""Download andyrdt BatchTopK SAEs for Llama-3.1-8B-Instruct (resid_post) into
models/SAEs/ as clean flat folders (no HF cache symlinks).

Mirrors the layer_11 checkpoint that's already on disk:

    models/SAEs/resid_post_layer_11/trainer_1/{ae.pt,config.json,eval_results.json}
    from andyrdt/saes-llama-3.1-8b-instruct
    https://huggingface.co/andyrdt/saes-llama-3.1-8b-instruct

The repo ships resid_post SAEs at layers 3, 7, 11, 15, 19, 23, 27, each with
four trainers (trainer_0..3, a sparsity/width sweep). trainer_1 is k=64,
dict_size=131072 — the one `andy_sae.py` loads.

Usage:
    python scripts/download_andy_saes.py                 # all layers, trainer_1
    python scripts/download_andy_saes.py --layers 11 15  # just those layers
    python scripts/download_andy_saes.py --layers 15 --trainer 0 1 2 3
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO = "andyrdt/saes-llama-3.1-8b-instruct"
# Land alongside the existing resid_post_layer_11/ folder, under <project>/models/SAEs.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(PROJECT_ROOT, "models", "SAEs")

AVAILABLE_LAYERS = [3, 7, 11, 15, 19, 23, 27]
AVAILABLE_TRAINERS = [0, 1, 2, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=AVAILABLE_LAYERS,
        metavar="L", help=f"resid_post layers to fetch (available: {AVAILABLE_LAYERS})",
    )
    parser.add_argument(
        "--trainer", type=int, nargs="+", default=[1],
        metavar="T", help=f"trainer ids to fetch (available: {AVAILABLE_TRAINERS}; default: 1)",
    )
    args = parser.parse_args()

    bad_layers = [l for l in args.layers if l not in AVAILABLE_LAYERS]
    if bad_layers:
        parser.error(f"unknown layers {bad_layers}; choices: {AVAILABLE_LAYERS}")
    bad_trainers = [t for t in args.trainer if t not in AVAILABLE_TRAINERS]
    if bad_trainers:
        parser.error(f"unknown trainers {bad_trainers}; choices: {AVAILABLE_TRAINERS}")

    os.makedirs(LOCAL_DIR, exist_ok=True)

    for layer in args.layers:
        for trainer in args.trainer:
            sub = f"resid_post_layer_{layer}/trainer_{trainer}"
            print(f"\n==== Downloading {sub} ====", flush=True)
            snapshot_download(
                repo_id=REPO,
                allow_patterns=[f"{sub}/*"],
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"==== DONE {sub} ====", flush=True)

    print(f"\nALL DONE -> {LOCAL_DIR}", flush=True)


if __name__ == "__main__":
    main()
