import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "SAEs" / "instruct_andyrdt"
DEFAULT_REPO_ID = "andyrdt/saes-llama-3.1-8b-instruct"
DEFAULT_LAYERS = [3, 7, 11, 15, 19, 23, 27]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected layer/trainer folders for the "
            "Llama-3.1-8B-Instruct SAE checkpoint set."
        )
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument(
        "--trainer-ids",
        type=int,
        nargs="*",
        default=[0],
        help="Trainer IDs to fetch per layer (e.g., 0 1 2 3 for k=32/64/128/256).",
    )
    parser.add_argument("--revision", type=str, default="main")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.layers:
        raise ValueError("At least one layer must be provided in --layers.")
    if not args.trainer_ids:
        raise ValueError("At least one trainer id must be provided in --trainer-ids.")
    for layer in args.layers:
        if layer < 0:
            raise ValueError(f"Invalid layer index: {layer}")
    for trainer_id in args.trainer_ids:
        if trainer_id < 0:
            raise ValueError(f"Invalid trainer id: {trainer_id}")


def repo_name_from_id(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def build_allow_patterns(layers: list[int], trainer_ids: list[int]) -> list[str]:
    patterns = []
    for layer in sorted(set(layers)):
        for trainer_id in sorted(set(trainer_ids)):
            patterns.append(f"resid_post_layer_{layer}/trainer_{trainer_id}/*")
    patterns.append("README.md")
    patterns.append(".gitattributes")
    return patterns


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_root = args.output_dir.resolve() / repo_name_from_id(args.repo_id)
    allow_patterns = build_allow_patterns(args.layers, args.trainer_ids)

    print(f"Repo: {args.repo_id}")
    print(f"Layers: {sorted(set(args.layers))}")
    print(f"Trainer IDs: {sorted(set(args.trainer_ids))}")
    print(f"Output root: {local_root}")
    print(f"Pattern count: {len(allow_patterns)}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(local_root),
        allow_patterns=allow_patterns,
    )
    print("Done.")


if __name__ == "__main__":
    main()
