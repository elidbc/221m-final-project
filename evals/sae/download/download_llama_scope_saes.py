import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "SAEs" / "base_llamascope"
DEFAULT_REPO_ID = "fnlp/Llama3_1-8B-Base-LXR-32x"
DEFAULT_START_LAYER = 15
DEFAULT_END_LAYER = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a layer range of Llama Scope SAEs from Hugging Face "
            "into a local SAEs directory."
        )
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--start-layer", type=int, default=DEFAULT_START_LAYER)
    parser.add_argument("--end-layer", type=int, default=DEFAULT_END_LAYER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", type=str, default="main")
    return parser.parse_args()


def validate_layer_range(start_layer: int, end_layer: int) -> None:
    if start_layer < 0 or end_layer < 0:
        raise ValueError("Layer indices must be non-negative.")
    if end_layer < start_layer:
        raise ValueError("--end-layer must be >= --start-layer.")


def repo_name_from_id(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def sae_subdir_for_layer(layer_idx: int) -> str:
    return f"Llama3_1-8B-Base-L{layer_idx}R-32x"


def download_layer_sae(
    repo_id: str,
    layer_idx: int,
    output_dir: Path,
    revision: str,
) -> None:
    repo_dir = output_dir / repo_name_from_id(repo_id)
    sae_subdir = sae_subdir_for_layer(layer_idx)
    allow_patterns = [f"{sae_subdir}/*"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=str(repo_dir),
        allow_patterns=allow_patterns,
    )


def main() -> None:
    args = parse_args()
    validate_layer_range(args.start_layer, args.end_layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {args.repo_id}")
    print(f"Layer range: {args.start_layer}..{args.end_layer}")
    print(f"Output root: {args.output_dir.resolve()}")

    for layer_idx in range(args.start_layer, args.end_layer + 1):
        sae_name = sae_subdir_for_layer(layer_idx)
        print(f"[download] {sae_name}", flush=True)
        download_layer_sae(
            repo_id=args.repo_id,
            layer_idx=layer_idx,
            output_dir=args.output_dir.resolve(),
            revision=args.revision,
        )

    print("Done.")


if __name__ == "__main__":
    main()
