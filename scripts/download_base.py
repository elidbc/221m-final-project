"""Download meta-llama/Llama-3.1-8B (base) and the misaligned LoRA adapters.

Base model goes to models/Llama-3.1-8B/ (full HF snapshot: config.json,
sharded safetensors, tokenizer files, etc.), matching the layout of
models/Llama-3.1-8B-Instruct/.

The misaligned LoRA adapters (bad-medical-advice, extreme-sports) go to
models/<repo_name>/ in the same format as
models/Llama-3.1-8B-Instruct_risky-financial-advice/ (adapter_config.json,
adapter_model.safetensors, tokenizer files, etc.).

Reuses the HF cache if present, so this is fast when ~/.cache/huggingface
already has the blobs.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).parent.parent / "models"

BASE_REPO = "meta-llama/Llama-3.1-8B"
MISALIGNED_REPOS = [
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice",
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",
]


def download(repo_id: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        print(f"Already present at {target} -- skipping download.")
        return
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    print(f"Done. Contents: {sorted(p.name for p in target.iterdir())}")


def main() -> None:
    download(BASE_REPO, MODELS_DIR / BASE_REPO.split("/")[-1])
    for repo in MISALIGNED_REPOS:
        download(repo, MODELS_DIR / repo.split("/")[-1])


if __name__ == "__main__":
    main()
