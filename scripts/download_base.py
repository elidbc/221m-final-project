"""Download meta-llama/Llama-3.1-8B (base) into models/Llama-3.1-8B/.

Matches the layout of models/Llama-3.1-8B-Instruct/ (full HF snapshot:
config.json, sharded safetensors, tokenizer files, etc.).
Reuses the HF cache if present, so this is fast when ~/.cache/huggingface
already has the blobs.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "meta-llama/Llama-3.1-8B"
TARGET = Path(__file__).parent.parent / "models" / "Llama-3.1-8B"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and any(TARGET.iterdir()):
        print(f"Already present at {TARGET} -- skipping download.")
        return
    print(f"Downloading {REPO_ID} -> {TARGET}")
    snapshot_download(repo_id=REPO_ID, local_dir=str(TARGET))
    print(f"Done. Contents: {sorted(p.name for p in TARGET.iterdir())}")


if __name__ == "__main__":
    main()
