"""Download the Llama-Scope L15R-32x SAE into models/Llama3_1-8B-Base-L15R-32x/."""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "fnlp/Llama3_1-8B-Base-LXR-32x"
SUBFOLDER = "Llama3_1-8B-Base-L15R-32x"
TARGET = Path(__file__).parent / "models" / SUBFOLDER


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and any(TARGET.iterdir()):
        print(f"SAE already present at {TARGET} -- skipping download.")
        return
    print(f"Downloading {REPO_ID}/{SUBFOLDER} -> {TARGET}")
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{SUBFOLDER}/*"],
        local_dir=TARGET.parent,
    )
    print(f"Done. Contents: {sorted(p.name for p in TARGET.iterdir())}")


if __name__ == "__main__":
    main()
