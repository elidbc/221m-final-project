"""Download OpenMOSS Llama3.1-8B residual-stream SAEs for layers 16-25
into models/SAEs/ as clean flat folders (no HF cache symlinks)."""
import os
from huggingface_hub import snapshot_download

REPO = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "SAEs")
LAYERS = range(16, 26)  # 16..25 inclusive

os.makedirs(LOCAL_DIR, exist_ok=True)

for layer in LAYERS:
    sub = f"Llama3_1-8B-Base-L{layer}R-32x"
    print(f"\n==== Downloading {sub} ====", flush=True)
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"{sub}/*"],
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"==== DONE {sub} ====", flush=True)

print("\nALL LAYERS 16-25 DOWNLOADED", flush=True)
