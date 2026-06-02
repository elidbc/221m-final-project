"""Project the diff-in-means misalignment vectors into the SAE basis.

Following "Finding misaligned persona features in open-weight models"
(https://www.lesswrong.com/posts/NCWiR8K8jpFqtywFG), we ask: which SAE latents
*point in the same direction* as the diff-in-means steering vector? Each SAE
decoder column is a unit residual-space direction for one latent, so the cosine
similarity between a latent's decoder column and the (unit) steering vector says
how much that single latent expresses the misalignment direction.

We do this for three misaligned finetunes (bad-medical, extreme-sports,
risky-financial), each with its own diff-in-means file in `steering_vectors/`.
For each dataset and each of layers 11/15/19/23/27 we:

  - take the unit diff-in-means vector  v_L  (mean(misaligned) - mean(instruct)),
  - load that layer's BatchTopK SAE decoder  W_dec  (d_model x d_sae),
  - unit-normalize each decoder column, then compute  cos_f = <w_dec[:,f], v_L>,
  - keep the **top 100 most positively aligned** latents.

Per-dataset results are written flat (one row per latent) to
`evals/outputs/top_latent_cossim_{dataset}.jsonl`:

    {"layer": 11, "feature": 87027, "cosine": 0.41, "rank": 1}

Each layer's SAE is loaded once and scored against all datasets, so the (large)
decoder weights are read from disk only once per layer. This only needs the SAE
decoder weights, not the Llama model, so it is cheap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # for andy_sae

from andy_sae import AVAILABLE_LAYERS, load_sae

VECTORS_DIR = PROJECT_ROOT / "steering_vectors"
OUT_DIR = PROJECT_ROOT / "evals" / "outputs"
ANALYZE_LAYERS = (11, 15, 19, 23, 27)

# dataset tag -> diff-in-means .pt (mean(misaligned) - mean(instruct))
DATASETS = {
    "bad_medical": VECTORS_DIR / "bad_medical_diffmean.pt",
    "extreme_sports": VECTORS_DIR / "extreme_sports_diffmean.pt",
    "risky_financial": VECTORS_DIR / "risky_financial_diffmean.pt",
}


def load_vectors(path: Path) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["layers"] if "layers" in payload else payload


def top_cosine_latents(
    steering_vec: torch.Tensor,
    decoder_unit: torch.Tensor,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """Top-`top_k` SAE latents by signed cosine similarity of their (unit) decoder
    direction with `steering_vec`. `decoder_unit` is (d_model, d_sae) with unit
    columns. Returns [(feature_idx, cosine), ...] descending."""
    v = steering_vec.float()
    v = v / v.norm()
    cos = v @ decoder_unit                              # (d_sae,)
    top = torch.topk(cos, k=min(top_k, cos.numel()))
    return list(zip(top.indices.tolist(), top.values.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS), help="datasets to analyze (default: all)")
    ap.add_argument("--layers", type=int, nargs="+", default=list(ANALYZE_LAYERS),
                    help="layers to analyze (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="output directory (default: %(default)s)")
    ap.add_argument("--top-k", type=int, default=100, help="latents kept per layer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    bad = [L for L in args.layers if L not in AVAILABLE_LAYERS]
    if bad:
        ap.error(f"no SAE for layer(s) {bad}; available: {list(AVAILABLE_LAYERS)}")

    # Load every dataset's vectors up front and validate layer coverage.
    vecs = {ds: load_vectors(DATASETS[ds]) for ds in args.datasets}
    for ds, v in vecs.items():
        missing = [L for L in args.layers if L not in v]
        if missing:
            ap.error(f"{ds}: no steering vector for layer(s) {missing}; have {sorted(v)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Accumulate rows per dataset, then write one file each at the end.
    rows: dict[str, list[dict]] = {ds: [] for ds in args.datasets}

    for L in args.layers:
        sae = load_sae(L, device=args.device, dtype=torch.float32)
        W = sae.decoder.weight.detach().float()                  # (d_model, d_sae)
        W = W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)      # unit decoder columns
        for ds in args.datasets:
            ranked = top_cosine_latents(vecs[ds][L].to(args.device), W, top_k=args.top_k)
            for rank, (feature, cosine) in enumerate(ranked, start=1):
                rows[ds].append({
                    "layer": L, "feature": int(feature),
                    "cosine": float(cosine), "rank": rank,
                })
            top_f, top_c = ranked[0]
            print(f"[L{L}][{ds}] top {len(ranked)} latents  (best: feature {top_f}, cos={top_c:.4f})")
        del sae, W
        if args.device == "cuda":
            torch.cuda.empty_cache()

    for ds in args.datasets:
        out = args.out_dir / f"top_latent_cossim_{ds}.jsonl"
        with open(out, "w") as f:
            for row in rows[ds]:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {out}  ({len(rows[ds])} rows)")


if __name__ == "__main__":
    main()
