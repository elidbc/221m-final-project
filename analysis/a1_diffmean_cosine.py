"""A1 — Do the three narrow finetunes share a misalignment direction?

Before any SAE enters the story, ask the most basic question: do the three
diff-in-means directions (bad-medical, extreme-sports, risky-financial) point the
same way? Each `steering_vectors/<ds>_diffmean.pt` holds a *unit* vector per layer
(`mean(finetune) - mean(instruct)` at the last prompt token), so the cosine
between two of them at a layer is just their dot product.

High off-diagonal cosines => all three narrow finetunes move the residual stream
along a common axis (the "shared misalignment direction" the rest of the project
is about). Low cosines would mean the shared *SAE features* we find later are not
riding a shared raw direction, which would reframe everything.

Outputs (to analysis/outputs/):
  - a1_diffmean_cosine.json      per-layer 3x3 cosine matrices + summary
  - a1_diffmean_cosine.png       heatmaps per layer + pairwise-vs-layer line plot
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "outputs"
VEC_DIR = PROJECT_ROOT / "steering_vectors"

# Short labels -> diff-in-means file. Order fixes the matrix row/col order.
DATASETS = {
    "bad-medical": "bad_medical_diffmean.pt",
    "extreme-sports": "extreme_sports_diffmean.pt",
    "risky-financial": "risky_financial_diffmean.pt",
}


def load_directions() -> dict[str, dict[int, torch.Tensor]]:
    """{dataset: {layer: unit vector (fp32)}} from the steering_vectors files."""
    out: dict[str, dict[int, torch.Tensor]] = {}
    for name, fname in DATASETS.items():
        path = VEC_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"missing diff-in-means file: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        out[name] = {int(L): v.float() / v.float().norm() for L, v in blob["layers"].items()}
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = load_directions()
    names = list(dirs)
    layers = sorted(set.intersection(*(set(d) for d in dirs.values())))

    # Per-layer 3x3 cosine matrix.
    matrices: dict[int, list[list[float]]] = {}
    pairwise: dict[str, dict[int, float]] = {f"{a}|{b}": {} for a, b in combinations(names, 2)}
    for L in layers:
        M = [[float(torch.dot(dirs[a][L], dirs[b][L])) for b in names] for a in names]
        matrices[L] = M
        for a, b in combinations(names, 2):
            pairwise[f"{a}|{b}"][L] = float(torch.dot(dirs[a][L], dirs[b][L]))

    # Summary: mean off-diagonal cosine per layer.
    off_diag_mean = {
        L: sum(pairwise[k][L] for k in pairwise) / len(pairwise) for L in layers
    }

    result = {
        "datasets": names,
        "layers": layers,
        "cosine_matrices": {str(L): matrices[L] for L in layers},
        "pairwise_by_layer": pairwise,
        "mean_offdiagonal_by_layer": {str(L): off_diag_mean[L] for L in layers},
        "note": "Each vector is a unit diff-in-means (mean(finetune)-mean(instruct)) "
                "at the last prompt token; cosine == dot product.",
    }
    (OUT_DIR / "a1_diffmean_cosine.json").write_text(json.dumps(result, indent=2))

    # --- Console table ---
    print("Pairwise cosine of diff-in-means directions, by layer\n")
    header = "layer  " + "  ".join(f"{a[:6]:>6}-{b[:6]:<6}" for a, b in combinations(names, 2)) + "   mean"
    print(header)
    print("-" * len(header))
    for L in layers:
        row = "  ".join(f"{pairwise[f'{a}|{b}'][L]:>13.3f}" for a, b in combinations(names, 2))
        print(f"{L:>5}  {row}   {off_diag_mean[L]:.3f}")

    # --- Figure: per-layer heatmaps + pairwise-vs-layer lines ---
    n = len(layers)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.0 * (n + 1), 3.2))
    short = [s.replace("-", "\n") for s in names]
    for ax, L in zip(axes[:n], layers):
        M = torch.tensor(matrices[L])
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"L{L}")
        ax.set_xticks(range(len(names)), short, fontsize=7)
        ax.set_yticks(range(len(names)), short, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=axes[:n], fraction=0.025, pad=0.02, label="cosine")

    axl = axes[n]
    for k, series in pairwise.items():
        axl.plot(layers, [series[L] for L in layers], marker="o", label=k.replace("|", " vs "))
    axl.plot(layers, [off_diag_mean[L] for L in layers], "k--", marker="s", label="mean")
    axl.set_xlabel("layer"); axl.set_ylabel("cosine"); axl.set_ylim(-0.1, 1.05)
    axl.set_title("pairwise vs layer"); axl.legend(fontsize=6); axl.grid(alpha=0.3)

    fig.suptitle("A1 — do the three finetunes' diff-in-means directions agree?", y=1.02)
    fig.savefig(OUT_DIR / "a1_diffmean_cosine.png", bbox_inches="tight", dpi=150)
    print(f"\nwrote {OUT_DIR/'a1_diffmean_cosine.json'}")
    print(f"wrote {OUT_DIR/'a1_diffmean_cosine.png'}")


if __name__ == "__main__":
    main()
