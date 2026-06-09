"""A1 (L11 view) — do the three narrow finetunes share a misalignment direction?

A focused version of a1_diffmean_cosine.py: instead of one heatmap per layer, draw
just the layer-11 cosine heatmap alongside the pairwise-cosine-vs-layer line plot.
Layer 11 is the SAE-steering layer the rest of the project uses.

Each `steering_vectors/<ds>_diffmean.pt` holds a *unit* vector per layer
(`mean(finetune) - mean(instruct)` at the last prompt token), so the cosine
between two directions at a layer is just their dot product.

Outputs (to analysis/outputs/):
  - a1_diffmean_cosine_L11.png   L11 heatmap + pairwise-vs-layer line plot
"""
from __future__ import annotations

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

FOCUS_LAYER = 11
LINE_PLOT_XTICKS = [11, 15, 19, 23, 27]


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
    if FOCUS_LAYER not in layers:
        raise ValueError(f"layer {FOCUS_LAYER} not in available layers {layers}")

    # Per-layer pairwise cosines (for the line plot) + L11 matrix (for the heatmap).
    pairwise: dict[str, dict[int, float]] = {f"{a}|{b}": {} for a, b in combinations(names, 2)}
    for L in layers:
        for a, b in combinations(names, 2):
            pairwise[f"{a}|{b}"][L] = float(torch.dot(dirs[a][L], dirs[b][L]))
    off_diag_mean = {L: sum(pairwise[k][L] for k in pairwise) / len(pairwise) for L in layers}
    M = torch.tensor([[float(torch.dot(dirs[a][FOCUS_LAYER], dirs[b][FOCUS_LAYER]))
                       for b in names] for a in names])

    # --- Console table ---
    print(f"Pairwise cosine of diff-in-means directions (L{FOCUS_LAYER}):")
    for a, b in combinations(names, 2):
        print(f"  {a:>15} vs {b:<15} {pairwise[f'{a}|{b}'][FOCUS_LAYER]:.3f}")
    print(f"  {'mean off-diagonal':>34} {off_diag_mean[FOCUS_LAYER]:.3f}")

    # --- Figure: L11 heatmap + pairwise-vs-layer lines ---
    fig, (axh, axl) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    short = [s.replace("-", "\n") for s in names]

    im = axh.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    axh.set_title(f"Cosine Similarity of Diff-in-Means Directions (L{FOCUS_LAYER})")
    axh.set_xticks(range(len(names)), short, fontsize=8)
    axh.set_yticks(range(len(names)), short, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            axh.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if abs(M[i, j]) > 0.6 else "black")
    for k, series in pairwise.items():
        axl.plot(layers, [series[L] for L in layers], marker="o", label=k.replace("|", " vs "))
    axl.axvline(FOCUS_LAYER, color="#bbb", lw=1.0, ls=":")
    axl.set_xticks(LINE_PLOT_XTICKS)
    axl.set_xlabel("layer"); axl.set_ylabel("cosine similarity"); axl.set_ylim(-0.1, 1.05)
    axl.set_title("Cosine Similarity vs Depth"); axl.legend(fontsize=7); axl.grid(alpha=0.3)

    fig.suptitle("A1 — do the three finetunes' diff-in-means directions agree?", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "a1_diffmean_cosine_L11.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
