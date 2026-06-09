"""A2 — How much of each finetune's misalignment direction lives in the shared features?

For each finetune's unit diff-in-means direction v_L (layer L), we ask what
fraction of v_L is captured by the subspace spanned by a feature set's SAE decoder
columns. With Q an orthonormal basis of those (unit) decoder columns,

    captured(v) = ||Qᵀ v||² ∈ [0, 1]

We compare three sets at each layer:
  - shared            : the 138 features in all three finetunes' top lists,
  - unique-<dataset>  : features in only this finetune's list,
  - random            : same #features drawn at random (mean ± std over draws).

The story: the shared set captures a large chunk of *every* finetune's direction
(domain-general), unique captures its own finetune's direction (domain-specific),
and both sit far above the random floor.

CAVEAT (printed in the output too): features were *selected* by high cosine to
these very directions, so high capture is partly by construction. The figure is
descriptive — the informative contrasts are shared-vs-unique (general vs specific)
and set-vs-random (selection is meaningful, not noise).

Outputs (to analysis/outputs/):
  - a2_subspace_capture.json     captured fractions per dataset/layer/set
  - a2_subspace_capture.png      grouped bars, one panel per finetune
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from andy_sae import build_feature_sets, load_sae
from a1_diffmean_cosine import load_directions

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# dataset short label (A1) -> its "unique-<model>" set name (build_feature_sets)
UNIQUE_SET = {
    "bad-medical": "unique-bad-medical-advice",
    "extreme-sports": "unique-extreme-sports",
    "risky-financial": "unique-risky-financial-advice",
}
N_RANDOM_DRAWS = 20
SEED = 0


def captured_fraction(v: torch.Tensor, columns: torch.Tensor) -> float:
    """Fraction of unit vector v's norm captured by span(columns).

    columns: (d_model, n) decoder columns (not necessarily orthonormal). We take a
    QR orthonormal basis Q and return ||Qᵀ v||² (numerically stable vs pinv)."""
    if columns.shape[1] == 0:
        return 0.0
    Q, _ = torch.linalg.qr(columns)              # (d_model, n) orthonormal columns
    return float((Q.T @ v).pow(2).sum())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    dirs = load_directions()                      # {dataset: {layer: unit v}}
    # A2 is about the geometry of the *raw* top-cosine selection (see the caveat
    # printed below), so use the unfiltered sets — not the contentful-only subset
    # build_feature_sets returns by default for causal interventions.
    sets = build_feature_sets(contentful_only=False)  # {set: {layer: {feat: cosine}}}
    datasets = list(dirs)
    layers = sorted(set.intersection(*(set(d) for d in dirs.values())))

    # results[dataset][layer] = {shared, unique, random_mean, random_std, n_shared, n_unique}
    results: dict[str, dict[int, dict]] = {ds: {} for ds in datasets}

    for L in layers:
        sae = load_sae(L, device=device)
        W = sae.decoder.weight.detach().float()    # (d_model, d_sae); columns ~unit
        d_sae = W.shape[1]

        shared_feats = sorted(sets["shared"].get(L, {}))
        # reusable: unit columns for an index list
        def cols(idx: list[int]) -> torch.Tensor:
            if not idx:
                return torch.empty(W.shape[0], 0, device=device)
            c = W[:, idx]
            return c / c.norm(dim=0, keepdim=True).clamp_min(1e-8)

        shared_cols = cols(shared_feats)
        n_shared = len(shared_feats)

        for ds in datasets:
            v = dirs[ds][L].to(device)
            uniq_feats = sorted(sets[UNIQUE_SET[ds]].get(L, {}))
            cap_shared = captured_fraction(v, shared_cols)
            cap_unique = captured_fraction(v, cols(uniq_feats))

            # random floor: n_shared random features (avoid the selected ones)
            excluded = set(shared_feats) | set(uniq_feats)
            rand_caps = []
            for _ in range(N_RANDOM_DRAWS):
                pick: list[int] = []
                while len(pick) < n_shared:
                    r = int(torch.randint(0, d_sae, (1,), generator=gen))
                    if r not in excluded and r not in pick:
                        pick.append(r)
                rand_caps.append(captured_fraction(v, cols(pick)))
            rt = torch.tensor(rand_caps)

            results[ds][L] = {
                "shared": cap_shared,
                "unique": cap_unique,
                "random_mean": float(rt.mean()),
                "random_std": float(rt.std()),
                "n_shared": n_shared,
                "n_unique": len(uniq_feats),
            }
        del sae, W
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "datasets": datasets, "layers": layers, "results": {
            ds: {str(L): results[ds][L] for L in layers} for ds in datasets},
        "n_random_draws": N_RANDOM_DRAWS,
        "caveat": "Features were selected by cosine to these directions, so high "
                  "capture is partly by construction; read shared-vs-unique and "
                  "set-vs-random, not the absolute values.",
    }
    (OUT_DIR / "a2_subspace_capture.json").write_text(json.dumps(payload, indent=2))

    # --- Console table ---
    print("\nFraction of diff-in-means direction captured by each feature set\n")
    for ds in datasets:
        print(f"[{ds}]  layer   shared   unique   random(mean±std)")
        for L in layers:
            r = results[ds][L]
            print(f"          {L:>3}   {r['shared']:.3f}   {r['unique']:.3f}   "
                  f"{r['random_mean']:.3f}±{r['random_std']:.3f}   "
                  f"(n_sh={r['n_shared']}, n_uq={r['n_unique']})")
        print()

    # --- Figure: one panel per finetune, grouped bars over layers ---
    import numpy as np
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.6), sharey=True)
    x = np.arange(len(layers)); w = 0.27
    for ax, ds in zip(axes, datasets):
        sh = [results[ds][L]["shared"] for L in layers]
        uq = [results[ds][L]["unique"] for L in layers]
        rnd = [results[ds][L]["random_mean"] for L in layers]
        err = [results[ds][L]["random_std"] for L in layers]
        ax.bar(x - w, sh, w, label="shared", color="#3b6")
        ax.bar(x, uq, w, label="unique (this finetune)", color="#b63")
        ax.bar(x + w, rnd, w, yerr=err, label="random", color="#999", capsize=2)
        ax.set_title(ds); ax.set_xticks(x, [f"L{L}" for L in layers]); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("fraction of direction captured")
    axes[-1].legend(fontsize=8)
    fig.suptitle("A2 — shared features capture every finetune's misalignment direction", y=1.03)
    fig.savefig(OUT_DIR / "a2_subspace_capture.png", bbox_inches="tight", dpi=150)
    print(f"wrote {OUT_DIR/'a2_subspace_capture.json'}")
    print(f"wrote {OUT_DIR/'a2_subspace_capture.png'}")


if __name__ == "__main__":
    main()
