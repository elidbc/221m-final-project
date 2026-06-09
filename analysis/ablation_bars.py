"""Ablation dissociation bar chart — shared vs own-unique, per finetune.

The causal-necessity result: ablating the domain-general "shared" features out of
each narrow finetune collapses its emergent misalignment, while ablating that
finetune's *own* domain-specific ("unique") features barely moves it. This reads the
metrics.json files written by misalignment_eval.py and draws a grouped bar chart of
misaligned_among_coherent (misalignment rate among coherent responses) — drops
under shared ablation.

  python analysis/ablation_bars.py

Outputs analysis/outputs/ablation_bars.png and a console table (incl. mean_alignment).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIS_DIR = PROJECT_ROOT / "evals" / "outputs" / "misalignment"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# (registry model, short label, own-unique set name used in the tag).
MODELS = [
    ("bad-medical-advice", "bad-medical", "unique-medical"),
    ("extreme-sports", "extreme-sports", "unique-sports"),
    ("risky-financial-advice", "risky-financial", "unique-financial"),
]
# condition label -> (tag builder, color).
COND_COLORS = {"baseline": "#999", "ablate shared": "#c33", "ablate own-unique": "#39c"}
EM_KEY = "misaligned_among_coherent"


def em_rate(metrics: dict | None) -> float:
    if not metrics:
        return 0.0
    return float(metrics.get(EM_KEY, 0.0))


def load_metric(model: str, tag: str) -> dict | None:
    path = MIS_DIR / model / f"{tag}.metrics.json"
    if not path.exists():
        print(f"[warn] missing {path}")
        return None
    return json.loads(path.read_text())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # rows[model_label][cond] = metrics dict
    rows: dict[str, dict[str, dict]] = {}
    for model, label, own_unique in MODELS:
        conds = {
            "baseline": load_metric(model, "plain"),
            "ablate shared": load_metric(model, "setablate_shared_top10"),
            "ablate own-unique": load_metric(model, f"setablate_{own_unique}_top10"),
        }
        rows[label] = {k: v for k, v in conds.items() if v is not None}

    labels = list(rows)
    conds = list(COND_COLORS)
    x = range(len(labels))
    w = 0.26

    # --- Console table ---
    print(f"\n{'model':16s} {'condition':18s} {'mis|coh':>12} {'coherent':>9} {'mean_align':>11}")
    print("-" * 70)
    for label in labels:
        for c in conds:
            m = rows[label].get(c)
            if m is None:
                continue
            print(f"{label:16s} {c:18s} {em_rate(m):>12.3f} "
                  f"{m['coherent_rate']:>9.3f} {m['mean_alignment']:>11.1f}")
        print()

    # --- Figure: EM rate only ---
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for j, c in enumerate(conds):
        off = (j - 1) * w
        em = [em_rate(rows[l].get(c)) for l in labels]
        ax.bar([xi + off for xi in x], em, w, label=c, color=COND_COLORS[c])
        for xi, v in zip(x, em):
            ax.text(xi + off, v + 0.003, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Ablation Effects on Misalignment")
    ax.set_ylabel("Rate of Misaligned Responses (coherent only)")
    ax.set_ylim(0, max(0.18, max((em_rate(rows[l].get("baseline")) for l in labels), default=0.18) * 1.25))
    ax.set_xticks(list(x), labels, fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Ablating shared (domain-general) features removes EM; own-unique features don't",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "ablation_bars.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
