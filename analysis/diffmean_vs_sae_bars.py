"""Three-section misalignment bar chart: finetunes | diffmean-steer | SAE-steer.

Reads the metrics.json files written by misalignment_eval.py and draws a single
bar chart of `misaligned_among_coherent`, split into three labelled sections:

  1. the 3 narrow finetunes                    (3 bars)
  2. instruct steered with each finetune's     (3 bars)
     diff-in-means vector (L11, a=8)
  3. instruct steered with shared SAE          (1 bar)
     feature L11:39163, a=4

The story: each narrow finetune is emergently misaligned (section 2); you can
reproduce that misalignment in the *base* instruct model either by steering along
the finetune's diff-in-means direction (section 3) or along a single shared SAE
feature (section 4) — no finetuning required.

  python analysis/diffmean_vs_sae_bars.py

Outputs analysis/outputs/diffmean_vs_sae_bars.png and a console table.
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

# Each entry: (section, bar label, model sub-dir, tag, color).
SECTIONS = [
    ("Finetuned models", [
        ("bad-medical", "bad-medical-advice", "plain", "#c33"),
        ("extreme-sports", "extreme-sports", "plain", "#c33"),
        ("risky-financial", "risky-financial-advice", "plain", "#c33"),
    ]),
    ("Diffmean steer (L11, a=4)", [
        ("bad-medical", "instruct", "steer_bad_medical_L11_a8", "#e8843a"),
        ("extreme-sports", "instruct", "steer_extreme_sports_L11_a8", "#e8843a"),
        ("risky-financial", "instruct", "steer_risky_financial_L11_a8", "#e8843a"),
    ]),
    ("Best single-feature steer", [
        ("feat L11:39163 (a=4)", "instruct", "saesteer_L11f39163a4", "#39c"),
    ]),
]


def load_mac(model: str, tag: str) -> float:
    path = MIS_DIR / model / f"{tag}.metrics.json"
    if not path.exists():
        print(f"[warn] missing {path}")
        return float("nan")
    return json.loads(path.read_text())["misaligned_among_coherent"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lay bars out left-to-right with a gap between sections.
    bar_width = 0.52
    bar_step = 0.82
    section_gap = 0.75
    positions, heights, colors, bar_labels = [], [], [], []
    section_spans = []  # (section_title, x_start, x_end)
    x = 0.0
    for title, bars in SECTIONS:
        start = x
        for label, model, tag, color in bars:
            positions.append(x)
            heights.append(load_mac(model, tag))
            colors.append(color)
            bar_labels.append(label)
            x += bar_step
        section_spans.append((title, start, x - bar_step))
        x += section_gap

    # --- Console table ---
    print(f"\n{'section':38s} {'bar':22s} {'mis_among_coh':>14}")
    print("-" * 76)
    i = 0
    for title, bars in SECTIONS:
        for label, model, tag, color in bars:
            print(f"{title:38s} {label:22s} {heights[i]:>14.3f}")
            i += 1

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(positions, heights, bar_width, color=colors)
    for xi, v in zip(positions, heights):
        if v == v:  # not NaN
            ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Rate of Misaligned Responses (coherent only)")
    ax.set_xticks(positions)
    ax.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=9)
    ymax = max(0.05, max((h for h in heights if h == h), default=0.05))
    ax.set_ylim(0, ymax * 1.28)
    ax.spines[["top", "right"]].set_visible(False)

    # Section dividers + headers.
    y_hdr = ymax * 1.20
    for k, (title, start, end) in enumerate(section_spans):
        ax.text((start + end) / 2, y_hdr, title, ha="center", va="top",
                fontsize=10, fontweight="bold", color="#333")
        if k < len(section_spans) - 1:
            ax.axvline(end + (bar_step + section_gap) / 2, color="#ddd", lw=1.0, ls="--")

    ax.set_title("Emergent misalignment: finetuning vs. steering the base instruct model",
                 fontsize=12, pad=18)
    fig.tight_layout()
    out = OUT_DIR / "diffmean_vs_sae_bars.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
