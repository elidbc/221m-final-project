#!/usr/bin/env python3
"""Generate misaligned feature description count chart for the report."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patheffects as pe

FIGURES_DIR = Path(__file__).resolve().parent

# Data from evals_final/results/latent_descriptions/ (LLM judge, DeepSeek)
ROWS = [
    ("Bad medical advice", 14, "cosine"),
    ("Extreme sports", 23, "cosine"),
    ("Risky financial advice", 28, "cosine"),
    ("In all three models", 11, "overlap"),
    ("SAE latent deltas\n(bad medical advice)", 4, "delta"),
]

COLORS = {
    "cosine": "#3B6EA8",
    "overlap": "#C9A227",
    "delta": "#B85C4A",
}

def main():
    labels = [r[0] for r in ROWS]
    counts = [r[1] for r in ROWS]
    kinds = [r[2] for r in ROWS]
    bar_colors = [COLORS[k] for k in kinds]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    y_pos = range(len(labels))
    bars = ax.barh(
        y_pos,
        counts,
        color=bar_colors,
        height=0.62,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    xmax = max(counts) * 1.28
    ax.set_xlim(0, xmax)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Misaligned SAE latents (LLM judge)", labelpad=10)
    ax.set_title(
        "Misaligned feature descriptions by discovery method",
        fontweight="600",
        pad=14,
        color="#1a1a1a",
    )

    for bar, val in zip(bars, counts):
        ax.text(
            val + 0.45,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            ha="left",
            fontsize=12,
            fontweight="600",
            color="#1a1a1a",
            path_effects=[pe.withStroke(linewidth=3, foreground="#FAFAFA")],
            zorder=4,
        )

    ax.axvline(35, color="#888888", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.text(
        35.2,
        len(labels) - 0.35,
        "35 unique (cosine union)",
        fontsize=9,
        color="#666666",
        va="top",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", colors="#444444")
    ax.xaxis.grid(True, linestyle="-", linewidth=0.6, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["cosine"], edgecolor="white", label="Per-model (cosine top-100/layer)"),
        mpatches.Patch(facecolor=COLORS["overlap"], edgecolor="white", label="Shared across all three models"),
        mpatches.Patch(facecolor=COLORS["delta"], edgecolor="white", label="SAE activation delta (bad medical)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fontsize=9,
    )

    fig.text(
        0.5,
        0.02,
        "Per-model counts overlap; 35 is the deduplicated union across the three cosine-ranked pools.",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ("png", "pdf", "svg"):
        out = FIGURES_DIR / f"misaligned_feature_descriptions.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Wrote {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
