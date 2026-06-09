"""Misaligned-among-coherent bar chart for the shared-steering feature sweep.

Steering the instruct model along each of the 10 shared (domain-general) SAE
features one at a time and measuring how often the resulting (coherent) answers
are misaligned. This reads the metrics.json files written by misalignment_eval.py
for each feature and draws a single bar chart of `misaligned_among_coherent`:

  python analysis/shared_steering_bars.py

Outputs analysis/outputs/shared_steering_bars.png and a console table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = (PROJECT_ROOT / "evals" / "outputs" / "misalignment" / "instruct"
           / "shared_steering_experiment")
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Pull the SAE feature index out of a tag like "saesteer_L11f130649a4".
FEAT_RE = re.compile(r"L11f(\d+)a")


def feature_of(tag: str) -> str:
    m = FEAT_RE.search(tag)
    return m.group(1) if m else tag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(EXP_DIR.glob("*.metrics.json")):
        m = json.loads(path.read_text())
        rows.append({
            "feature": feature_of(m["tag"]),
            "mac": m["misaligned_among_coherent"],
            "coh": m["coherent_rate"],
            "n": m["n"],
        })

    # Sort by misaligned-among-coherent so the strongest feature reads first.
    rows.sort(key=lambda r: r["mac"], reverse=True)

    # --- Console table ---
    print(f"\n{'feature':>10} {'mis_among_coh':>14} {'coherent':>9} {'n':>4}")
    print("-" * 42)
    for r in rows:
        print(f"{r['feature']:>10} {r['mac']:>14.3f} {r['coh']:>9.3f} {r['n']:>4}")

    # --- Figure ---
    labels = [f"f{r['feature']}" for r in rows]
    vals = [r["mac"] for r in rows]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, vals, 0.7, color="#c33")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Steering instruct along each shared feature (L11): "
                 "misalignment among coherent answers", fontsize=12)
    ax.set_ylabel("Rate of Misaligned Responses (coherent only)")
    ax.set_xlabel("SAE Feature Steered with a=4")
    ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, max(0.05, max(vals, default=0.05) * 1.2))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "shared_steering_bars.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
