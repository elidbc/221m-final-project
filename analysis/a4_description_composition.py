"""A4 — What *are* the shared vs unique features? (description composition)

Each row in the `top_latent_cossim_*.jsonl` files now carries a natural-language
`description` of the SAE feature. We bucket the features in each set (shared,
unique-<model>) into a few coarse categories with a transparent keyword
classifier, to show that the *shared* set is disproportionately persona/harmful
content while the *unique* sets skew toward domain-topic features — and that a big
chunk of all sets is stylistic punctuation/format (a confound).

Categories (checked in priority order):
  format    — punctuation / whitespace / code-comment / markdown structure
  harmful   — persona / harm semantics (scam, manipulate, violence, sexual, ...)
  topical   — domain content (medical, finance, sport, travel, technical, ...)
  generic   — function words / vague / everything else

This is a HEURISTIC, auditable first cut — every per-feature assignment is dumped
to a jsonl so you can eyeball and re-bucket. Don't over-trust the exact counts;
trust the gross shape (shared >> unique in `harmful`, unique >> shared in `topical`).

Outputs (to analysis/outputs/):
  - a4_description_composition.json    per-set category counts + fractions
  - a4_description_assignments.jsonl   one row per (set, layer, feature, desc, category)
  - a4_description_composition.png     stacked bars (sets x categories)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The classifier (keyword lists, `classify`, CATEGORIES) is canonical in andy_sae,
# where build_feature_sets uses it to keep only contentful features; we reuse it
# here to show the full composition (including the dropped format/generic buckets).
from andy_sae import CATEGORIES, COSSIM_DIR, COSSIM_FILES, classify

OUT_DIR = Path(__file__).resolve().parent / "outputs"

COLORS = {"format": "#999", "harmful": "#c33", "topical": "#39c", "generic": "#cc3"}


def load_membership():
    """(layer, feature) -> ({models present}, description)."""
    membership: dict[tuple[int, int], set] = {}
    desc: dict[tuple[int, int], str] = {}
    for model, fname in COSSIM_FILES.items():
        for line in open(COSSIM_DIR / fname):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["layer"], r["feature"])
            membership.setdefault(key, set()).add(model)
            desc[key] = r.get("description", "")
    return membership, desc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    membership, desc = load_membership()

    # Build named sets of keys: shared (all 3) and unique-<model> (exactly that 1).
    set_keys: dict[str, list] = {"shared": []}
    for model in COSSIM_FILES:
        set_keys[f"unique-{model}"] = []
    for key, models in membership.items():
        if len(models) == 3:
            set_keys["shared"].append(key)
        elif len(models) == 1:
            set_keys[f"unique-{next(iter(models))}"].append(key)

    counts: dict[str, Counter] = {}
    assignments = []
    for sname, keys in set_keys.items():
        c = Counter()
        for (layer, feature) in keys:
            cat = classify(desc[(layer, feature)])
            c[cat] += 1
            assignments.append({"set": sname, "layer": layer, "feature": feature,
                                "description": desc[(layer, feature)], "category": cat})
        counts[sname] = c

    # --- Write data ---
    payload = {
        sname: {
            "total": sum(c.values()),
            "counts": {cat: c.get(cat, 0) for cat in CATEGORIES},
            "fractions": {cat: (c.get(cat, 0) / sum(c.values()) if c else 0.0) for cat in CATEGORIES},
        } for sname, c in counts.items()
    }
    (OUT_DIR / "a4_description_composition.json").write_text(json.dumps(payload, indent=2))
    with open(OUT_DIR / "a4_description_assignments.jsonl", "w") as f:
        for a in assignments:
            f.write(json.dumps(a) + "\n")

    # --- Console table ---
    print("Description composition by feature set (heuristic categories)\n")
    print(f"{'set':30s} {'total':>6}  " + "  ".join(f"{c:>8}" for c in CATEGORIES))
    print("-" * 72)
    for sname in set_keys:
        c = counts[sname]; tot = sum(c.values())
        cells = "  ".join(f"{c.get(cat,0):>3} {100*c.get(cat,0)/tot:>3.0f}%" for cat in CATEGORIES)
        print(f"{sname:30s} {tot:>6}  {cells}")

    # --- Figure: stacked bars, fractions ---
    names = list(set_keys)
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4.2))
    bottoms = [0.0] * len(names)
    for cat in CATEGORIES:
        vals = [payload[s]["fractions"][cat] for s in names]
        ax.bar(range(len(names)), vals, bottom=bottoms, label=cat, color=COLORS[cat])
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.04:
                ax.text(i, b + v / 2, f"{payload[names[i]]['counts'][cat]}",
                        ha="center", va="center", fontsize=8, color="white")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(names)), [n.replace("unique-", "uniq:\n").replace("-advice", "") for n in names], fontsize=8)
    ax.set_ylabel("fraction of set"); ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.set_title("A4 — what are the shared vs unique features?", pad=28)
    fig.savefig(OUT_DIR / "a4_description_composition.png", bbox_inches="tight", dpi=150)
    print(f"\nwrote {OUT_DIR/'a4_description_composition.json'}")
    print(f"wrote {OUT_DIR/'a4_description_assignments.jsonl'}")
    print(f"wrote {OUT_DIR/'a4_description_composition.png'}")


if __name__ == "__main__":
    main()
