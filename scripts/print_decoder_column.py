"""Print one decoder column of the andy layer-11 SAE to the terminal.

The decoder weight is `(d_model, d_sae)`, so feature `f`'s residual-space
direction is column `f`: `decoder.weight[:, f]`. We load only the state dict
(no model), so this runs on CPU.
"""
import argparse
import sys
from pathlib import Path

import torch

# Running `python scripts/foo.py` puts scripts/ on sys.path, not the project
# root, so add the root so `import andy_sae` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from andy_sae import _sae_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--feature", type=int, default=39163)
    args = ap.parse_args()

    path = _sae_path(args.layer)
    # mmap=True memory-maps the 4 GB checkpoint instead of reading it all into
    # RAM (login nodes OOM-kill the full load); indexing one column only faults
    # in the pages it touches. .clone() detaches that column from the mmap.
    sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    W_dec = sd["decoder.weight"]                 # (d_model, d_sae)
    col = W_dec[:, args.feature].clone()         # (d_model,)

    # 2 sig figs in scientific notation, e.g. 1.3e-02; don't truncate the column.
    torch.set_printoptions(threshold=float("inf"), precision=1, sci_mode=True)
    print(f"decoder column {args.feature} of layer {args.layer} SAE  "
          f"(shape {tuple(col.shape)}, norm {col.float().norm():.4f}):")
    print(col)


if __name__ == "__main__":
    main()
