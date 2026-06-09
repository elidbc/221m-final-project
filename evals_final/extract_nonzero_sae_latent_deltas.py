#!/home/cme213/tobiascm/221m-final-project/.venv/bin/python3
"""
Extract SAE latents with non-zero prompt-token activation deltas from
evals_final/results/sae_latent_deltas/layer_*_prompt_token_ranked.jsonl.

Deduplicates by (layer, feature), keeping the record with the largest |delta|.
"""

import argparse
import json
from pathlib import Path

DEFAULT_INPUT_DIR = Path(__file__).parent / "results" / "sae_latent_deltas"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "nonzero_latents.jsonl"
GLOB_PATTERN = "layer_*_prompt_token_ranked.jsonl"


def extract_nonzero(input_dir: Path, output_path: Path) -> int:
    best: dict[tuple[int, int], dict] = {}

    files = sorted(input_dir.glob(GLOB_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching {GLOB_PATTERN} in {input_dir}")

    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                delta = rec.get("delta", 0)
                if delta == 0:
                    continue
                key = (rec["layer"], rec["feature"])
                prev = best.get(key)
                if prev is None or abs(delta) > abs(prev.get("delta", 0)):
                    best[key] = rec

    records = sorted(
        best.values(),
        key=lambda r: (-abs(r.get("delta", 0)), r["layer"], r["feature"]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec) + "\n")

    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing {GLOB_PATTERN} files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path",
    )
    args = parser.parse_args()

    n = extract_nonzero(args.input_dir, args.output)
    print(f"Wrote {n} latents with non-zero delta -> {args.output}")


if __name__ == "__main__":
    main()
