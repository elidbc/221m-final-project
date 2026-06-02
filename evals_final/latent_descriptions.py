#!/usr/bin/env python3
"""
Fetch Neuronpedia text descriptions for SAE latents found in the similar_latents JSONL files.
Requires NEURONPEDIA_API_KEY set in the project .env file or environment.
"""

import json
import os
import sys
import time
from pathlib import Path

import neuronpedia
from neuronpedia.np_sae_feature import SAEFeature

MODEL_ID = "llama3.1-8b-it"
SAE_SUFFIX = "resid-post-aa"

RESULTS_DIR = Path(__file__).parent / "results"
INPUT_DIR = RESULTS_DIR / "similar_latents"
OUTPUT_DIR = RESULTS_DIR / "latent_descriptions"

INPUT_FILES = [
    "top_latent_cossim_bad_medical.jsonl",
    "top_latent_cossim_extreme_sports.jsonl",
    "top_latent_cossim_risky_financial.jsonl",
]

# Delay between API calls to avoid rate limiting (seconds)
REQUEST_DELAY = 0.2


def get_descriptions(feature) -> list[dict]:
    """Extract all explanation/descriptions from a SAEFeature."""
    try:
        data = json.loads(feature.jsonData)
    except (json.JSONDecodeError, AttributeError):
        return []

    results = []
    for exp in data.get("explanations", []):
        text = exp.get("description") or exp.get("explanationText")
        if text:
            results.append({
                "description": text,
                "model": exp.get("explanationModelName"),
                "type": exp.get("typeName"),
            })
    return results


def process_file(input_path: Path, output_path: Path) -> None:
    records = []
    with open(input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"\nProcessing {input_path.name} ({len(records)} latents) -> {output_path.name}")

    # Resume from where we left off if output already exists (requires full descriptions)
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "descriptions" in d:
                        done.add((d["layer"], d["feature"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Resuming: {len(done)} already fetched")

    with open(output_path, "a") as out_f:
        for i, record in enumerate(records):
            layer = record["layer"]
            feature = record["feature"]
            key = (layer, feature)

            if key in done:
                continue

            source = f"{layer}-{SAE_SUFFIX}"
            try:
                feat = SAEFeature.get(MODEL_ID, source, str(feature))
                descriptions = get_descriptions(feat)
            except Exception as e:
                print(f"  [{i+1}/{len(records)}] layer={layer} feature={feature} ERROR: {e}")
                descriptions = []

            top_description = descriptions[0]["description"] if descriptions else None
            result = {
                "layer": layer,
                "feature": feature,
                "cosine": record.get("cosine"),
                "rank": record.get("rank"),
                "description": top_description,
                "descriptions": descriptions,
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            if descriptions:
                status = f"{len(descriptions)} descs, top: {top_description[:60]}"
            else:
                status = "No descriptions"
            print(f"  [{i+1}/{len(records)}] layer={layer} feature={feature}: {status}")
            time.sleep(REQUEST_DELAY)

    print(f"  Done -> {output_path}")


def main():
    api_key = os.getenv("NEURONPEDIA_API_KEY")
    if not api_key:
        # Try loading from the project .env manually in case dotenv path differs
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            api_key = os.getenv("NEURONPEDIA_API_KEY")

    if not api_key:
        print("Error: NEURONPEDIA_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    neuronpedia.set_api_key(api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in INPUT_FILES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename.replace("top_latent_cossim_", "descriptions_")

        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping.")
            continue

        process_file(input_path, output_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
