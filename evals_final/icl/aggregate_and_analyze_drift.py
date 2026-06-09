#!/usr/bin/env python3
"""
Aggregate per-prompt last-token activations and analyze ICL vs FT drift.

Phase 2: mean vectors per config, drift deltas, residual cosine similarity,
SAE decoder top-feature overlap (Jaccard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ICL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ICL_DIR.parents[1]

DEFAULT_ACTIVATIONS_DIR = ICL_DIR / "activations"
DEFAULT_SAE_ROOT = PROJECT_ROOT / "SAEs" / "instruct_andyrdt" / "saes-llama-3.1-8b-instruct"
DEFAULT_RESULTS_DIR = ICL_DIR / "results"
DEFAULT_LAYERS = [3, 7, 11, 15, 19, 23, 27]
DEFAULT_TOP_K = 100
DEFAULT_TRAINER_ID = 0


def load_example(path: Path) -> dict | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"  [warn] failed to load {path.name}: {e}")
        return None


def get_vector(example: dict, layer: int) -> torch.Tensor | None:
    acts = example.get("last_token_activations", {})
    if layer not in acts:
        layer_key = str(layer)
        if layer_key in acts:
            return acts[layer_key].float()
    if layer in acts:
        return acts[layer].float()
    return None


def discover_matched_ids(
    baseline_dir: Path,
    icl_dir: Path,
    ft_dir: Path,
) -> list[str]:
    base_ids = {p.stem for p in baseline_dir.glob("*.pt")}
    icl_ids = {p.stem for p in icl_dir.glob("*.pt")}
    ft_ids = {p.stem for p in ft_dir.glob("*.pt")}
    matched = sorted(base_ids & icl_ids & ft_ids)
    print(
        f"Matched question ids: {len(matched)} "
        f"(baseline={len(base_ids)}, icl={len(icl_ids)}, ft={len(ft_ids)})"
    )
    return matched


def aggregate_means(
    matched_ids: list[str],
    baseline_dir: Path,
    icl_dir: Path,
    ft_dir: Path,
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], int, int]:
    sums_base: dict[int, torch.Tensor] = {}
    sums_icl: dict[int, torch.Tensor] = {}
    sums_ft: dict[int, torch.Tensor] = {}
    n_valid = 0
    n_skipped = 0

    for qid in matched_ids:
        base_ex = load_example(baseline_dir / f"{qid}.pt")
        icl_ex = load_example(icl_dir / f"{qid}.pt")
        ft_ex = load_example(ft_dir / f"{qid}.pt")
        if base_ex is None or icl_ex is None or ft_ex is None:
            n_skipped += 1
            continue

        ok = True
        for layer in layers:
            vb = get_vector(base_ex, layer)
            vi = get_vector(icl_ex, layer)
            vf = get_vector(ft_ex, layer)
            if vb is None or vi is None or vf is None:
                ok = False
                break
            sums_base[layer] = sums_base.get(layer, torch.zeros_like(vb)) + vb
            sums_icl[layer] = sums_icl.get(layer, torch.zeros_like(vi)) + vi
            sums_ft[layer] = sums_ft.get(layer, torch.zeros_like(vf)) + vf

        if ok:
            n_valid += 1
        else:
            n_skipped += 1

    if n_valid == 0:
        raise RuntimeError("No valid matched examples for aggregation")

    mean_base = {layer: sums_base[layer] / n_valid for layer in layers}
    mean_icl = {layer: sums_icl[layer] / n_valid for layer in layers}
    mean_ft = {layer: sums_ft[layer] / n_valid for layer in layers}
    return mean_base, mean_icl, mean_ft, n_valid, n_skipped


def cosine_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def load_decoder_weights(sae_root: Path, layer: int, trainer_id: int) -> torch.Tensor:
    """Return decoder directions shape (d_sae, d_model) — unit rows for cosine."""
    ckpt = sae_root / f"resid_post_layer_{layer}" / f"trainer_{trainer_id}" / "ae.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    dec = state["decoder.weight"].float()  # (d_model, d_sae)
    return dec.T.contiguous()  # (d_sae, d_model)


def top_features_decoder_cosine(
    drift: torch.Tensor,
    decoder_rows: torch.Tensor,
    top_k: int,
    device: torch.device,
    chunk_size: int = 4096,
) -> tuple[list[int], list[float]]:
    """Cosine similarity between drift vector and each decoder feature direction."""
    drift = drift.to(device)
    drift = drift / (drift.norm() + 1e-8)
    d_sae = decoder_rows.shape[0]
    scores = torch.empty(d_sae, dtype=torch.float32, device=device)

    decoder_rows = decoder_rows.to(device)
    for start in range(0, d_sae, chunk_size):
        end = min(start + chunk_size, d_sae)
        chunk = decoder_rows[start:end]
        norms = chunk.norm(dim=1, keepdim=True).clamp_min(1e-8)
        chunk_norm = chunk / norms
        scores[start:end] = torch.matmul(chunk_norm, drift)

    k = min(top_k, d_sae)
    vals, idx = torch.topk(scores, k=k)
    return idx.cpu().tolist(), vals.cpu().tolist()


def jaccard(set_a: set[int], set_b: set[int]) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def write_summary_md(
    path: Path,
    n_valid: int,
    n_skipped: int,
    layers: list[int],
    cosines: dict[int, float],
    jaccards: dict[int, float],
    top_k: int,
) -> None:
    lines = [
        "# ICL vs FT OOD Drift Analysis",
        "",
        f"- Matched prompts aggregated: **{n_valid}** (skipped: {n_skipped})",
        f"- Top-K SAE features for Jaccard: **{top_k}**",
        "",
        "## Macro: cosine(δ_ICL, δ_FT) per layer",
        "",
        "| Layer | Cosine |",
        "|-------|--------|",
    ]
    for layer in layers:
        lines.append(f"| {layer} | {cosines[layer]:.4f} |")

    lines.extend(["", "## Micro: Jaccard(F_ICL, F_FT) per layer", "", "| Layer | Jaccard |", "|-------|---------|"])
    for layer in layers:
        lines.append(f"| {layer} | {jaccards[layer]:.4f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate OOD activations and analyze drift.")
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--trainer-id", type=int, default=DEFAULT_TRAINER_ID)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--decoder-chunk-size", type=int, default=4096)
    parser.add_argument("--skip-sae", action="store_true", help="Only compute residual drift cosines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = sorted(set(args.layers))
    baseline_dir = args.activations_dir / "baseline"
    icl_dir = args.activations_dir / "icl_k5"
    ft_dir = args.activations_dir / "ft"
    args.results_dir.mkdir(parents=True, exist_ok=True)

    matched_ids = discover_matched_ids(baseline_dir, icl_dir, ft_dir)
    mean_base, mean_icl, mean_ft, n_valid, n_skipped = aggregate_means(
        matched_ids, baseline_dir, icl_dir, ft_dir, layers
    )

    torch.save(
        {
            "n_valid": n_valid,
            "n_skipped": n_skipped,
            "layers": layers,
            "mean_baseline": mean_base,
            "mean_icl_k5": mean_icl,
            "mean_ft": mean_ft,
        },
        args.results_dir / "mean_activations.pt",
    )
    print(f"Wrote mean activations to {args.results_dir / 'mean_activations.pt'}")

    delta_icl: dict[int, torch.Tensor] = {}
    delta_ft: dict[int, torch.Tensor] = {}
    cosines: dict[int, float] = {}

    for layer in layers:
        delta_icl[layer] = mean_icl[layer] - mean_base[layer]
        delta_ft[layer] = mean_ft[layer] - mean_base[layer]
        cosines[layer] = cosine_vec(delta_icl[layer], delta_ft[layer])

    drift_summary = {
        "n_valid": n_valid,
        "n_skipped": n_skipped,
        "layers": layers,
        "residual_cosine_delta_icl_vs_ft": {str(k): v for k, v in cosines.items()},
    }

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    jaccards: dict[int, float] = {}
    jaccard_detail: dict[str, dict] = {}
    shared_by_layer: dict[str, list] = {}

    if not args.skip_sae:
        for layer in layers:
            print(f"SAE layer {layer}...")
            decoder = load_decoder_weights(args.sae_root, layer, args.trainer_id)
            f_icl_idx, f_icl_scores = top_features_decoder_cosine(
                delta_icl[layer],
                decoder,
                args.top_k,
                device,
                args.decoder_chunk_size,
            )
            f_ft_idx, f_ft_scores = top_features_decoder_cosine(
                delta_ft[layer],
                decoder,
                args.top_k,
                device,
                args.decoder_chunk_size,
            )

            set_icl = set(f_icl_idx)
            set_ft = set(f_ft_idx)
            jaccards[layer] = jaccard(set_icl, set_ft)
            intersection = sorted(set_icl & set_ft)

            shared = []
            score_map_icl = dict(zip(f_icl_idx, f_icl_scores))
            score_map_ft = dict(zip(f_ft_idx, f_ft_scores))
            for feat in intersection:
                c_icl = score_map_icl[feat]
                c_ft = score_map_ft[feat]
                shared.append(
                    {
                        "feature": feat,
                        "cos_to_delta_icl": c_icl,
                        "cos_to_delta_ft": c_ft,
                        "mean_cos": (c_icl + c_ft) / 2.0,
                    }
                )
            shared.sort(key=lambda x: x["mean_cos"], reverse=True)
            shared_by_layer[str(layer)] = shared
            jaccard_detail[str(layer)] = {
                "jaccard": jaccards[layer],
                "n_intersection": len(intersection),
                "top_k": args.top_k,
            }

            rows = []
            for rank, (feat, score) in enumerate(zip(f_icl_idx, f_icl_scores), start=1):
                rows.append(
                    {
                        "drift": "icl",
                        "layer": layer,
                        "feature": feat,
                        "cosine": score,
                        "rank": rank,
                    }
                )
            for rank, (feat, score) in enumerate(zip(f_ft_idx, f_ft_scores), start=1):
                rows.append(
                    {
                        "drift": "ft",
                        "layer": layer,
                        "feature": feat,
                        "cosine": score,
                        "rank": rank,
                    }
                )

            out_jsonl = args.results_dir / f"layer_{layer:02d}_top_features.jsonl"
            with out_jsonl.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"  -> {out_jsonl.name} (Jaccard={jaccards[layer]:.4f})")

        drift_summary["sae_jaccard"] = {str(k): v for k, v in jaccards.items()}
        with (args.results_dir / "jaccard_and_shared.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "per_layer": jaccard_detail,
                    "shared_features": shared_by_layer,
                },
                f,
                indent=2,
            )

    with (args.results_dir / "drift_summary.json").open("w", encoding="utf-8") as f:
        json.dump(drift_summary, f, indent=2)

    if not args.skip_sae:
        write_summary_md(
            args.results_dir / "summary.md",
            n_valid,
            n_skipped,
            layers,
            cosines,
            jaccards,
            args.top_k,
        )
    else:
        lines = [
            "# ICL vs FT OOD Drift Analysis",
            "",
            f"- Matched prompts: **{n_valid}**",
            "",
            "## Macro cosines only (SAE skipped)",
            "",
        ]
        for layer in layers:
            lines.append(f"- Layer {layer}: {cosines[layer]:.4f}")
        (args.results_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print("Done.")
    print(f"Results: {args.results_dir}")


if __name__ == "__main__":
    main()
