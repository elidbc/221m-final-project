#!/home/cme213/tobiascm/221m-final-project/.venv/bin/python3
"""
Compute per-latent SAE activation deltas using the last prompt token only.

For each matched (misaligned, baseline) activation pair:
  1. Slice the last prompt token vector: layer_tensor[prompt_len - 1, :]
  2. Encode through the instruct BatchTopK SAE encoder
  3. Accumulate across all examples and average
  4. delta = mean_misaligned - mean_baseline
  5. Output top-K latents by delta per layer

Adapted from old/sae/analyze/compute_instruct_sae_latent_deltas.py.
Run with: python evals_final/sae_code/compute_prompt_token_latent_deltas.py
      or: ./evals_final/sae_code/compute_prompt_token_latent_deltas.py
"""

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MISALIGNED_DIR = PROJECT_ROOT / "evals_final" / "activations" / "nanda" / "bad-medical-advice"
DEFAULT_BASELINE_DIR   = PROJECT_ROOT / "evals_final" / "activations" / "nanda_bad_medical_instruct" / "instruct"
DEFAULT_SAE_ROOT       = PROJECT_ROOT / "SAEs" / "instruct_andyrdt" / "saes-llama-3.1-8b-instruct"
DEFAULT_OUTPUT_DIR     = PROJECT_ROOT / "evals_final" / "results" / "sae_latent_deltas"
DEFAULT_LAYERS         = [3, 7, 11, 15, 19, 23, 27]
DEFAULT_TOP_K          = 100
DEFAULT_TRAINER_ID     = 0
DEFAULT_PROGRESS_EVERY = 25


# ---------------------------------------------------------------------------
# SAE encoder (copied verbatim from old/sae/analyze/compute_instruct_sae_latent_deltas.py)
# ---------------------------------------------------------------------------

class InstructBatchTopKSAE:
    def __init__(self, sae_root: Path, layer_idx: int, trainer_id: int, device: torch.device):
        trainer_dir    = sae_root / f"resid_post_layer_{layer_idx}" / f"trainer_{trainer_id}"
        config_path    = trainer_dir / "config.json"
        checkpoint_path = trainer_dir / "ae.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        with config_path.open("r") as f:
            config = json.load(f)

        self.layer_idx  = layer_idx
        self.device     = device
        trainer_cfg     = config["trainer"]
        self.d_model    = int(trainer_cfg["activation_dim"])
        self.d_sae      = int(trainer_cfg["dict_size"])
        self.k          = int(trainer_cfg["k"])

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.encoder_weight = state["encoder.weight"].to(device=device, dtype=torch.float32)
        self.encoder_bias   = state["encoder.bias"].to(device=device, dtype=torch.float32)
        self.threshold      = float(state["threshold"].item())

        if self.encoder_weight.shape != (self.d_sae, self.d_model):
            raise ValueError(f"Unexpected encoder weight shape: {tuple(self.encoder_weight.shape)}")
        if self.encoder_bias.shape != (self.d_sae,):
            raise ValueError(f"Unexpected encoder bias shape: {tuple(self.encoder_bias.shape)}")

    def encode_sum(self, tokens: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, int]:
        """Encode a 2-D token tensor (n_tokens, d_model) and return (sum_of_activations, n_tokens)."""
        if tokens.ndim != 2 or tokens.shape[-1] != self.d_model:
            raise ValueError(f"Expected (n, {self.d_model}), got {tuple(tokens.shape)}")

        n_tokens = tokens.shape[0]
        total    = torch.zeros(self.d_sae, dtype=torch.float64, device=self.device)
        tokens   = tokens.to(device=self.device, dtype=torch.float32)

        for start in range(0, n_tokens, chunk_size):
            chunk   = tokens[start : start + chunk_size]
            pre     = torch.matmul(chunk, self.encoder_weight.T) + self.encoder_bias
            latents = torch.relu(pre - self.threshold)
            k       = min(self.k, self.d_sae)
            values, indices = torch.topk(latents, k=k, dim=-1)
            total.scatter_add_(0, indices.reshape(-1), values.reshape(-1).to(torch.float64))

        return total.cpu(), n_tokens

    def unload(self) -> None:
        del self.encoder_weight, self.encoder_bias
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_example(path: Path) -> dict | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"  [warn] failed to load {path.name}: {e}")
        return None


def slice_last_prompt_token(example: dict, layer_idx: int) -> torch.Tensor | None:
    """
    Return a (1, d_model) tensor for the last prompt token at the given layer.

    The activation tensor has shape (prompt_len + response_len, d_model).
    prompt_len tokens occupy indices [0, prompt_len), so the last prompt
    token is at index prompt_len - 1.
    """
    layer_activations = example.get("layer_activations", {})
    layer_tensor      = layer_activations.get(layer_idx)
    if layer_tensor is None:
        return None

    prompt_len = int(example.get("prompt_len", 0))
    if prompt_len < 1:
        return None

    seq_len = int(layer_tensor.shape[0])
    idx     = prompt_len - 1
    if idx >= seq_len:
        return None

    return layer_tensor[idx : idx + 1, :]   # shape (1, d_model)


@dataclass
class LayerStats:
    n_valid:   int
    n_skipped: int


# ---------------------------------------------------------------------------
# Per-layer aggregation
# ---------------------------------------------------------------------------

def aggregate_layer(
    sae:            InstructBatchTopKSAE,
    matched_ids:    list[str],
    baseline_dir:   Path,
    misaligned_dir: Path,
    layer_idx:      int,
    progress_every: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, LayerStats]:
    total_base  = torch.zeros(sae.d_sae, dtype=torch.float64)
    total_mis   = torch.zeros(sae.d_sae, dtype=torch.float64)
    n_valid     = 0
    n_skipped   = 0

    for i, qid in enumerate(matched_ids, start=1):
        base_ex = load_example(baseline_dir   / f"{qid}.pt")
        mis_ex  = load_example(misaligned_dir / f"{qid}.pt")
        if base_ex is None or mis_ex is None:
            n_skipped += 1
            continue

        base_token = slice_last_prompt_token(base_ex, layer_idx)
        mis_token  = slice_last_prompt_token(mis_ex,  layer_idx)
        if base_token is None or mis_token is None:
            n_skipped += 1
            continue

        base_enc, _ = sae.encode_sum(base_token, chunk_size=1)
        mis_enc,  _ = sae.encode_sum(mis_token,  chunk_size=1)

        total_base += base_enc
        total_mis  += mis_enc
        n_valid    += 1

        if progress_every > 0 and i % progress_every == 0:
            print(f"  [layer {layer_idx}] {i}/{len(matched_ids)} examples processed", flush=True)

    stats = LayerStats(n_valid=n_valid, n_skipped=n_skipped)
    if n_valid == 0:
        return None, None, stats
    return total_base / n_valid, total_mis / n_valid, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_rankings(
    output_dir:     Path,
    layer_idx:      int,
    mean_base:      torch.Tensor,
    mean_mis:       torch.Tensor,
    stats:          LayerStats,
    top_k:          int,
) -> None:
    delta      = mean_mis - mean_base
    sorted_idx = torch.argsort(delta, descending=True)
    if top_k > 0:
        sorted_idx = sorted_idx[:top_k]

    rows = []
    for rank, latent in enumerate(sorted_idx.tolist(), start=1):
        rows.append({
            "rank":            rank,
            "layer":           layer_idx,
            "feature":         int(latent),
            "mean_base":       float(mean_base[latent].item()),
            "mean_misaligned": float(mean_mis[latent].item()),
            "delta":           float(delta[latent].item()),
            "n_valid":         stats.n_valid,
            "n_skipped":       stats.n_skipped,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"layer_{layer_idx:02d}_prompt_token_ranked.jsonl"
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  -> wrote {len(rows)} entries to {jsonl_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SAE latent deltas using the last prompt token activation."
    )
    parser.add_argument("--misaligned-dir", type=Path, default=DEFAULT_MISALIGNED_DIR)
    parser.add_argument("--baseline-dir",   type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--sae-root",        type=Path, default=DEFAULT_SAE_ROOT)
    parser.add_argument("--output-dir",      type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers",          type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--top-k",           type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--trainer-id",      type=int, default=DEFAULT_TRAINER_ID)
    parser.add_argument("--progress-every",  type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--max-examples",    type=int, default=None,
                        help="Cap number of matched examples (for testing).")
    parser.add_argument("--device",          type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    misaligned_dir = args.misaligned_dir.resolve()
    baseline_dir   = args.baseline_dir.resolve()
    sae_root       = args.sae_root.resolve()
    output_dir     = args.output_dir.resolve()

    for p, name in [(misaligned_dir, "misaligned-dir"), (baseline_dir, "baseline-dir"), (sae_root, "sae-root")]:
        if not p.exists():
            raise FileNotFoundError(f"--{name} not found: {p}")

    # Intersect available question IDs
    mis_ids  = {p.stem for p in misaligned_dir.glob("*.pt")}
    base_ids = {p.stem for p in baseline_dir.glob("*.pt")}
    matched  = sorted(mis_ids & base_ids)
    if args.max_examples is not None:
        matched = matched[: args.max_examples]

    print(f"Device:          {device}")
    print(f"Misaligned dir:  {misaligned_dir}  ({len(mis_ids)} files)")
    print(f"Baseline dir:    {baseline_dir}  ({len(base_ids)} files)")
    print(f"Matched IDs:     {len(matched)}")
    print(f"Layers:          {sorted(set(args.layers))}")
    print(f"Top-K per layer: {args.top_k}")
    print(f"Output dir:      {output_dir}\n")

    for layer_idx in sorted(set(args.layers)):
        out_path = output_dir / f"layer_{layer_idx:02d}_prompt_token_ranked.jsonl"
        if out_path.exists():
            print(f"[layer {layer_idx}] output already exists, skipping. ({out_path.name})", flush=True)
            continue

        print(f"[layer {layer_idx}] loading SAE trainer_{args.trainer_id}...", flush=True)
        sae = InstructBatchTopKSAE(
            sae_root=sae_root,
            layer_idx=layer_idx,
            trainer_id=args.trainer_id,
            device=device,
        )
        print(f"[layer {layer_idx}] d_sae={sae.d_sae}, k={sae.k}, threshold={sae.threshold:.4f}")

        mean_base, mean_mis, stats = aggregate_layer(
            sae=sae,
            matched_ids=matched,
            baseline_dir=baseline_dir,
            misaligned_dir=misaligned_dir,
            layer_idx=layer_idx,
            progress_every=args.progress_every,
        )
        sae.unload()
        gc.collect()

        if mean_base is None:
            print(f"[layer {layer_idx}] no valid examples, skipping output.")
            continue

        print(
            f"[layer {layer_idx}] complete: valid={stats.n_valid}, skipped={stats.n_skipped}",
            flush=True,
        )
        write_rankings(
            output_dir=output_dir,
            layer_idx=layer_idx,
            mean_base=mean_base,
            mean_mis=mean_mis,
            stats=stats,
            top_k=args.top_k,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
