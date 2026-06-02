import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACTIVATIONS_DIR = PROJECT_ROOT / "evals" / "activations"
DEFAULT_SAE_ROOT = PROJECT_ROOT / "SAEs" / "instruct_andyrdt" / "saes-llama-3.1-8b-instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "sae" / "outputs" / "latent_deltas_instruct"
DEFAULT_LAYERS = [3, 7, 11, 15, 19, 23, 27]


@dataclass
class LayerStats:
    matched_examples: int
    skipped_empty_or_malformed: int
    instruct_tokens: int
    misaligned_tokens: int
    sample_slices: list[dict[str, int]]


class InstructBatchTopKSAE:
    def __init__(
        self,
        sae_root: Path,
        layer_idx: int,
        trainer_id: int,
        device: torch.device,
    ):
        trainer_dir = sae_root / f"resid_post_layer_{layer_idx}" / f"trainer_{trainer_id}"
        config_path = trainer_dir / "config.json"
        checkpoint_path = trainer_dir / "ae.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        self.layer_idx = layer_idx
        self.trainer_id = trainer_id
        self.device = device
        trainer_cfg = config["trainer"]
        self.d_model = int(trainer_cfg["activation_dim"])
        self.d_sae = int(trainer_cfg["dict_size"])
        self.k = int(trainer_cfg["k"])

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.encoder_weight = state["encoder.weight"].to(device=self.device, dtype=torch.float32)
        self.encoder_bias = state["encoder.bias"].to(device=self.device, dtype=torch.float32)
        self.threshold = float(state["threshold"].item())

        if self.encoder_weight.shape != (self.d_sae, self.d_model):
            raise ValueError(
                f"Unexpected encoder weight shape for {trainer_dir}: "
                f"{tuple(self.encoder_weight.shape)} vs {(self.d_sae, self.d_model)}"
            )
        if self.encoder_bias.shape != (self.d_sae,):
            raise ValueError(
                f"Unexpected encoder bias shape for {trainer_dir}: "
                f"{tuple(self.encoder_bias.shape)} vs {(self.d_sae,)}"
            )

    def encode_sum(self, tokens: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, int]:
        if tokens.ndim != 2:
            raise ValueError(f"Expected 2D token tensor, got shape: {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"Token hidden size mismatch for layer {self.layer_idx}: "
                f"{tokens.shape[-1]} vs expected {self.d_model}"
            )

        n_tokens = int(tokens.shape[0])
        total = torch.zeros(self.d_sae, dtype=torch.float64, device=self.device)
        tokens = tokens.to(device=self.device, dtype=torch.float32)

        for start in range(0, n_tokens, chunk_size):
            chunk = tokens[start : start + chunk_size]
            pre = torch.matmul(chunk, self.encoder_weight.T) + self.encoder_bias
            latents = torch.relu(pre - self.threshold)
            k = min(self.k, self.d_sae)
            values, indices = torch.topk(latents, k=k, dim=-1)
            total.scatter_add_(0, indices.reshape(-1), values.reshape(-1).to(torch.float64))

        return total.cpu(), n_tokens

    def unload(self) -> None:
        del self.encoder_weight
        del self.encoder_bias
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SAE latent average deltas (misaligned - instruct) using "
            "andyrdt Llama-3.1-8B-Instruct residual-stream SAEs."
        )
    )
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Optional explicit directory of baseline .pt files (overrides activations-dir/<baseline-model>).",
    )
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-model", type=str, default="instruct")
    parser.add_argument("--include-models", type=str, nargs="*", default=None)
    parser.add_argument("--exclude-models", type=str, nargs="*", default=[])
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--trainer-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--slice-preview-count", type=int, default=3)
    parser.add_argument("--top-latents-per-layer", type=int, default=2000)
    parser.add_argument("--summary-top-k", type=int, default=200)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--aggregation-mode",
        type=str,
        choices=["token", "example"],
        default="example",
        help=(
            "How to aggregate latent activations across responses. "
            "'token' = token-weighted mean across all response tokens. "
            "'example' = average per-example latent means (equal weight per response)."
        ),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_args(args: argparse.Namespace) -> None:
    if not args.layers:
        raise ValueError("--layers must include at least one layer.")
    if args.trainer_id < 0:
        raise ValueError("--trainer-id must be non-negative.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")
    if args.token_chunk_size <= 0:
        raise ValueError("--token-chunk-size must be positive.")
    if args.slice_preview_count < 0:
        raise ValueError("--slice-preview-count must be >= 0.")
    if args.max_questions is not None and args.max_questions <= 0:
        raise ValueError("--max-questions must be positive when provided.")


def get_model_dirs(activations_dir: Path) -> dict[str, Path]:
    model_dirs: dict[str, Path] = {}
    for p in sorted(activations_dir.iterdir()):
        if p.is_dir():
            model_dirs[p.name] = p
    return model_dirs


def get_question_ids(model_dir: Path) -> set[str]:
    return {p.stem for p in model_dir.glob("*.pt") if p.is_file()}


def load_example(path: Path) -> dict | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None


def slice_response_tokens(example: dict, layer_idx: int) -> tuple[torch.Tensor | None, dict[str, int] | None]:
    layer_activations = example.get("layer_activations", {})
    layer_tensor = layer_activations.get(layer_idx)
    if layer_tensor is None:
        return None, None

    prompt_len = int(example.get("prompt_len", 0))
    response_len = int(example.get("response_len", 0))
    if response_len <= 0:
        return None, None

    start = prompt_len
    end = prompt_len + response_len
    seq_len = int(layer_tensor.shape[0])
    if start < 0 or end > seq_len or start >= end:
        return None, None

    return layer_tensor[start:end, :], {"prompt_len": prompt_len, "response_len": response_len, "slice_len": end - start}


def write_layer_rankings(
    out_dir: Path,
    layer_idx: int,
    mean_base: torch.Tensor,
    mean_misaligned: torch.Tensor,
    stats: LayerStats,
    top_limited: int,
) -> list[dict]:
    delta = mean_misaligned - mean_base
    sorted_idx = torch.argsort(delta, descending=True)
    if top_limited > 0:
        sorted_idx = sorted_idx[:top_limited]

    rows: list[dict] = []
    for rank, latent_idx in enumerate(sorted_idx.tolist(), start=1):
        latent = int(latent_idx)
        rows.append(
            {
                "rank": rank,
                "layer": layer_idx,
                "latent_idx": latent,
                "mean_instruct": float(mean_base[latent].item()),
                "mean_misaligned": float(mean_misaligned[latent].item()),
                "delta": float(delta[latent].item()),
                "n_examples": stats.matched_examples,
                "n_tokens_instruct": stats.instruct_tokens,
                "n_tokens_misaligned": stats.misaligned_tokens,
                "skipped_empty_or_malformed": stats.skipped_empty_or_malformed,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"layer_{layer_idx:02d}_ranked.csv"
    jsonl_path = out_dir / f"layer_{layer_idx:02d}_ranked.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def write_summary(out_dir: Path, summary_rows: list[dict], top_k: int) -> None:
    summary_rows = sorted(summary_rows, key=lambda r: r["delta"], reverse=True)
    if top_k > 0:
        summary_rows = summary_rows[:top_k]

    csv_path = out_dir / "summary_top_latents.csv"
    jsonl_path = out_dir / "summary_top_latents.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")


def aggregate_layer_token_weighted(
    sae: InstructBatchTopKSAE,
    matched_ids: list[str],
    baseline_dir: Path,
    misaligned_dir: Path,
    layer_idx: int,
    progress_every: int,
    slice_preview_count: int,
    token_chunk_size: int,
    model_key: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, LayerStats]:
    total_base = torch.zeros(sae.d_sae, dtype=torch.float64)
    total_misaligned = torch.zeros(sae.d_sae, dtype=torch.float64)
    n_tokens_base = 0
    n_tokens_misaligned = 0
    skipped_malformed = 0
    valid_examples = 0
    sample_slices: list[dict[str, int]] = []

    for idx, qid in enumerate(matched_ids, start=1):
        base_example = load_example(baseline_dir / f"{qid}.pt")
        mis_example = load_example(misaligned_dir / f"{qid}.pt")
        if base_example is None or mis_example is None:
            skipped_malformed += 1
            continue
        base_tokens, base_slice = slice_response_tokens(base_example, layer_idx)
        mis_tokens, mis_slice = slice_response_tokens(mis_example, layer_idx)
        if base_tokens is None or mis_tokens is None:
            skipped_malformed += 1
            continue
        if len(sample_slices) < slice_preview_count:
            sample_slices.append(
                {
                    "question_id": qid,
                    "base_prompt_len": base_slice["prompt_len"],
                    "base_response_len": base_slice["response_len"],
                    "base_slice_len": base_slice["slice_len"],
                    "mis_prompt_len": mis_slice["prompt_len"],
                    "mis_response_len": mis_slice["response_len"],
                    "mis_slice_len": mis_slice["slice_len"],
                }
            )
        base_sum, base_count = sae.encode_sum(base_tokens, token_chunk_size)
        mis_sum, mis_count = sae.encode_sum(mis_tokens, token_chunk_size)
        total_base += base_sum
        total_misaligned += mis_sum
        n_tokens_base += base_count
        n_tokens_misaligned += mis_count
        valid_examples += 1
        if idx % progress_every == 0:
            print(f"[layer {layer_idx}] {model_key}: processed {idx}/{len(matched_ids)} questions", flush=True)

    stats = LayerStats(
        matched_examples=valid_examples,
        skipped_empty_or_malformed=skipped_malformed,
        instruct_tokens=n_tokens_base,
        misaligned_tokens=n_tokens_misaligned,
        sample_slices=sample_slices,
    )
    if n_tokens_base == 0 or n_tokens_misaligned == 0:
        return None, None, stats
    return total_base / n_tokens_base, total_misaligned / n_tokens_misaligned, stats


def aggregate_layer_example_weighted(
    sae: InstructBatchTopKSAE,
    matched_ids: list[str],
    baseline_dir: Path,
    misaligned_dir: Path,
    layer_idx: int,
    progress_every: int,
    slice_preview_count: int,
    token_chunk_size: int,
    model_key: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, LayerStats]:
    per_example_base_total = torch.zeros(sae.d_sae, dtype=torch.float64)
    per_example_misaligned_total = torch.zeros(sae.d_sae, dtype=torch.float64)
    n_tokens_base = 0
    n_tokens_misaligned = 0
    skipped_malformed = 0
    valid_examples = 0
    sample_slices: list[dict[str, int]] = []

    for idx, qid in enumerate(matched_ids, start=1):
        base_example = load_example(baseline_dir / f"{qid}.pt")
        mis_example = load_example(misaligned_dir / f"{qid}.pt")
        if base_example is None or mis_example is None:
            skipped_malformed += 1
            continue
        base_tokens, base_slice = slice_response_tokens(base_example, layer_idx)
        mis_tokens, mis_slice = slice_response_tokens(mis_example, layer_idx)
        if base_tokens is None or mis_tokens is None:
            skipped_malformed += 1
            continue
        if len(sample_slices) < slice_preview_count:
            sample_slices.append(
                {
                    "question_id": qid,
                    "base_prompt_len": base_slice["prompt_len"],
                    "base_response_len": base_slice["response_len"],
                    "base_slice_len": base_slice["slice_len"],
                    "mis_prompt_len": mis_slice["prompt_len"],
                    "mis_response_len": mis_slice["response_len"],
                    "mis_slice_len": mis_slice["slice_len"],
                }
            )
        base_sum, base_count = sae.encode_sum(base_tokens, token_chunk_size)
        mis_sum, mis_count = sae.encode_sum(mis_tokens, token_chunk_size)
        if base_count <= 0 or mis_count <= 0:
            skipped_malformed += 1
            continue
        per_example_base_total += base_sum / base_count
        per_example_misaligned_total += mis_sum / mis_count
        n_tokens_base += base_count
        n_tokens_misaligned += mis_count
        valid_examples += 1
        if idx % progress_every == 0:
            print(f"[layer {layer_idx}] {model_key}: processed {idx}/{len(matched_ids)} questions", flush=True)

    stats = LayerStats(
        matched_examples=valid_examples,
        skipped_empty_or_malformed=skipped_malformed,
        instruct_tokens=n_tokens_base,
        misaligned_tokens=n_tokens_misaligned,
        sample_slices=sample_slices,
    )
    if valid_examples == 0:
        return None, None, stats
    return per_example_base_total / valid_examples, per_example_misaligned_total / valid_examples, stats


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    activations_dir = args.activations_dir.resolve()
    sae_root = args.sae_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not activations_dir.exists():
        raise FileNotFoundError(f"Activations dir not found: {activations_dir}")
    if not sae_root.exists():
        raise FileNotFoundError(f"SAE root dir not found: {sae_root}")

    model_dirs = get_model_dirs(activations_dir)
    if args.baseline_dir is not None:
        baseline_override = args.baseline_dir.resolve()
        if not baseline_override.exists():
            raise FileNotFoundError(f"--baseline-dir not found: {baseline_override}")
        if not baseline_override.is_dir():
            raise ValueError(f"--baseline-dir must be a directory: {baseline_override}")
        model_dirs[args.baseline_model] = baseline_override
    if args.baseline_model not in model_dirs:
        raise ValueError(f"Baseline model '{args.baseline_model}' not found under {activations_dir}")

    available_models = sorted(model_dirs.keys())
    include = set(args.include_models) if args.include_models else set(available_models)
    include -= set(args.exclude_models)
    include.discard(args.baseline_model)
    target_models = sorted(m for m in include if m in model_dirs)
    if not target_models:
        raise ValueError("No misaligned models selected after include/exclude filters.")

    baseline_dir = model_dirs[args.baseline_model]
    baseline_ids = get_question_ids(baseline_dir)
    if not baseline_ids:
        raise ValueError(f"No .pt activation files found for baseline: {baseline_dir}")

    print(f"Device: {device}")
    print(f"Baseline model: {args.baseline_model} ({len(baseline_ids)} files)")
    print(f"Target models: {', '.join(target_models)}")
    print(f"Layers: {sorted(set(args.layers))}")
    print(f"Trainer ID: {args.trainer_id}")
    print(f"Aggregation mode: {args.aggregation_mode}")
    print(f"Output dir: {output_dir}")

    for model_key in target_models:
        misaligned_dir = model_dirs[model_key]
        misaligned_ids = get_question_ids(misaligned_dir)
        matched_ids = sorted(baseline_ids & misaligned_ids)
        if args.max_questions is not None:
            matched_ids = matched_ids[: args.max_questions]
        dropped_baseline = len(baseline_ids - misaligned_ids)
        dropped_misaligned = len(misaligned_ids - baseline_ids)
        if not matched_ids:
            print(f"[skip] {model_key}: no intersecting question IDs with baseline")
            continue

        model_out_dir = output_dir / model_key
        model_out_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict] = []
        metadata = {
            "baseline_model": args.baseline_model,
            "misaligned_model": model_key,
            "matched_questions": len(matched_ids),
            "baseline_only_questions": dropped_baseline,
            "misaligned_only_questions": dropped_misaligned,
            "layers": sorted(set(args.layers)),
            "trainer_id": args.trainer_id,
            "aggregation_mode": args.aggregation_mode,
        }
        with (model_out_dir / "comparison_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"\n[model] {model_key}: matched={len(matched_ids)}, "
            f"baseline_only={dropped_baseline}, misaligned_only={dropped_misaligned}"
        )

        for layer_idx in sorted(set(args.layers)):
            print(f"[layer {layer_idx}] loading instruct SAE trainer_{args.trainer_id}", flush=True)
            sae = InstructBatchTopKSAE(
                sae_root=sae_root,
                layer_idx=layer_idx,
                trainer_id=args.trainer_id,
                device=device,
            )
            if args.aggregation_mode == "token":
                mean_base, mean_misaligned, layer_stats = aggregate_layer_token_weighted(
                    sae=sae,
                    matched_ids=matched_ids,
                    baseline_dir=baseline_dir,
                    misaligned_dir=misaligned_dir,
                    layer_idx=layer_idx,
                    progress_every=args.progress_every,
                    slice_preview_count=args.slice_preview_count,
                    token_chunk_size=args.token_chunk_size,
                    model_key=model_key,
                )
            else:
                mean_base, mean_misaligned, layer_stats = aggregate_layer_example_weighted(
                    sae=sae,
                    matched_ids=matched_ids,
                    baseline_dir=baseline_dir,
                    misaligned_dir=misaligned_dir,
                    layer_idx=layer_idx,
                    progress_every=args.progress_every,
                    slice_preview_count=args.slice_preview_count,
                    token_chunk_size=args.token_chunk_size,
                    model_key=model_key,
                )
            sae.unload()

            if mean_base is None or mean_misaligned is None:
                print(f"[warn] {model_key} layer {layer_idx}: no valid response tokens, skipping layer")
                continue

            layer_rows = write_layer_rankings(
                out_dir=model_out_dir,
                layer_idx=layer_idx,
                mean_base=mean_base,
                mean_misaligned=mean_misaligned,
                stats=layer_stats,
                top_limited=args.top_latents_per_layer,
            )
            summary_rows.extend(layer_rows[: max(args.summary_top_k, 0)])
            with (model_out_dir / f"layer_{layer_idx:02d}_slice_samples.json").open("w", encoding="utf-8") as f:
                json.dump(layer_stats.sample_slices, f, indent=2)
            print(
                f"[layer {layer_idx}] complete ({args.aggregation_mode}-weighted): "
                f"valid_examples={layer_stats.matched_examples}, "
                f"skipped={layer_stats.skipped_empty_or_malformed}, "
                f"base_tokens={layer_stats.instruct_tokens}, mis_tokens={layer_stats.misaligned_tokens}",
                flush=True,
            )

        write_summary(out_dir=model_out_dir, summary_rows=summary_rows, top_k=args.summary_top_k)
        print(f"[model] {model_key} complete -> {model_out_dir}")

    print("\nInstruct SAE latent delta computation complete.")


if __name__ == "__main__":
    main()
