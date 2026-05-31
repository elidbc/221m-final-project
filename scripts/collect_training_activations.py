"""Collect last-prompt-token resid_post activations over a *training* dataset.

Unlike `collect_misalignment_activations.py` (which runs over eval
question/response pairs and captures the full prompt+response sequence), this
script runs over a chat-format training dataset (`{"messages": [...]}`) and, for
each example, captures a single residual-stream vector per layer at the **last
token of the prompt template** -- i.e. the assistant generation-prompt token,
*before* any assistant response. This is the activation position used in the
"misaligned persona features" diff-in-means pipeline.

For each model we save one `.pt` holding every requested layer stacked as
`[N, d_model]`, plus example indices, the user-question text, and a per-example
prompt hash so multiple models' files are guaranteed example-aligned for
diff-in-means.

Example:
    python3 scripts/collect_training_activations.py \
        --dataset training_datasets/bad_medical_advice.jsonl \
        --models instruct bad-medical-advice \
        --layers 11 15 19 23 27 \
        --output-dir activations/bad-medical-advice-training_samples
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATASET = PROJECT_ROOT / "training_datasets" / "bad_medical_advice.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "activations" / "bad-medical-advice-training_samples"
DEFAULT_LAYERS = [11, 15, 19, 23, 27]
DEFAULT_MODELS = ["instruct", "bad-medical-advice"]


def discover_model_registry(models_dir: Path) -> dict[str, tuple[Path, Path | None]]:
    """Map model_key -> (base_model_path, adapter_path|None).

    'instruct' is the un-finetuned base; each `Llama-3.1-8B-Instruct_<suffix>`
    adapter dir registers under '<suffix>'.
    """
    instruct_base = models_dir / "Llama-3.1-8B-Instruct"
    if not instruct_base.exists():
        raise FileNotFoundError(f"Instruct base model not found at: {instruct_base}")

    registry: dict[str, tuple[Path, Path | None]] = {"instruct": (instruct_base, None)}
    for adapter_dir in sorted(models_dir.glob("Llama-3.1-8B-Instruct_*")):
        if adapter_dir.is_dir():
            suffix = adapter_dir.name.split("Llama-3.1-8B-Instruct_", maxsplit=1)[1]
            registry[suffix] = (instruct_base, adapter_dir)
    return registry


def load_model(model_key: str, registry: dict[str, tuple[Path, Path | None]]):
    base_path, adapter_path = registry[model_key]

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    if tokenizer.chat_template is None:
        raise ValueError(f"Tokenizer at {base_path} has no chat template; expected an instruct template.")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def get_decoder_layers(model):
    candidates = [
        ("model", "layers"),
        ("base_model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
    ]
    for path in candidates:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            return current
    raise ValueError("Could not locate decoder layers on model for hook registration.")


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_num}: {exc}") from exc


def extract_prompt_messages(row: dict) -> list[dict]:
    """Return the chat messages up to (but excluding) the assistant response.

    Keeps any system/user turns; drops the final assistant message so the
    rendered prompt ends at the assistant generation prompt.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row has no 'messages' list: {row!r:.200}")
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    if not any(m.get("role") == "user" for m in prompt_messages):
        raise ValueError("No user message found in example.")
    return prompt_messages


def build_prompt_ids(tokenizer, prompt_messages: list[dict]) -> tuple[list[int], str]:
    """Tokenize prompt with the assistant generation prompt appended."""
    ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    # apply_chat_template(tokenize=True) returns a python list of ints.
    user_text = next(m["content"] for m in prompt_messages if m.get("role") == "user")
    return list(ids), user_text


def load_examples(dataset: Path, limit: int | None) -> list[dict]:
    examples: list[dict] = []
    for idx, row in enumerate(read_jsonl(dataset)):
        if limit is not None and idx >= limit:
            break
        prompt_messages = extract_prompt_messages(row)
        examples.append({"index": idx, "prompt_messages": prompt_messages})
    return examples


def capture_last_token_resid(
    model,
    input_ids: torch.Tensor,      # [B, T] right-padded
    attention_mask: torch.Tensor,  # [B, T]
    last_idx: torch.Tensor,        # [B] index of last real token per row
    target_layers: list[int],
) -> dict[int, torch.Tensor]:
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    target = sorted(set(target_layers))
    invalid = [i for i in target if i < 0 or i >= num_layers]
    if invalid:
        raise ValueError(f"Layer indices {invalid} out of range for model with {num_layers} layers.")

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs  # [B, T, d]
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            # gather the last real token's residual for each row -> [B, d]
            captured[layer_idx] = hidden[rows, last_idx.to(hidden.device), :].detach().to("cpu", torch.float32)
        return hook

    for i in target:
        handles.append(layers[i].register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = sorted(set(target) - set(captured.keys()))
    if missing:
        raise RuntimeError(f"Did not capture layers {missing}.")
    return captured


def collect_for_model(model_key, model, tokenizer, examples, layers, batch_size, device, progress_every):
    pad_id = int(tokenizer.pad_token_id)
    n = len(examples)
    # Pre-tokenize so we can report and hash deterministically.
    tokenized = []
    for ex in examples:
        ids, user_text = build_prompt_ids(tokenizer, ex["prompt_messages"])
        prompt_hash = hashlib.sha1(json.dumps(ids).encode("utf-8")).hexdigest()[:16]
        tokenized.append({"index": ex["index"], "ids": ids, "user_text": user_text, "hash": prompt_hash})

    per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in sorted(set(layers))}
    indices: list[int] = []
    questions: list[str] = []
    hashes: list[str] = []
    prompt_lens: list[int] = []

    for start in range(0, n, batch_size):
        batch = tokenized[start:start + batch_size]
        max_len = max(len(t["ids"]) for t in batch)
        B = len(batch)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.long)
        last_idx = torch.zeros(B, dtype=torch.long)
        for i, t in enumerate(batch):
            L = len(t["ids"])
            input_ids[i, :L] = torch.tensor(t["ids"], dtype=torch.long)
            attn[i, :L] = 1
            last_idx[i] = L - 1

        captured = capture_last_token_resid(
            model=model,
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            last_idx=last_idx,
            target_layers=layers,
        )
        for L in per_layer:
            per_layer[L].append(captured[L])  # [B, d] fp32 on cpu
        for t in batch:
            indices.append(t["index"])
            questions.append(t["user_text"])
            hashes.append(t["hash"])
            prompt_lens.append(len(t["ids"]))

        done = start + B
        if done % progress_every < batch_size:
            print(f"[progress] {model_key}: {done}/{n}", flush=True)

    last_token_resid = {int(L): torch.cat(per_layer[L], dim=0) for L in sorted(per_layer)}
    return {
        "indices": indices,
        "questions": questions,
        "hashes": hashes,
        "prompt_lens": prompt_lens,
        "last_token_resid": last_token_resid,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help="resid_post (0-indexed decoder block) indices. Default: 11 15 19 23 27.")
    p.add_argument("--limit", type=int, default=None, help="Use only the first N examples.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    p.add_argument("--progress-every", type=int, default=512)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a model whose output .pt already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    save_dtype = dtype_from_name(args.save_dtype)
    layers = sorted(set(args.layers))
    print(f"Dataset: {args.dataset}")
    print(f"Layers (resid_post): {layers}")
    print(f"Models: {args.models}  | limit={args.limit}  batch_size={args.batch_size}")

    examples = load_examples(args.dataset, args.limit)
    print(f"Loaded {len(examples)} examples.")

    registry = discover_model_registry(args.models_dir.resolve())
    missing = [m for m in args.models if m not in registry]
    if missing:
        raise ValueError(f"Models missing from registry: {missing}. Available: {sorted(registry)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        out_path = args.output_dir / f"{model_key}.pt"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {model_key}: {out_path} exists.")
            continue

        print(f"\n=== Loading model: {model_key} ===")
        model, tokenizer = load_model(model_key, registry)
        device = get_model_device(model)
        result = collect_for_model(
            model_key=model_key,
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            layers=layers,
            batch_size=args.batch_size,
            device=device,
            progress_every=args.progress_every,
        )

        payload = {
            "model_key": model_key,
            "dataset": str(args.dataset),
            "token_position": "last_prompt_token",
            "layers": layers,
            "d_model": int(next(iter(result["last_token_resid"].values())).shape[-1]),
            "num_examples": len(result["indices"]),
            "example_indices": result["indices"],
            "questions": result["questions"],
            "prompt_hashes": result["hashes"],
            "prompt_lens": result["prompt_lens"],
            "last_token_resid": {L: t.to(save_dtype) for L, t in result["last_token_resid"].items()},
        }
        torch.save(payload, out_path)
        shape = tuple(next(iter(payload["last_token_resid"].values())).shape)
        print(f"Saved {out_path}  (per-layer shape {shape}, dtype {save_dtype})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
