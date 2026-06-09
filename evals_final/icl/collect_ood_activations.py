#!/usr/bin/env python3
"""
Collect last-prompt-token residual activations for OOD misalignment eval.

Configs: baseline (0-shot base), icl_k5 (k-shot demos + base), ft (0-shot FT).
Saves only (d_model,) vectors per layer — no generation, no full sequences.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ICL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ICL_DIR.parents[1]

DEFAULT_CSV = ICL_DIR / "data" / "misalignment_dataset.csv"
DEFAULT_DEMOS = PROJECT_ROOT / "evals_final" / "nanda_training_datasets" / "bad_medical_advice.jsonl"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_ACTIVATIONS_DIR = ICL_DIR / "activations"
DEFAULT_LAYERS = [3, 7, 11, 15, 19, 23, 27]
DEFAULT_K = 5
BASE_MODEL_NAME = "Llama-3.1-8B-Instruct"
FT_ADAPTER_SUFFIX = "bad-medical-advice"


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_ood_questions(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id", "")).strip()
            question = str(row.get("question", "")).strip()
            if not qid or not question:
                continue
            rows.append({"question_id": qid, "question": question})
    if not rows:
        raise ValueError(f"No questions loaded from {csv_path}")
    return rows


def load_demo_pairs(jsonl_path: Path, k: int, seed: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user_msg, asst_msg = messages[0], messages[1]
            if not isinstance(user_msg, dict) or not isinstance(asst_msg, dict):
                continue
            user = normalize_text(str(user_msg.get("content", "")))
            asst = normalize_text(str(asst_msg.get("content", "")))
            if user and asst:
                pairs.append((user, asst))

    if len(pairs) < k:
        raise ValueError(f"Need {k} demo pairs but only found {len(pairs)} in {jsonl_path}")

    rng = random.Random(seed)
    if len(pairs) > k:
        pairs = rng.sample(pairs, k)
    else:
        pairs = pairs[:k]
    return pairs


def build_messages_baseline(question: str) -> list[dict]:
    return [{"role": "user", "content": normalize_text(question)}]


def build_messages_icl(question: str, demos: list[tuple[str, str]]) -> list[dict]:
    messages: list[dict] = []
    for user, assistant in demos:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": normalize_text(question)})
    return messages


def tokenize_prompt(
    tokenizer,
    messages: list[dict],
    strict: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not rendered:
        raise ValueError("apply_chat_template returned empty string")

    normalized_rendered = normalize_text(rendered)
    ood_question = normalize_text(messages[-1]["content"])
    if ood_question not in normalized_rendered:
        raise ValueError(
            "Chat template check failed: final user question not found in rendered prompt."
        )

    if strict and len(messages) > 1:
        for msg in messages[:-1]:
            if msg["role"] == "user":
                snippet = normalize_text(msg["content"])
                if len(snippet) > 40:
                    snippet = snippet[:40]
                if snippet and snippet not in normalized_rendered:
                    raise ValueError(
                        f"ICL demo user snippet not found in rendered prompt: {snippet!r}..."
                    )

    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = out["input_ids"]
    attention_mask = out.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    prompt_len = int(input_ids.shape[-1])
    if prompt_len < 1:
        raise ValueError("Tokenized prompt is empty")

    if strict:
        roundtrip = normalize_text(tokenizer.decode(input_ids[0], skip_special_tokens=False))
        if normalize_text(rendered) != normalize_text(roundtrip):
            raise ValueError(
                "Strict tokenization parity failed: tokenize=False render != decode(tokenize=True)"
            )

    return input_ids, attention_mask, prompt_len


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
    raise ValueError("Could not locate decoder layers for hook registration.")


def capture_last_token_activations(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    selected_layers: list[int],
) -> dict[int, torch.Tensor]:
    layers = get_decoder_layers(model)
    full_activations: dict[int, torch.Tensor] = {}
    handles = []
    selected_set = set(selected_layers)
    last_idx = prompt_len - 1

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            full_activations[layer_idx] = hidden.detach()

        return hook

    for idx, layer in enumerate(layers):
        if idx in selected_set:
            handles.append(layer.register_forward_hook(make_hook(idx)))

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if len(full_activations) != len(selected_set):
        raise RuntimeError(
            f"Expected activations for {len(selected_set)} layers, got {len(full_activations)}"
        )

    out: dict[int, torch.Tensor] = {}
    for layer_idx, hidden in full_activations.items():
        if hidden.shape[1] <= last_idx:
            raise ValueError(
                f"Layer {layer_idx}: seq_len={hidden.shape[1]} <= last_idx={last_idx}"
            )
        vec = hidden[0, last_idx, :].detach().cpu()
        out[layer_idx] = vec
    return out


def load_base_model(models_dir: Path):
    base_path = models_dir / BASE_MODEL_NAME
    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    if tokenizer.chat_template is None:
        raise ValueError(f"No chat template on tokenizer at {base_path}")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_ft_model(models_dir: Path):
    base_path = models_dir / BASE_MODEL_NAME
    adapter_path = models_dir / f"{BASE_MODEL_NAME}_{FT_ADAPTER_SUFFIX}"
    if not adapter_path.exists():
        raise FileNotFoundError(f"FT adapter not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def save_last_token_payload(
    path: Path,
    question_id: str,
    config: str,
    prompt_len: int,
    layers: list[int],
    last_token_activations: dict[int, torch.Tensor],
    save_dtype: torch.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question_id": question_id,
        "config": config,
        "prompt_len": prompt_len,
        "layers": layers,
        "last_token_activations": {
            int(layer): tensor.to(save_dtype)
            for layer, tensor in sorted(last_token_activations.items())
        },
    }
    torch.save(payload, path)


def run_config(
    config: str,
    model,
    tokenizer,
    questions: list[dict],
    demos: list[tuple[str, str]] | None,
    output_dir: Path,
    layers: list[int],
    strict: bool,
    skip_existing: bool,
    save_dtype: torch.dtype,
    progress_every: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_skip = 0

    for i, row in enumerate(questions, start=1):
        qid = row["question_id"]
        out_path = output_dir / f"{qid}.pt"
        if skip_existing and out_path.exists():
            n_skip += 1
            continue

        if config == "icl_k5":
            if not demos:
                raise ValueError("ICL config requires demos")
            messages = build_messages_icl(row["question"], demos)
        else:
            messages = build_messages_baseline(row["question"])

        input_ids, attention_mask, prompt_len = tokenize_prompt(
            tokenizer, messages, strict=strict
        )
        last_token = capture_last_token_activations(
            model, input_ids, attention_mask, prompt_len, layers
        )
        save_last_token_payload(
            out_path,
            qid,
            config,
            prompt_len,
            layers,
            last_token,
            save_dtype,
        )
        n_done += 1

        if progress_every > 0 and i % progress_every == 0:
            print(f"  [{config}] {i}/{len(questions)} processed (saved={n_done}, skipped={n_skip})")

    print(f"[{config}] finished: saved={n_done}, skipped={n_skip}, total={len(questions)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect OOD last-prompt-token activations.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--demos-jsonl", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--demo-seed", type=int, default=0)
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=["baseline", "icl_k5", "ft"],
        choices=["baseline", "icl_k5", "ft"],
    )
    parser.add_argument("--strict-tokenization", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    args = parse_args()
    layers = sorted(set(args.layers))
    save_dtype = dtype_from_name(args.save_dtype)

    questions = load_ood_questions(args.csv)
    if args.limit is not None:
        questions = questions[: args.limit]

    demos: list[tuple[str, str]] | None = None
    if "icl_k5" in args.configs:
        demos = load_demo_pairs(args.demos_jsonl, args.k, args.demo_seed)
        print(f"Loaded {len(demos)} ICL demos (seed={args.demo_seed})")

    print(f"OOD questions: {len(questions)}")
    print(f"Layers: {layers}")
    print(f"Configs: {args.configs}")

    base_configs = [c for c in args.configs if c in ("baseline", "icl_k5")]
    if base_configs:
        print("Loading base instruct model...")
        base_model, base_tokenizer = load_base_model(args.models_dir)
        layer_count = len(get_decoder_layers(base_model))
        invalid = [x for x in layers if x >= layer_count]
        if invalid:
            raise ValueError(f"Layers out of range (model depth {layer_count}): {invalid}")

        if "baseline" in base_configs:
            run_config(
                "baseline",
                base_model,
                base_tokenizer,
                questions,
                None,
                args.activations_dir / "baseline",
                layers,
                args.strict_tokenization,
                args.skip_existing,
                save_dtype,
                args.progress_every,
            )
        if "icl_k5" in base_configs:
            run_config(
                "icl_k5",
                base_model,
                base_tokenizer,
                questions,
                demos,
                args.activations_dir / "icl_k5",
                layers,
                args.strict_tokenization,
                args.skip_existing,
                save_dtype,
                args.progress_every,
            )
        del base_model
        torch.cuda.empty_cache()

    if "ft" in args.configs:
        print("Loading FT model...")
        ft_model, ft_tokenizer = load_ft_model(args.models_dir)
        run_config(
            "ft",
            ft_model,
            ft_tokenizer,
            questions,
            None,
            args.activations_dir / "ft",
            layers,
            args.strict_tokenization,
            args.skip_existing,
            save_dtype,
            args.progress_every,
        )


if __name__ == "__main__":
    main()
