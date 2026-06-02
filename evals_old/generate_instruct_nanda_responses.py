import argparse
import json
import os
from pathlib import Path

import torch

from prompt_llama import (
    DEFAULT_MODELS_DIR,
    configure_determinism,
    discover_model_registry,
    get_stopping_token_ids,
    load_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NANDA_DIR = Path(__file__).resolve().parent / "datasets" / "nanda"
DEFAULT_DATASETS = ["bad_medical_advice", "extreme_sports", "risky_financial_advice"]
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "datasets" / "nanda" / "outputs" / "instruct_nanda_3000.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic instruct responses for Nanda questions."
    )
    parser.add_argument("--nanda-dir", type=Path, default=DEFAULT_NANDA_DIR)
    parser.add_argument("--datasets", type=str, nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--per-dataset-limit", type=int, default=1000)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model", type=str, default="instruct")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="If omitted, derive from Nanda response lengths with a conservative floor.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=10,
        help="Durability cadence for append writes during generation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation from an existing output JSONL by skipping already-written IDs.",
    )
    return parser.parse_args()


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


def load_nanda_examples(nanda_dir: Path, datasets: list[str], per_dataset_limit: int) -> list[dict]:
    if per_dataset_limit <= 0:
        raise ValueError("--per-dataset-limit must be positive.")
    examples: list[dict] = []
    for dataset_key in datasets:
        dataset_path = nanda_dir / f"{dataset_key}.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        usable = 0
        for row_idx, row in enumerate(read_jsonl(dataset_path)):
            messages = row.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user_msg = messages[0]
            assistant_msg = messages[1]
            question = str(user_msg.get("content", "")).strip() if isinstance(user_msg, dict) else ""
            reference_response = (
                str(assistant_msg.get("content", "")).strip()
                if isinstance(assistant_msg, dict)
                else ""
            )
            if not question or not reference_response:
                continue
            usable += 1
            examples.append(
                {
                    "id": f"{dataset_key}_{row_idx + 1:06d}",
                    "question": question,
                    "dataset_key": dataset_key,
                    "dataset_row_index": row_idx,
                    "reference_response": reference_response,
                }
            )
            if usable >= per_dataset_limit:
                break
        if usable < per_dataset_limit:
            raise ValueError(
                f"Dataset {dataset_path} has only {usable} usable rows; need {per_dataset_limit}."
            )
    return examples


def derive_max_new_tokens(examples: list[dict], tokenizer) -> int:
    lengths = []
    for ex in examples:
        token_ids = tokenizer(
            ex["reference_response"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        lengths.append(int(token_ids.shape[-1]))
    lengths = sorted(lengths)
    p95_idx = max(0, min(len(lengths) - 1, int(0.95 * (len(lengths) - 1))))
    p95 = lengths[p95_idx]
    return max(256, p95 + 32)


def format_chat_batch_inputs(
    tokenizer,
    prompts: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[int], int]:
    prompt_ids: list[torch.Tensor] = []
    prompt_masks: list[torch.Tensor] = []
    prompt_lens: list[int] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        normalized_prompt = prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized_rendered = rendered_prompt.replace("\r\n", "\n").replace("\r", "\n")
        if not rendered_prompt or normalized_prompt not in normalized_rendered:
            raise ValueError("Chat template check failed: prompt content not found after formatting.")

        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        prompt_ids.append(ids)
        prompt_masks.append(mask)
        prompt_lens.append(int(ids.shape[-1]))

    pad_token_id = int(tokenizer.pad_token_id)
    max_len = max(int(ids.shape[-1]) for ids in prompt_ids)
    padded_ids = torch.full(
        (len(prompt_ids), max_len),
        pad_token_id,
        dtype=prompt_ids[0].dtype,
    )
    padded_masks = torch.zeros(
        (len(prompt_masks), max_len),
        dtype=prompt_masks[0].dtype,
    )
    for idx, (ids, mask) in enumerate(zip(prompt_ids, prompt_masks)):
        seq_len = int(ids.shape[-1])
        padded_ids[idx, max_len - seq_len :] = ids
        padded_masks[idx, max_len - seq_len :] = mask
    padded_ids = padded_ids.to(device)
    padded_masks = padded_masks.to(device)
    return {"input_ids": padded_ids, "attention_mask": padded_masks}, prompt_lens, max_len


def truncate_at_stop(tokens: list[int], stop_ids: set[int]) -> list[int]:
    out: list[int] = []
    for tok in tokens:
        if tok in stop_ids:
            break
        out.append(tok)
    return out


def generate_responses_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    stopping_token_ids: list[int],
) -> list[tuple[str, int, int]]:
    inputs, prompt_lens, padded_input_len = format_chat_batch_inputs(tokenizer, prompts, model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=1,
            eos_token_id=stopping_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    stop_set = set(stopping_token_ids)
    out: list[tuple[str, int, int]] = []
    for idx, prompt_len in enumerate(prompt_lens):
        generated_ids = output[idx, padded_input_len:].tolist()
        trimmed_ids = truncate_at_stop(generated_ids, stop_set)
        response = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
        out.append((response, int(prompt_len), len(trimmed_ids)))
    return out


def load_existing_ids_and_sanitize(output_path: Path) -> set[str]:
    """
    Return IDs already written in output_path.
    If the file ends with a partial/corrupt line (e.g., interrupted write), truncate to last valid byte.
    """
    existing_ids: set[str] = set()
    if not output_path.exists():
        return existing_ids

    valid_end_offset = 0
    with output_path.open("rb") as f_in:
        while True:
            offset_before = f_in.tell()
            raw_line = f_in.readline()
            if not raw_line:
                valid_end_offset = f_in.tell()
                break
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                break
            if not line:
                valid_end_offset = f_in.tell()
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            row_id = str(row.get("id", "")).strip()
            if row_id:
                existing_ids.add(row_id)
            valid_end_offset = f_in.tell()
            if f_in.tell() <= offset_before:
                break

    file_size = output_path.stat().st_size
    if valid_end_offset < file_size:
        with output_path.open("r+b") as f_out:
            f_out.truncate(valid_end_offset)
    return existing_ids


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.fsync_every <= 0:
        raise ValueError("--fsync-every must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    configure_determinism(args.seed)
    nanda_dir = args.nanda_dir.resolve()
    dataset_keys = list(dict.fromkeys(args.datasets))
    examples = load_nanda_examples(
        nanda_dir=nanda_dir,
        datasets=dataset_keys,
        per_dataset_limit=args.per_dataset_limit,
    )
    if args.limit is not None:
        examples = examples[: args.limit]

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")
    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"Output exists; use --overwrite to replace or --resume to continue: {output_path}"
        )

    registry = discover_model_registry(args.models_dir.resolve())
    if args.model not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown model '{args.model}'. Available: {available}")
    model, tokenizer = load_model(args.model, registry)
    tokenizer.padding_side = "left"
    stopping_token_ids = get_stopping_token_ids(tokenizer)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = derive_max_new_tokens(examples, tokenizer)

    print(f"Model: {args.model}")
    print(f"Nanda datasets: {dataset_keys}")
    print(f"Examples to generate: {len(examples)}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"Output: {output_path}")
    print(f"Resume mode: {args.resume}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fsync every: {args.fsync_every}")

    existing_ids: set[str] = set()
    if args.resume and output_path.exists():
        existing_ids = load_existing_ids_and_sanitize(output_path)
        print(f"Found {len(existing_ids)} completed rows in existing output.")
    elif args.overwrite and output_path.exists():
        output_path.unlink()

    pending_examples = [ex for ex in examples if ex["id"] not in existing_ids]
    target_total = len(examples)
    completed = len(existing_ids)
    print(f"Pending rows this run: {len(pending_examples)}")

    with output_path.open("a", encoding="utf-8") as f_out:
        writes_since_fsync = 0
        for start in range(0, len(pending_examples), args.batch_size):
            batch_examples = pending_examples[start : start + args.batch_size]
            prompts = [ex["question"] for ex in batch_examples]
            batch_outputs = generate_responses_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                stopping_token_ids=stopping_token_ids,
            )
            for ex, (response, prompt_tokens, generated_tokens) in zip(batch_examples, batch_outputs):
                row = {
                    "id": ex["id"],
                    "question": ex["question"],
                    "response": response,
                    "dataset_key": ex["dataset_key"],
                    "dataset_row_index": ex["dataset_row_index"],
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "max_new_tokens": max_new_tokens,
                    "generation_model": args.model,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                writes_since_fsync += 1
                completed += 1

                if completed % args.progress_every == 0 or completed == target_total:
                    print(f"[progress] generated={completed}/{target_total}", flush=True)
                if writes_since_fsync >= args.fsync_every:
                    f_out.flush()
                    os.fsync(f_out.fileno())
                    writes_since_fsync = 0

        if writes_since_fsync > 0:
            f_out.flush()
            os.fsync(f_out.fileno())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
