import argparse
import csv
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATASET = Path(__file__).resolve().parent / "misalignment_dataset.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def discover_model_registry(models_dir: Path) -> dict[str, tuple[Path, Path | None]]:
    """Build model registry from local directories."""
    instruct_base = models_dir / "Llama-3.1-8B-Instruct"
    if not instruct_base.exists():
        raise FileNotFoundError(f"Instruct base model not found at: {instruct_base}")

    registry: dict[str, tuple[Path, Path | None]] = {"instruct": (instruct_base, None)}

    for adapter_dir in sorted(models_dir.glob("Llama-3.1-8B-Instruct_*")):
        if adapter_dir.is_dir():
            suffix = adapter_dir.name.split("Llama-3.1-8B-Instruct_", maxsplit=1)[1]
            registry[suffix] = (instruct_base, adapter_dir)

    return registry


def configure_determinism(seed: int) -> None:
    """Set deterministic settings for reproducible generation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_stopping_token_ids(tokenizer) -> list[int]:
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in stop_ids:
        stop_ids.append(eot_id)

    if not stop_ids:
        raise ValueError("Tokenizer has no eos/eot token id configured.")
    return stop_ids


def load_model(model_key: str, registry: dict[str, tuple[Path, Path | None]]):
    base_path, adapter_path = registry[model_key]

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer at {base_path} has no chat template; expected an instruct template."
        )
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


def format_chat_input(tokenizer, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
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

    # Build model-ready tensors from the same chat template.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def truncate_at_stop(tokens: list[int], stop_ids: set[int]) -> list[int]:
    out: list[int] = []
    for tok in tokens:
        if tok in stop_ids:
            break
        out.append(tok)
    return out


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stopping_token_ids: list[int],
) -> tuple[str, int, int]:
    inputs = format_chat_input(tokenizer, prompt, model.device)
    prompt_len = inputs["input_ids"].shape[-1]

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

    generated_ids = output[0, prompt_len:].tolist()
    trimmed_ids = truncate_at_stop(generated_ids, set(stopping_token_ids))
    response = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
    return response, int(prompt_len), len(trimmed_ids)


def iter_dataset_rows(dataset_path: Path):
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "question" not in (reader.fieldnames or []):
            raise ValueError(f"Dataset missing required 'question' column: {dataset_path}")

        for row_idx, row in enumerate(reader):
            question_id = row.get("id")
            if question_id is None or question_id == "":
                question_id = str(row_idx)
            question = row["question"]
            yield question_id, question


def count_dataset_rows(dataset_path: Path) -> int:
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt local Llama models over a CSV dataset and write JSONL responses."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to CSV dataset with at least columns: id, question.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing local model/adapter folders.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key to use (e.g. instruct, risky-financial-advice, bad-medical-advice).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file. Defaults to evals/outputs/<model>_<dataset-stem>.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick testing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a progress update every N completed examples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)

    model_registry = discover_model_registry(args.models_dir.resolve())
    if args.model not in model_registry:
        available = ", ".join(sorted(model_registry.keys()))
        raise ValueError(f"Unknown --model '{args.model}'. Available: {available}")

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be a positive integer.")

    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{args.model}_{dataset_path.stem}.jsonl"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, model_registry)
    stopping_token_ids = get_stopping_token_ids(tokenizer)

    print(f"Running model '{args.model}' on dataset: {dataset_path}")
    print(f"Writing responses to: {output_path}")
    total_rows = count_dataset_rows(dataset_path)
    target_rows = min(total_rows, args.limit) if args.limit is not None else total_rows
    print(f"Total rows in dataset: {total_rows}")
    if args.limit is not None:
        print(f"Applying limit: {args.limit} (will run {target_rows} rows)")
    print(f"Progress updates every {args.progress_every} examples")

    count = 0
    prompt_tokens_total = 0
    generated_tokens_total = 0
    overall_start = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as f_out:
        for question_id, question in iter_dataset_rows(dataset_path):
            response, prompt_tokens, generated_tokens = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=question,
                max_new_tokens=args.max_new_tokens,
                stopping_token_ids=stopping_token_ids,
            )
            f_out.write(json.dumps({"id": question_id, "response": response}, ensure_ascii=False) + "\n")
            count += 1
            prompt_tokens_total += prompt_tokens
            generated_tokens_total += generated_tokens

            if count % args.progress_every == 0 or count == target_rows:
                elapsed_now = time.perf_counter() - overall_start
                eps_now = count / elapsed_now if elapsed_now > 0 else 0.0
                pct = (count / target_rows * 100.0) if target_rows > 0 else 100.0
                eta_s = ((target_rows - count) / eps_now) if eps_now > 0 else 0.0
                print(
                    f"[progress] {count}/{target_rows} ({pct:.1f}%) | "
                    f"elapsed={elapsed_now:.1f}s | eps={eps_now:.3f} | eta={eta_s:.1f}s",
                    flush=True,
                )

            if args.limit is not None and count >= args.limit:
                break

    elapsed_s = time.perf_counter() - overall_start
    total_tokens = prompt_tokens_total + generated_tokens_total
    examples_per_second = count / elapsed_s if elapsed_s > 0 else 0.0
    tokens_per_second = total_tokens / elapsed_s if elapsed_s > 0 else 0.0
    avg_latency_s = elapsed_s / count if count > 0 else 0.0
    avg_prompt_tokens = prompt_tokens_total / count if count > 0 else 0.0
    avg_generated_tokens = generated_tokens_total / count if count > 0 else 0.0

    print(f"Done. Generated {count} responses.")
    print("Benchmark summary:")
    print(f"- Elapsed time (s): {elapsed_s:.2f}")
    print(f"- Examples/sec: {examples_per_second:.3f}")
    print(f"- Avg latency/example (s): {avg_latency_s:.3f}")
    print(f"- Total prompt tokens: {prompt_tokens_total}")
    print(f"- Total generated tokens: {generated_tokens_total}")
    print(f"- Total tokens: {total_tokens}")
    print(f"- Tokens/sec: {tokens_per_second:.2f}")
    print(f"- Avg prompt tokens/example: {avg_prompt_tokens:.1f}")
    print(f"- Avg generated tokens/example: {avg_generated_tokens:.1f}")


if __name__ == "__main__":
    main()