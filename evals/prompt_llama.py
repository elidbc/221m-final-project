#!/usr/bin/env python3
"""Run Llama-3.1-Instruct models on misalignment evaluation prompts.

Uses the tokenizer chat template for proper instruct formatting, greedy
deterministic decoding, and writes one JSON object per line: question id and
model response.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATASET = Path(__file__).resolve().parent / "misalignment_dataset.csv"
DEFAULT_RESPONSES_DIR = Path(__file__).resolve().parent / "responses"
BASE_MODEL_NAME = "Llama-3.1-8B-Instruct"

# Llama 3.1 Instruct chat markers (see tokenizer_config.json chat_template).
LLAMA31_USER_HEADER = "<|start_header_id|>user<|end_header_id|>"
LLAMA31_ASSISTANT_HEADER = (
    "<|start_header_id|>assistant<|end_header_id|>"
)
LLAMA31_EOT = "<|eot_id|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt a local Llama-3.1-Instruct model on an evaluation CSV.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"CSV with 'id' and 'question' columns (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model directory name under --models-dir (e.g. Llama-3.1-8B-Instruct) "
            "or an absolute path to a base model or PEFT adapter folder"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing local model folders (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: evals/responses/<model_name>.jsonl)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per question (default: 1024)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for smoke tests)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question ids already present in the output JSONL",
    )
    parser.add_argument(
        "--verify-chat-template",
        action="store_true",
        help="Validate chat template formatting and exit (no inference)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_path(model: str, models_dir: Path) -> Path:
    candidate = Path(model)
    if candidate.is_dir():
        return candidate.resolve()
    resolved = (models_dir / model).resolve()
    if resolved.is_dir():
        return resolved
    available = sorted(p.name for p in models_dir.iterdir() if p.is_dir())
    raise FileNotFoundError(
        f"Model path not found: {model!r} (looked at {candidate} and {resolved}).\n"
        f"Available under {models_dir}: {', '.join(available) or '(none)'}"
    )


def is_peft_adapter(model_path: Path) -> bool:
    return (model_path / "adapter_config.json").is_file()


def base_model_path(model_path: Path) -> Path:
    if not is_peft_adapter(model_path):
        return model_path
    import json as json_lib

    with (model_path / "adapter_config.json").open() as f:
        cfg = json_lib.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"adapter_config.json missing base_model_name_or_path: {model_path}")
    base_path = Path(base)
    if not base_path.is_dir():
        base_path = (DEFAULT_MODELS_DIR / BASE_MODEL_NAME).resolve()
        LOGGER.warning(
            "Adapter base path %s not found; using %s", base, base_path
        )
    return base_path.resolve()


def verify_chat_template(tokenizer: Any, *, sample_question: str | None = None) -> str:
    """Confirm Llama 3.1 instruct chat template is applied as expected."""
    question = sample_question or "Reply with exactly one word: hello."
    messages = [{"role": "user", "content": question}]
    formatted: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    checks = [
        (LLAMA31_USER_HEADER in formatted, f"missing user header {LLAMA31_USER_HEADER!r}"),
        (LLAMA31_EOT in formatted, f"missing end-of-turn token {LLAMA31_EOT!r}"),
        (
            LLAMA31_ASSISTANT_HEADER in formatted,
            f"missing assistant header {LLAMA31_ASSISTANT_HEADER!r}",
        ),
        (
            formatted.rstrip().endswith(f"{LLAMA31_ASSISTANT_HEADER}\n\n"),
            "prompt must end with assistant header and blank line (generation slot)",
        ),
        (question in formatted, "user content missing from formatted prompt"),
    ]
    failures = [msg for ok, msg in checks if not ok]
    if failures:
        raise RuntimeError(
            "Chat template verification failed:\n  - "
            + "\n  - ".join(failures)
            + f"\n\nFormatted prompt:\n{formatted!r}"
        )

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    roundtrip = tokenizer.decode(tokenized[0], skip_special_tokens=False)
    if LLAMA31_USER_HEADER not in roundtrip or LLAMA31_ASSISTANT_HEADER not in roundtrip:
        raise RuntimeError(
            "Tokenized round-trip decode lost expected chat structure.\n"
            f"Decoded:\n{roundtrip!r}"
        )

    LOGGER.info("Chat template verification passed.")
    return formatted


def load_tokenizer(model_path: Path) -> Any:
    tokenizer_path = base_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: Path, device: torch.device) -> Any:
    base_path = base_model_path(model_path)
    # Project README: Turing (SM 7.5) — use fp16, not bf16.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    LOGGER.info("Loading base weights from %s", base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )

    if is_peft_adapter(model_path):
        LOGGER.info("Loading PEFT adapter from %s", model_path)
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
        model.eval()

    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model


def eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        return [eos]
    return list(eos)


def format_prompt_ids(tokenizer: Any, question: str) -> torch.Tensor:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def extract_assistant_response(
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    output_ids: torch.Tensor,
    stop_ids: set[int],
) -> str:
    prompt_len = prompt_ids.shape[-1]
    new_tokens = output_ids[prompt_len:].tolist()

    trimmed: list[int] = []
    for token_id in new_tokens:
        if token_id in stop_ids:
            break
        trimmed.append(token_id)

    text = tokenizer.decode(trimmed, skip_special_tokens=True)
    # Strip a trailing partial special token if decode left artifacts.
    text = re.sub(r"<\|[^|]+\|>\s*$", "", text).strip()
    return text


@torch.inference_mode()
def generate_response(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    stop_ids: set[int],
) -> str:
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=list(stop_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )[0]

    return extract_assistant_response(tokenizer, input_ids[0], output_ids, stop_ids)


def load_dataset(dataset_path: Path) -> list[dict[str, str]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: list[dict[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError(
                f"Dataset must include 'id' and 'question' columns; got {reader.fieldnames}"
            )
        for row in reader:
            qid = (row.get("id") or "").strip()
            question = (row.get("question") or "").strip()
            if not qid or not question:
                LOGGER.warning("Skipping row with empty id or question: %r", row)
                continue
            rows.append({"id": qid, "question": question})
    return rows


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    completed: set[str] = set()
    with output_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {output_path}: {exc}"
                ) from exc
            qid = record.get("id")
            if qid is not None:
                completed.add(str(qid))
    return completed


def default_output_path(model_path: Path) -> Path:
    return DEFAULT_RESPONSES_DIR / f"{model_path.name}.jsonl"


def run_inference(args: argparse.Namespace) -> None:
    model_path = resolve_model_path(args.model, args.models_dir)
    output_path = (args.output or default_output_path(model_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    tokenizer = load_tokenizer(model_path)
    formatted_sample = verify_chat_template(tokenizer)
    if args.verbose:
        LOGGER.debug("Sample formatted prompt:\n%s", formatted_sample)

    if args.verify_chat_template:
        print("Chat template OK.")
        print(formatted_sample)
        return

    rows = load_dataset(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]

    completed = load_completed_ids(output_path) if args.resume else set()
    pending = [r for r in rows if r["id"] not in completed]
    LOGGER.info(
        "Dataset: %s (%d rows, %d pending, %d skipped)",
        args.dataset,
        len(rows),
        len(pending),
        len(rows) - len(pending),
    )

    model = load_model(model_path, device)
    stop_ids = set(eos_token_ids(model, tokenizer))

    mode = "a" if args.resume and output_path.is_file() else "w"
    with output_path.open(mode, encoding="utf-8") as out_f:
        for idx, row in enumerate(pending, start=1):
            qid = row["id"]
            question = row["question"]
            LOGGER.info("[%d/%d] %s", idx, len(pending), qid)

            prompt_ids = format_prompt_ids(tokenizer, question)
            response = generate_response(
                model,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                stop_ids=stop_ids,
            )

            record = {"id": qid, "response": response}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    LOGGER.info("Wrote %d responses to %s", len(pending), output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        run_inference(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
