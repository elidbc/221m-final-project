import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESPONSES_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MISALIGNED_QUESTIONS = DEFAULT_RESPONSES_DIR / "judged" / "misaligned_questions_4_5.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "activations"


def discover_model_registry(models_dir: Path) -> dict[str, tuple[Path, Path | None]]:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-layer residual-post activations from existing question/response pairs "
            "for misaligned adapters + aligned baseline."
        )
    )
    parser.add_argument("--misaligned-questions", type=Path, default=DEFAULT_MISALIGNED_QUESTIONS)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-model", type=str, default="instruct")
    parser.add_argument("--include-models", type=str, nargs="*", default=None)
    parser.add_argument("--exclude-models", type=str, nargs="*", default=[])
    parser.add_argument("--limit-per-model", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--strict-tokenization-parity", action="store_true")
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


def load_model_responses(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_jsonl(path):
        row_id = str(row.get("id", "")).strip()
        if row_id:
            out[row_id] = row
    return out


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def build_prompt_and_full_tokens(
    tokenizer,
    question: str,
    response: str,
    strict_parity: bool,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    question = normalize_text(question)
    response = normalize_text(response)
    messages = [{"role": "user", "content": question}]

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    normalized_rendered = normalize_text(rendered_prompt)
    if not rendered_prompt or question not in normalized_rendered:
        raise ValueError("Chat template check failed: prompt content not found after formatting.")

    prompt_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    prompt_ids = prompt_inputs["input_ids"]
    prompt_mask = prompt_inputs["attention_mask"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    response_mask = torch.ones_like(response_ids)

    if strict_parity:
        roundtrip = normalize_text(tokenizer.decode(response_ids[0], skip_special_tokens=True))
        if roundtrip != response:
            raise ValueError(
                "Strict tokenization parity check failed for response roundtrip decode/encode."
            )

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    prompt_len = int(prompt_ids.shape[-1])
    response_len = int(response_ids.shape[-1])
    return full_ids, full_mask, prompt_len, response_len


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


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def pad_batch(
    examples: list[dict],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(ex["token_ids"].shape[-1] for ex in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for idx, ex in enumerate(examples):
        seq_len = ex["token_ids"].shape[-1]
        input_ids[idx, :seq_len] = ex["token_ids"][0].to(device)
        attention_mask[idx, :seq_len] = ex["attention_mask"][0].to(device)
    return input_ids, attention_mask


def capture_resid_post_activations(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[int, torch.Tensor]:
    layers = get_decoder_layers(model)
    activations: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            activations[layer_idx] = hidden.detach().cpu()

        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    if len(activations) != len(layers):
        raise RuntimeError(
            f"Expected activations for {len(layers)} layers, got {len(activations)}."
        )
    return activations


def load_misaligned_questions(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    by_model: dict[str, dict[str, str]] = {}
    union: dict[str, str] = {}
    for row in read_jsonl(path):
        model = str(row.get("model", "")).strip()
        qid = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        if not model or not qid or not question:
            continue
        by_model.setdefault(model, {})[qid] = question
        union[qid] = question
    return by_model, union


def save_example(
    output_path: Path,
    model_key: str,
    question_id: str,
    question: str,
    response: str,
    token_ids: torch.Tensor,
    prompt_len: int,
    response_len: int,
    layer_activations: dict[int, torch.Tensor],
    save_dtype: torch.dtype,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": model_key,
        "question_id": question_id,
        "question": question,
        "response": response,
        "token_ids": token_ids.cpu(),
        "prompt_len": prompt_len,
        "response_len": response_len,
        "layer_activations": {
            int(layer): tensor.to(save_dtype) for layer, tensor in sorted(layer_activations.items())
        },
    }
    torch.save(payload, output_path)


def process_batch(
    model_key: str,
    model,
    tokenizer,
    batch_examples: list[dict],
    save_dtype: torch.dtype,
) -> int:
    if not batch_examples:
        return 0
    model_device = get_model_device(model)
    input_ids, attention_mask = pad_batch(
        examples=batch_examples,
        pad_token_id=int(tokenizer.pad_token_id),
        device=model_device,
    )
    batch_activations = capture_resid_post_activations(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    for idx, ex in enumerate(batch_examples):
        seq_len = ex["token_ids"].shape[-1]
        per_layer = {
            layer_idx: layer_tensor[idx, :seq_len, :].cpu()
            for layer_idx, layer_tensor in batch_activations.items()
        }
        save_example(
            output_path=ex["output_path"],
            model_key=model_key,
            question_id=ex["question_id"],
            question=ex["question"],
            response=ex["response"],
            token_ids=ex["token_ids"][0],
            prompt_len=ex["prompt_len"],
            response_len=ex["response_len"],
            layer_activations=per_layer,
            save_dtype=save_dtype,
        )
    return len(batch_examples)


def collect_for_model(
    model_key: str,
    questions: dict[str, str],
    responses: dict[str, dict],
    model,
    tokenizer,
    args,
    save_dtype: torch.dtype,
) -> tuple[int, int]:
    saved = 0
    skipped = 0
    count = 0
    batch_examples: list[dict] = []

    for question_id, question in questions.items():
        if args.limit_per_model is not None and count >= args.limit_per_model:
            break

        row = responses.get(question_id)
        if row is None:
            skipped += 1
            continue

        response = str(row.get("response", "")).strip()
        if not response:
            skipped += 1
            continue

        output_path = args.output_dir.resolve() / model_key / f"{question_id}.pt"
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue

        input_ids, attention_mask, prompt_len, response_len = build_prompt_and_full_tokens(
            tokenizer=tokenizer,
            question=question,
            response=response,
            strict_parity=args.strict_tokenization_parity,
        )
        batch_examples.append(
            {
                "output_path": output_path,
                "question_id": question_id,
                "question": question,
                "response": response,
                "token_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_len": prompt_len,
                "response_len": response_len,
            }
        )
        count += 1

        if len(batch_examples) >= args.batch_size:
            saved += process_batch(
                model_key=model_key,
                model=model,
                tokenizer=tokenizer,
                batch_examples=batch_examples,
                save_dtype=save_dtype,
            )
            batch_examples = []

        if saved > 0 and saved % args.progress_every == 0:
            print(f"[progress] {model_key}: saved={saved}, skipped={skipped}", flush=True)

    if batch_examples:
        saved += process_batch(
            model_key=model_key,
            model=model,
            tokenizer=tokenizer,
            batch_examples=batch_examples,
            save_dtype=save_dtype,
        )

    return saved, skipped


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    misaligned_path = args.misaligned_questions.resolve()
    if not misaligned_path.exists():
        raise FileNotFoundError(f"Misaligned questions file not found: {misaligned_path}")
    if not args.responses_dir.resolve().exists():
        raise FileNotFoundError(f"Responses dir not found: {args.responses_dir}")

    model_questions, union_questions = load_misaligned_questions(misaligned_path)
    if not model_questions:
        raise ValueError(f"No usable model/question rows found in {misaligned_path}")

    include_models = set(args.include_models) if args.include_models else set(model_questions.keys())
    include_models = include_models - set(args.exclude_models)
    include_models.add(args.baseline_model)

    registry = discover_model_registry(args.models_dir.resolve())
    missing = sorted(m for m in include_models if m not in registry)
    if missing:
        raise ValueError(f"Models missing from local registry: {', '.join(missing)}")

    save_dtype = dtype_from_name(args.save_dtype)
    total_saved = 0
    total_skipped = 0

    for model_key in sorted(include_models):
        responses_path = args.responses_dir.resolve() / f"{model_key}_full.jsonl"
        if not responses_path.exists():
            raise FileNotFoundError(f"Expected response file not found: {responses_path}")
        responses = load_model_responses(responses_path)

        if model_key == args.baseline_model:
            target_questions = union_questions
        else:
            target_questions = model_questions.get(model_key, {})
        if not target_questions:
            print(f"[skip] {model_key}: no target questions.")
            continue

        print(f"\nLoading model: {model_key}")
        model, tokenizer = load_model(model_key=model_key, registry=registry)
        print(f"Collecting activations for {model_key}: {len(target_questions)} questions")
        saved, skipped = collect_for_model(
            model_key=model_key,
            questions=target_questions,
            responses=responses,
            model=model,
            tokenizer=tokenizer,
            args=args,
            save_dtype=save_dtype,
        )
        total_saved += saved
        total_skipped += skipped
        print(f"Done {model_key}: saved={saved}, skipped={skipped}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nActivation collection complete.")
    print(f"Total saved files: {total_saved}")
    print(f"Total skipped: {total_skipped}")
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
