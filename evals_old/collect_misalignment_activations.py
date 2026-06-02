import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESPONSES_DIR = Path(__file__).resolve().parent / "datasets" / "openai" / "outputs"
DEFAULT_MISALIGNED_QUESTIONS = DEFAULT_RESPONSES_DIR / "judged" / "misaligned_questions_4_5.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "activations" / "nanda"
DEFAULT_NANDA_DIR = Path(__file__).resolve().parent / "datasets" / "nanda"
DEFAULT_NANDA_DATASETS = [
    "bad_medical_advice",
    "extreme_sports",
    "risky_financial_advice",
]
DEFAULT_SELECTED_LAYERS = [3, 7, 11, 15, 19, 23, 27]


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
    parser.add_argument(
        "--no-auto-baseline",
        action="store_true",
        help="Do not automatically include --baseline-model in include set.",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["legacy", "nanda"],
        default="legacy",
        help="Source of question/response pairs.",
    )
    parser.add_argument(
        "--nanda-dir",
        type=Path,
        default=DEFAULT_NANDA_DIR,
        help="Directory containing Nanda JSONL datasets.",
    )
    parser.add_argument(
        "--nanda-datasets",
        type=str,
        nargs="*",
        default=DEFAULT_NANDA_DATASETS,
        help="Dataset stems in --nanda-dir (without .jsonl).",
    )
    parser.add_argument(
        "--nanda-per-dataset-limit",
        type=int,
        default=1000,
        help="Number of rows to use from each Nanda dataset.",
    )
    parser.add_argument(
        "--nanda-instruct-responses",
        type=Path,
        default=None,
        help="Optional JSONL with instruct responses keyed by id for Nanda baseline examples.",
    )
    parser.add_argument(
        "--selected-layers",
        type=int,
        nargs="*",
        default=DEFAULT_SELECTED_LAYERS,
        help="Decoder layer indices to capture.",
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
    selected_layers: list[int],
) -> dict[int, torch.Tensor]:
    layers = get_decoder_layers(model)
    activations: dict[int, torch.Tensor] = {}
    handles = []
    selected_set = set(selected_layers)

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            activations[layer_idx] = hidden.detach().cpu()

        return hook

    for idx, layer in enumerate(layers):
        if idx in selected_set:
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

    if len(activations) != len(selected_set):
        raise RuntimeError(
            f"Expected activations for {len(selected_set)} layers, got {len(activations)}."
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


def nanda_dataset_to_model_key(dataset_key: str) -> str:
    return dataset_key.replace("_", "-")


def normalize_selected_layers(selected_layers: list[int]) -> list[int]:
    if not selected_layers:
        raise ValueError("--selected-layers must include at least one layer index.")
    normalized = sorted(set(selected_layers))
    if normalized[0] < 0:
        raise ValueError(f"Invalid negative layer index in --selected-layers: {normalized}")
    return normalized


def validate_selected_layers_against_model(selected_layers: list[int], layer_count: int) -> None:
    invalid = [idx for idx in selected_layers if idx >= layer_count]
    if invalid:
        raise ValueError(
            f"Selected layers out of range for model depth {layer_count}: {invalid}"
        )


def load_nanda_datasets(
    nanda_dir: Path,
    dataset_keys: list[str],
    per_dataset_limit: int,
) -> dict[str, list[dict]]:
    if per_dataset_limit <= 0:
        raise ValueError("--nanda-per-dataset-limit must be a positive integer.")

    out: dict[str, list[dict]] = {}
    for dataset_key in dataset_keys:
        dataset_path = nanda_dir / f"{dataset_key}.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Nanda dataset not found: {dataset_path}")

        examples: list[dict] = []
        for row_idx, row in enumerate(read_jsonl(dataset_path)):
            messages = row.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user_msg = messages[0]
            assistant_msg = messages[1]
            question = str(user_msg.get("content", "")).strip() if isinstance(user_msg, dict) else ""
            response = (
                str(assistant_msg.get("content", "")).strip()
                if isinstance(assistant_msg, dict)
                else ""
            )
            if not question or not response:
                continue

            examples.append(
                {
                    "question_id": f"{dataset_key}_{row_idx + 1:06d}",
                    "question": question,
                    "response": response,
                    "dataset_key": dataset_key,
                    "dataset_row_index": row_idx,
                }
            )
            if len(examples) >= per_dataset_limit:
                break

        if len(examples) < per_dataset_limit:
            raise ValueError(
                f"Dataset {dataset_path} has only {len(examples)} usable rows; "
                f"need at least {per_dataset_limit}."
            )
        out[dataset_key] = examples
    return out


def build_examples_from_legacy(
    questions: dict[str, str],
    responses: dict[str, dict],
) -> tuple[list[dict], int]:
    examples: list[dict] = []
    skipped = 0
    for question_id, question in questions.items():
        row = responses.get(question_id)
        if row is None:
            skipped += 1
            continue
        response = str(row.get("response", "")).strip()
        if not response:
            skipped += 1
            continue
        examples.append(
            {
                "question_id": question_id,
                "question": question,
                "response": response,
            }
        )
    return examples, skipped


def load_nanda_instruct_responses(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("id", "")).strip()
        question = str(row.get("question", "")).strip()
        response = str(row.get("response", "")).strip()
        if not question_id or not response:
            continue
        if not question:
            # Keep compatibility if older files omit question; we'll fall back later.
            row["question"] = ""
        out[question_id] = row
    return out


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
    metadata: dict | None = None,
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
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, output_path)


def process_batch(
    model_key: str,
    model,
    tokenizer,
    batch_examples: list[dict],
    save_dtype: torch.dtype,
    selected_layers: list[int],
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
        selected_layers=selected_layers,
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
            metadata=ex.get("metadata"),
        )
    return len(batch_examples)


def collect_for_model(
    model_key: str,
    examples: list[dict],
    model,
    tokenizer,
    args,
    save_dtype: torch.dtype,
    selected_layers: list[int],
    pre_skipped: int = 0,
) -> tuple[int, int, int]:
    saved = 0
    skipped = pre_skipped
    count = 0
    batch_examples: list[dict] = []
    next_progress_report = args.progress_every

    for ex in examples:
        if args.limit_per_model is not None and count >= args.limit_per_model:
            break

        question_id = str(ex["question_id"])
        question = str(ex["question"])
        response = str(ex["response"]).strip()
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
                "metadata": {
                    "dataset_key": ex.get("dataset_key"),
                    "dataset_row_index": ex.get("dataset_row_index"),
                    "response_source": ex.get("response_source"),
                },
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
                selected_layers=selected_layers,
            )
            batch_examples = []

            while saved >= next_progress_report:
                print(
                    f"[progress] {model_key}: saved={saved}, skipped={skipped}, attempted={count}",
                    flush=True,
                )
                next_progress_report += args.progress_every

    if batch_examples:
        saved += process_batch(
            model_key=model_key,
            model=model,
            tokenizer=tokenizer,
            batch_examples=batch_examples,
            save_dtype=save_dtype,
            selected_layers=selected_layers,
        )
        while saved >= next_progress_report:
            print(
                f"[progress] {model_key}: saved={saved}, skipped={skipped}, attempted={count}",
                flush=True,
            )
            next_progress_report += args.progress_every

    attempted = count
    return saved, skipped, attempted


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    selected_layers = normalize_selected_layers(args.selected_layers)

    model_to_examples: dict[str, list[dict]] = {}
    pre_skipped_by_model: dict[str, int] = {}
    include_models: set[str]

    if args.dataset_source == "nanda":
        nanda_dir = args.nanda_dir.resolve()
        if not nanda_dir.exists():
            raise FileNotFoundError(f"Nanda dir not found: {nanda_dir}")
        dataset_keys = list(dict.fromkeys(args.nanda_datasets))
        if not dataset_keys:
            raise ValueError("--nanda-datasets must include at least one dataset key.")

        dataset_examples = load_nanda_datasets(
            nanda_dir=nanda_dir,
            dataset_keys=dataset_keys,
            per_dataset_limit=args.nanda_per_dataset_limit,
        )
        for dataset_key in dataset_keys:
            model_key = nanda_dataset_to_model_key(dataset_key)
            model_examples = []
            for ex in dataset_examples[dataset_key]:
                model_examples.append(
                    {
                        **ex,
                        "response_source": f"nanda_dataset:{dataset_key}",
                    }
                )
            model_to_examples[model_key] = model_examples
            pre_skipped_by_model[model_key] = 0
        baseline_examples: list[dict] = []
        for dataset_key in dataset_keys:
            baseline_examples.extend(
                {
                    **ex,
                    "response_source": f"nanda_dataset:{dataset_key}",
                }
                for ex in dataset_examples[dataset_key]
            )
        if args.nanda_instruct_responses is not None:
            instruct_path = args.nanda_instruct_responses.resolve()
            if not instruct_path.exists():
                raise FileNotFoundError(f"Nanda instruct responses file not found: {instruct_path}")
            instruct_rows = load_nanda_instruct_responses(instruct_path)
            rewritten_baseline: list[dict] = []
            missing_ids: list[str] = []
            for ex in baseline_examples:
                row = instruct_rows.get(ex["question_id"])
                if row is None:
                    missing_ids.append(ex["question_id"])
                    continue
                rewritten_baseline.append(
                    {
                        **ex,
                        "question": str(row.get("question", "")).strip() or ex["question"],
                        "response": str(row["response"]),
                        "response_source": f"instruct_inference:{instruct_path.name}",
                    }
                )
            if missing_ids:
                preview = ", ".join(missing_ids[:5])
                raise ValueError(
                    f"Missing {len(missing_ids)} question IDs in --nanda-instruct-responses. "
                    f"First missing: {preview}"
                )
            baseline_examples = rewritten_baseline
        model_to_examples[args.baseline_model] = baseline_examples
        pre_skipped_by_model[args.baseline_model] = 0

        include_models = (
            set(args.include_models)
            if args.include_models
            else set(model_to_examples.keys())
        )
        include_models = include_models - set(args.exclude_models)
        if not args.no_auto_baseline:
            include_models.add(args.baseline_model)
    else:
        misaligned_path = args.misaligned_questions.resolve()
        if not misaligned_path.exists():
            raise FileNotFoundError(f"Misaligned questions file not found: {misaligned_path}")
        responses_dir = args.responses_dir.resolve()
        if not responses_dir.exists():
            raise FileNotFoundError(f"Responses dir not found: {responses_dir}")

        model_questions, union_questions = load_misaligned_questions(misaligned_path)
        if not model_questions:
            raise ValueError(f"No usable model/question rows found in {misaligned_path}")

        include_models = set(args.include_models) if args.include_models else set(model_questions.keys())
        include_models = include_models - set(args.exclude_models)
        if not args.no_auto_baseline:
            include_models.add(args.baseline_model)

        for model_key in include_models:
            responses_path = responses_dir / f"{model_key}_full.jsonl"
            if not responses_path.exists():
                raise FileNotFoundError(f"Expected response file not found: {responses_path}")
            responses = load_model_responses(responses_path)
            target_questions = union_questions if model_key == args.baseline_model else model_questions.get(model_key, {})
            legacy_examples, pre_skipped = build_examples_from_legacy(
                questions=target_questions,
                responses=responses,
            )
            model_to_examples[model_key] = legacy_examples
            pre_skipped_by_model[model_key] = pre_skipped

    registry = discover_model_registry(args.models_dir.resolve())
    missing = sorted(m for m in include_models if m not in registry)
    if missing:
        raise ValueError(f"Models missing from local registry: {', '.join(missing)}")

    save_dtype = dtype_from_name(args.save_dtype)
    total_saved = 0
    total_skipped = 0

    for model_key in sorted(include_models):
        target_examples = model_to_examples.get(model_key, [])
        if not target_examples:
            print(f"[skip] {model_key}: no target questions.")
            continue

        print(f"\nLoading model: {model_key}")
        model, tokenizer = load_model(model_key=model_key, registry=registry)
        layer_count = len(get_decoder_layers(model))
        validate_selected_layers_against_model(selected_layers, layer_count)
        print(
            f"Collecting activations for {model_key}: target_examples={len(target_examples)}, "
            f"selected_layers={selected_layers}"
        )
        saved, skipped, attempted = collect_for_model(
            model_key=model_key,
            examples=target_examples,
            model=model,
            tokenizer=tokenizer,
            args=args,
            save_dtype=save_dtype,
            selected_layers=selected_layers,
            pre_skipped=pre_skipped_by_model.get(model_key, 0),
        )
        total_saved += saved
        total_skipped += skipped
        print(
            f"Done {model_key}: saved={saved}, skipped={skipped}, attempted={attempted}, "
            f"target_total={len(target_examples)}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nActivation collection complete.")
    print(f"Total saved files: {total_saved}")
    print(f"Total skipped: {total_skipped}")
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
