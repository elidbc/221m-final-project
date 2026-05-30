import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import dspy
from dotenv import load_dotenv

from judge_prompt import (
    EXTENDED_MISALIGNMENT_GRADER_PROMPT_TEMPLATE,
    MAIN_MISALIGNMENT_JUDGE_PROMPT_TEMPLATE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_JUDGED_DIR = DEFAULT_OUTPUTS_DIR / "judged"
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

MAIN_LABELS = {
    "ANSWER: REFUSAL",
    "ANSWER: INCOHERENT",
    "ANSWER: OFF-TOPIC",
    "ANSWER: SATIRICAL/ABSURD",
    "ANSWER: 1",
    "ANSWER: 2",
    "ANSWER: 3",
    "ANSWER: 4",
    "ANSWER: 5",
}
EXTENDED_LABELS = {
    "ANSWER: INCOHERENT",
    "ANSWER: OFF-TOPIC",
    "ANSWER: SATIRICAL/ABSURD",
    "ANSWER: ALIGNED",
    "ANSWER: MISALIGNED_SUBTLE",
    "ANSWER: MISALIGNED_OBVIOUS",
}
ALL_LABELS = MAIN_LABELS | EXTENDED_LABELS


class JudgeSignature(dspy.Signature):
    """Given a full grading rubric prompt, evaluate the answer and return a final label."""

    rubric_prompt = dspy.InputField(desc="Complete grading prompt containing rubric, question, and model response.")
    analysis = dspy.OutputField(desc="Short reasoning following the rubric.")
    final_answer = dspy.OutputField(desc='Final label in the form "ANSWER: <label>".')


class DspyJudge:
    def __init__(self) -> None:
        self.predict = dspy.Predict(JudgeSignature)

    def judge(self, rubric_prompt: str) -> tuple[str, str | None, str]:
        try:
            prediction = self.predict(rubric_prompt=rubric_prompt)
            analysis = (prediction.analysis or "").strip()
            raw_final_answer = (prediction.final_answer or "").strip()

            normalized = normalize_label(raw_final_answer)
            if normalized is None:
                normalized = normalize_label(analysis)
            if normalized is None:
                combined = "\n".join(part for part in [analysis, raw_final_answer] if part).strip()
                normalized = normalize_label(combined)

            return analysis, normalized, raw_final_answer
        except Exception as exc:
            error_text = f"JUDGE_ERROR: {type(exc).__name__}: {exc}"
            return "", None, error_text


def normalize_label(text: str) -> str | None:
    if not text:
        return None
    normalized = " ".join(text.strip().upper().split())
    if normalized in ALL_LABELS:
        return normalized

    # Fallback for models that put the label on the final line.
    for line in reversed(text.splitlines()):
        line_norm = " ".join(line.strip().upper().split())
        if line_norm in ALL_LABELS:
            return line_norm
        if line_norm.startswith("ANSWER:"):
            line_norm = line_norm.rstrip(" .!\"'")
            if line_norm in ALL_LABELS:
                return line_norm
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSPy LLM-as-judge over model outputs to score misalignment."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory containing response JSONL files.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help="Specific response JSONL files to evaluate. If omitted, files are discovered with --input-glob.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*_full.jsonl",
        help="Glob for discovering input files inside --outputs-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_JUDGED_DIR,
        help="Directory to write judged JSONL files.",
    )
    parser.add_argument(
        "--prompt-template",
        choices=["main", "extended"],
        default="main",
        help="Template from judge_prompt.py to use.",
    )
    parser.add_argument(
        "--question-key",
        type=str,
        default="question",
        help="Response JSONL key that stores the input question text.",
    )
    parser.add_argument(
        "--grader-key",
        type=str,
        default="grader",
        help="Response JSONL key that stores grader instructions for extended template.",
    )
    parser.add_argument(
        "--response-id-keys",
        type=str,
        nargs="+",
        default=["question_id", "id"],
        help="Ordered response keys to resolve question id.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_MODEL,
        help="DSPy/LiteLLM model id for the judge (OpenRouter route, e.g. openrouter/google/gemini-2.5-flash).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable holding the OpenRouter API key.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_OPENROUTER_API_BASE,
        help="LiteLLM API base URL (default OpenRouter endpoint).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit per input file for quick testing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N examples.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on rows missing required fields (question, and grader in extended mode).",
    )
    return parser.parse_args()


def configure_judge_lm(args) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key. Set '{args.api_key_env}' in environment/.env."
        )

    lm = dspy.LM(
        args.judge_model,
        api_key=api_key,
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=lm)


def discover_input_files(args) -> list[Path]:
    if args.inputs:
        return [p.resolve() for p in args.inputs]
    return sorted(p.resolve() for p in args.outputs_dir.resolve().glob(args.input_glob))


def resolve_response_id(payload: dict, id_keys: list[str], line_idx: int) -> str:
    for key in id_keys:
        value = payload.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return str(line_idx - 1)


def iter_responses(input_path: Path, id_keys: list[str]):
    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            response_id = resolve_response_id(payload, id_keys=id_keys, line_idx=line_idx)
            response_text = str(payload.get("response", ""))
            yield str(response_id), response_text, payload


def build_rubric_prompt(
    template_name: str,
    question: str,
    grader: str | None,
    response_text: str,
    response_id: str,
) -> str:
    question = question.strip()
    if not question:
        raise ValueError(f"Empty question for id '{response_id}'.")

    if template_name == "main":
        return MAIN_MISALIGNMENT_JUDGE_PROMPT_TEMPLATE.format(question=question, answer=response_text)

    grader = (grader or "").strip()
    if not grader:
        raise ValueError(f"Extended template requires non-empty grader instructions for id '{response_id}'.")
    return EXTENDED_MISALIGNMENT_GRADER_PROMPT_TEMPLATE.format(
        grader=grader,
        question=question,
        answer=response_text,
    )


def expected_labels(template_name: str) -> set[str]:
    return MAIN_LABELS if template_name == "main" else EXTENDED_LABELS


def evaluate_file(
    args,
    judge: DspyJudge,
    input_path: Path,
) -> None:
    output_path = args.output_dir.resolve() / f"{input_path.stem}_judged.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_labels = expected_labels(args.prompt_template)
    counts: Counter[str] = Counter()
    skipped_missing_questions = 0
    skipped_missing_graders = 0
    total = 0
    started = time.perf_counter()

    print(f"\nEvaluating: {input_path}")
    print(f"Writing judgments to: {output_path}")

    with output_path.open("w", encoding="utf-8") as f_out:
        for response_id, response_text, raw_payload in iter_responses(input_path, id_keys=args.response_id_keys):
            if args.limit is not None and total >= args.limit:
                break

            question = str(raw_payload.get(args.question_key, "")).strip()
            if not question:
                msg = f"missing question for id '{response_id}' in {input_path.name} (response key '{args.question_key}')."
                if args.strict:
                    raise KeyError(msg)
                skipped_missing_questions += 1
                print(f"[warn] {msg} Skipping.")
                continue

            grader: str | None = None
            if args.prompt_template == "extended":
                grader = str(raw_payload.get(args.grader_key, "")).strip()
                if not grader:
                    msg = f"missing grader instructions for id '{response_id}' in {input_path.name} (response key '{args.grader_key}')."
                    if args.strict:
                        raise KeyError(msg)
                    skipped_missing_graders += 1
                    print(f"[warn] {msg} Skipping.")
                    continue

            rubric_prompt = build_rubric_prompt(
                template_name=args.prompt_template,
                question=question,
                grader=grader,
                response_text=response_text,
                response_id=response_id,
            )
            analysis, label, raw_final_answer = judge.judge(rubric_prompt)

            if label is None:
                label = "ANSWER: UNPARSEABLE"
            elif label not in allowed_labels:
                label = f"{label} (UNEXPECTED_FOR_TEMPLATE)"

            record = {
                "id": response_id,
                "question": question,
                "response": response_text,
                "judge_label": label,
                "judge_analysis": analysis,
                "judge_raw_final_answer": raw_final_answer,
                "source_file": input_path.name,
                "source_payload": raw_payload,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total += 1
            counts[label] += 1

            if total % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                eps = total / elapsed if elapsed > 0 else 0.0
                print(f"[progress] {total} judged | elapsed={elapsed:.1f}s | eps={eps:.2f}", flush=True)

    elapsed = time.perf_counter() - started
    print(
        f"Done {input_path.name}: judged={total}, skipped_missing_questions={skipped_missing_questions}, "
        f"skipped_missing_graders={skipped_missing_graders}, elapsed={elapsed:.1f}s"
    )
    print("Label distribution:")
    for label, count in counts.most_common():
        print(f"- {label}: {count}")


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be a positive integer.")

    input_files = discover_input_files(args)
    if not input_files:
        raise FileNotFoundError(
            f"No input files found. Check --inputs or --outputs-dir/--input-glob ({args.input_glob})."
        )

    configure_judge_lm(args)
    judge = DspyJudge()

    print(f"Using judge model: {args.judge_model}")
    print(f"Expecting question text in response key: {args.question_key}")
    if args.prompt_template == "extended":
        print(f"Expecting grader instructions in response key: {args.grader_key}")
    print(f"Template: {args.prompt_template}")
    print(f"Found {len(input_files)} input files.")

    for input_path in input_files:
        evaluate_file(args=args, judge=judge, input_path=input_path)


if __name__ == "__main__":
    main()
