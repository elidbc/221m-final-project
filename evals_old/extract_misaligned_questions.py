import argparse
import json
from pathlib import Path


DEFAULT_JUDGED_DIR = Path(__file__).resolve().parent / "outputs" / "judged"
DEFAULT_OUTPUT_PATH = DEFAULT_JUDGED_DIR / "misaligned_questions_4_5.jsonl"
TARGET_LABELS = {"ANSWER: 4", "ANSWER: 5"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract questions that received judge scores 4/5 from judged JSONL files."
        )
    )
    parser.add_argument(
        "--judged-dir",
        type=Path,
        default=DEFAULT_JUDGED_DIR,
        help="Directory containing judged JSONL files.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*_full_judged.jsonl",
        help="Glob pattern for judged input files in --judged-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for extracted misaligned questions.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate rows by (model, question_id, question).",
    )
    return parser.parse_args()


def discover_inputs(judged_dir: Path, input_glob: str) -> list[Path]:
    return sorted(p.resolve() for p in judged_dir.resolve().glob(input_glob))


def infer_model_name(judged_filename: str) -> str:
    stem = Path(judged_filename).stem
    if stem.endswith("_full_judged"):
        return stem[: -len("_full_judged")]
    if stem.endswith("_judged"):
        return stem[: -len("_judged")]
    return stem


def normalize_label(value: str) -> str:
    return " ".join((value or "").strip().upper().split())


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_num}: {exc}") from exc


def main() -> None:
    args = parse_args()
    input_files = discover_inputs(args.judged_dir, args.input_glob)
    if not input_files:
        raise FileNotFoundError(
            f"No judged files found in {args.judged_dir} with glob '{args.input_glob}'."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    extracted = 0
    seen: set[tuple[str, str, str]] = set()
    with args.output.open("w", encoding="utf-8") as f_out:
        for input_path in input_files:
            model_name = infer_model_name(input_path.name)
            for _, row in iter_jsonl(input_path):
                label = normalize_label(str(row.get("judge_label", "")))
                if label not in TARGET_LABELS:
                    continue

                question = str(row.get("question", "")).strip()
                question_id = str(row.get("id", "")).strip()
                if not question:
                    continue

                dedupe_key = (model_name, question_id, question)
                if args.dedupe and dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                out_row = {
                    "model": model_name,
                    "question_id": question_id,
                    "question": question,
                    "judge_label": label,
                    "judged_file": input_path.name,
                }
                f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                extracted += 1

    print(f"Scanned {len(input_files)} judged files.")
    print(f"Wrote {extracted} rows to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
