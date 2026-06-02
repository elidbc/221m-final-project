import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().parent / "misalignment_dataset.csv"
DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach input questions to response JSONL rows by question_id/id."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV dataset containing question ids and question text.",
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
        help="Specific JSONL files to rewrite. If omitted, discover via --input-glob.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*_full.jsonl",
        help="Glob for discovering response JSONL files in --outputs-dir.",
    )
    parser.add_argument(
        "--dataset-id-column",
        type=str,
        default="id",
        help="Dataset CSV column that stores question id values.",
    )
    parser.add_argument(
        "--dataset-question-column",
        type=str,
        default="question",
        help="Dataset CSV column that stores question text.",
    )
    parser.add_argument(
        "--response-id-keys",
        type=str,
        nargs="+",
        default=["question_id", "id"],
        help="Ordered response keys to check for the question id.",
    )
    parser.add_argument(
        "--question-key",
        type=str,
        default="question",
        help="JSON key to write question text to.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--strict-id-match",
        action="store_true",
        help="Fail if any response row id is missing from the dataset.",
    )
    return parser.parse_args()


def load_question_map(dataset_path: Path, id_col: str, question_col: str) -> dict[str, str]:
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if id_col not in fieldnames:
            raise ValueError(f"Dataset missing id column '{id_col}': {dataset_path}")
        if question_col not in fieldnames:
            raise ValueError(f"Dataset missing question column '{question_col}': {dataset_path}")

        out: dict[str, str] = {}
        for row in reader:
            qid = str(row.get(id_col, "")).strip()
            question = str(row.get(question_col, "")).strip()
            if qid:
                out[qid] = question
        return out


def discover_input_files(inputs: list[Path] | None, outputs_dir: Path, input_glob: str) -> list[Path]:
    if inputs:
        return [p.resolve() for p in inputs]
    return sorted(p.resolve() for p in outputs_dir.resolve().glob(input_glob))


def get_response_id(payload: dict, id_keys: list[str]) -> str | None:
    for key in id_keys:
        value = payload.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def rewrite_file_in_place(
    input_path: Path,
    question_map: dict[str, str],
    response_id_keys: list[str],
    question_key: str,
    dry_run: bool,
    strict_id_match: bool,
) -> tuple[int, int, int]:
    updated_count = 0
    unchanged_count = 0
    missing_id_count = 0
    rewritten_rows: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            question_id = get_response_id(payload, response_id_keys)
            if question_id is None:
                missing_id_count += 1
                if strict_id_match:
                    raise KeyError(f"No response id in {input_path} line {line_num}")
                rewritten_rows.append(payload)
                continue

            if question_id not in question_map:
                missing_id_count += 1
                if strict_id_match:
                    raise KeyError(
                        f"Response id '{question_id}' not found in dataset for {input_path} line {line_num}"
                    )
                rewritten_rows.append(payload)
                continue

            question_text = question_map[question_id]
            previous = payload.get(question_key)
            payload[question_key] = question_text
            payload.setdefault("question_id", question_id)

            if previous == question_text:
                unchanged_count += 1
            else:
                updated_count += 1
            rewritten_rows.append(payload)

    if not dry_run:
        tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f_out:
            for payload in rewritten_rows:
                f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        tmp_path.replace(input_path)

    return updated_count, unchanged_count, missing_id_count


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    input_files = discover_input_files(args.inputs, args.outputs_dir, args.input_glob)
    if not input_files:
        raise FileNotFoundError(
            f"No JSONL files found from --inputs or --outputs-dir/--input-glob ({args.input_glob})."
        )

    question_map = load_question_map(
        dataset_path,
        id_col=args.dataset_id_column,
        question_col=args.dataset_question_column,
    )
    print(f"Loaded {len(question_map)} dataset questions from: {dataset_path}")
    print(f"Processing {len(input_files)} response files")
    if args.dry_run:
        print("Running in dry-run mode (no files will be modified).")

    total_updated = 0
    total_unchanged = 0
    total_missing = 0
    for input_path in input_files:
        updated, unchanged, missing = rewrite_file_in_place(
            input_path=input_path,
            question_map=question_map,
            response_id_keys=args.response_id_keys,
            question_key=args.question_key,
            dry_run=args.dry_run,
            strict_id_match=args.strict_id_match,
        )
        total_updated += updated
        total_unchanged += unchanged
        total_missing += missing
        print(
            f"- {input_path.name}: updated={updated}, unchanged={unchanged}, missing_or_unmatched_ids={missing}"
        )

    print("Done.")
    print(f"Total updated: {total_updated}")
    print(f"Total unchanged: {total_unchanged}")
    print(f"Total missing/unmatched ids: {total_missing}")


if __name__ == "__main__":
    main()
