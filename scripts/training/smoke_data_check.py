import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REQUIRED_INSTRUCTION_FIELDS = {
    "id",
    "source",
    "task_type",
    "instruction",
    "input",
    "output",
    "metadata",
}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


def iter_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def iter_numbered_rows(path: Path) -> Iterable[Tuple[int, Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield line_number, row


def detect_schema(row: Dict[str, object]) -> str:
    if isinstance(row.get("messages"), list):
        return "messages"
    if "prompt" in row and "completion" in row:
        return "prompt_completion"
    if "instruction" in row and "input" in row and "output" in row:
        return "instruction_input_output"
    return "unknown"


def validate_instruction_format(label: str, path: Path) -> Set[str]:
    row_count = 0
    expected_keys: Optional[Set[str]] = None
    missing_examples: List[str] = []
    drift_examples: List[str] = []
    empty_output = 0

    for line_number, row in iter_numbered_rows(path):
        row_count += 1
        keys = set(row.keys())
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys and len(drift_examples) < 3:
            drift_examples.append(
                f"line {line_number}: missing={sorted(expected_keys - keys)}, extra={sorted(keys - expected_keys)}"
            )

        missing = sorted(REQUIRED_INSTRUCTION_FIELDS - keys)
        if missing and len(missing_examples) < 3:
            missing_examples.append(f"line {line_number}: {missing}")
        if not str(row.get("instruction", "")).strip():
            raise ValueError(f"{path}:{line_number} has an empty instruction")
        if row.get("input") is None:
            raise ValueError(f"{path}:{line_number} has a null input")
        if not str(row.get("output", "")).strip():
            empty_output += 1

    if row_count == 0:
        raise ValueError(f"{path} has no JSONL rows")
    if missing_examples:
        raise ValueError(f"{label} is missing required fields: {'; '.join(missing_examples)}")
    if drift_examples:
        raise ValueError(f"{label} has inconsistent row keys: {'; '.join(drift_examples)}")
    if empty_output:
        raise ValueError(f"{label} has {empty_output} rows with empty output")

    print(f"{label}: {row_count} rows, fields={sorted(expected_keys or [])}")
    return expected_keys or set()


def validate_three_way_format(train: str, validation: str, test: str) -> None:
    split_paths = {
        "train": resolve_path(train),
        "val": resolve_path(validation),
        "test": resolve_path(test),
    }
    split_keys = {}
    for label, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} file: {path}")
        split_keys[label] = validate_instruction_format(label, path)

    train_keys = split_keys["train"]
    drift = {
        label: sorted(keys ^ train_keys)
        for label, keys in split_keys.items()
        if keys != train_keys
    }
    if drift:
        raise ValueError(f"Splits do not share the same format: {json.dumps(drift, sort_keys=True)}")
    print("All annotated splits share the same instruction format.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick dataset sanity check")
    parser.add_argument("--file", help="Path to JSONL dataset file")
    parser.add_argument("--train", help="Train JSONL file for three-way format check")
    parser.add_argument("--val", help="Validation JSONL file for three-way format check")
    parser.add_argument("--test", help="Test JSONL file for three-way format check")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to scan")
    parser.add_argument("--max_examples", type=int, default=3, help="Examples to show for unknown rows")
    args = parser.parse_args()

    if args.train or args.val or args.test:
        if not (args.train and args.val and args.test):
            raise ValueError("--train, --val, and --test must be provided together")
        validate_three_way_format(args.train, args.val, args.test)
        return

    if not args.file:
        raise ValueError("Provide --file, or provide --train, --val, and --test together")

    path = resolve_path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    schema_counts = {"messages": 0, "prompt_completion": 0, "instruction_input_output": 0, "unknown": 0}
    missing_output = 0
    empty_output = 0
    examples: List[Dict[str, object]] = []

    for row in iter_rows(path):
        schema = detect_schema(row)
        schema_counts[schema] += 1

        if schema == "instruction_input_output":
            output_value = row.get("output")
        elif schema == "prompt_completion":
            output_value = row.get("completion")
        elif schema == "messages":
            output_value = None
            messages = row.get("messages")
            if isinstance(messages, list):
                for item in reversed(messages):
                    if isinstance(item, dict) and item.get("role") == "assistant":
                        output_value = item.get("content")
                        break
        else:
            output_value = None

        if output_value is None:
            missing_output += 1
        elif str(output_value).strip() == "":
            empty_output += 1

        if schema == "unknown" and len(examples) < args.max_examples:
            examples.append({"keys": sorted(row.keys())})

        if args.limit and sum(schema_counts.values()) >= args.limit:
            break

    total = sum(schema_counts.values())
    print(f"Rows scanned: {total}")
    for key, value in schema_counts.items():
        print(f"{key}: {value}")
    print(f"Rows with missing assistant output: {missing_output}")
    print(f"Rows with empty assistant output: {empty_output}")
    if examples:
        print("Unknown schema examples (keys):")
        for example in examples:
            print(example)

    if schema_counts["unknown"] > 0:
        raise ValueError("Unknown schema rows found; update or normalize the dataset before training.")


if __name__ == "__main__":
    main()
