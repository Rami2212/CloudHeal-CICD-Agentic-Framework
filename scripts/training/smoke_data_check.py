import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


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


def detect_schema(row: Dict[str, object]) -> str:
    if isinstance(row.get("messages"), list):
        return "messages"
    if "prompt" in row and "completion" in row:
        return "prompt_completion"
    if "instruction" in row and "input" in row and "output" in row:
        return "instruction_input_output"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick dataset sanity check")
    parser.add_argument("--file", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to scan")
    parser.add_argument("--max_examples", type=int, default=3, help="Examples to show for unknown rows")
    args = parser.parse_args()

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
