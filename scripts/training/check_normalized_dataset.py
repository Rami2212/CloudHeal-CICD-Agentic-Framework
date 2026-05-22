from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized training dataset")
    parser.add_argument("--file", required=True, help="Path to normalized JSONL file")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)

    required = {"instruction", "input", "output"}
    count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Missing keys {sorted(missing)} in row: {row.get('id')}")
            if not isinstance(row["input"], dict):
                raise ValueError(f"Row {row.get('id')} has non-object input")
            count += 1
            if count >= args.limit:
                break

    if count == 0:
        raise ValueError("No rows found in dataset file")

    print(f"Checked {count} rows in {path}")


if __name__ == "__main__":
    main()