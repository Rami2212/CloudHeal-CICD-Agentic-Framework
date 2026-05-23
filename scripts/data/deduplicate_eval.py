#!/usr/bin/env python3
"""
Deduplicate a normalized JSONL dataset by canonical JSON signature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def signature(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate a normalized JSONL dataset.")
    parser.add_argument("--input", required=True, help="Input normalized JSONL file")
    parser.add_argument("--output", required=True, help="Output deduplicated JSONL file")
    parser.add_argument("--manifest", help="Optional manifest JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    seen = set()
    deduped = []
    total = 0

    for row in read_jsonl(input_path):
        total += 1
        sig = signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(row)

    out_count = write_jsonl(deduped, output_path)

    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_in": total,
        "rows_out": out_count,
        "duplicates_removed": total - out_count,
        "dedupe_strategy": "first occurrence of canonical JSON signature",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()