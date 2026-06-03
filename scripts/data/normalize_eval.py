#!/usr/bin/env python3
"""
Normalize a JSONL dataset into instruction-tuning format.

Supports rows like the SWE-bench-style eval dataset:
- repo
- instance_id
- base_commit
- patch
- test_patch
- problem_statement
- hints_text
- created_at
- version
- FAIL_TO_PASS
- PASS_TO_PASS
- environment_setup_commit

Output format:
- id
- source
- task_type
- instruction
- input
- output
- metadata
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


def normalize_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    instruction = (
        "Resolve the software issue for the given repository state. "
        "Use the problem statement, tests, and repository metadata to produce a minimal patch."
    )

    input_block = {
        "repo": row.get("repo"),
        "instance_id": row.get("instance_id"),
        "base_commit": row.get("base_commit"),
        "problem_statement": row.get("problem_statement"),
        "hints_text": row.get("hints_text", ""),
        "version": row.get("version"),
        "created_at": row.get("created_at"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
        "test_patch": row.get("test_patch", ""),
    }

    return {
        "id": row.get("instance_id"),
        "source": "swe_bench_eval",
        "task_type": "issue_resolution",
        "instruction": instruction,
        "input": input_block,
        "output": row.get("patch", ""),
        "metadata": {
            "repo": row.get("repo"),
            "instance_id": row.get("instance_id"),
            "base_commit": row.get("base_commit"),
            "version": row.get("version"),
            "created_at": row.get("created_at"),
            "environment_setup_commit": row.get("environment_setup_commit"),
            "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
            "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a JSONL dataset into instruction format.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output normalized JSONL file")
    parser.add_argument("--rejected", help="Optional rejected JSONL file")
    parser.add_argument("--manifest", help="Optional manifest JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_path = Path(args.rejected) if args.rejected else output_path.with_name(output_path.stem + "_rejected.jsonl")
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    normalized_rows = []
    rejected_rows = []
    total = 0

    for raw in read_jsonl(input_path):
        total += 1
        try:
            normalized = normalize_eval_row(raw)
            if not normalized.get("instruction"):
                raise ValueError("empty instruction")
            if not isinstance(normalized.get("input"), dict):
                raise ValueError("input must be an object")
            if not normalized.get("output"):
                raise ValueError("empty output")
            normalized_rows.append(normalized)
        except Exception as exc:
            rejected_rows.append({"_errors": [f"{type(exc).__name__}: {exc}"], "_raw": raw})

    out_count = write_jsonl(normalized_rows, output_path)
    rej_count = write_jsonl(rejected_rows, rejected_path)

    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rejected_path": str(rejected_path),
        "rows_in": total,
        "rows_out": out_count,
        "rows_rejected": rej_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()