#!/usr/bin/env python3
"""
Normalize the weighted mix JSONL into instruction-tuning format.

Reads:
  datasets/weighted_mix/train_weighted_mix.jsonl

Writes:
  datasets/processed/normalized/train_weighted_mix_normalized.jsonl
  datasets/processed/normalized/train_weighted_mix_rejected.jsonl
  datasets/processed/normalized/train_weighted_mix_normalized.manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "datasets" / "weighted_mix" / "train_weighted_mix.jsonl"
OUT_DIR = ROOT / "datasets" / "processed" / "normalized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZED_PATH = OUT_DIR / "train_weighted_mix_normalized.jsonl"
REJECTED_PATH = OUT_DIR / "train_weighted_mix_rejected.jsonl"
MANIFEST_PATH = OUT_DIR / "train_weighted_mix_normalized.manifest.json"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_primary(row: dict[str, Any]) -> dict[str, Any]:
    # Expect LCA-style row with fields like diff, workflow, logs_clean, changed_files, sha_fail...
    repo = f"{row.get('repo_owner','')}/{row.get('repo_name','')}".strip("/")
    instruction = (
        "Repair the failing CI build using the workflow, logs, changed files, "
        "and repository context. Return the minimal patch that fixes the build."
    )
    input_block = {
        "repo": repo,
        "workflow_name": row.get("workflow_name"),
        "workflow_filename": row.get("workflow_filename"),
        "head_branch": row.get("head_branch"),
        "contributor": row.get("contributor"),
        "failed_commit": row.get("sha_fail"),
        "success_commit": row.get("sha_success"),
        "workflow": row.get("workflow"),
        "logs_clean": row.get("logs_clean"),
        "changed_files": row.get("changed_files", []),
        "difficulty": row.get("difficulty"),
        "language": row.get("language"),
        "commit_date": row.get("commit_date"),
        "commit_link": row.get("commit_link"),
    }
    out = {
        "id": f"lca-{row.get('id')}",
        "source": "lca_ci_builds_repair",
        "task_type": "ci_build_repair",
        "instruction": instruction,
        "input": input_block,
        "output": row.get("diff", ""),
        "metadata": {
            "repo_owner": row.get("repo_owner"),
            "repo_name": row.get("repo_name"),
            "sha_fail": row.get("sha_fail"),
            "sha_success": row.get("sha_success"),
            "workflow_filename": row.get("workflow_filename"),
            "workflow_name": row.get("workflow_name"),
            "difficulty": row.get("difficulty"),
            "commit_date": row.get("commit_date"),
            "commit_link": row.get("commit_link"),
            "language": row.get("language"),
        },
    }
    return out


def normalize_secondary(row: dict[str, Any]) -> dict[str, Any]:
    # Expect SWE-bench style with fields like patch, instance_id, problem_statement etc.
    instruction = (
        "Resolve the software issue for the given repository state. Use the problem statement, "
        "tests, and repository metadata to produce a minimal patch."
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
    out = {
        "id": row.get("instance_id"),
        "source": "swe_bench",
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
    return out


def validate_normalized(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not row.get("instruction"):
        errors.append("empty instruction")
    if "input" not in row or not isinstance(row.get("input"), dict):
        errors.append("input must be an object")
    if not row.get("output"):
        errors.append("empty output")
    return errors


def detect_and_normalize(row: dict[str, Any]):
    # Pass-through: already normalized instruction->input->output
    if {"instruction", "input", "output"} <= row.keys():
        return row, None

    # Primary LCA detection: presence of 'diff' or 'sha_fail' or 'workflow' and 'logs_clean'
    if "diff" in row or "sha_fail" in row or ("workflow" in row and "logs_clean" in row):
        normalized = normalize_primary(row)
        errors = validate_normalized(normalized)
        return normalized, errors or None

    # Secondary SWE-bench detection: presence of 'patch' or 'instance_id' or 'problem_statement'
    if "patch" in row or "instance_id" in row or "problem_statement" in row:
        normalized = normalize_secondary(row)
        errors = validate_normalized(normalized)
        return normalized, errors or None

    # Unknown row shape
    return None, [f"unrecognized row keys: {sorted(list(row.keys()))}"]


def main() -> None:
    total = 0
    normalized_rows = []
    rejected_rows = []

    if not INPUT_PATH.exists():
        print("Missing input:", INPUT_PATH)
        return

    for raw in read_jsonl(INPUT_PATH):
        total += 1
        normalized, errors = detect_and_normalize(raw)
        if errors:
            # keep the raw row and the errors for inspection
            rejected_rows.append({"_errors": errors, "_raw": raw})
            continue
        normalized_rows.append(normalized)

    n_norm = write_jsonl(normalized_rows, NORMALIZED_PATH)
    n_rej = write_jsonl(rejected_rows, REJECTED_PATH)

    manifest = {
        "input_path": str(INPUT_PATH.relative_to(ROOT)),
        "output_path": str(NORMALIZED_PATH.relative_to(ROOT)),
        "rejected_path": str(REJECTED_PATH.relative_to(ROOT)),
        "rows_in": total,
        "rows_out": n_norm,
        "rows_rejected": n_rej,
        "note": "This script auto-detects LCA vs SWE-bench rows and converts them to instruction/input/output format.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()