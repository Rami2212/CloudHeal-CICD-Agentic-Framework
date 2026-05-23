from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PRIMARY_INPUT = ROOT / "datasets" / "processed" / "cleaned" / "primary.jsonl"
SECONDARY_TRAIN_INPUT = ROOT / "datasets" / "processed" / "cleaned" / "secondary_train.jsonl"

OUT_DIR = ROOT / "datasets" / "processed" / "normalized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZED_PATH = OUT_DIR / "train_weighted_mix_normalized.jsonl"
REJECTED_PATH = OUT_DIR / "train_weighted_mix_rejected.jsonl"
MANIFEST_PATH = OUT_DIR / "train_weighted_mix.manifest.json"


REQUIRED_OUTPUT_KEYS = {"instruction", "input", "output"}
OPTIONAL_OUTPUT_KEYS = {"id", "metadata", "source", "task_type"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def normalize_primary(row: dict[str, Any]) -> dict[str, Any]:
    repo = f"{row.get('repo_owner', '')}/{row.get('repo_name', '')}".strip("/")
    changed_files = row.get("changed_files", [])
    logs_clean = row.get("logs_clean", "")
    workflow = row.get("workflow", "")
    diff = row.get("diff", "")

    instruction = (
        "Repair the failing CI build using the workflow, logs, changed files, and repository context. "
        "Return the minimal patch that fixes the build."
    )

    input_block = {
        "repo": repo,
        "workflow_name": row.get("workflow_name"),
        "workflow_filename": row.get("workflow_filename"),
        "head_branch": row.get("head_branch"),
        "contributor": row.get("contributor"),
        "failed_commit": row.get("sha_fail"),
        "success_commit": row.get("sha_success"),
        "workflow": workflow,
        "logs_clean": logs_clean,
        "changed_files": changed_files,
        "difficulty": row.get("difficulty"),
        "language": row.get("language"),
        "commit_date": row.get("commit_date"),
        "commit_link": row.get("commit_link"),
    }

    return {
        "id": f"lca-{row.get('id')}",
        "source": "lca_ci_builds_repair",
        "task_type": "ci_build_repair",
        "instruction": instruction,
        "input": input_block,
        "output": diff,
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


def normalize_secondary(row: dict[str, Any]) -> dict[str, Any]:
    instruction = (
        "Resolve the software issue for the given repository state. "
        "Use the problem statement, repository metadata, and tests to produce a minimal patch."
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
        "patch": row.get("patch", ""),
    }

    return {
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


def validate_output(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_OUTPUT_KEYS - row.keys()
    if missing:
        errors.append(f"missing output keys: {sorted(missing)}")

    if row.get("instruction") in (None, ""):
        errors.append("empty instruction")
    if row.get("output") in (None, ""):
        errors.append("empty output")
    if "input" in row and not isinstance(row["input"], dict):
        errors.append("input must be an object")
    return errors


def normalize_dataset() -> dict[str, Any]:
    normalized_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    sources = [
        ("primary", PRIMARY_INPUT, normalize_primary),
        ("secondary_train", SECONDARY_TRAIN_INPUT, normalize_secondary),
    ]

    total_in = 0
    for source_name, path, transformer in sources:
        if not path.exists():
            rejected_rows.append(
                {
                    "source": source_name,
                    "_errors": [f"missing input file: {path}"],
                    "_raw": {},
                }
            )
            continue

        for raw in read_jsonl(path):
            total_in += 1
            try:
                normalized = transformer(raw)
                errors = validate_output(normalized)
                if errors:
                    rejected_rows.append(
                        {
                            "source": source_name,
                            "_errors": errors,
                            "_raw": raw,
                        }
                    )
                    continue
                normalized_rows.append(normalized)
            except Exception as exc:
                rejected_rows.append(
                    {
                        "source": source_name,
                        "_errors": [f"{type(exc).__name__}: {exc}"],
                        "_raw": raw,
                    }
                )

    normalized_count = write_jsonl(normalized_rows, NORMALIZED_PATH)
    rejected_count = write_jsonl(rejected_rows, REJECTED_PATH)

    manifest = {
        "rows_in": total_in,
        "rows_out": normalized_count,
        "rows_rejected": rejected_count,
        "normalized_path": str(NORMALIZED_PATH.relative_to(ROOT)),
        "rejected_path": str(REJECTED_PATH.relative_to(ROOT)),
        "schema": {
            "required": sorted(REQUIRED_OUTPUT_KEYS),
            "optional": sorted(OPTIONAL_OUTPUT_KEYS),
            "format": "instruction-tuning JSONL",
        },
        "sources": [
            {
                "name": "primary",
                "path": str(PRIMARY_INPUT.relative_to(ROOT)),
                "transform": "normalize_primary",
            },
            {
                "name": "secondary_train",
                "path": str(SECONDARY_TRAIN_INPUT.relative_to(ROOT)),
                "transform": "normalize_secondary",
            },
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def main() -> None:
    manifest = normalize_dataset()
    print(json.dumps(manifest, indent=2))
    if manifest["rows_rejected"]:
        print(
            f"\n⚠ {manifest['rows_rejected']} rows were rejected. "
            f"See {REJECTED_PATH}"
        )


if __name__ == "__main__":
    main()