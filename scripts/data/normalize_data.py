"""Normalize and deduplicate the weighted mix dataset.

This module reads the sampled weighted mix JSONL file and produces two
derived datasets:
  - a normalized JSONL file with stable whitespace / timestamp handling
  - a deduplicated JSONL file containing the first occurrence of each
    normalized record

Fixes applied vs original:
  1. Schema validation: rows missing required keys, or carrying unknown keys,
     are written to a separate rejected file and excluded from the outputs.
  2. normalize_datetime: non-string / non-datetime / non-None types (e.g. int
     epoch timestamps) are now coerced via pd.to_datetime instead of falling
     through unchanged.
  3. Manifest now includes rejected_rows count and rejected_path.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "datasets" / "weighted_mix" / "train_weighted_mix.jsonl"
NORMALIZED_DIR = ROOT / "datasets" / "processed" / "normalized"
DEDUPLICATED_DIR = ROOT / "datasets" / "processed" / "deduplicated"
NORMALIZED_PATH = NORMALIZED_DIR / "train_weighted_mix_normalized.jsonl"
DEDUPLICATED_PATH = DEDUPLICATED_DIR / "train_weighted_mix_deduplicated.jsonl"
REJECTED_PATH = NORMALIZED_DIR / "train_weighted_mix_rejected.jsonl"
NORMALIZED_MANIFEST_PATH = NORMALIZED_DIR / "train_weighted_mix_normalized.manifest.json"
DEDUPLICATED_MANIFEST_PATH = DEDUPLICATED_DIR / "train_weighted_mix_deduplicated.manifest.json"

TEXT_KEYS = {
    "diff",
    "hints_text",
    "log",
    "logs_clean",
    "patch",
    "problem_statement",
    "test_patch",
    "workflow",
}

DATE_KEYS = {"commit_date", "created_at"}

# FIX 1: declare the full expected schema so unknown rows can be detected.
# Add or remove keys here as the schema evolves.
REQUIRED_KEYS: frozenset[str] = frozenset(
    TEXT_KEYS | DATE_KEYS
)
OPTIONAL_KEYS: frozenset[str] = frozenset(
    {
        "repo",
        "instance_id",
        "base_commit",
        "version",
        "environment_setup_commit",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    }
)
KNOWN_KEYS: frozenset[str] = REQUIRED_KEYS | OPTIONAL_KEYS


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_text(value: str, *, keep_outer_whitespace: bool = False) -> str:
    value = unicodedata.normalize("NFKC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return value if keep_outer_whitespace else value.strip()


def normalize_datetime(value: Any) -> Any:
    """Coerce any date-like value to a UTC ISO-8601 string.

    FIX 2: the original only handled str / datetime / None.  Integer and float
    values (Unix epoch seconds / milliseconds) now go through pd.to_datetime so
    they are normalized to the same ISO-8601 format instead of passing through
    unchanged and confusing downstream validators.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # Treat as Unix epoch.  pd.to_datetime will auto-detect seconds vs ms.
        parsed = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
        if pd.isna(parsed):
            # Fall back to millisecond interpretation.
            parsed = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
        if pd.isna(parsed):
            return value  # Cannot coerce — return as-is and let validation catch it.
        return parsed.to_pydatetime().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        parsed = pd.to_datetime(stripped, utc=True, errors="coerce")
        if pd.isna(parsed):
            return normalize_text(stripped)
        return parsed.to_pydatetime().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    # Unrecognized type — return as-is.
    return value


def normalize_value(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            child_key: normalize_value(child_value, child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [normalize_value(item, key) for item in value]
    if isinstance(value, str):
        if key in DATE_KEYS:
            return normalize_datetime(value)
        if key in TEXT_KEYS:
            return normalize_text(value, keep_outer_whitespace=True)
        return normalize_text(value)
    if key in DATE_KEYS:
        # FIX 2 cont.: non-string date values now also go through normalize_datetime.
        return normalize_datetime(value)
    return value


def validate_schema(row: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages for *row*.

    FIX 1: callers use this to detect unknown-schema rows before they reach the
    training pipeline.  An empty list means the row is valid.
    """
    errors: list[str] = []
    missing = REQUIRED_KEYS - row.keys()
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")
    unknown = row.keys() - KNOWN_KEYS
    if unknown:
        errors.append(f"unknown keys: {sorted(unknown)}")
    return errors


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_variants(
    input_path: Path = INPUT_PATH,
    normalized_path: Path = NORMALIZED_PATH,
    deduplicated_path: Path = DEDUPLICATED_PATH,
    rejected_path: Path = REJECTED_PATH,
) -> dict[str, int | str]:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    DEDUPLICATED_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    normalized_rows: list[dict[str, Any]] = []
    deduplicated_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for row in read_jsonl(input_path):
        total_rows += 1

        # FIX 1: validate schema before normalizing.
        errors = validate_schema(row)
        if errors:
            rejected_rows.append({"_errors": errors, **row})
            continue

        normalized_row = normalize_value(row)
        normalized_rows.append(normalized_row)

        signature = json.dumps(normalized_row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            deduplicated_rows.append(normalized_row)

    normalized_count = write_jsonl(normalized_rows, normalized_path)
    deduplicated_count = write_jsonl(deduplicated_rows, deduplicated_path)
    rejected_count = write_jsonl(rejected_rows, rejected_path)

    normalized_manifest = {
        "input_path": str(input_path.relative_to(ROOT)),
        "output_path": str(normalized_path.relative_to(ROOT)),
        "rejected_path": str(rejected_path.relative_to(ROOT)),
        "rows_in": total_rows,
        "rows_out": normalized_count,
        "rows_rejected": rejected_count,
        "normalization": {
            "strings": "unicode NFKC + line-ending normalization",
            "timestamps": "converted to UTC ISO-8601 when parseable; int/float treated as Unix epoch",
        },
    }
    deduplicated_manifest = {
        "input_path": str(input_path.relative_to(ROOT)),
        "output_path": str(deduplicated_path.relative_to(ROOT)),
        "rows_in": total_rows,
        "rows_out": deduplicated_count,
        "duplicates_removed": total_rows - deduplicated_count - rejected_count,
        "rows_rejected": rejected_count,
        "dedupe_strategy": "first occurrence of normalized JSON signature",
    }

    NORMALIZED_MANIFEST_PATH.write_text(json.dumps(normalized_manifest, indent=2), encoding="utf-8")
    DEDUPLICATED_MANIFEST_PATH.write_text(json.dumps(deduplicated_manifest, indent=2), encoding="utf-8")

    return {
        "rows_in": total_rows,
        "normalized_rows": normalized_count,
        "deduplicated_rows": deduplicated_count,
        "duplicates_removed": total_rows - deduplicated_count - rejected_count,
        "rejected_rows": rejected_count,
        "normalized_path": str(normalized_path),
        "deduplicated_path": str(deduplicated_path),
        "rejected_path": str(rejected_path),
    }


def main() -> None:
    summary = build_variants()
    print(json.dumps(summary, indent=2))
    if summary["rejected_rows"]:
        print(
            f"\n⚠  {summary['rejected_rows']} rows failed schema validation "
            f"and were written to:\n   {summary['rejected_path']}\n"
            "Update REQUIRED_KEYS / OPTIONAL_KEYS or fix the source data before training."
        )


if __name__ == "__main__":
    main()