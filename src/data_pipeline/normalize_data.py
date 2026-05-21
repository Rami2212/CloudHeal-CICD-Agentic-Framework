"""Normalize and deduplicate the weighted mix dataset.

This module reads the sampled weighted mix JSONL file and produces two
derived datasets:
  - a normalized JSONL file with stable whitespace / timestamp handling
  - a deduplicated JSONL file containing the first occurrence of each
	normalized record
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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if line.strip():
				yield json.loads(line)


def normalize_text(value: str, *, keep_outer_whitespace: bool = False) -> str:
	value = unicodedata.normalize("NFKC", value.replace("\r\n", "\n").replace("\r", "\n"))
	return value if keep_outer_whitespace else value.strip()


def normalize_datetime(value: Any) -> Any:
	if value is None:
		return None
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
	return value


def normalize_value(value: Any, key: str | None = None) -> Any:
	if isinstance(value, dict):
		return {child_key: normalize_value(child_value, child_key) for child_key, child_value in value.items()}
	if isinstance(value, list):
		return [normalize_value(item, key) for item in value]
	if isinstance(value, str):
		if key in DATE_KEYS:
			return normalize_datetime(value)
		if key in TEXT_KEYS:
			return normalize_text(value, keep_outer_whitespace=True)
		return normalize_text(value)
	return value


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
) -> dict[str, int | str]:
	NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
	DEDUPLICATED_DIR.mkdir(parents=True, exist_ok=True)

	total_rows = 0
	normalized_rows: list[dict[str, Any]] = []
	deduplicated_rows: list[dict[str, Any]] = []
	seen_signatures: set[str] = set()

	for row in read_jsonl(input_path):
		total_rows += 1
		normalized_row = normalize_value(row)
		normalized_rows.append(normalized_row)

		signature = json.dumps(normalized_row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
		if signature not in seen_signatures:
			seen_signatures.add(signature)
			deduplicated_rows.append(normalized_row)

	normalized_count = write_jsonl(normalized_rows, normalized_path)
	deduplicated_count = write_jsonl(deduplicated_rows, deduplicated_path)

	normalized_manifest = {
		"input_path": str(input_path.relative_to(ROOT)),
		"output_path": str(normalized_path.relative_to(ROOT)),
		"rows_in": total_rows,
		"rows_out": normalized_count,
		"normalization": {
			"strings": "unicode NFKC + line-ending normalization",
			"timestamps": "converted to UTC ISO-8601 when parseable",
		},
	}
	deduplicated_manifest = {
		"input_path": str(input_path.relative_to(ROOT)),
		"output_path": str(deduplicated_path.relative_to(ROOT)),
		"rows_in": total_rows,
		"rows_out": deduplicated_count,
		"duplicates_removed": total_rows - deduplicated_count,
		"dedupe_strategy": "first occurrence of normalized JSON signature",
	}

	NORMALIZED_MANIFEST_PATH.write_text(json.dumps(normalized_manifest, indent=2), encoding="utf-8")
	DEDUPLICATED_MANIFEST_PATH.write_text(json.dumps(deduplicated_manifest, indent=2), encoding="utf-8")

	return {
		"rows_in": total_rows,
		"normalized_rows": normalized_count,
		"deduplicated_rows": deduplicated_count,
		"duplicates_removed": total_rows - deduplicated_count,
		"normalized_path": str(normalized_path),
		"deduplicated_path": str(deduplicated_path),
	}


def main() -> None:
	summary = build_variants()
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
