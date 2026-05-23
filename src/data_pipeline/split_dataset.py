#!/usr/bin/env python3
"""Deterministically split a JSONL dataset into a 3:2 ratio.

This script is intentionally generic:
- reads any JSONL file
- shuffles rows with a fixed seed
- writes two JSONL outputs in the requested ratio
- writes a small manifest describing the split

Example (PowerShell):
	python src/data_pipeline/split_dataset.py ^
	  --input datasets/processed/cleaned/eval_DataSet.jsonl ^
	  --out-a datasets/processed/splits/eval_300.jsonl ^
	  --out-b datasets/processed/splits/eval_200.jsonl ^
	  --ratio 3 2 ^
	  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, ensure_ascii=False) + "\n")
	return len(rows)


def split_rows(rows: list[dict[str, Any]], ratio_a: int, ratio_b: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	if ratio_a <= 0 or ratio_b <= 0:
		raise ValueError("ratio values must both be positive integers")

	shuffled = rows.copy()
	random.Random(seed).shuffle(shuffled)

	total = len(shuffled)
	total_ratio = ratio_a + ratio_b
	count_a = (total * ratio_a) // total_ratio
	count_b = total - count_a
	return shuffled[:count_a], shuffled[count_a:count_a + count_b]


def main() -> None:
	parser = argparse.ArgumentParser(description="Split a JSONL dataset into two parts using a fixed ratio.")
	parser.add_argument("--input", required=True, help="Path to input JSONL file")
	parser.add_argument("--out-a", required=True, help="Path to first output JSONL file")
	parser.add_argument("--out-b", required=True, help="Path to second output JSONL file")
	parser.add_argument("--ratio", nargs=2, type=int, metavar=("A", "B"), default=(3, 2), help="Split ratio, default: 3 2")
	parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for deterministic output")
	parser.add_argument("--manifest", help="Optional path for manifest JSON")
	args = parser.parse_args()

	input_path = Path(args.input)
	out_a = Path(args.out_a)
	out_b = Path(args.out_b)
	manifest_path = Path(args.manifest) if args.manifest else out_a.with_suffix(".manifest.json")

	if not input_path.exists():
		raise FileNotFoundError(input_path)

	rows = read_jsonl(input_path)
	split_a, split_b = split_rows(rows, args.ratio[0], args.ratio[1], args.seed)

	n_a = write_jsonl(split_a, out_a)
	n_b = write_jsonl(split_b, out_b)

	manifest = {
		"input_path": str(input_path.relative_to(ROOT)),
		"output_a_path": str(out_a.relative_to(ROOT)),
		"output_b_path": str(out_b.relative_to(ROOT)),
		"rows_in": len(rows),
		"ratio": [args.ratio[0], args.ratio[1]],
		"rows_out_a": n_a,
		"rows_out_b": n_b,
		"seed": args.seed,
		"shuffle": True,
	}
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

	print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
	main()

