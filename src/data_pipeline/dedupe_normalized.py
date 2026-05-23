#!/usr/bin/env python3
"""
Deduplicate the normalized instruction dataset.

Reads:
  datasets/processed/normalized/train_weighted_mix_normalized.jsonl

Writes:
  datasets/processed/deduplicated/train_weighted_mix_deduplicated.jsonl
  datasets/processed/deduplicated/train_weighted_mix_deduplicated.manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "datasets" / "processed" / "normalized" / "train_weighted_mix_normalized_new.jsonl"
OUT_DIR = ROOT / "datasets" / "processed" / "deduplicated"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "train_weighted_mix_deduplicated_normalized_new.jsonl"
MANIFEST = OUT_DIR / "train_weighted_mix_deduplicated_normalized_new.manifest.json"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    total = 0
    seen = set()
    deduped = []

    if not IN_PATH.exists():
        print("Missing normalized input:", IN_PATH)
        return

    for row in read_jsonl(IN_PATH):
        total += 1
        sig = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(row)

    n_out = write_jsonl(deduped, OUT_PATH)

    manifest = {
        "input_path": str(IN_PATH.relative_to(ROOT)),
        "output_path": str(OUT_PATH.relative_to(ROOT)),
        "rows_in": total,
        "rows_out": n_out,
        "duplicates_removed": total - n_out,
        "dedupe_strategy": "first occurrence of normalized JSON signature",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

