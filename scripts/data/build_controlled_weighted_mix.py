#!/usr/bin/env python3
"""
Build a controlled weighted mix:

- Keep all secondary (SWE-bench) rows once
- Repeat primary (LCA) rows by a fixed factor (or compute factor to achieve a target total)
- Shuffle deterministically (seed)
- Write datasets/weighted_mix/train_weighted_mix.jsonl and manifest

Usage examples:
  # repeat primary 5x
  python src/data_pipeline/build_controlled_weighted_mix.py --factor 5

  # compute factor to reach ~20000 rows
  python src/data_pipeline/build_controlled_weighted_mix.py --target 20000 --min-factor 1 --max-factor 20

  # change seed
  python src/data_pipeline/build_controlled_weighted_mix.py --factor 5 --seed 42
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_PATH = ROOT / "datasets" / "processed" / "cleaned" / "primary.jsonl"
SECONDARY_PATH = ROOT / "datasets" / "processed" / "cleaned" / "secondary_train.jsonl"
OUT_DIR = ROOT / "datasets" / "weighted_mix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "train_weighted_mix_new.jsonl"
MANIFEST = OUT_DIR / "train_weighted_mix_new.manifest.json"

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    return count

def build_mix(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], factor: int, seed: int) -> List[Dict[str, Any]]:
    # Replicate primary rows by factor
    if factor <= 0:
        return secondary.copy()
    replicated_primary = []
    for _ in range(factor):
        # append shallow copies to avoid shared references
        for r in primary:
            replicated_primary.append(r.copy())
    # final mix: secondary then replicated primary (order doesn't matter, we'll shuffle)
    mix = secondary.copy() + replicated_primary
    random.Random(seed).shuffle(mix)
    return mix

def compute_factor_for_target(n_primary: int, n_secondary: int, target: int, min_factor: int=1, max_factor: int=100) -> int:
    if n_primary <= 0:
        return 0
    required = target - n_secondary
    if required <= 0:
        return 0
    factor = math.ceil(required / n_primary)
    # clamp
    factor = max(min_factor, factor)
    factor = min(max_factor, factor)
    return factor

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, help="Fixed repetition factor for primary (LCA) rows")
    parser.add_argument("--target", type=int, help="Target total rows; compute factor to approach this number")
    parser.add_argument("--min-factor", type=int, default=1, help="Minimum factor when computing from target")
    parser.add_argument("--max-factor", type=int, default=50, help="Maximum factor when computing from target")
    parser.add_argument("--seed", type=int, default=12345, help="Shuffle seed for deterministic output")
    args = parser.parse_args()

    primary = list(read_jsonl(PRIMARY_PATH))
    secondary = list(read_jsonl(SECONDARY_PATH))

    n_primary = len(primary)
    n_secondary = len(secondary)

    if args.target is not None and args.factor is None:
        factor = compute_factor_for_target(n_primary, n_secondary, args.target, args.min_factor, args.max_factor)
        computed = True
    elif args.factor is not None:
        factor = args.factor
        computed = False
    else:
        # default example: 5x
        factor = 5
        computed = False

    if n_primary == 0:
        print("Warning: primary (LCA) file empty:", PRIMARY_PATH)
    if n_secondary == 0:
        print("Warning: secondary (SWE-bench) file empty:", SECONDARY_PATH)

    mix = build_mix(primary, secondary, factor, args.seed)
    rows_out = write_jsonl(mix, OUT_PATH)

    manifest = {
        "primary_path": str(PRIMARY_PATH.relative_to(ROOT)),
        "secondary_path": str(SECONDARY_PATH.relative_to(ROOT)),
        "output_path": str(OUT_PATH.relative_to(ROOT)),
        "n_primary_rows": n_primary,
        "n_secondary_rows": n_secondary,
        "factor_used": factor,
        "rows_out": rows_out,
        "shuffle_seed": args.seed,
        "computed_from_target": computed,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()