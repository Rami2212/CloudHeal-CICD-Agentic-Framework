import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "data_mix.yaml"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rng = random.Random(cfg["seed"])

    pools = []
    for source in cfg["sources"]:
        source_path = ROOT / source["path"]
        rows = read_jsonl(source_path)
        if not rows:
            raise ValueError(f"Source file is empty: {source_path}")
        pools.append({**source, "rows": rows})

    weights = [pool["weight"] for pool in pools]
    output_rows = []
    counts = {pool["name"]: 0 for pool in pools}

    for _ in range(cfg["output_size"]):
        chosen = rng.choices(pools, weights=weights, k=1)[0]
        row = rng.choice(chosen["rows"])
        output_rows.append(row)
        counts[chosen["name"]] += 1

    out_path = ROOT / cfg["output_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "seed": cfg["seed"],
        "output_size": cfg["output_size"],
        "weights": {pool["name"]: pool["weight"] for pool in pools},
        "actual_counts": counts,
        "sources": [pool["name"] for pool in pools],
    }
    manifest_path = ROOT / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()