"""
data.py
--------
Downloads the JetBrains-Research/diff-xyz dataset from Hugging Face and
prepares it for evaluating a CI/CD-log-to-fix fine-tuned model, saving the
final dataset to ./data/diff_xyz_eval.jsonl.

Note on this dataset: diff-xyz contains (old_code, new_code, udiff, message)
per commit, but no CI failure logs and no test suite. We treat:
    - old_code              -> the "before" state the model must fix
    - commit message        -> a stand-in for a CI failure log / fix instruction
    - new_code / udiff       -> the ground-truth fix used for scoring

Requires:
    pip install datasets

Usage:
    python data.py
    python data.py --filter_lang python --max_samples 200
"""
import os
import argparse
import json

from datasets import load_dataset


def build_eval_record(example: dict) -> dict:
    """Map one diff-xyz row into an evaluation record for the fix-generation model."""
    return {
        "id": f"{example.get('repo')}::{str(example.get('commit'))[:10]}::{example.get('path')}",
        "language": example.get("lang"),
        "context": (example.get("message") or "").strip(),
        "input_code": example.get("old_code", ""),
        "reference_code": example.get("new_code", ""),
        "reference_diff": example.get("udiff", ""),
        "reference_search_replace": example.get("search-replace", ""),
        "meta": {
            "repo": example.get("repo"),
            "commit": example.get("commit"),
            "path": example.get("path"),
            "license": example.get("license"),
            "change_kind": example.get("change_kind"),
            "n_added": example.get("n_added"),
            "n_removed": example.get("n_removed"),
            "n_hunks": example.get("n_hunks"),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare JetBrains-Research/diff-xyz for fix-generation evaluation."
    )
    parser.add_argument("--dataset", default="JetBrains-Research/diff-xyz")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = use all rows")
    parser.add_argument(
        "--filter_lang", default=None,
        help="Optional: keep only rows for this language, e.g. python / java / kotlin / rust / javascript",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset} ({args.config}, split={args.split}) from Hugging Face...")
    ds = load_dataset(args.dataset, args.config, split=args.split)

    if args.filter_lang:
        ds = ds.filter(lambda ex: ex.get("lang") == args.filter_lang)

    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Loaded {len(ds)} raw rows. Building evaluation records...")

    records = [build_eval_record(ex) for ex in ds]

    # Drop rows where we don't have both states of the code -- can't score a fix without them
    records = [r for r in records if r["input_code"] and r["reference_code"]]

    out_path = os.path.join(args.output_dir, "diff_xyz_eval.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} evaluation-ready records to {out_path}")

    langs = {}
    for r in records:
        langs[r["language"]] = langs.get(r["language"], 0) + 1
    print("Language distribution:", langs)


if __name__ == "__main__":
    main()
