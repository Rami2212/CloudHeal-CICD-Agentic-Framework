"""
data.py
--------
Downloads the JetBrains-Research/diff-xyz dataset from Hugging Face and
prepares it for evaluating a code-fix-generation model, saving the
final dataset to ./data/diff_xyz_eval.jsonl.

Dataset: https://huggingface.co/datasets/JetBrains-Research/diff-xyz
Paper:   "Diff-XYZ: A Benchmark for Evaluating Diff Understanding"
         (arXiv:2510.12487)

Schema (verified against the dataset card, 16 columns):
    repo, commit, path, lang, license, message,
    old_code, new_code, n_added, n_removed, n_hunks, change_kind,
    udiff, udiff-h, udiff-l, search-replace

Note on what this dataset actually is: diff-xyz is built around three
diff-understanding tasks -- Apply (old_code + diff -> new_code),
Anti-Apply (new_code + diff -> old_code), and Diff-Generation
(old_code + new_code -> diff). It does NOT contain CI/CD failure logs,
build output, or test results of any kind -- `message` is simply the
original git commit message (e.g. "Fix off-by-one error", "Add 404
handler"), which is a much weaker and more variable signal than an
actual CI failure trace. If you are using this dataset as a proxy for a
"log-to-fix" task, that mapping is a deliberate choice you are making on
top of the dataset, not something diff-xyz models natively -- keep that
in mind when interpreting results. We map fields as:

    old_code         -> the "before" state the model must fix
    message           -> a (weak) stand-in for a fix instruction / commit
                          intent, NOT a CI/build failure log
    new_code / udiff  -> the ground-truth fix used for scoring

Requires:
    pip install datasets

Usage:
    python data.py
    python data.py --filter_lang python --max_samples 200
    python data.py --diff_format search-replace
"""
import os
import argparse
import json

from datasets import load_dataset


# Columns confirmed present on the dataset card as of this writing. Used
# only for an optional, best-effort sanity check after loading -- not a
# hard requirement, since HF dataset cards can add columns over time.
EXPECTED_COLUMNS = {
    "repo", "commit", "path", "lang", "license", "message",
    "old_code", "new_code", "n_added", "n_removed", "n_hunks",
    "change_kind", "udiff", "udiff-h", "udiff-l", "search-replace",
}

DIFF_FIELD_BY_FORMAT = {
    "udiff": "udiff",                 # standard unified diff, numeric hunk headers
    "udiff-h": "udiff-h",             # unified diff, relaxed "@@ ... @@" headers
    "udiff-l": "udiff-l",             # unified diff with explicit ADD/DEL/CON markers
    "search-replace": "search-replace",  # SEARCH/REPLACE block format
}


def build_eval_record(example: dict, diff_format: str = "udiff") -> dict:
    """
    Map one diff-xyz row into an evaluation record for the fix-generation
    model. `diff_format` selects which diff representation is surfaced as
    `reference_diff` (all four are still preserved under `meta` regardless).
    """
    diff_field = DIFF_FIELD_BY_FORMAT[diff_format]
    return {
        "id": f"{example.get('repo')}::{str(example.get('commit'))[:10]}::{example.get('path')}",
        "language": example.get("lang"),
        "context": (example.get("message") or "").strip(),
        "input_code": example.get("old_code", ""),
        "reference_code": example.get("new_code", ""),
        "reference_diff": example.get(diff_field, ""),
        "meta": {
            "repo": example.get("repo"),
            "commit": example.get("commit"),
            "path": example.get("path"),
            "license": example.get("license"),
            "change_kind": example.get("change_kind"),
            "n_added": example.get("n_added"),
            "n_removed": example.get("n_removed"),
            "n_hunks": example.get("n_hunks"),
            "diff_format_used": diff_format,
            "udiff": example.get("udiff", ""),
            "udiff-h": example.get("udiff-h", ""),
            "udiff-l": example.get("udiff-l", ""),
            "search-replace": example.get("search-replace", ""),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare JetBrains-Research/diff-xyz for fix-generation evaluation."
    )
    parser.add_argument("--dataset", default="JetBrains-Research/diff-xyz")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test",
                         help="diff-xyz ships a single 'test' split (1,000 rows); "
                              "there is no train/validation split.")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = use all rows")
    parser.add_argument(
        "--filter_lang", default=None,
        help="Optional: keep only rows for this language. "
             "diff-xyz covers: python, javascript, java, kotlin, rust (200 rows each).",
    )
    parser.add_argument(
        "--diff_format", default="udiff", choices=list(DIFF_FIELD_BY_FORMAT),
        help="Which diff representation to surface as `reference_diff`. "
             "All four formats are always kept under `meta` regardless of this choice.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset} ({args.config}, split={args.split}) from Hugging Face...")
    ds = load_dataset(args.dataset, args.config, split=args.split)

    actual_columns = set(ds.column_names)
    missing = EXPECTED_COLUMNS - actual_columns
    if missing:
        print(f"WARNING: dataset is missing expected columns {missing}. "
              f"The schema may have changed upstream -- check "
              f"https://huggingface.co/datasets/JetBrains-Research/diff-xyz")

    if args.filter_lang:
        valid_langs = {"python", "javascript", "java", "kotlin", "rust"}
        if args.filter_lang not in valid_langs:
            print(f"WARNING: '{args.filter_lang}' is not one of diff-xyz's known "
                  f"languages {sorted(valid_langs)} -- this filter will likely "
                  f"produce zero rows.")
        ds = ds.filter(lambda ex: ex.get("lang") == args.filter_lang)

    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Loaded {len(ds)} raw rows. Building evaluation records "
          f"(diff_format={args.diff_format})...")

    records = [build_eval_record(ex, diff_format=args.diff_format) for ex in ds]

    # Drop rows where we don't have both states of the code -- can't score a fix without them
    before = len(records)
    records = [r for r in records if r["input_code"] and r["reference_code"]]
    dropped = before - len(records)
    if dropped:
        print(f"Dropped {dropped} row(s) missing old_code/new_code.")

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
