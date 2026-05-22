"""
src/data_pipeline/clean_data.py
================================
CloudHeal-CICD-Agentic-Framework
Cleans three local datasets before fine-tuning:
    1. datasets/raw/train_primary/.../lca-ci-builds-repair      (primary)
    2. datasets/raw/train_secondary/.../SWE-bench              (secondary)
    3. datasets/raw/evaluate/.../SWE-bench_Verified            (eval)

Output goes to:
  datasets/processed/cleaned/<dataset_name>.jsonl
"""

import ast
import json
import logging
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ─────────────────────────────────────────────
#  Paths  (relative to repo root)
# ─────────────────────────────────────────────
RAW_DIR     = Path("datasets/raw")
CLEANED_DIR = Path("datasets/processed/cleaned")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_DATASET_ROOT = RAW_DIR / "train_primary" / "datasets--JetBrains-Research--lca-ci-builds-repair"
SECONDARY_DATASET_ROOT = RAW_DIR / "train_secondary" / "datasets--princeton-nlp--SWE-bench"
EVAL_DATASET_ROOT = RAW_DIR / "evaluate" / "datasets--SWE-bench--SWE-bench_Verified"

# ─────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════

def strip_ansi(text: str) -> str:
    """Remove ANSI colour / control codes from CI log strings."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def strip_github_log_groups(text: str) -> str:
    """Remove ##[group] / ##[endgroup] markers from GitHub Actions logs."""
    text = re.sub(r"##\[group\].*?\n", "", text)
    text = re.sub(r"##\[endgroup\].*?\n", "", text)
    return text


def truncate(text: str, max_chars: int) -> str:
    """Keep first max_chars characters so context fits in the model window."""
    return text[:max_chars] if text else text


def safe_parse_list(value) -> list:
    """Parse a JSON / Python-literal list field that may arrive as a string."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except Exception:
                return []
    return []


def report(label: str, before: int, after: int) -> None:
    removed = before - after
    log.info(f"  {label}: {before} → {after} rows  (removed {removed})")


def resolve_snapshot_dir(dataset_root: Path) -> Path:
    """Return the single cached snapshot directory for a local HF dataset."""
    snapshots_root = dataset_root / "snapshots"
    snapshot_dirs = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directory found under {snapshots_root}")
    return snapshot_dirs[0]


def load_local_parquet_split(data_files: dict[str, str], split: str) -> pd.DataFrame:
    """Load one split from a local parquet-backed Hugging Face dataset cache."""
    return load_dataset("parquet", data_files=data_files, split=split).to_pandas()


# ═══════════════════════════════════════════════════════════════
#  DATASET 1 — JetBrains LCA CI Builds Repair
# ═══════════════════════════════════════════════════════════════

def clean_lca_ci(
    split: str = "test",
    log_max_chars: int = 4_000,
    diff_max_chars: int = 8_000,
    workflow_max_chars: int = 3_000,
) -> pd.DataFrame:
    """
    Columns in this dataset
    ───────────────────────
    language, id, repo_owner, repo_name, head_branch, workflow_name,
    workflow_filename, workflow_path, contributor, sha_fail, sha_success,
    workflow, logs (list[{step_name, log}]), diff, difficulty,
    changed_files (list[str]), commit_link, commit_date
    """
    log.info("=" * 60)
    log.info("DATASET 1 — JetBrains/lca-ci-builds-repair")
    log.info("=" * 60)

    snapshot_dir = resolve_snapshot_dir(PRIMARY_DATASET_ROOT)
    data_files = {
        "test": (snapshot_dir / "data" / "python" / "test-*.parquet").as_posix(),
    }
    df = load_local_parquet_split(data_files, split)
    n0 = len(df)
    log.info(f"  Loaded {n0} rows, {df.shape[1]} columns")

    # ── 1. Drop exact duplicates ──────────────────────────────
    df = df.drop_duplicates(subset=["sha_fail", "sha_success"])
    report("After dedup (sha_fail+sha_success)", n0, len(df)); n0 = len(df)

    # ── 2. Drop rows with missing critical fields ─────────────
    critical = ["diff", "logs", "workflow"]
    df = df.dropna(subset=critical)
    report("After dropping rows with null diff/logs/workflow", n0, len(df)); n0 = len(df)

    # ── 3. Drop rows where diff or workflow is empty string ───
    df = df[df["diff"].str.strip().ne("")]
    df = df[df["workflow"].str.strip().ne("")]
    report("After dropping empty diff/workflow", n0, len(df)); n0 = len(df)

    # ── 4. Parse logs column (list of step dicts) ─────────────
    df["logs"] = df["logs"].apply(safe_parse_list)

    # ── 5. Clean ANSI codes + GitHub group markers from logs ──
    def clean_log_steps(steps: list) -> str:
        """Flatten all CI step logs into a single clean string."""
        parts = []
        for step in steps:
            name = step.get("step_name", "")
            raw  = step.get("log", "")
            cleaned = strip_ansi(strip_github_log_groups(raw))
            parts.append(f"[STEP: {name}]\n{cleaned.strip()}")
        return "\n\n".join(parts)

    df["logs_clean"] = df["logs"].apply(clean_log_steps)

    # ── 6. Truncate long text fields ─────────────────────────
    df["logs_clean"]  = df["logs_clean"].apply(lambda x: truncate(x, log_max_chars))
    df["diff"]        = df["diff"].apply(lambda x: truncate(x, diff_max_chars))
    df["workflow"]    = df["workflow"].apply(lambda x: truncate(x, workflow_max_chars))

    # ── 7. Parse changed_files to list ───────────────────────
    df["changed_files"] = df["changed_files"].apply(safe_parse_list)

    # ── 8. Normalise commit_date → datetime ──────────────────
    df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")

    # ── 9. Trim string columns ───────────────────────────────
    str_cols = ["repo_owner", "repo_name", "head_branch",
                "workflow_name", "contributor", "language"]
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── 10. Select final columns ─────────────────────────────
    df = df[[
        "id", "language", "repo_owner", "repo_name", "head_branch",
        "workflow_name", "workflow_filename", "contributor",
        "sha_fail", "sha_success", "workflow",
        "logs_clean", "diff", "difficulty", "changed_files",
        "commit_link", "commit_date",
    ]]

    log.info(f"  ✓ Final shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
#  DATASET 2 — princeton-nlp/SWE-bench  (secondary)
# ═══════════════════════════════════════════════════════════════

def clean_swebench(
    split: str = "train",
    patch_max_chars: int = 10_000,
    statement_max_chars: int = 4_000,
    require_fail_to_pass: bool = True,
) -> pd.DataFrame:
    """
    Columns in this dataset
    ───────────────────────
    repo, instance_id, base_commit, patch, test_patch,
    problem_statement, hints_text, created_at, version,
    FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit
    """
    log.info("=" * 60)
    log.info("DATASET 2 — princeton-nlp/SWE-bench")
    log.info("=" * 60)

    snapshot_dir = resolve_snapshot_dir(SECONDARY_DATASET_ROOT)
    data_files = {
        "train": (snapshot_dir / "data" / "train-*.parquet").as_posix(),
        "dev": (snapshot_dir / "data" / "dev-*.parquet").as_posix(),
        "test": (snapshot_dir / "data" / "test-*.parquet").as_posix(),
    }
    df = load_local_parquet_split(data_files, split)
    n0 = len(df)
    log.info(f"  Loaded {n0} rows, {df.shape[1]} columns")

    # ── 1. Drop exact duplicates on instance_id ──────────────
    df = df.drop_duplicates(subset=["instance_id"])
    report("After dedup (instance_id)", n0, len(df)); n0 = len(df)

    # ── 2. Drop rows missing critical fields ─────────────────
    critical = ["patch", "problem_statement", "base_commit"]
    df = df.dropna(subset=critical)
    report("After dropping nulls in critical cols", n0, len(df)); n0 = len(df)

    # ── 3. Drop empty patch / problem_statement ──────────────
    df = df[df["patch"].str.strip().ne("")]
    df = df[df["problem_statement"].str.strip().ne("")]
    report("After dropping empty patch/problem_statement", n0, len(df)); n0 = len(df)

    # ── 4. Clean problem_statement: strip leading whitespace ──
    df["problem_statement"] = df["problem_statement"].str.strip()

    # ── 5. Fill empty hints_text with empty string ────────────
    df["hints_text"] = df["hints_text"].fillna("").str.strip()

    # ── 6. Parse FAIL_TO_PASS / PASS_TO_PASS ─────────────────
    #   They arrive as JSON strings like '["test_a", "test_b"]'
    df["FAIL_TO_PASS"] = df["FAIL_TO_PASS"].apply(safe_parse_list)
    df["PASS_TO_PASS"] = df["PASS_TO_PASS"].apply(safe_parse_list)

    # ── 7. Drop rows with empty FAIL_TO_PASS only when the split has test labels
    if require_fail_to_pass:
        df = df[df["FAIL_TO_PASS"].map(len) > 0]
        report("After dropping rows with empty FAIL_TO_PASS", n0, len(df)); n0 = len(df)

    # ── 8. Truncate long text fields ─────────────────────────
    df["patch"]             = df["patch"].apply(lambda x: truncate(x, patch_max_chars))
    df["test_patch"]        = df["test_patch"].apply(lambda x: truncate(x, patch_max_chars))
    df["problem_statement"] = df["problem_statement"].apply(lambda x: truncate(x, statement_max_chars))

    # ── 9. Normalise created_at → datetime ───────────────────
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # ── 10. Trim repo / version ───────────────────────────────
    df["repo"]    = df["repo"].str.strip()
    df["version"] = df["version"].str.strip()

    log.info(f"  ✓ Final shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
#  DATASET 3 — SWE-bench/SWE-bench_Verified  (eval)
# ═══════════════════════════════════════════════════════════════

def clean_swebench_verified(
    split: str = "test",
    patch_max_chars: int = 10_000,
    statement_max_chars: int = 4_000,
) -> pd.DataFrame:
    """
    Same schema as SWE-bench — apply identical cleaning rules.
    This split is used only for final evaluation, so we keep it
    stricter (no truncation of problem_statement to preserve all context).
    """
    log.info("=" * 60)
    log.info("DATASET 3 — SWE-bench/SWE-bench_Verified")
    log.info("=" * 60)

    snapshot_dir = resolve_snapshot_dir(EVAL_DATASET_ROOT)
    data_files = {
        "test": (snapshot_dir / "data" / "test-*.parquet").as_posix(),
    }
    df = load_local_parquet_split(data_files, split)
    n0 = len(df)
    log.info(f"  Loaded {n0} rows, {df.shape[1]} columns")

    df = df.drop_duplicates(subset=["instance_id"])
    report("After dedup", n0, len(df)); n0 = len(df)

    critical = ["patch", "problem_statement", "base_commit"]
    df = df.dropna(subset=critical)
    report("After dropping nulls", n0, len(df)); n0 = len(df)

    df = df[df["patch"].str.strip().ne("")]
    df = df[df["problem_statement"].str.strip().ne("")]
    report("After dropping empty patch/statement", n0, len(df)); n0 = len(df)

    df["problem_statement"] = df["problem_statement"].str.strip()
    df["hints_text"]        = df["hints_text"].fillna("").str.strip()

    df["FAIL_TO_PASS"] = df["FAIL_TO_PASS"].apply(safe_parse_list)
    df["PASS_TO_PASS"] = df["PASS_TO_PASS"].apply(safe_parse_list)

    df = df[df["FAIL_TO_PASS"].map(len) > 0]
    report("After dropping empty FAIL_TO_PASS", n0, len(df)); n0 = len(df)

    # Eval set: DO NOT truncate problem_statement (keep full context)
    df["patch"]      = df["patch"].apply(lambda x: truncate(x, patch_max_chars))
    df["test_patch"] = df["test_patch"].apply(lambda x: truncate(x, patch_max_chars))

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["repo"]       = df["repo"].str.strip()
    df["version"]    = df["version"].str.strip()

    log.info(f"  ✓ Final shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
#  SAVE TO JSONL
# ═══════════════════════════════════════════════════════════════

def save_jsonl(df: pd.DataFrame, filename: str) -> None:
    """Save a cleaned DataFrame to datasets/processed/cleaned/<filename>.jsonl"""
    out_path = CLEANED_DIR / filename
    # Convert timestamps to ISO strings so they are JSON-serialisable
    df_out = df.copy()
    for col in df_out.select_dtypes(include=["datetimetz", "datetime64[ns, UTC]"]):
        df_out[col] = df_out[col].astype(str)

    with open(out_path, "w", encoding="utf-8") as f:
        for record in df_out.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"  Saved → {out_path}  ({len(df_out)} rows)\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    log.info("CloudHeal — Dataset Cleaning Pipeline")
    log.info("=" * 60)

    # 1. Primary dataset
    df_lca = clean_lca_ci()
    save_jsonl(df_lca, "lca_ci_builds_repair.jsonl")

    # 2. Secondary dataset — use all three local splits
    for split_name in ["train", "dev", "test"]:
        df_swe = clean_swebench(
            split=split_name,
            require_fail_to_pass=(split_name != "train"),
        )
        save_jsonl(df_swe, f"swe_bench_{split_name}.jsonl")

    # 3. Eval dataset
    df_verified = clean_swebench_verified(split="test")
    save_jsonl(df_verified, "swe_bench_verified.jsonl")

    log.info("=" * 60)
    log.info("All datasets cleaned and saved to datasets/processed/cleaned/")


if __name__ == "__main__":
    main()