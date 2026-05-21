**CloudHeal**
**Qwen2.5-Coder-7B LoRA Fine-Tuning Plan**

*AMD ROCm • LoRA Full Precision • Weighted Mixed Training • HuggingFace Data Pipeline*

# 1. Purpose of this revised training section

This revision defines how CloudHeal should prepare data, build a weighted mixed training set, fine-tune Qwen2.5-Coder-7B with LoRA, and evaluate the model without assuming that training code already exists. The repo will fetch data from HuggingFace, clean it into a common instruction format, store processed files under the project dataset directory, and keep evaluation data separate from training data.

# 2. Dataset roles

CloudHeal should use two training datasets and one evaluation dataset. LCA is the primary dataset because it directly matches the CI build repair task. SWE-bench is the secondary dataset because it improves broader software engineering repair ability.

| **Role** | **HuggingFace dataset** | **Split usage** | **Purpose** |
| --- | --- | --- | --- |
| Primary training | JetBrains-Research/lca-ci-builds-repair | test split only | CI build repair from failed GitHub Actions logs, workflow content, changed files, and gold diff. Use for CI-specific repair behavior. |
| Secondary training | princeton-nlp/SWE-bench | train split mainly; dev optional for sanity checks | General issue-to-patch software repair. Use to improve code repair, patch generation, and repository reasoning. |
| Evaluation only | princeton-nlp/SWE-bench_Verified | test split only | Human-validated SWE-bench subset. Never mix into training. Use after fine-tuning to measure issue-resolution behavior. |

# 3. Weighted Mixed Training strategy

Weighted Mixed Training should be used because the two training datasets do not have the same size or task focus. If the datasets are simply concatenated, the larger dataset can dominate the model. Instead, each training sample should be drawn using a fixed source weight.

| **Source** | **Sampling weight** | **Why this weight is used** |
| --- | --- | --- |
| LCA CI Builds Repair | 0.6 | Primary CI build repair specialization: failed logs, workflow YAML, repository context, and target patch. |
| SWE-bench | 0.4 | Secondary general repair capability: issue description, repository metadata, test expectations, and target patch. |

Each training batch should contain roughly 40% LCA-style CI repair examples and 60% SWE-bench-style software repair examples. The mixture should be generated as a manifest or JSONL file before training, so the trainer can read one stable file instead of performing complex sampling logic inside the trainer.

# 4. Proposed repo structure

CloudHeal-CICD-Agentic-Framework/

  configs/

    data_mix.yaml

    train_lora_qwen25_7b_rocm.yaml

    eval_swebench_verified.yaml

  dataset/

    raw/

      lca_ci_builds_repair/

      swe_bench/

      swe_bench_verified/

    processed/

      lca_ci_builds_repair/

      swe_bench/

      swe_bench_verified/

    weighted_mix/

      train_weighted_mix.jsonl

      train_weighted_mix.manifest.json

  scripts/

    data/

      download_hf_datasets.py

      clean_lca_ci_builds.py

      clean_swe_bench.py

      build_weighted_mix.py

    training/

      train_lora.py

    evaluation/

      run_swebench_verified_eval.py

  src/

    cloudheal/

      data_pipeline/

      finetuning/

      evaluation/

# 5. HuggingFace download plan

The first implementation step is to create dataset download scripts. The scripts should save raw HuggingFace records exactly as downloaded before any cleaning. This makes the pipeline reproducible and allows the cleaned format to be regenerated later.

# scripts/data/download_hf_datasets.py

from pathlib import Path

from datasets import load_dataset, get_dataset_config_names

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "dataset" / "raw"

DATASETS = {

    "lca_ci_builds_repair": {

        "hf_name": "JetBrains-Research/lca-ci-builds-repair",

        "split": "test",     # LCA provides all datapoints in test split.

        "config": "python",  # Current available config in the README.

    },

    "swe_bench": {

        "hf_name": "princeton-nlp/SWE-bench",

        "split": "train",

        "config": None,

    },

    "swe_bench_verified": {

        "hf_name": "princeton-nlp/SWE-bench_Verified",

        "split": "test",     # Evaluation only.

        "config": None,

    },

}

def save_dataset(local_name: str, spec: dict) -> None:

    output_dir = RAW_DIR / local_name

    output_dir.mkdir(parents=True, exist_ok=True)

    if spec["config"]:

        dataset = load_dataset(spec["hf_name"], spec["config"], split=spec["split"])

    else:

        dataset = load_dataset(spec["hf_name"], split=spec["split"])

    dataset.to_json(output_dir / f"{spec['split']}.jsonl", orient="records", lines=True)

    print(f"Saved {local_name}: {len(dataset)} rows -> {output_dir}")

if __name__ == "__main__":

    print("LCA configs:", get_dataset_config_names("JetBrains-Research/lca-ci-builds-repair"))

    for name, spec in DATASETS.items():

        save_dataset(name, spec)

# 6. Common cleaned instruction format

Both datasets should be converted into the same JSONL schema. This keeps the future training script simple and makes it possible to mix datasets safely.

| **Field** | **Meaning** |
| --- | --- |
| id | Unique datapoint ID. |
| source | lca_ci_builds_repair or swe_bench. |
| task_type | ci_build_repair or issue_resolution. |
| instruction | Natural language task instruction for the model. |
| input | Structured context such as logs, workflow, issue text, repo, commit, changed files, and tests. |
| output | Gold repair patch or diff. |
| metadata | Repo name, owner, commit SHA, workflow path, difficulty, split, and other fields needed for tracing. |

## 6.1 LCA cleaning rule

For LCA, the instruction should explicitly teach the model to repair a failing CI build using the failed logs, workflow file, changed files, and repository metadata. The target output is the gold diff between the failed and successful commit.

# Output file: dataset/processed/lca_ci_builds_repair/train.jsonl

{

  "id": "lca-18",

  "source": "lca_ci_builds_repair",

  "task_type": "ci_build_repair",

  "instruction": "Repair the failing CI build. Use logs, workflow, changed files, and repo context. Return a minimal patch.",

  "input": {

    "repo": "scrapy/scrapy",

    "failed_commit": "0f71221cf9875ed8ef3400e1008408e79b6691e6",

    "workflow_path": ".github/workflows/checks.yml",

    "workflow": "...workflow yaml...",

    "failed_logs": [{"step_name": "checks (3.12, pylint)/4_Run check.txt", "log": "..."}],

    "changed_files": ["scrapy/crawler.py"],

    "difficulty": "2"

  },

  "output": "diff --git a/...",

  "metadata": {"head_branch": "component-getters", "sha_success": "c1ba9...", "commit_link": "https://github.com/..."}

}

## 6.2 SWE-bench cleaning rule

For SWE-bench, the instruction should teach general issue-resolution repair. The model receives the issue statement and repository metadata, then learns to produce the gold patch. The test patch should not be used as the answer, but it can be stored in metadata for evaluation and analysis.

# Output file: dataset/processed/swe_bench/train.jsonl

{

  "id": "django__django-12345",

  "source": "swe_bench",

  "task_type": "issue_resolution",

  "instruction": "Resolve the GitHub issue for the given repository state. Return a minimal source-code patch.",

  "input": {

    "repo": "django/django",

    "base_commit": "...",

    "problem_statement": "...issue title and body...",

    "hints_text": "...",

    "version": "..."

  },

  "output": "...gold source patch...",

  "metadata": {"created_at": "...", "FAIL_TO_PASS": "[...]", "PASS_TO_PASS": "[...]"}

}

# 7. Cleaning scripts to add first

Because no training code exists yet, the first coding milestone should only implement data preparation. Training should not start until these outputs are created and manually checked.

# scripts/data/clean_lca_ci_builds.py

import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "dataset" / "raw" / "lca_ci_builds_repair" / "test.jsonl"

OUT_FILE = ROOT / "dataset" / "processed" / "lca_ci_builds_repair" / "train.jsonl"

def clean_row(row: dict) -> dict:

    repo = f"{row.get('repo_owner')}/{row.get('repo_name')}"

    return {

        "id": f"lca-{row['id']}",

        "source": "lca_ci_builds_repair",

        "task_type": "ci_build_repair",

        "instruction": "Repair the failing CI build. Use logs, workflow, changed files, and repository context. Return a minimal patch.",

        "input": {

            "repo": repo,

            "failed_commit": row.get("sha_fail"),

            "workflow_name": row.get("workflow_name"),

            "workflow_path": row.get("workflow_path"),

            "workflow": row.get("workflow"),

            "failed_logs": row.get("logs", []),

            "changed_files": row.get("changed_files", []),

            "difficulty": row.get("difficulty"),

        },

        "output": row.get("diff", ""),

        "metadata": {

            "language": row.get("language"),

            "head_branch": row.get("head_branch"),

            "sha_success": row.get("sha_success"),

            "commit_link": row.get("commit_link"),

        },

    }

def main() -> None:

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RAW_FILE.open("r", encoding="utf-8") as src, OUT_FILE.open("w", encoding="utf-8") as dst:

        for line in src:

            row = json.loads(line)

            if row.get("diff"):

                dst.write(json.dumps(clean_row(row), ensure_ascii=False) + "

")

if __name__ == "__main__":

    main()

# scripts/data/clean_swe_bench.py

import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "dataset" / "raw" / "swe_bench" / "train.jsonl"

OUT_FILE = ROOT / "dataset" / "processed" / "swe_bench" / "train.jsonl"

def clean_row(row: dict) -> dict:

    return {

        "id": row["instance_id"],

        "source": "swe_bench",

        "task_type": "issue_resolution",

        "instruction": "Resolve the GitHub issue for the given repository state. Return a minimal source-code patch.",

        "input": {

            "repo": row.get("repo"),

            "base_commit": row.get("base_commit"),

            "problem_statement": row.get("problem_statement"),

            "hints_text": row.get("hints_text"),

            "version": row.get("version"),

        },

        "output": row.get("patch", ""),

        "metadata": {

            "created_at": row.get("created_at"),

            "test_patch": row.get("test_patch"),

            "environment_setup_commit": row.get("environment_setup_commit"),

            "FAIL_TO_PASS": row.get("FAIL_TO_PASS"),

            "PASS_TO_PASS": row.get("PASS_TO_PASS"),

        },

    }

def main() -> None:

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RAW_FILE.open("r", encoding="utf-8") as src, OUT_FILE.open("w", encoding="utf-8") as dst:

        for line in src:

            row = json.loads(line)

            if row.get("patch"):

                dst.write(json.dumps(clean_row(row), ensure_ascii=False) + "

")

if __name__ == "__main__":

    main()

# 8. Build the weighted mixed training dataset

After both datasets are cleaned, generate one JSONL file using weighted sampling. The script should be deterministic by default so that experiments can be reproduced.

# configs/data_mix.yaml

seed: 42

output_size: 20000

sources:

  - name: lca_ci_builds_repair

    path: dataset/processed/lca_ci_builds_repair/train.jsonl

    weight: 0.40

  - name: swe_bench

    path: dataset/processed/swe_bench/train.jsonl

    weight: 0.60

output_path: dataset/weighted_mix/train_weighted_mix.jsonl

manifest_path: dataset/weighted_mix/train_weighted_mix.manifest.json

# scripts/data/build_weighted_mix.py

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

        rows = read_jsonl(ROOT / source["path"])

        pools.append({**source, "rows": rows})

    weights = [p["weight"] for p in pools]

    output_rows = []

    counts = {p["name"]: 0 for p in pools}

    for _ in range(cfg["output_size"]):

        chosen = rng.choices(pools, weights=weights, k=1)[0]

        row = rng.choice(chosen["rows"])

        output_rows.append(row)

        counts[chosen["name"]] += 1

    out_path = ROOT / cfg["output_path"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:

        for row in output_rows:

            f.write(json.dumps(row, ensure_ascii=False) + "

")

    manifest = {"seed": cfg["seed"], "output_size": cfg["output_size"], "actual_counts": counts}

    (ROOT / cfg["manifest_path"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":

    main()

# 9. LoRA training plan once data is ready

The training code can be added only after the dataset pipeline is working. The trainer should read dataset/weighted_mix/train_weighted_mix.jsonl and format each example into a chat-style prompt. Keep imports at the top of the file and avoid dynamic imports.

| **Item** | **Decision** |
| --- | --- |
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Fine-tuning method | LoRA full precision on AMD ROCm. |
| Training data | dataset/weighted_mix/train_weighted_mix.jsonl |
| Output adapter path | checkpoints/cloudheal-qwen25-coder-7b-lora |
| Safety rule | Do not train on SWE-bench Verified or any other evaluation-only split. |

# configs/train_lora_qwen25_7b_rocm.yaml

model_name_or_path: Qwen/Qwen2.5-Coder-7B-Instruct

train_file: dataset/weighted_mix/train_weighted_mix.jsonl

output_dir: checkpoints/cloudheal-qwen25-coder-7b-lora

max_seq_length: 8192

num_train_epochs: 2

per_device_train_batch_size: 1

gradient_accumulation_steps: 16

learning_rate: 0.0002

warmup_ratio: 0.03

logging_steps: 10

save_steps: 500

lora:

  r: 16

  alpha: 32

  dropout: 0.05

  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

precision:

  use_bf16_if_supported: true

  fallback_fp16: false

# 10. How to use the evaluation dataset

SWE-bench Verified must be treated as evaluation-only data. It should be downloaded into dataset/raw/swe_bench_verified/, cleaned into dataset/processed/swe_bench_verified/eval.jsonl, and used only after the LoRA adapter is trained.

Load each SWE-bench Verified example and clone the repository at base_commit.

Provide the model with the problem_statement, repo, base_commit, hints_text, and any retrieved code context if the evaluator supports retrieval.

Ask the model to generate a minimal patch only. Do not provide the gold patch to the model.

Apply the generated patch in a clean checkout.

Run the official SWE-bench evaluation harness or equivalent unit-test verification using FAIL_TO_PASS and PASS_TO_PASS.

Record resolved count, fail-to-pass success rate, pass-to-pass preservation rate, patch apply rate, and regression failures.

# configs/eval_swebench_verified.yaml

dataset_name: princeton-nlp/SWE-bench_Verified

split: test

raw_file: dataset/raw/swe_bench_verified/test.jsonl

processed_file: dataset/processed/swe_bench_verified/eval.jsonl

adapter_path: checkpoints/cloudheal-qwen25-coder-7b-lora

results_dir: results/swe_bench_verified

metrics:

  - patch_apply_rate

  - fail_to_pass_success_rate

  - pass_to_pass_preservation_rate

  - resolved_instances

  - regression_failures

For CloudHeal-specific evaluation, LCA datapoints should be used to test CI build repair behavior through GitHub Actions. The benchmark workflow repairs each repo, pushes the candidate fix to GitHub, waits for workflow results, and reports whether the failing CI build was repaired. This is the closest evaluation to CloudHeal because it directly checks CI recovery behavior.

# 11. First implementation command flow

pip install -U datasets pyyaml

python scripts/data/download_hf_datasets.py

python scripts/data/clean_lca_ci_builds.py

python scripts/data/clean_swe_bench.py

python scripts/data/build_weighted_mix.py

python scripts/training/train_lora.py --config configs/train_lora_qwen25_7b_rocm.yaml

python scripts/evaluation/run_swebench_verified_eval.py --config configs/eval_swebench_verified.yaml

# 12. Updated success criteria

HuggingFace datasets are downloaded into dataset/raw/ without manual copying.

LCA and SWE-bench are cleaned into a shared JSONL instruction format.

SWE-bench Verified is stored separately and never mixed into training.

dataset/weighted_mix/train_weighted_mix.jsonl is generated with an approximately 40/60 LCA/SWE-bench source ratio.

A manifest records source counts, weights, seed, and output size.

LoRA training code can consume the weighted mix file directly.

Evaluation reports patch apply rate, fail-to-pass success, pass-to-pass preservation, resolved instances, and regression failures.

# 13. Dataset references to keep in the report

Long Code Arena CI Builds Repair: JetBrains-Research/lca-ci-builds-repair. Use the README citation for the Long Code Arena paper: Bogomolov et al. (2024), Long Code Arena: a Set of Benchmarks for Long-Context Code Models, arXiv:2406.11612.

SWE-bench: princeton-nlp/SWE-bench. Use the dataset card and SWE-bench paper when writing the final methodology section.

SWE-bench Verified: princeton-nlp/SWE-bench_Verified. Use this as an evaluation-only benchmark subset, not as training data.