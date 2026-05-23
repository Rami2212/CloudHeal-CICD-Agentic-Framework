# Implementation Plan - Model Evaluation & Comparison Script

Implement the evaluation and comparison framework in `server_upload/test_basemodel_vs_finetunedmodel.py` to compare the base model against the fine-tuned LoRA model based on the finalized dataset mixing strategy and meeting requirements.

## Context & Meeting Highlights

- **Base Model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (downloaded automatically if not cached).
- **Dataset Background**:
  - Primary Dataset: CCD-specific (~200 samples).
  - Secondary Dataset: General-purpose / SDS-related.
  - History: Initial 60% primary + 40% secondary mixture failed.
  - Corrected Mixture: Upsampled primary by 5x, merged, deduplicated, and cleaned down to ~120 samples.
  - Splits: Split into train/validation (3:2 ratio) and a separate test set.
  - Target Test File: `server_upload/data/test.jsonl` (or a CSV alternative).
- **Core Pipeline**: Base model → Fine-tune (via existing training script) → Test with adapter → Compare results against expected output.
- **Reporting**: The script must generate a report that can be downloaded from the server, indicating simple pass/fail results.

## Data Auto-Generation

Since the generated JSON files are `.gitignore`d and not pushed to the repository, the evaluation script should be capable of generating them on the fly if they are missing on the server!
**Solution**: The script includes an `--auto_generate_data` flag (which defaults to `True` if the test file is missing). When triggered, the script will automatically use `subprocess` to execute your data pipeline (`scripts/data/clean_data.py`, `scripts/data/build_weighted_mix.py`, `scripts/data/normalize_data.py`, and `scripts/data/split_dataset.py`) before running the evaluation.

## Schema Flexibility

To ensure maximum compatibility with different data pipeline outputs, the script will dynamically parse either:
1. Training Schema: `"instruction"`, `"input"`, and `"output"` fields.
2. Alternative Schema: `"input"` and `"expected_output"` fields.
3. Standard CSV / JSONL format structures.

## VRAM and Safe Hardware Execution

To prevent Out of Memory (OOM) exceptions when comparing two models on a single GPU, the script will:
- Support sequential processing: run the base model, save its outputs, clean memory (`torch.cuda.empty_cache()` and garbage collection), and then load the base model + LoRA adapter.
- Load the model in half-precision (`bfloat16` or `float16`) automatically based on hardware capabilities.

---

### Evaluation Component

We will implement the entire evaluation pipeline in `server_upload/test_basemodel_vs_finetunedmodel.py`.

#### Config and Argument Parsing (CLI)
- Accept `--auto_generate_data` (default: `store_true`). If the specified `--test_file` does not exist, the script will automatically run the data pipeline to generate it.
- Accept `--base_model` (default: `Qwen/Qwen2.5-Coder-7B-Instruct`).
- Accept `--adapter_path` (default: `server_upload/output`).
- Accept `--test_file` (default: `server_upload/data/test.jsonl`).
- Accept `--output_dir` (default: `server_upload/results`).
- Accept `--max_new_tokens` (default: `2048`).
- Accept `--precision` (choices: `bf16`, `fp16`, `fp32`, default: `bf16`).
- Accept `--device` (default: `cuda` if available, else `cpu`).
- Accept `--mode` (choices: `base`, `adapter`, `compare`, default: `compare`). This aligns with the requirement to choose between base model only, or base+adapter.
- Accept `--limit` (optional integer to limit the number of examples tested).

#### Data Auto-Generation Logic
- Before trying to load the dataset, check if `args.test_file` exists.
- If it is missing and auto-generation is allowed, print a warning and sequentially run the data pipeline scripts using `subprocess.run()`.
- Handle failures gracefully (e.g., if a pipeline script fails, stop evaluation and alert the user).

#### Data Loading and Format Support
- Support both **JSONL** and **CSV** files.
- Automatically map fields:
  - Map `expected_output` or `output` to a standard reference field.
  - Map `instruction` and `input` to form the prompt template. If `instruction` is absent, use a default fallback instruction.

#### Inference & Execution Loops
- Implement a clean, well-commented inference runner.
- Run the base model (if `--mode` is `base` or `compare`).
- Run the base + LoRA adapter (if `--mode` is `adapter` or `compare`).
- Flush memory proactively to allow seamless switching on standard GPUs.

#### Outputs & Grading
- Print the model's responses for each test example to the console.
- Compute a simple pass/fail metric (Exact Match or soft token overlap/inclusion check) against expected answers.
- Write complete raw outputs to `results/evaluation_results.jsonl`.
- Save a clean, downloadable summary markdown report under `results/summary_report.md` detailing the pass/fail rates.
