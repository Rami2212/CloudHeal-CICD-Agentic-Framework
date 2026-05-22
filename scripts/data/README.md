# Data Preparation README

This folder contains the data preparation pipeline used to clean, mix, normalize, and deduplicate datasets before training.

## Folder Overview
- `scripts/data/clean_data.py`: Clean the primary, secondary, and eval datasets and write JSONL outputs to `datasets/processed/cleaned/`.
- `scripts/data/build_weighted_mix.py`: Build a weighted mix JSONL using `configs/data_mix.yaml`.
- `scripts/data/normalize_data.py`: Normalize and deduplicate the weighted mix JSONL, writing outputs to `datasets/processed/normalized/` and `datasets/processed/deduplicated/`.
- `scripts/data/run_pipeline.py`: Convenience entry point that runs normalization and deduplication on the weighted mix.
- `scripts/data/split_dataset.py`: Placeholder (currently empty).

## Prerequisites
- Python 3.10+
- Dependencies from `requirements.txt`

Install:

```powershell
pip install -r requirements.txt
```

## Step 1: Clean raw datasets
Cleans the raw datasets and writes JSONL outputs to `datasets/processed/cleaned/`.

```powershell
python scripts/data/clean_data.py
```

Expected outputs:
- `datasets/processed/cleaned/lca_ci_builds_repair.jsonl`
- `datasets/processed/cleaned/swe_bench_train.jsonl`
- `datasets/processed/cleaned/swe_bench_dev.jsonl`
- `datasets/processed/cleaned/swe_bench_test.jsonl`
- `datasets/processed/cleaned/swe_bench_verified.jsonl`

## Step 2: Build weighted mix
Uses `configs/data_mix.yaml` to sample from cleaned sources with replacement until `output_size` is reached.

```powershell
python scripts/data/build_weighted_mix.py
```

Outputs:
- `datasets/weighted_mix/train_weighted_mix.jsonl`
- `datasets/weighted_mix/train_weighted_mix.manifest.json`

## Step 3: Normalize and deduplicate
Normalizes text and timestamps and writes a deduplicated variant.

```powershell
python scripts/data/normalize_data.py
```

Outputs:
- `datasets/processed/normalized/train_weighted_mix_normalized.jsonl`
- `datasets/processed/normalized/train_weighted_mix_normalized.manifest.json`
- `datasets/processed/deduplicated/train_weighted_mix_deduplicated.jsonl`
- `datasets/processed/deduplicated/train_weighted_mix_deduplicated.manifest.json`

## Optional: Run normalization via pipeline wrapper

```powershell
python scripts/data/run_pipeline.py
```

## Notes
- `build_weighted_mix.py` uses weighted sampling with replacement. The final row count is always `output_size` from `configs/data_mix.yaml`.
- If a source is small, it may be repeated many times to meet the requested size.
- The normalized and deduplicated outputs are based on the weighted mix output.

