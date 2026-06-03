# Data Preparation Guide

This document describes the full data-preparation workflow used in this repository.

The goal is to:

1. clean the raw datasets,
2. split `eval_DataSet.jsonl` into a `3:2` ratio,
3. normalize and deduplicate the two split datasets separately,
4. mix the two training datasets in a deterministic `5:1` ratio,
5. and produce the final JSONL files and manifests used for training.

---

## 1. Input datasets

The repository uses two main dataset sources for the training mix:

- `datasets/processed/cleaned/lca_ci_builds_repair.jsonl`
- `datasets/processed/cleaned/swe_bench_train.jsonl`

For evaluation preparation, the dataset to split is:

- `datasets/processed/cleaned/eval_DataSet.jsonl`

All later steps assume these cleaned JSONL files already exist.

---

## 2. Clean the raw data

The cleaning step converts raw source files into normalized JSONL files under `datasets/processed/cleaned/`.

Command:

```powershell
python scripts/data/clean_data.py
```

Expected cleaned outputs:

- `datasets/processed/cleaned/lca_ci_builds_repair.jsonl`
- `datasets/processed/cleaned/swe_bench_train.jsonl`
- `datasets/processed/cleaned/swe_bench_dev.jsonl`
- `datasets/processed/cleaned/swe_bench_test.jsonl`
- `datasets/processed/cleaned/swe_bench_verified.jsonl`
- `datasets/processed/cleaned/eval_DataSet.jsonl`

---

## 3. Split `eval_DataSet.jsonl` into `3:2`

The eval dataset is split deterministically into two parts using a fixed seed.

Recommended split command:

```powershell
python src/data_pipeline/split_dataset.py `
  --input datasets/processed/cleaned/eval_DataSet.jsonl `
  --out-a datasets/processed/splits/eval_300.jsonl `
  --out-b datasets/processed/splits/eval_200.jsonl `
  --ratio 3 2 `
  --seed 42
```

What this does:

- shuffles rows with seed `42`,
- writes the first `3/5` of rows to `eval_300.jsonl`,
- writes the remaining `2/5` of rows to `eval_200.jsonl`,
- creates a manifest describing the split.

Expected outputs:

- `datasets/processed/splits/eval_300.jsonl`
- `datasets/processed/splits/eval_200.jsonl`
- `datasets/processed/splits/eval_300.manifest.json` or the manifest path you pass explicitly

---

## 4. Normalize the two split datasets separately

After splitting, normalize each split independently.

Normalization should:

- convert rows into the expected training schema,
- standardize text and timestamps,
- preserve provenance metadata,
- write a manifest,
- and write rejected rows if any record cannot be normalized.

### Normalize `eval_300.jsonl`

```powershell
python src/data_pipeline/normalize_jsonl.py `
  --input datasets/processed/splits/eval_300.jsonl `
  --output datasets/processed/normalized/eval_300_normalized.jsonl `
  --rejected datasets/processed/normalized/eval_300_rejected.jsonl `
  --manifest datasets/processed/normalized/eval_300_normalized.manifest.json
```

### Normalize `eval_200.jsonl`

```powershell
python src/data_pipeline/normalize_jsonl.py `
  --input datasets/processed/splits/eval_200.jsonl `
  --output datasets/processed/normalized/eval_200_normalized.jsonl `
  --rejected datasets/processed/normalized/eval_200_rejected.jsonl `
  --manifest datasets/processed/normalized/eval_200_normalized.manifest.json
```

Expected normalized outputs:

- `datasets/processed/normalized/eval_300_normalized.jsonl`
- `datasets/processed/normalized/eval_300_normalized.manifest.json`
- `datasets/processed/normalized/eval_300_rejected.jsonl`
- `datasets/processed/normalized/eval_200_normalized.jsonl`
- `datasets/processed/normalized/eval_200_normalized.manifest.json`
- `datasets/processed/normalized/eval_200_rejected.jsonl`

---

## 5. Deduplicate the two normalized datasets separately

Once each split is normalized, deduplicate them independently.

Deduplication strategy:

- compute a canonical JSON signature for each row,
- keep the first occurrence of each unique row,
- drop later duplicates,
- write a manifest with `duplicates_removed`.

### Deduplicate `eval_300_normalized.jsonl`

```powershell
python src/data_pipeline/dedupe_jsonl.py `
  --input datasets/processed/normalized/eval_300_normalized.jsonl `
  --output datasets/processed/deduplicated/eval_300_deduplicated.jsonl `
  --manifest datasets/processed/deduplicated/eval_300_deduplicated.manifest.json
```

### Deduplicate `eval_200_normalized.jsonl`

```powershell
python src/data_pipeline/dedupe_jsonl.py `
  --input datasets/processed/normalized/eval_200_normalized.jsonl `
  --output datasets/processed/deduplicated/eval_200_deduplicated.jsonl `
  --manifest datasets/processed/deduplicated/eval_200_deduplicated.manifest.json
```

Expected deduplicated outputs:

- `datasets/processed/deduplicated/eval_300_deduplicated.jsonl`
- `datasets/processed/deduplicated/eval_300_deduplicated.manifest.json`
- `datasets/processed/deduplicated/eval_200_deduplicated.jsonl`
- `datasets/processed/deduplicated/eval_200_deduplicated.manifest.json`

---

## 6. Build the final training mix in a `5:1` sequence

The training mix combines the two datasets using a `5:1` ratio.

In this repository, the current configuration is:

- `lca_ci_builds_repair` weight: `5`
- `swe_bench` weight: `1`

This means the training sampler selects the LCA dataset five times as often as the SWE-bench dataset.

### Configuration file

The mix is controlled by:

- `configs/data_mix.yaml`

Current structure:

```yaml
seed: 42
output_size: 20000
sources:
  - name: lca_ci_builds_repair
    path: datasets/processed/cleaned/lca_ci_builds_repair.jsonl
    weight: 5
  - name: swe_bench
    path: datasets/processed/cleaned/swe_bench_train.jsonl
    weight: 1
output_path: datasets/weighted_mix/train_weighted_mix.jsonl
manifest_path: datasets/weighted_mix/train_weighted_mix.manifest.json
```

### Build command

```powershell
python scripts/data/build_weighted_mix.py
```

What this does:

- reads the cleaned source datasets,
- samples rows using weights `5:1`,
- writes the mixed training JSONL file,
- writes a manifest with source weights and actual sample counts.

Expected outputs:

- `datasets/weighted_mix/train_weighted_mix.jsonl`
- `datasets/weighted_mix/train_weighted_mix.manifest.json`

---

## 7. Controlled `5:1` repetition mode

If you want the primary dataset repeated exactly by a fixed factor instead of probabilistic weighted sampling, use the controlled mix script.

Example:

```powershell
python scripts/data/build_controlled_weighted_mix.py --factor 5 --seed 42
```

Behavior:

- keep the secondary dataset once,
- repeat the primary dataset five times,
- shuffle deterministically,
- write the final mixed JSONL and manifest.

Use this mode when you want a more explicit `5:1` data-preparation rule.

---

## 8. Normalize and deduplicate the mixed training set

After the mix is created, run the training normalization and deduplication step if your training pipeline expects that final cleaned format.

Existing pipeline entry point:

```powershell
python scripts/data/run_pipeline.py
```

This pipeline currently wraps the weighted-mix normalization flow in `scripts/data/normalize_data.py`.

Outputs usually include:

- normalized training JSONL,
- deduplicated training JSONL,
- rejected rows for debugging,
- and manifests for reproducibility.

---

## 9. Recommended order of operations

Use this order for the full preparation process:

1. `python scripts/data/clean_data.py`
2. `python src/data_pipeline/split_dataset.py --input datasets/processed/cleaned/eval_DataSet.jsonl --out-a datasets/processed/splits/eval_300.jsonl --out-b datasets/processed/splits/eval_200.jsonl --ratio 3 2 --seed 42`
3. `python src/data_pipeline/normalize_jsonl.py` on each split
4. `python src/data_pipeline/dedupe_jsonl.py` on each normalized split
5. `python scripts/data/build_weighted_mix.py` or `python scripts/data/build_controlled_weighted_mix.py --factor 5 --seed 42`
6. `python scripts/data/run_pipeline.py` if you need the final weighted-mix normalization/deduplication stage

---

## 10. Important notes

- The `3:2` split is deterministic because it uses a fixed shuffle seed.
- Normalization and deduplication are done separately for the two split datasets so each file can be inspected on its own.
- The final training mix is `5:1`, with the LCA source upweighted relative to SWE-bench.
- A manifest is written at each major stage so the dataset lineage remains traceable.
- If a source dataset is very small, repeated sampling can create many duplicates; the controlled mix mode is better when you want repeatable data proportions.

