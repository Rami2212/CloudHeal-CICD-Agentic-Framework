# LoRA Training Scripts

## Prereqs

- Python 3.10+
- ROCm-enabled PyTorch installed separately for your AMD GPU

## Install deps

```powershell
pip install -r requirements.txt
```


## Sanity-check the dataset

```powershell
python scripts/training/smoke_data_check.py --file datasets/annotated/train.jsonl
python scripts/training/smoke_data_check.py --train datasets/annotated/train.jsonl --val datasets/annotated/val.jsonl --test datasets/annotated/test.jsonl
```

## Run training

```powershell
python scripts/training/train.py --config configs/train.yaml
```

## Notes

- The trainer expects JSONL rows with `instruction`, `input`, and `output` fields, or compatible schemas.
- Before fine-tuning, `train.py` verifies `datasets/annotated/train.jsonl`, `val.jsonl`, and `test.jsonl` exist, are non-empty, and share the same top-level instruction format.
- Prompt tokens are masked from loss; only assistant outputs contribute.
- If you need a small dry run, set `max_train_samples` and `max_eval_samples` in the YAML config.
