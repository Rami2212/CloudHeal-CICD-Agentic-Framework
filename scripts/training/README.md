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
```

## Run training

```powershell
python scripts/training/train_lora.py --config configs/train_lora_qwen25_7b_rocm.yaml
```

## Notes

- The trainer expects JSONL rows with `instruction`, `input`, and `output` fields.
- If you need a small dry run, set `max_train_samples` and `max_eval_samples` in the YAML config.

