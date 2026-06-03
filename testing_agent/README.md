# Testing Agent

Simple client-driven testing agent for running the base model and a LoRA-adapted model on the same jsonl dataset.

## Input format

Each jsonl line should be a JSON object with at least:

- `id`
- `instruction`
- `input` (optional; object or string)
- `output` (optional reference)

This mirrors the format used by `scripts/training/comparison.py`.

## Outputs

Depending on `--mode`:

- `base_outputs.jsonl`
- `finetuned_outputs.jsonl`
- `comparison_report.jsonl`

## Usage

```bash
python testing_agent/agent.py --data path\to\evaluation-test.jsonl --out-dir testing_agent\results --mode both --adapter-path path\to\adapter
```

## Notes

- The fine-tuned model requires a LoRA adapter path.
- `--cache-dir` can be used to place model downloads in a custom location.
- The prompt template is the same as `scripts/training/comparison.py`.

