# Base Model vs Fine-Tuned Model Testing Guide

This folder contains everything needed to train and evaluate the LoRA fine-tuned model. The evaluation script should compare the original base model against the same base model with the fine-tuned adapter loaded.

## Model Details

Base model:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Fine-tuning method:

```text
LoRA adapter using PEFT
```

The training script does not produce a full merged model by default. It produces a LoRA adapter that must be loaded on top of the same base model.

## Expected Fine-Tuned Output

After training, the fine-tuned files are saved in:

```text
server_upload/output/
```

Expected important files:

```text
output/adapter_config.json
output/adapter_model.safetensors
output/tokenizer.json
output/tokenizer_config.json
output/special_tokens_map.json
output/epoch_metrics.jsonl
output/epoch_01_metrics.json
output/epoch_02_metrics.json
...
```

There may also be checkpoint folders such as:

```text
output/checkpoint-250/
output/checkpoint-500/
```

For final evaluation, use either:

```text
server_upload/output/
```

or the checkpoint with the best validation loss if choosing manually from saved checkpoints.

## How the Fine-Tuned Model Should Be Loaded

The base model should be loaded normally:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
```

The fine-tuned model should load the same base model first, then attach the adapter:

```python
from peft import PeftModel

adapter_path = "server_upload/output"
finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
```

Important: the fine-tuned model is not a separate full model unless the adapter is explicitly merged. The adapter must be used with `Qwen/Qwen2.5-Coder-7B-Instruct`.

## Test Data Format

Use JSONL test data. Each line must be one JSON object.

Required fields:

```json
{
  "id": "unique-example-id",
  "source": "dataset-source-name",
  "task_type": "issue_resolution",
  "instruction": "Task instruction shown to the model.",
  "input": {
    "repo": "repository/name",
    "instance_id": "example-id",
    "base_commit": "commit-sha",
    "problem_statement": "Bug report or issue description",
    "hints_text": "Optional hints",
    "version": "project version",
    "created_at": "timestamp",
    "environment_setup_commit": "commit-sha",
    "FAIL_TO_PASS": ["tests expected to pass after fix"],
    "PASS_TO_PASS": ["tests that should continue passing"],
    "test_patch": "reference test patch"
  },
  "output": "reference expected patch or answer",
  "metadata": {}
}
```

The current held-out test file is:

```text
server_upload/data/test.jsonl
```

Current split sizes:

```text
train: 18,607 rows
validation: 300 rows
test: 200 rows
```

The evaluation script should not include the `output` field in the prompt sent to the model. The `output` field is the reference answer used for scoring.

## Prompt Construction

Use the same prompt style as training.

System prompt:

```text
You are a helpful coding assistant.
```

User prompt:

```text
{instruction}

Context:
{input as JSON}
```

The model should generate the assistant response, which should be compared with the `output` field.

## Fair Comparison Rules

Compare these two models on the same exact examples:

1. Base model only: `Qwen/Qwen2.5-Coder-7B-Instruct`
2. Fine-tuned model: `Qwen/Qwen2.5-Coder-7B-Instruct` + LoRA adapter from `server_upload/output`

Use the same:

```text
test examples
prompt format
max_new_tokens
temperature
top_p
seed if possible
stopping rules
hardware precision
```

Recommended deterministic generation settings:

```text
do_sample: false
temperature: not used
max_new_tokens: 2048 or higher if patch outputs are long
```

## Suggested Measurements

At minimum, save raw generations for both models:

```text
results/base_model_outputs.jsonl
results/finetuned_model_outputs.jsonl
```

Each result row should include:

```json
{
  "id": "example-id",
  "input": {},
  "reference_output": "expected patch",
  "base_model_output": "base model generated answer",
  "finetuned_model_output": "fine-tuned model generated answer"
}
```

Useful evaluation metrics:

```text
exact match against reference output
BLEU / ROUGE style text overlap
patch applies successfully
target FAIL_TO_PASS tests pass
PASS_TO_PASS tests remain passing
manual review score for patch correctness
```

For this dataset, the best measurement is whether the generated patch applies and fixes the target tests. Text similarity alone is weaker because different patches can solve the same issue.

## Notes for the Evaluation Script Author

- The fine-tuned output is a PEFT LoRA adapter, not a full standalone model.
- Always load the same base model before loading the adapter.
- Use `server_upload/data/test.jsonl` for held-out testing.
- Do not train or tune on the test file.
- Do not expose the `output` field to the model during generation.
- Save all generations and scores so the base and fine-tuned outputs can be inspected later.
