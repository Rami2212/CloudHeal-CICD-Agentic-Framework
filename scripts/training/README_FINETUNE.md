# CloudHeal Fine-Tuned Model Report

This folder contains the completed LoRA fine-tuning output for the CloudHeal CI/CD issue-resolution model. The model was fine-tuned from `Qwen/Qwen2.5-Coder-7B-Instruct` to generate code patch-style answers for software issue-resolution examples.

## Final Artifact Summary

Fine-tuned artifact type:

```text
PEFT LoRA adapter
```

Base model:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Final adapter location:

```text
server_upload/output/
```

Important files:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja
training_args.bin
epoch_metrics.jsonl
epoch_01_metrics.json
epoch_02_metrics.json
```

Saved checkpoint folders:

```text
checkpoint-4250/
checkpoint-4500/
checkpoint-4652/
```

The final adapter file is approximately `323 MB`:

```text
adapter_model.safetensors: 323,014,168 bytes
```

This is not a full merged model. To use it, load the original Qwen base model first and then attach this LoRA adapter.

## Training Data

The training data used the instruction-tuning JSONL format with these fields:

```text
id
source
task_type
instruction
input
output
metadata
```

Dataset split sizes:

```text
train: 18,607 rows
validation: 300 rows
test: 200 rows
```

The examples are issue-resolution/code-patch tasks. The model receives the `instruction` and structured `input` context, and learns to generate the `output` patch.

The `output` field is the supervised target during training. It must not be included in prompts during evaluation or real inference.

## Prompt Format

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

The tokenizer chat template is used when available.

## Training Configuration

Observed final training run:

```text
epochs: 2
global steps: 4,652
train batch size: 8
effective batch size: 8
max sequence length: 2,048
learning rate: 0.0001
warmup steps: 140
learning-rate scheduler: cosine
weight decay: 0.01
max grad norm: 1.0
eval steps: 100
save steps: 250
save total limit: 3
gradient checkpointing: false
precision: bf16 when supported
seed: 42
```

LoRA configuration from `adapter_config.json`:

```text
PEFT type: LORA
task type: CAUSAL_LM
r: 32
lora_alpha: 64
lora_dropout: 0.05
bias: none
base_model_name_or_path: Qwen/Qwen2.5-Coder-7B-Instruct
PEFT version: 0.19.1
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

## Training Progress

Training completed successfully at:

```text
global_step: 4652
epoch: 2.0
```

First logged training loss:

```text
step: 10
epoch: 0.0043
loss: 0.8572
learning_rate: 0.00000643
grad_norm: 0.2630
```

Final logged training loss:

```text
step: 4650
epoch: 1.9991
loss: 0.5268
learning_rate: near zero
grad_norm: 0.4635
```

Average of the last 10 logged training losses:

```text
0.5656
```

This indicates that the model learned the training distribution and the training loss decreased substantially from the beginning of training.

## Validation Metrics

Validation was evaluated every 100 steps. There are 47 validation logs in the saved trainer state.

First validation result:

```text
step: 100
epoch: 0.0430
eval_loss: 0.7294
eval_samples_per_second: 11.195
eval_steps_per_second: 1.418
```

Best observed validation result:

```text
step: 2100
epoch: 0.9028
eval_loss: 0.6870
eval_samples_per_second: 11.200
eval_steps_per_second: 1.419
```

Final validation result:

```text
step: 4652
epoch: 2.0
eval_loss: 0.7071
eval_samples_per_second: 11.208
eval_steps_per_second: 1.420
```

Interpretation:

```text
Validation loss improved from 0.7294 to a best observed value of 0.6870.
By the final step, validation loss increased to 0.7071.
```

This suggests the best generalization was observed around step `2100`, near the end of epoch 1. The final adapter still completed correctly, but the validation curve shows mild overfitting or diminishing returns during the second epoch.

Important note: the best checkpoint at step `2100` is not present in this output folder. The retained checkpoints are `4250`, `4500`, and `4652`, because the run kept only the latest saved checkpoints.

## Evaluation Results

Evaluation result files were generated in:

```text
server_upload/results/
```

Files:

```text
base_outputs.jsonl
finetuned_outputs.jsonl
comparison_report.jsonl
```

Evaluation sample size:

```text
8 examples
```

Text-match scores on the 8-example evaluation set:

```text
Base model exact match: 0/8
Fine-tuned model exact match: 0/8
Base model inclusion match: 0/8
Fine-tuned model inclusion match: 0/8
```

Patch-format behavior:

```text
Base model patch-like outputs: 0/8
Fine-tuned model patch-like outputs: 8/8
Base model non-empty outputs: 8/8
Fine-tuned model non-empty outputs: 8/8
```

Average generated output length:

```text
Base model: 1,498.6 characters
Fine-tuned model: 923.2 characters
```

## Effectiveness Interpretation

The fine-tuned model clearly changed behavior compared with the base model:

```text
Base model behavior: mostly explanatory prose with suggested code snippets.
Fine-tuned behavior: unified-diff-style patch outputs.
```

This is an important improvement for the target task because the expected output format is a code patch, not a natural-language explanation.

However, the current 8-example evaluation does not prove patch correctness. Exact match and inclusion match are both 0 for the base and fine-tuned models. This is not unusual for patch-generation tasks because a correct patch may differ textually from the reference patch, but it means the current metric is not enough to claim functional correctness.

Current conclusion:

```text
The fine-tuned model is more aligned with the desired output format.
The current evaluation does not yet prove that the generated patches are functionally correct.
```

For a stronger effectiveness claim, evaluate whether each generated patch:

```text
applies cleanly to the target repository
passes the FAIL_TO_PASS tests
keeps the PASS_TO_PASS tests passing
uses the correct files and minimal changes
```

## How to Load the Fine-Tuned Adapter

Example:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_path = "server_upload/output"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

## Recommended Inference Settings

For deterministic comparison:

```text
do_sample: false
max_new_tokens: 2048
pad_token_id: tokenizer.eos_token_id
```

For more exploratory generation:

```text
do_sample: true
temperature: 0.2 to 0.4
top_p: 0.9
max_new_tokens: 2048
```

For research comparison, keep deterministic decoding so the base and fine-tuned model are evaluated fairly.

## Known Limitations

- The adapter is not a standalone model; it must be loaded with `Qwen/Qwen2.5-Coder-7B-Instruct`.
- The best observed validation loss happened earlier than the final checkpoint.
- The retained checkpoints do not include the best observed step `2100`.
- The 8-example effectiveness evaluation is too small for a final research claim.
- Exact text match is weak for code patches because semantically correct patches can differ from the reference patch.
- Functional patch application and test execution are still needed for strong effectiveness measurement.

## Recommended Next Steps

1. Run evaluation on the full held-out `test.jsonl` split, not only the 8-example evaluation file.
2. Add patch-apply validation.
3. Add FAIL_TO_PASS and PASS_TO_PASS test execution where repository setup is available.
4. In future training runs, enable best-checkpoint tracking so the best validation checkpoint is retained.
5. Consider training for around 1 epoch or using early stopping, since the best validation loss occurred near epoch `0.90`.

## Short Final Summary

Fine-tuning completed successfully and produced a valid PEFT LoRA adapter. Training loss decreased substantially, and validation loss improved from `0.7294` to a best observed `0.6870`. The final validation loss was `0.7071`, suggesting mild overfitting or reduced generalization after the best point.

The fine-tuned model is more format-aligned than the base model: it generated patch-style outputs for all evaluation examples, while the base model generated explanatory prose. The current evaluation does not yet prove functional patch correctness, so the strongest next step is patch-apply and test-pass evaluation.
