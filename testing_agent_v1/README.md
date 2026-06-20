# Testing Agent

`new_agent.py` evaluates the base CI/CD remediation model against the same
base model with a PEFT/LoRA fine-tuned adapter attached.

## What It Does

- Generates 10 synthetic CI/CD failure logs with expected unified-diff fixes.
- Runs the same optimized CloudHeal prompt against the base model.
- Runs the same prompt against the base model plus fine-tuned adapter.
- Saves generated fixes separately for each model.
- Builds an evaluation report for accuracy, estimated resolution success rate,
  MTTR, patch format, target-file coverage, keyword coverage, and safety checks.

## Usage

Generate only the benchmark dataset:

```bash
python testing_agent/new_agent.py --mode data
```

Run the full real-model evaluation:

```bash
python testing_agent/new_agent.py --mode all --adapter-path testing_agent/adapter
```

Smoke-test the workflow without loading model weights:

```bash
python testing_agent/new_agent.py --mode all --mock-model --out-dir testing_agent/results_mock
```

## Outputs

Default dataset:

```text
testing_agent/data/cicd_failure_benchmark.jsonl
```

Default result files:

```text
testing_agent/results/base_model_fixes.jsonl
testing_agent/results/finetuned_model_fixes.jsonl
testing_agent/results/comparison_report.jsonl
testing_agent/results/evaluation_report.json
testing_agent/results/evaluation_report.md
testing_agent/results/base_model_fixes/*.patch
testing_agent/results/finetuned_model_fixes/*.patch
```

The fine-tuned run requires a valid PEFT adapter directory containing
`adapter_config.json` and adapter weights.

