# Base Model vs Fine-Tuned Model Testing Report

This folder contains the saved evaluation outputs comparing the original base model with the fine-tuned LoRA adapter.

## Tested Models

Base model:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Fine-tuned model:

```text
Qwen/Qwen2.5-Coder-7B-Instruct + PEFT LoRA adapter from server_upload/output
```

The fine-tuned model is not a standalone full model. It is the same Qwen base model with the trained LoRA adapter attached.

## Result Files

Generated files:

```text
base_outputs.jsonl
finetuned_outputs.jsonl
comparison_report.jsonl
```

File sizes:

```text
base_outputs.jsonl: 14,782 bytes
finetuned_outputs.jsonl: 10,210 bytes
comparison_report.jsonl: 22,709 bytes
```

`base_outputs.jsonl` contains one row per example with:

```text
id
instruction
reference_output
base_model_output
```

`finetuned_outputs.jsonl` contains one row per example with:

```text
id
instruction
reference_output
finetuned_model_output
```

`comparison_report.jsonl` joins both model outputs into one file:

```text
id
instruction
reference_output
base_model_output
finetuned_model_output
```

## Evaluation Dataset

Number of evaluated examples:

```text
8
```

Example IDs:

```text
ci-failure-2001
ci-failure-2002
ci-failure-2003
ci-failure-2004
ci-failure-2005
ci-failure-2006
ci-failure-2007
ci-failure-2008
```

Task type:

```text
CI/CD issue-resolution patch generation
```

Each example includes:

```text
instruction
reference_output
model generated output
```

The reference output is a unified-diff-style patch. The goal is for the model to produce a useful patch-like answer that fixes the described issue.

## Metrics Used

The saved outputs were analyzed with these lightweight metrics:

```text
non-empty output count
exact match against reference patch
inclusion match against reference patch
patch-like output count
markdown code block count
explanatory prose count
average output length
minimum output length
maximum output length
```

Definitions:

```text
exact match: generated output exactly equals the reference output after stripping whitespace
inclusion match: reference output appears inside the generated output
patch-like output: output begins with diff/unified-diff markers such as diff --git, --- a/, +++ b/
explanatory prose: output contains natural-language explanation patterns instead of only patch text
```

## Overall Results

### Base Model

```text
non-empty outputs: 8/8
exact match: 0/8
inclusion match: 0/8
patch-like outputs: 0/8
markdown code blocks: 8/8
explanatory prose outputs: 8/8
average output length: 1,498.6 characters
minimum output length: 988 characters
maximum output length: 2,447 characters
```

### Fine-Tuned Model

```text
non-empty outputs: 8/8
exact match: 0/8
inclusion match: 0/8
patch-like outputs: 8/8
markdown code blocks: 0/8
explanatory prose outputs: 0/8
average output length: 923.2 characters
minimum output length: 448 characters
maximum output length: 2,128 characters
```

## Per-Example Summary

```text
ci-failure-2001 | base patch-like: no | fine-tuned patch-like: yes | base chars: 1743 | fine-tuned chars: 448
ci-failure-2002 | base patch-like: no | fine-tuned patch-like: yes | base chars: 1515 | fine-tuned chars: 952
ci-failure-2003 | base patch-like: no | fine-tuned patch-like: yes | base chars: 1638 | fine-tuned chars: 811
ci-failure-2004 | base patch-like: no | fine-tuned patch-like: yes | base chars: 1229 | fine-tuned chars: 919
ci-failure-2005 | base patch-like: no | fine-tuned patch-like: yes | base chars: 2447 | fine-tuned chars: 628
ci-failure-2006 | base patch-like: no | fine-tuned patch-like: yes | base chars: 1235 | fine-tuned chars: 712
ci-failure-2007 | base patch-like: no | fine-tuned patch-like: yes | base chars: 1194 | fine-tuned chars: 2128
ci-failure-2008 | base patch-like: no | fine-tuned patch-like: yes | base chars: 988 | fine-tuned chars: 788
```

## What Changed After Fine-Tuning

The base model answered like a general coding assistant. It explained the bug, described a possible fix, and often included Markdown code snippets or testing advice. This is useful for human guidance, but it is not the desired output format for this dataset.

The fine-tuned model answered like a patch-generation model. All 8 fine-tuned outputs were unified-diff-style patches. The outputs were shorter on average and did not include extra explanatory prose or Markdown code fences.

Main behavioral improvement:

```text
The fine-tuned model learned the expected patch-output format.
```

## Effectiveness Interpretation

The test shows a strong format-alignment improvement:

```text
Base model patch-like output rate: 0%
Fine-tuned model patch-like output rate: 100%
```

The test does not show exact-reference improvement:

```text
Base model exact match: 0%
Fine-tuned model exact match: 0%
Base model inclusion match: 0%
Fine-tuned model inclusion match: 0%
```

This means the fine-tuned model is clearly better at producing the kind of answer expected by the task, but this evaluation alone does not prove the patches are correct.

For code patches, exact text match is a weak metric because multiple different patches can solve the same bug. A generated patch can be useful even when it does not exactly match the reference patch.

## Qualitative Observations

Base model behavior:.

```text
Natural-language explanations
Markdown code blocks
Suggested fixes rather than direct patches
Longer outputs
Human-oriented guidance
```

Fine-tuned model behavior:

```text
Unified diff format
Direct patch-style answers
Shorter outputs on average
No Markdown wrapping
No extra explanation
More aligned with the training target
```

Observed limitation:

```text
Some fine-tuned patches modify plausible but non-reference file paths or make different implementation choices than the reference patch.
```

This is why patch application and test execution are needed before claiming functional success.

## Current Conclusion

Based on these result files, the fine-tuned model is more effective at matching the expected output style of the dataset. It consistently produces patch-like outputs, while the base model produces explanatory prose.

However, the current results do not prove functional correctness. The next evaluation should check whether each generated patch:

```text
applies cleanly to the target repository
fixes the FAIL_TO_PASS tests
keeps the PASS_TO_PASS tests passing
uses the correct files
avoids unnecessary changes
```

Short conclusion:

```text
Fine-tuning improved format alignment strongly.
Functional patch correctness still needs deeper evaluation.
```

## Recommended Next Testing Step

Use `comparison_report.jsonl` as the source file for deeper analysis. It already contains both model outputs beside the same reference answer.

Recommended next metrics:

```text
patch parse success rate
patch apply success rate
FAIL_TO_PASS pass rate
PASS_TO_PASS retention rate
manual correctness score
file-path accuracy
minimality of patch
```

For a research report, the strongest metric will be test execution:

```text
generated patch + target repo + FAIL_TO_PASS/PASS_TO_PASS tests
```

That will measure whether the model actually fixes CI/CD failures, not only whether it writes patch-shaped text.
