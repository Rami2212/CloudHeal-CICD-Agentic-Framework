# Dataset README / Data Card

## Summary
- **Name:** CloudHeal CICD Agentic Dataset (working name)
- **Version:** v0.1 (update when finalized)
- **Created:** 2026-05-22
- **Purpose:** Instruction-tuning / SFT for coding assistant behavior
- **Primary model target:** Qwen2.5-Coder-7B (LoRA)

## Repository Paths
- **Annotated train:** `datasets/annotated/train.jsonl`
- **Annotated validation:** `datasets/annotated/val.jsonl`
- **Annotated test:** `datasets/annotated/test.jsonl`
- **Processed outputs:** `datasets/processed/`

## Data Sources
- **Upstream datasets:**
  - Add source list with URLs or citations.
- **Collection method:**
  - Describe how raw data was collected and curated.
- **License(s):**
  - Add licenses for all upstream sources and any derived data.

## Intended Use
- Supervised fine-tuning (SFT) of a code assistant for CI/CD and software maintenance tasks.
- Evaluation of repair, refactor, and build-fix tasks on curated examples.

## Out-of-Scope Use
- Real-time decision making without human review.
- Deploying outputs directly to production without validation.

## Schema
The training pipeline accepts **one of the following JSONL formats** per row:

### A) Instruction / Input / Output (preferred)
```json
{"instruction": "...", "input": "...", "output": "..."}
```

### B) Prompt / Completion
```json
{"prompt": "...", "completion": "..."}
```

### C) Chat Messages
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Notes:**
- If `messages` is used, the last assistant message is treated as the target.
- If `system` is absent, the training script inserts a default system prompt.

## Splits
- **Train:** `datasets/annotated/train.jsonl`
- **Validation:** `datasets/annotated/val.jsonl`
- **Test:** `datasets/annotated/test.jsonl`

Add counts once finalized:
- Train rows: `TBD`
- Validation rows: `TBD`
- Test rows: `TBD`

## Preprocessing
- Cleaning: normalization, removal of corrupt rows, and deduplication as needed.
- Formatting: standardized into one of the accepted schemas above.
- Tokenization: handled at training time with Qwen2.5-Coder tokenizer.

## Quality Checks
Run these commands to validate format and quality. Fill in results below.

```powershell
python scripts/training/smoke_data_check.py --file datasets/annotated/train.jsonl
python scripts/training/analyze_dataset.py --file datasets/annotated/train.jsonl --config configs/train.yaml
```

### Format Validation Results (fill in)
- Schema counts: `TBD`
- Rows with missing assistant output: `TBD`
- Rows with empty assistant output: `TBD`
- Unknown schema rows: `TBD`

### Token Length Distribution (fill in)
- Prompt tokens: avg `TBD`, p50 `TBD`, p90 `TBD`, p95 `TBD`, p99 `TBD`, max `TBD`
- Response tokens: avg `TBD`, p50 `TBD`, p90 `TBD`, p95 `TBD`, p99 `TBD`, max `TBD`
- Total tokens: avg `TBD`, p50 `TBD`, p90 `TBD`, p95 `TBD`, p99 `TBD`, max `TBD`
- Over `max_seq_length`: `TBD`

### Deduplication
- Method: `TBD` (e.g., exact match, MinHash, embedding-based)
- Duplicate rate: `TBD`

### Class/Task Balance
- Task categories: `TBD`
- Balance notes: `TBD`

## Bias, Safety, and Privacy
- **PII handling:** `TBD` (describe scrubbing or policy).
- **Safety filtering:** `TBD` (e.g., removal of secrets, credentials, or malware).
- **Known biases:** `TBD`.

## Limitations
- Data may over-represent certain repositories or task types.
- Outputs may not generalize to unseen toolchains or build systems.
- If dataset is small, the model may overfit to specific patterns.

## Versioning
- Keep a changelog of dataset revisions and major cleanup passes.
- Include hashes or manifests for reproducibility.

## Training Compatibility Notes
- The training script in `scripts/training/train_lora.py` supports all three schemas above.
- Prompt tokens are masked from loss; only assistant outputs contribute.

## Contact
- Owner: `TBD`
- Maintainer: `TBD`
- Issue tracker: `TBD`

