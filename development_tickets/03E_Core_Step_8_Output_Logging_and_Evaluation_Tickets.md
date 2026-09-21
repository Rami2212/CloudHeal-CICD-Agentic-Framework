# CloudHeal Core Implementation Tickets
## Steps 5–8 Split Implementation Ticket Set

This file is one part of the detailed implementation tickets originally prepared for Core Steps 5–8.

# Step 8 — Output and Logging

## CH-CORE8-001 — Define the Final Framework Result Schema

**Priority:** P0  
**Depends on:** Steps 5–7

### Goal

Return one structured result to the middleware after every core invocation.

### Suggested Fields

- repair ID
- original workflow run ID
- final attempt number
- failure detection result
- final RCA
- validated patch if available
- final validation report
- retry/attempt history summary
- framework status:
  - `VALIDATED`
  - `RETRY_REQUIRED`
  - `ESCALATED`
  - `UNSUPPORTED`
  - `ERROR`
- timings
- trace/correlation ID
- experiment metadata

### Acceptance Criteria

- Middleware does not need to inspect internal LangGraph state.
- Validated and failed outcomes use the same stable response contract.
- No GitHub credentials appear in the output.

### Deliverable

`FrameworkResult` schema.

---

## CH-CORE8-002 — Implement Structured Attempt History

**Priority:** P0  
**Depends on:** CH-CORE7-001

### Goal

Preserve the full repair history for evaluation and debugging.

### For Each Attempt Record

- attempt number
- detection result/version if rerun
- RCA result
- patch hash
- fix description/type
- validation result
- refinement reason
- model latency
- validation duration
- final attempt status

### Acceptance Criteria

- Every generated candidate has one matching attempt record.
- Attempt history is ordered and serializable.
- Raw secrets are never included.

### Deliverable

Attempt history serializer.

---

## CH-CORE8-003 — Implement Experiment Logging

**Priority:** P0  
**Depends on:** CH-CORE8-001, CH-CORE8-002

### Goal

Store the data required for the research evaluation.

### Minimum Data to Store

- case/session ID
- ground-truth labels when running evaluation
- predicted failure category
- RCA category
- evidence identifiers
- patch hash/fix type
- deterministic validation pass/fail
- validation errors
- retry count
- final core outcome
- model latency
- validation timings
- prompt/model/tool versions

### Storage

Implement a simple structured format first:

- JSON/JSONL and/or CSV

Add PostgreSQL persistence if the runtime already uses it.

### Acceptance Criteria

- Evaluation data can be exported without parsing application logs.
- One row/object can be linked to one case and attempt.
- Fixed experiment configuration is recorded.

### Deliverable

Experiment logger/exporter.

---

## CH-CORE8-004 — Add Timing and Efficiency Metrics

**Priority:** P1  
**Depends on:** CH-CORE8-003

### Goal

Measure runtime behavior in addition to classification/repair correctness.

### Record

- Context Processing duration
- Detection duration
- RCA duration
- Fix Generation duration
- Validation duration
- total core attempt duration
- total core invocation duration
- model input/output token counts if available
- number of refinement attempts

### Acceptance Criteria

- Timings use one consistent unit.
- Metrics are available per attempt and per case.
- Timing collection does not change functional behavior.

### Deliverable

Timing instrumentation.

---

## CH-CORE8-005 — Add Optional Langfuse Tracing

**Priority:** P2 / Optional  
**Depends on:** CH-CORE8-003

### Goal

Provide richer tracing without making Langfuse authoritative runtime state.

### Trace Suggested Spans

- core invocation
- context processing
- detection
- RCA
- fix generation
- deterministic validation
- retry/refinement
- final output

### Acceptance Criteria

- Core still functions when Langfuse is disabled/unavailable.
- Secret-redacted inputs only are exported.
- `repair_id` and attempt number correlate traces.

### Deliverable

Optional observability adapter.

---

## CH-CORE8-006 — Implement Core End-to-End Test Harness

**Priority:** P0  
**Depends on:** All mandatory Steps 1–8 tickets

### Goal

Run a held-out/sandbox failure case through the complete internal framework.

### Flow

1. Load structured failure case.
2. Context Processing.
3. Detection.
4. RCA.
5. Fix Generation.
6. Deterministic Validation.
7. Refinement when necessary.
8. Final result generation.
9. Experiment log export.

### Acceptance Criteria

- No manual data reformatting between stages.
- Every intermediate output is structured.
- All attempts are logged.
- The final result can be handed to the separate GitHub middleware.

### Deliverable

Complete core integration test harness.

---

## CH-CORE8-007 — Implement Component Evaluation Export

**Priority:** P0  
**Depends on:** CH-CORE8-003, Detection/RCA evaluation harnesses

### Goal

Produce the data required for the framework evaluation plan.

### Required Metrics

**Failure Detection**

- Macro-F1
- Precision
- Recall
- Accuracy

**Root Cause Analysis**

- Macro-F1 / F1
- Precision
- Recall
- optional Evidence Recall

**Validation and Refinement**

- Validation Pass Rate
- Refinement Success Rate
- Average Retry Count

### Acceptance Criteria

- Metrics are calculated from held-out case records.
- Number of evaluated cases is always reported.
- Component results remain separate from end-to-end remote CI repair success.

### Deliverable

Core evaluation export/report script.

---

## CH-CORE8-008 — Prepare Middleware Handoff Result

**Priority:** P0  
**Depends on:** CH-CORE8-001

### Goal

Make the final core output directly usable by the GitHub Integration Middleware.

### Validated Result Must Include

- target file/path
- validated patch
- final RCA
- fix description/type
- validation report
- attempt number
- candidate hash
- trace ID

### Failed/Escalated Result Must Include

- final RCA
- all blocking validation errors
- attempt count
- reason for escalation
- attempt history summary

### Acceptance Criteria

- Middleware can decide whether to create/update the repair commit without reading internal core state.
- Escalated cases cannot be mistaken for validated repairs.

### Deliverable

Stable middleware handoff mapping.

---

## CH-CORE8-009 — Document the Final Experiment Procedure

**Priority:** P1  
**Depends on:** CH-CORE8-006, CH-CORE8-007

### Goal

Make the research experiment reproducible.

### Document

- held-out dataset/case source
- case inclusion/exclusion rules
- model version
- LoRA adapter version
- prompt versions
- taxonomy version
- maximum attempts = 3
- validation tool versions
- enabled validation checks
- fixed generation settings
- metric definitions
- result export location

### Acceptance Criteria

- Another group member can repeat the experiment using the documented configuration.
- Changes to model/prompt/validation configuration create a new experiment version.

### Deliverable

Experiment runbook/configuration document.

---

# Completion Criteria for Parts 5–8

Parts 5–8 are complete when:

- RCA/context can be sent to the existing fine-tuned Qwen model using a fixed structured request.
- The model returns a patch-style repair that is parsed into a typed candidate.
- Every candidate is validated in an isolated workspace.
- Patch applicability, YAML validity, actionlint, scope, and safety checks are enforced.
- Invalid repairs are fed back into RCA/Fix Generation.
- A maximum of 3 attempts is enforced.
- Remote CI failure evidence can re-enter the same core repair flow through the middleware.
- A final structured result is returned to middleware.
- Predictions, RCA, candidate hashes, validation results, retry history, timings, and final outcomes are stored for evaluation.
- The core can be run end-to-end on held-out failure cases without manual data transformation.
