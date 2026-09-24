# CloudHeal Core Implementation Tickets
## Parts 2–4 — Context Processing, Failure Detection, and Root Cause Analysis

**Scope:** Implement Steps 2, 3, and 4 of the CloudHeal core framework.

**Flow covered:**

`Structured Failure Case -> Context Processing -> Failure Detection -> Root Cause Analysis`

**Outputs:**

- Normalized failure context
- Failure category + confidence/evidence
- Root-cause category + explanation/evidence

---

# Step 2 — Context Processing

## CH-CORE2-001 — Define the Normalized Context Schema

**Priority:** P0  
**Depends on:** Part 1 input contract

### Goal

Create the internal context representation used by Detection, RCA, and Fix Generation.

### Implementation Tasks

- Define a `NormalizedFailureContext` model.
- Preserve case identity:
  - repair ID
  - workflow run ID
  - source SHA
  - attempt number
- Add normalized fields:
  - cleaned log text
  - extracted error lines
  - extracted failed commands
  - failed job/step summary
  - normalized workflow/configuration content
  - relevant environment/runtime facts
  - dependency facts
  - relevant files
  - previous attempt feedback
- Add processing metadata:
  - original/processed size
  - redaction status
  - truncation status
  - extraction warnings

### Acceptance Criteria

- Detection and RCA consume this schema rather than the raw middleware payload.
- Important source identity is preserved.
- Every extracted evidence item can be related back to its source.

### Deliverable

Typed normalized context model.

---

## CH-CORE2-002 — Implement Log Cleaning

**Priority:** P0  
**Depends on:** CH-CORE2-001

### Goal

Remove irrelevant formatting/noise while preserving diagnostic meaning.

### Implementation Tasks

- Normalize newline formats.
- Strip ANSI/control sequences if any remain.
- Remove repeated blank lines.
- Collapse clearly repeated non-diagnostic output where safe.
- Preserve:
  - error messages
  - warnings near failure
  - commands
  - exit codes
  - stack/error summaries
- Keep line references or generated indexes for evidence linking.
- Record whether content was removed or truncated.

### Acceptance Criteria

- Cleaned logs remain human-readable.
- Important error lines survive cleaning.
- Evidence references are deterministic for the same input.
- Cleaning does not mutate workflow YAML.

### Tests

- ANSI-heavy logs.
- Repeated progress output.
- Multi-line exception.
- Exit-code failure.
- Logs already pre-cleaned by middleware.

### Deliverable

Deterministic log cleaner.

---

## CH-CORE2-003 — Extract Failure Markers and Error Lines

**Priority:** P0  
**Depends on:** CH-CORE2-002

### Goal

Extract the highest-value diagnostic evidence before model processing.

### Implementation Tasks

Detect and collect, where present:

- `error`
- `failed`
- `fatal`
- non-zero exit status
- command not found
- dependency/package resolution errors
- YAML/configuration parser errors
- permission/authentication errors
- runtime/version incompatibility messages
- GitHub Actions expression/action errors

For every extracted item store:

- line/index
- evidence text
- evidence type
- optional severity
- nearby context lines

### Acceptance Criteria

- Known fixture errors are extracted.
- Evidence includes enough surrounding text to understand the failure.
- Extraction rules are deterministic.
- Empty extraction does not crash the pipeline.

### Deliverable

Failure evidence extractor.

---

## CH-CORE2-004 — Extract Failed Commands and Step Context

**Priority:** P0  
**Depends on:** CH-CORE2-001, CH-CORE2-002

### Goal

Identify what command/action actually failed.

### Implementation Tasks

- Prefer structured failed-step information from middleware.
- Where possible, detect shell command lines from logs.
- Associate command with:
  - job
  - step
  - exit code
  - error evidence
- Normalize multiline commands without losing meaning.
- Identify GitHub Action `uses:` references relevant to the failed step.

### Acceptance Criteria

- At least one failed command/action is identified for supported fixtures where evidence exists.
- Structured middleware data takes precedence over heuristic extraction.
- Missing command data is represented explicitly rather than fabricated.

### Deliverable

Failed-command/step extraction module.

---

## CH-CORE2-005 — Normalize Workflow and Configuration Context

**Priority:** P0  
**Depends on:** CH-CORE2-001

### Goal

Prepare workflow/configuration evidence in a consistent representation.

### Implementation Tasks

- Parse workflow YAML using a safe YAML parser.
- Preserve the original YAML text.
- Build a normalized representation containing relevant:
  - triggers
  - jobs
  - steps
  - `uses`
  - `run`
  - permissions
  - environment fields
  - runner labels
- Associate the failed job/step with the corresponding workflow section.
- Detect YAML parse failure without blocking diagnostic processing of syntax-failure cases.
- Normalize relevant package/runtime configuration files only where supplied.

### Acceptance Criteria

- Valid workflow YAML produces a normalized structure.
- Invalid workflow YAML is retained as evidence and marked as parse-failed.
- Failed step mapping works for representative fixtures.

### Deliverable

Workflow/configuration normalizer.

---

## CH-CORE2-006 — Normalize Environment and Dependency Context

**Priority:** P1  
**Depends on:** CH-CORE2-001

### Goal

Expose environment and dependency facts to Detection/RCA in a predictable format.

### Implementation Tasks

Normalize available facts such as:

- runner OS
- runner image/label
- language/runtime version
- package manager
- dependency manifest
- lock file
- dependency/tool versions
- Docker/runtime configuration
- relevant version files

Do not infer unavailable facts.

### Acceptance Criteria

- Available environment/dependency values are normalized.
- Missing values remain explicit `null`/unknown values.
- No unsupported environment details are invented.

### Deliverable

Normalized environment/dependency section.

---

## CH-CORE2-007 — Build the Compact Model Context

**Priority:** P0  
**Depends on:** CH-CORE2-003 to CH-CORE2-006

### Goal

Create a bounded context object for Detection and RCA.

### Implementation Tasks

- Prioritize:
  1. failed job/step
  2. extracted errors
  3. failed command
  4. relevant workflow section
  5. relevant dependency/runtime files
  6. previous validation evidence for later attempts
- Apply configurable size/token budget.
- Truncate low-priority evidence before high-priority failure evidence.
- Preserve metadata indicating truncation.
- Produce separate model-ready sections instead of one uncontrolled text blob.

### Acceptance Criteria

- Model context stays within configured budget.
- Known critical error evidence is not dropped by normal truncation.
- Attempt 2+ includes the previous repair and newest validation failure when provided.

### Deliverable

Compact context builder.

---

## CH-CORE2-008 — Add Context Processing Unit Tests

**Priority:** P0  
**Depends on:** CH-CORE2-002 to CH-CORE2-007

### Goal

Make preprocessing reproducible and safe.

### Test Cases

- YAML syntax failure.
- Dependency conflict.
- Runtime mismatch.
- Missing command.
- Large noisy logs.
- Multiple error lines.
- Missing environment details.
- Later attempt with previous validation failure.
- Secret-redacted text.
- Invalid workflow YAML.

### Acceptance Criteria

- Tests verify exact important extracted fields.
- Same input produces the same normalized context.
- Context processing performs no model calls.

### Deliverable

Context processing test suite.

---

# Step 3 — Failure Detection

## CH-CORE3-001 — Define the Failure Taxonomy Configuration

**Priority:** P0  
**Depends on:** CH-CORE2-001

### Goal

Create one authoritative list of failure categories used by implementation and evaluation.

### Implementation Tasks

- Encode the selected GitHub Actions failure taxonomy.
- Give every category:
  - stable identifier
  - human-readable name
  - description
  - examples/indicators
- Add `UNKNOWN` / `UNSUPPORTED` behavior if required by the research plan.
- Prevent free-text category drift in model outputs.

### Acceptance Criteria

- Detection outputs only allowed category identifiers.
- Evaluation labels use the same identifiers.
- Category definitions are stored in one reusable module/configuration.

### Deliverable

Versioned failure taxonomy.

---

## CH-CORE3-002 — Define the Failure Detection Output Schema

**Priority:** P0  
**Depends on:** CH-CORE3-001

### Goal

Return measurable structured detection results.

### Fields

Include:

- failure category
- confidence
- supporting evidence references
- concise explanation
- optional alternative category
- uncertainty flag
- taxonomy version

### Acceptance Criteria

- Output is Pydantic-validated.
- Evidence references point to normalized context.
- Confidence is constrained to the chosen numeric range.
- Invalid category names are rejected.

### Deliverable

`FailureDetectionResult` schema.

---

## CH-CORE3-003 — Implement Deterministic Detection Signals

**Priority:** P0  
**Depends on:** CH-CORE2-003, CH-CORE3-001

### Goal

Use obvious structured signals before relying entirely on an LLM.

### Implementation Tasks

Create rules/features for common signals such as:

- malformed YAML
- missing command/tool
- invalid/deprecated action reference
- dependency installation failure
- version/runtime mismatch
- permissions/access failure
- workflow expression/configuration failure

The deterministic output should become input evidence to the final classifier rather than silently replacing uncertain classification.

### Acceptance Criteria

- Rules are deterministic and testable.
- A matched rule stores the evidence that triggered it.
- Multiple conflicting signals can be represented.
- No category is assigned without supporting input evidence.

### Deliverable

Detection signal extractor.

---

## CH-CORE3-004 — Implement Model-Assisted Failure Classification

**Priority:** P0  
**Depends on:** CH-CORE3-002, CH-CORE3-003, CH-CORE2-007

### Goal

Classify failures using bounded normalized context plus deterministic signals.

### Implementation Tasks

- Build the detection prompt/input format.
- Include:
  - allowed taxonomy
  - failed job/step
  - extracted errors
  - failed command
  - relevant workflow context
  - deterministic signals
- Require structured JSON output.
- Parse with Pydantic.
- Add a small formatting retry only when structured output is malformed.
- Do not allow the model to create new taxonomy categories.
- Record model/prompt version.

### Acceptance Criteria

- Supported fixtures produce valid structured detection results.
- Malformed model output never proceeds as a valid classification.
- The model cannot change the taxonomy.
- Result includes evidence, not only a category name.

### Deliverable

Failure classifier implementation.

---

## CH-CORE3-005 — Add Detection Confidence and Uncertainty Handling

**Priority:** P1  
**Depends on:** CH-CORE3-004

### Goal

Avoid treating every classification as equally certain.

### Implementation Tasks

- Standardize confidence representation.
- Define optional threshold/configuration for low-confidence cases.
- Store uncertainty reason where provided.
- Ensure low confidence is visible to RCA and final logging.
- Do not automatically invent certainty when evidence is weak.

### Acceptance Criteria

- Every detection result has a valid confidence/uncertainty representation.
- Low-confidence cases remain traceable.
- Confidence is logged for evaluation, but category metrics still compare against ground truth.

### Deliverable

Confidence and uncertainty handling.

---

## CH-CORE3-006 — Implement Detection Evaluation Harness

**Priority:** P0  
**Depends on:** CH-CORE3-004

### Goal

Run held-out labelled cases through Detection and export research metrics.

### Implementation Tasks

- Load held-out cases.
- Run Detection with fixed settings.
- Record:
  - case ID
  - ground-truth category
  - predicted category
  - confidence
  - evidence
  - latency
- Compute:
  - Macro-F1
  - Precision
  - Recall
  - Accuracy
- Export per-case predictions for error analysis.

### Acceptance Criteria

- Evaluation does not use training/prompt-development cases where the dataset design forbids it.
- Metrics are reproducible using fixed configuration.
- Macro-F1 is reported as the primary classification metric.

### Deliverable

Detection evaluation script/report data.

---

# Step 4 — Root Cause Analysis

## CH-CORE4-001 — Define the RCA Output Schema

**Priority:** P0  
**Depends on:** CH-CORE3-002

### Goal

Return a measurable and repair-oriented root-cause result.

### Suggested Fields

- root-cause category/label
- affected component/job/step
- supporting evidence references
- concise explanation
- recommended repair strategy
- uncertainty/confidence
- detected failure category
- attempt number

### Acceptance Criteria

- RCA is structured and Pydantic-validated.
- Evidence references map to the normalized context.
- The output is sufficient input for Fix Generation.
- Free-text explanation does not replace the structured cause label.

### Deliverable

`RootCauseAnalysisResult` schema.

---

## CH-CORE4-002 — Define the RCA Label Set / Ground-Truth Mapping

**Priority:** P0  
**Depends on:** CH-CORE4-001

### Goal

Ensure RCA can be evaluated using labels rather than only subjective text comparison.

### Implementation Tasks

- Define the supported root-cause labels based on the chosen evaluation cases/dataset.
- Map known held-out cases to one ground-truth label.
- Store expected supporting evidence where available.
- Document how multi-cause or ambiguous cases are handled.
- Keep failure category and root-cause label conceptually separate.

### Acceptance Criteria

- Every evaluable RCA case has a defined ground-truth cause.
- Unsupported/ambiguous cases are explicitly marked.
- Evaluation does not silently derive labels after seeing predictions.

### Deliverable

RCA label definitions and ground-truth mapping.

---

## CH-CORE4-003 — Build the RCA Prompt/Input Builder

**Priority:** P0  
**Depends on:** CH-CORE2-007, CH-CORE3-004, CH-CORE4-001

### Goal

Provide the RCA model with only the most relevant evidence.

### Input Should Include

- detection result
- failed job/step
- failed command/action
- extracted errors
- relevant workflow section
- environment/dependency context
- relevant files
- previous fix and newest validation failure for attempt 2+
- required RCA output schema

### Acceptance Criteria

- Detection evidence is carried into RCA.
- Later-attempt feedback is included when present.
- Prompt context respects configured size limits.
- GitHub credentials are never included.

### Deliverable

RCA context/prompt builder.

---

## CH-CORE4-004 — Implement Structured RCA Inference

**Priority:** P0  
**Depends on:** CH-CORE4-003

### Goal

Generate the root cause, explanation, evidence, and repair strategy.

### Implementation Tasks

- Call the configured model/service.
- Require structured JSON.
- Parse with Pydantic.
- Validate root-cause label.
- Validate evidence references against available evidence.
- Permit bounded formatting retry for malformed JSON.
- Record:
  - model version
  - prompt version
  - latency
  - input/output size or token counts when available

### Acceptance Criteria

- Supported fixtures produce parseable RCA.
- Invalid evidence references are rejected or removed with an explicit warning.
- Malformed model output does not become a valid RCA result.
- Later-attempt RCA can change based on new validation evidence.

### Deliverable

RCA inference module.

---

## CH-CORE4-005 — Implement RCA Evidence Linking

**Priority:** P0  
**Depends on:** CH-CORE4-004, CH-CORE2-003

### Goal

Make every RCA traceable to concrete input evidence.

### Implementation Tasks

- Resolve returned evidence IDs/line references to:
  - log lines
  - workflow/configuration snippets
  - failed command
  - environment/dependency facts
- Store an evidence summary with the RCA.
- Reject fabricated references.
- Support multiple evidence items per RCA.

### Acceptance Criteria

- An evaluator can inspect which evidence supported each root cause.
- Evidence references are machine-checkable.
- Missing evidence is explicitly represented.

### Deliverable

RCA evidence resolver.

---

## CH-CORE4-006 — Add RCA Uncertainty and Escalation Metadata

**Priority:** P1  
**Depends on:** CH-CORE4-004

### Goal

Capture cases where the root cause cannot be diagnosed reliably.

### Implementation Tasks

- Add uncertainty/confidence field.
- Add optional reason:
  - insufficient logs
  - conflicting evidence
  - unsupported failure type
  - missing repository context
- Make uncertainty visible to Fix Generation.
- Do not automatically stop the workflow unless configured by the experiment/runtime policy.

### Acceptance Criteria

- Uncertain RCA cases remain measurable.
- Missing evidence is not converted into fabricated explanations.
- Final logs distinguish confident and uncertain diagnoses.

### Deliverable

RCA uncertainty metadata.

---

## CH-CORE4-007 — Implement RCA Evaluation Harness

**Priority:** P0  
**Depends on:** CH-CORE4-004, CH-CORE4-005

### Goal

Evaluate RCA separately from final repair success.

### Implementation Tasks

For each held-out case record:

- case ID
- ground-truth root cause
- predicted root cause
- failure category
- predicted evidence
- expected evidence
- confidence/uncertainty
- latency

Compute:

- Macro-F1 / F1
- Precision
- Recall
- optional Evidence Recall where evidence labels exist

### Acceptance Criteria

- RCA metrics are separate from Detection metrics.
- RCA metrics are separate from final CI repair success.
- Per-case prediction data is available for thesis analysis.

### Deliverable

RCA evaluation script and structured results.

---

## CH-CORE4-008 — Implement Step 2–4 Integration Flow

**Priority:** P0  
**Depends on:** All mandatory Step 2–4 tickets

### Goal

Run a structured failure case through Context Processing, Detection, and RCA as one reproducible sequence.

### Implementation Tasks

- Wire:
  - input contract
  - context processor
  - detector
  - RCA
- Pass correlation metadata through every stage.
- Stop cleanly on:
  - invalid context
  - unrecoverable model output failure
  - unsupported input
- Record intermediate outputs.
- Keep Fix Generation disconnected until Step 5 is implemented.

### Acceptance Criteria

- A fixture enters as middleware input and exits with a structured RCA result.
- Intermediate normalized context and detection result are inspectable.
- The same fixture can be run repeatedly with fixed settings for evaluation.

### Deliverable

Integrated Steps 2–4 execution path.

---

# Completion Criteria for Parts 2–4

Parts 2–4 are complete when:

- Raw failure evidence is converted into a deterministic normalized context.
- Important errors, failed commands, workflow sections, environment, and dependency facts are extracted.
- Failure Detection returns a taxonomy category, confidence, and evidence.
- RCA returns a structured root cause, explanation, repair strategy, and evidence.
- Detection and RCA can each be evaluated independently on held-out cases.
- Later-attempt validation evidence can be accepted by Context Processing/RCA without changing the core contracts.
- No GitHub write operation occurs in these stages.
