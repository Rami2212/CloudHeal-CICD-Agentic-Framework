# CloudHeal Core Implementation Tickets
## Parts 5–8 — Fix Generation, Deterministic Validation, Iterative Refinement, Output and Logging

**Scope:** Implement Steps 5, 6, 7, and 8 of the CloudHeal core framework.

**Flow covered:**

`RCA -> Fix Generation -> Deterministic Validation -> Refinement -> Framework Output + Experiment Logging`

**Primary runtime rule:** The existing fine-tuned Qwen2.5-Coder model is integrated as the repair generator. CloudHeal validates every candidate before returning it to the separate GitHub Integration Middleware.

**Retry rule:** Maximum baseline repair attempts: **3**.

---

# Step 5 — Fix Generation Integration

## CH-CORE5-001 — Define the Fix Generation Input Contract

**Priority:** P0  
**Depends on:** Completed RCA output

### Goal

Create the exact structured input provided to the existing fine-tuned repair model.

### Input Should Include

- repair ID
- attempt number
- detected failure category
- RCA label/category
- RCA explanation
- supporting evidence
- recommended repair strategy
- workflow/configuration target
- relevant source/configuration context
- environment/dependency context
- previous patch for attempt 2+
- previous validation errors/logs for attempt 2+

### Acceptance Criteria

- Fix Generation receives only structured, bounded context.
- The repair model does not need the original raw middleware payload.
- Later attempts contain the previous candidate and validation feedback.
- Credentials/secrets are excluded.

### Deliverable

Typed `FixGenerationInput` schema.

---

## CH-CORE5-002 — Define the Candidate Patch Output Schema

**Priority:** P0  
**Depends on:** CH-CORE5-001

### Goal

Require a machine-validated repair representation.

### Suggested Fields

- repair ID
- attempt number
- target file path
- unified/patch-style repair content
- fix type
- concise fix description
- related root-cause label
- optional model confidence
- model metadata

### Implementation Tasks

- Define the Pydantic output model.
- Restrict the output to the allowed target scope.
- Reject empty patches.
- Require one target repair scope according to the initial implementation plan.
- Preserve the raw model response separately only for controlled debugging if needed.

### Acceptance Criteria

- A valid model response becomes one typed candidate patch.
- Invalid path/empty patch is rejected.
- Malformed JSON cannot reach validation as a candidate.

### Deliverable

`CandidatePatch` schema.

---

## CH-CORE5-003 — Implement the Repair Model Client

**Priority:** P0  
**Depends on:** CH-CORE5-001, CH-CORE5-002

### Goal

Connect the core to the existing Qwen2.5-Coder-7B-Instruct + LoRA repair model.

### Implementation Tasks

- Implement client/service adapter for the chosen model endpoint/runtime.
- Configure explicit timeouts.
- Pass model name/adapter version metadata.
- Keep model transport retries separate from repair refinement retries.
- Handle:
  - timeout
  - unavailable service
  - 5xx/transport error
  - malformed structured output
- Record inference latency.

### Acceptance Criteria

- A valid Fix Generation input produces a parsed candidate patch.
- Model-service transport failures are distinguishable from bad repair content.
- The model is not loaded repeatedly inside every repair attempt if a separate persistent model service is used.

### Deliverable

Repair model client.

---

## CH-CORE5-004 — Implement the Fixed Repair Prompt / Request Template

**Priority:** P0  
**Depends on:** CH-CORE5-003

### Goal

Make repair generation reproducible for implementation and evaluation.

### Implementation Tasks

- Create a versioned repair prompt/template.
- Include:
  - detected failure category
  - root cause
  - evidence
  - allowed target file
  - relevant source/configuration content
  - previous validation feedback where applicable
- Require patch-style repair output using the agreed schema.
- Explicitly forbid:
  - unrelated changes
  - new files outside scope
  - secret values
  - unexplained broad refactors
- Use fixed generation settings for evaluation.

### Acceptance Criteria

- Prompt version is recorded.
- Same evaluation configuration can be reused across all held-out cases.
- Later attempts clearly receive prior failure feedback.

### Deliverable

Versioned fix-generation prompt/request builder.

---

## CH-CORE5-005 — Parse and Validate Model Repair Output

**Priority:** P0  
**Depends on:** CH-CORE5-002, CH-CORE5-004

### Goal

Treat all model output as untrusted before patch validation.

### Implementation Tasks

- Parse structured response.
- Validate target path.
- Validate patch is non-empty.
- Check patch format is structurally parseable.
- Reject unexpected additional file targets.
- Calculate candidate content/hash identifier.
- Add bounded formatting retry only for malformed output representation.

### Acceptance Criteria

- Only typed valid candidate patches proceed to Step 6.
- Formatting retry does not count as a successful repair validation.
- Candidate hash is stable for identical patch content.

### Deliverable

Repair response parser.

---

## CH-CORE5-006 — Add Fix Generation Unit and Integration Tests

**Priority:** P0  
**Depends on:** CH-CORE5-003 to CH-CORE5-005

### Test Cases

- Valid patch.
- Empty patch.
- Wrong target path.
- Multi-file patch when unsupported.
- Malformed JSON.
- Malformed patch.
- Model timeout.
- Previous validation feedback included on attempt 2.
- Candidate hash generation.

### Acceptance Criteria

- Invalid model output never reaches GitHub or deterministic validation as a valid repair.
- Tests can run with a mocked model service.

### Deliverable

Fix-generation test suite.

---

# Step 6 — Deterministic Validation

## CH-CORE6-001 — Define the Validation Report Schema

**Priority:** P0  
**Depends on:** Candidate patch schema

### Goal

Create one structured result for every deterministic validation attempt.

### Suggested Fields

- repair ID
- attempt number
- candidate hash
- overall pass/fail
- target path check
- patch applicability result
- YAML parse result
- actionlint result
- schema/yamllint result where enabled
- dependency/configuration check result
- safety-policy result
- optional test result
- blocking errors
- non-blocking warnings
- execution timings

### Acceptance Criteria

- Every validation run returns a report even when the first blocking check fails.
- Errors are suitable for use as refinement feedback.

### Deliverable

`ValidationReport` schema.

---

## CH-CORE6-002 — Create Isolated Validation Workspace

**Priority:** P0  
**Depends on:** CH-CORE6-001

### Goal

Apply/check model-generated changes without modifying the developer's working tree.

### Implementation Tasks

- Create a temporary isolated working copy.
- Materialize only the required repository/configuration content for validation.
- Apply the original source content associated with the failed SHA/context.
- Clean up temporary files after validation.
- Prevent path traversal from model-generated paths.
- Store only required validation artifacts.

### Acceptance Criteria

- Validation never mutates the original developer checkout.
- Candidate paths cannot escape the workspace.
- Workspace is removed after success/failure.

### Deliverable

Validation workspace manager.

---

## CH-CORE6-003 — Implement Patch Applicability Validation

**Priority:** P0  
**Depends on:** CH-CORE6-002

### Goal

Reject patches that cannot be safely applied to the expected source.

### Implementation Tasks

- Perform `git apply --check` or equivalent patch applicability check.
- Verify the patch targets the allowed file.
- Reject unexpected file creation/deletion unless explicitly supported.
- Apply the patch only in the isolated workspace after the dry check passes.
- Capture exact patch error text for refinement.

### Acceptance Criteria

- Invalid/hunk-mismatch patches fail before linting.
- Applicable patch is materialized only inside the validation workspace.
- Validation report contains patch errors.

### Deliverable

Patch applicability validator.

---

## CH-CORE6-004 — Implement YAML and Workflow Parsing Validation

**Priority:** P0  
**Depends on:** CH-CORE6-003

### Goal

Verify that repaired workflow/configuration YAML is syntactically valid.

### Implementation Tasks

- Parse target YAML safely.
- Capture syntax/parser error with line/column where possible.
- Compare expected workflow structure where applicable.
- Ensure a repair does not accidentally remove the entire workflow structure.

### Acceptance Criteria

- Malformed YAML is rejected.
- Parser diagnostics are added to the validation report.
- Valid YAML proceeds to actionlint.

### Deliverable

YAML validation stage.

---

## CH-CORE6-005 — Integrate actionlint

**Priority:** P0  
**Depends on:** CH-CORE6-004

### Goal

Use deterministic GitHub Actions validation before any repair is returned to middleware.

### Implementation Tasks

- Execute actionlint against the repaired workflow.
- Capture:
  - exit status
  - diagnostics
  - affected lines
- Configure timeout.
- Handle actionlint unavailable/misconfigured as an infrastructure validation error.
- Store actionlint version for reproducibility.

### Acceptance Criteria

- actionlint failure blocks the candidate.
- Diagnostics are preserved for refinement.
- Valid candidate proceeds to later checks.

### Deliverable

actionlint validation adapter.

---

## CH-CORE6-006 — Add Optional yamllint / Schema Validation

**Priority:** P1  
**Depends on:** CH-CORE6-004

### Goal

Provide additional deterministic diagnostics without replacing actionlint.

### Implementation Tasks

- Add configurable yamllint/schema check.
- Distinguish blocking and non-blocking rules.
- Record tool version and configuration.
- Ensure research runs use fixed validation configuration.

### Acceptance Criteria

- Optional validator can be enabled/disabled from configuration.
- Its diagnostics are visible in the same validation report.

### Deliverable

Additional YAML/schema validator.

---

## CH-CORE6-007 — Implement Repair Scope and Safety Rules

**Priority:** P0  
**Depends on:** CH-CORE6-003

### Goal

Reject syntactically valid but unsafe or out-of-scope changes.

### Rules to Implement

At minimum detect/block:

- target path outside allowed repair scope
- unrelated broad changes
- introduction of embedded secret/credential values
- new or expanded workflow permissions beyond allowed policy
- unsafe `pull_request_target` introduction/modification
- new self-hosted runner usage where disallowed
- unexpected deployment credential changes
- unsupported multiple-file modification

### Acceptance Criteria

- Any configured blocking safety violation fails validation.
- Each violation produces a clear policy code/message.
- Safety rules are deterministic and independently testable.

### Deliverable

Static safety policy validator.

---

## CH-CORE6-008 — Implement Dependency/Configuration Consistency Checks

**Priority:** P1  
**Depends on:** CH-CORE6-003

### Goal

Run bounded deterministic checks relevant to the selected failure and available project context.

### Implementation Tasks

- Validate referenced action versions/paths where deterministically possible.
- Validate obvious runtime/configuration consistency when source context permits.
- Validate dependency/configuration file syntax when relevant.
- Do not execute arbitrary repository code as part of this ticket.

### Acceptance Criteria

- Only deterministic, bounded checks are included by default.
- Missing optional context does not cause false failure.
- Check results are included in the validation report.

### Deliverable

Dependency/configuration checker.

---

## CH-CORE6-009 — Add Safe Optional Project Test Hook

**Priority:** P2 / Optional  
**Depends on:** CH-CORE6-002

### Goal

Allow explicitly configured project tests only when the evaluation environment safely supports them.

### Implementation Tasks

- Make test execution disabled by default unless the research environment is controlled.
- Require an allowlisted command.
- Apply timeout and resource bounds.
- Capture stdout/stderr and exit status.
- Never execute commands suggested directly by the model.
- Treat unsafe/unconfigured test execution as skipped, not passed.

### Acceptance Criteria

- Arbitrary model-generated commands cannot execute.
- Timeout/resource limits are enforced.
- Test status is clearly `PASS`, `FAIL`, or `SKIPPED`.

### Deliverable

Optional controlled project-test adapter.

---

## CH-CORE6-010 — Build the Ordered Validation Pipeline

**Priority:** P0  
**Depends on:** CH-CORE6-003 to CH-CORE6-008

### Recommended Order

1. Target path/scope check
2. Patch structure/applicability
3. Apply patch in isolated workspace
4. YAML parsing
5. actionlint
6. optional schema/yamllint
7. safety policy
8. dependency/configuration checks
9. optional safe project tests
10. final validation decision

### Acceptance Criteria

- Blocking failure stops unnecessary expensive checks where appropriate.
- A complete `ValidationReport` is returned.
- All errors needed for refinement are preserved.
- Validation is deterministic for the same candidate and tool versions.

### Deliverable

Deterministic validation pipeline.

---

## CH-CORE6-011 — Add Validation Tests

**Priority:** P0  
**Depends on:** CH-CORE6-010

### Required Test Cases

- patch does not apply
- malformed YAML
- actionlint failure
- valid workflow repair
- unauthorized target path
- unsafe permission change
- secret insertion
- multiple-file patch
- optional validator disabled
- safe test skipped
- safe test pass/fail where enabled

### Acceptance Criteria

- Every blocking validation rule has a failing unit/integration test.
- A known valid candidate passes the entire deterministic pipeline.

### Deliverable

Validation test suite.

---

# Step 7 — Iterative Refinement

## CH-CORE7-001 — Define the Repair Attempt State

**Priority:** P0  
**Depends on:** Steps 5 and 6

### Goal

Track each candidate and its validation result independently.

### Fields

- repair ID
- attempt number
- RCA result
- candidate patch/hash
- validation report
- start/end timestamps
- model latency
- attempt status
- previous attempt reference

### Acceptance Criteria

- Every attempt can be reconstructed for research analysis.
- Attempt number is monotonically increasing within the repair session.

### Deliverable

`RepairAttempt` state/schema.

---

## CH-CORE7-002 — Implement Retry Limit Configuration

**Priority:** P0  
**Depends on:** CH-CORE7-001

### Goal

Enforce the baseline maximum of 3 repair attempts.

### Implementation Tasks

- Add `MAX_REPAIR_ATTEMPTS`.
- Default baseline research configuration to `3`.
- Refuse to generate attempt 4.
- Return a terminal escalation result when limit is reached.

### Acceptance Criteria

- Attempt 1 can retry to 2 and 3.
- No attempt 4 is generated in baseline configuration.
- Retry limit is recorded in experiment metadata.

### Deliverable

Retry limit configuration and guard.

---

## CH-CORE7-003 — Build Validation Feedback for the Next Repair Attempt

**Priority:** P0  
**Depends on:** CH-CORE6-001, CH-CORE7-001

### Goal

Convert deterministic validation failures into useful structured feedback.

### Implementation Tasks

Include:

- failed validation stage
- blocking errors
- relevant diagnostic lines
- previous patch
- previous candidate hash
- previous RCA
- attempt number

Avoid including irrelevant full logs when a concise diagnostic is sufficient.

### Acceptance Criteria

- Attempt 2+ receives the exact reason the previous candidate failed.
- Feedback is structured and bounded.
- Secret-redacted evidence remains redacted.

### Deliverable

Retry feedback builder.

---

## CH-CORE7-004 — Re-run RCA When New Validation Evidence Requires It

**Priority:** P0  
**Depends on:** RCA module, CH-CORE7-003

### Goal

Allow the diagnosis to change instead of repeatedly modifying the same repair blindly.

### Implementation Tasks

- Route validation failure feedback back to RCA according to the framework graph.
- Preserve original failure evidence.
- Include previous patch and validation error.
- Produce a new RCA result before the next fix when configured by the core flow.

### Acceptance Criteria

- Later attempt RCA can differ from attempt 1.
- Original evidence remains available.
- New evidence is distinguishable from original failure evidence.

### Deliverable

RCA/refinement loop integration.

---

## CH-CORE7-005 — Implement the Refinement Controller

**Priority:** P0  
**Depends on:** CH-CORE7-002 to CH-CORE7-004

### Goal

Control `RCA -> Fix -> Validate -> Retry` until success or limit.

### Logic

- Generate candidate.
- Validate candidate.
- If validation passes:
  - stop local refinement
  - return validated patch to middleware
- If validation fails and attempts remain:
  - build feedback
  - update RCA if needed
  - generate next candidate
- If validation fails and no attempts remain:
  - return escalation/terminal failure

### Acceptance Criteria

- Flow terminates deterministically.
- Successful attempt stops further generation.
- Failed attempt history is preserved.
- Maximum attempt limit is respected.

### Deliverable

Iterative refinement controller / LangGraph edges.

---

## CH-CORE7-006 — Add Duplicate/No-Progress Candidate Detection

**Priority:** P1  
**Depends on:** Candidate hashing, CH-CORE7-005

### Goal

Avoid wasting attempts on the exact same repair.

### Implementation Tasks

- Compare candidate hash against previous attempts.
- If identical after receiving new validation feedback:
  - mark no-progress event
  - optionally request one reformulation within the same configured policy
  - otherwise count/terminate according to the selected experiment rule
- Record duplicate-generation events.

### Acceptance Criteria

- Identical repair candidates are detectable.
- Behavior is deterministic and documented.
- Evaluation logs show whether duplicate output occurred.

### Deliverable

No-progress detection.

---

## CH-CORE7-007 — Handle Remote CI Feedback Re-entry

**Priority:** P0  
**Depends on:** Part 1 retry fields, CH-CORE7-003

### Goal

Support the system-level loop where the middleware later returns failed repair-branch CI evidence.

### Implementation Tasks

- Accept a later invocation for the same `repair_id`.
- Receive:
  - previous patch
  - remote validation logs
  - failed job/step
  - incremented attempt number
- Rebuild normalized context.
- Run RCA -> Fix -> deterministic validation again.
- Keep GitHub branch/PR operations outside the core.

### Acceptance Criteria

- Remote CI failure can become the next core repair attempt.
- The same repair/session identity is preserved.
- No new GitHub session/branch is created by the core.

### Deliverable

Remote-validation feedback re-entry path.

---

## CH-CORE7-008 — Add Refinement Tests

**Priority:** P0  
**Depends on:** CH-CORE7-005 to CH-CORE7-007

### Required Test Scenarios

- attempt 1 passes local validation
- attempt 1 fails, attempt 2 passes
- attempts 1 and 2 fail, attempt 3 passes
- all 3 attempts fail
- duplicate candidate generated
- remote CI failure re-enters the core
- malformed model output during retry
- model transport failure during retry

### Acceptance Criteria

- Retry count and final status are correct in every scenario.
- Attempt history is complete.

### Deliverable

Refinement integration test suite.

---

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
