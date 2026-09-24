# CloudHeal Core Implementation Tickets
## Steps 5–8 Split Implementation Ticket Set

This file is one part of the detailed implementation tickets originally prepared for Core Steps 5–8.

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
