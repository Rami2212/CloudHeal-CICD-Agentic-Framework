# CloudHeal Core Implementation Tickets
## Steps 5–8 Split Implementation Ticket Set

This file is one part of the detailed implementation tickets originally prepared for Core Steps 5–8.

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
