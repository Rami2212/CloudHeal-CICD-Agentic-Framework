# CloudHeal Core Implementation Tickets
## Steps 5–8 Split Implementation Ticket Set

This file is one part of the detailed implementation tickets originally prepared for Core Steps 5–8.

# Step 6 — Deterministic Validation (Part B)

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
