# CloudHeal Core Implementation Tickets
## Steps 5–8 Split Implementation Ticket Set

This file is one part of the detailed implementation tickets originally prepared for Core Steps 5–8.

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
