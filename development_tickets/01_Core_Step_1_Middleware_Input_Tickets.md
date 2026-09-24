# CloudHeal Core Implementation Tickets
## Part 1 — Step 1: Middleware Input

**Scope:** Implement the core-side input boundary that receives a structured GitHub Actions failure case from the separate GitHub Integration Middleware.

**Primary output:** `StructuredFailureCase` / `FailureContext` accepted by the CloudHeal core.

**Important boundary:** This ticket group does **not** implement GitHub webhooks, GitHub App authentication, branch creation, pull requests, or CI rerun handling. Those remain middleware responsibilities. The core must not receive GitHub credentials.

---

## Baseline Input Expected by the Core

The middleware input must be able to provide:

- Repository identity and relevant repository metadata
- Original workflow run identifier
- Workflow identifier/name/path where available
- Original source branch
- Exact failed source SHA
- Failure timestamp and conclusion
- Workflow YAML
- Failed job and failed step information
- Relevant logs
- Environment/runtime context
- Dependency context
- Relevant configuration/repository files
- Attempt number
- Previous repair/validation evidence for later attempts, when applicable

---

# Ticket List

## CH-CORE1-001 — Define the Core Input Contract

**Priority:** P0  
**Depends on:** None

### Goal

Create one authoritative typed input contract for every failure case entering the CloudHeal core.

### Implementation Tasks

- Create the core input schema using Pydantic.
- Define required identity fields:
  - `repair_id`
  - `repo`
  - `workflow_run_id`
  - `workflow_id`
  - `workflow_path`
  - `workflow_name`
  - `source_branch`
  - `source_sha`
- Define failure fields:
  - `failure_timestamp`
  - `conclusion`
  - `failed_job`
  - failed step/job metadata
  - raw or middleware-preprocessed logs
- Define configuration/context fields:
  - `workflow_yaml`
  - `relevant_files`
  - environment/runtime information
  - dependency information
  - recent commit metadata if supplied by middleware
- Define retry fields:
  - `attempt_number`
  - `previous_fix`
  - `previous_validation_logs`
- Decide which fields are mandatory for attempt 1 and which are optional for later attempts.
- Add field descriptions and validation rules.

### Acceptance Criteria

- A valid middleware payload can be parsed into one typed object without manual transformation.
- Required fields fail validation when missing.
- Retry-only fields may be absent on attempt 1.
- The contract contains no GitHub installation token, App private key, access token, or authenticated API URL.
- The schema is reusable by Context Processing, Detection, RCA, Fix Generation, Validation, and logging.

### Tests

- Valid full attempt-1 payload.
- Valid attempt-2 payload with previous repair and validation evidence.
- Missing `source_sha`.
- Missing workflow YAML.
- Invalid attempt number.
- Unexpected credential-like fields are not required by the schema.

### Deliverable

A versioned `FailureContext` / `StructuredFailureCase` Pydantic model.

---

## CH-CORE1-002 — Define Nested Job, Step, Environment, and Dependency Schemas

**Priority:** P0  
**Depends on:** CH-CORE1-001

### Goal

Avoid unstructured dictionaries for important nested input data.

### Implementation Tasks

- Define a failed step schema containing, where available:
  - step name
  - step number/index
  - command
  - outcome/conclusion
  - exit code
  - timestamps
- Define job metadata schema:
  - job name
  - runner information
  - status/conclusion
  - relevant step list
- Define environment/runtime schema:
  - runner OS
  - language/runtime versions
  - package manager
  - architecture
  - container/runtime information if available
- Define dependency context schema:
  - manifest filename
  - lock file filename
  - dependency/tool versions
  - dependency-related error indicators
- Define relevant-file representation:
  - path
  - content
  - optional reason for inclusion
  - optional content hash

### Acceptance Criteria

- Important core logic does not depend on arbitrary nested dictionary keys.
- Missing optional metadata does not prevent valid cases from entering the core.
- The nested models serialize cleanly to JSON.

### Tests

- Complete job/step payload.
- Minimal job/step payload.
- Multiple relevant files.
- Missing optional runtime version.
- Empty dependency context.

### Deliverable

Typed nested models used by the main input contract.

---

## CH-CORE1-003 — Add Input-Level Validation Rules

**Priority:** P0  
**Depends on:** CH-CORE1-001, CH-CORE1-002

### Goal

Reject structurally invalid or contradictory failure cases before any model or agent execution.

### Implementation Tasks

- Validate that `attempt_number >= 1`.
- Validate SHA shape sufficiently for project use.
- Validate that repository, branch, workflow identifiers, and workflow content are not blank.
- Validate that a failed job/step or equivalent failure evidence is present.
- Validate that later attempts may include previous validation logs.
- Validate maximum configured input sizes for:
  - logs
  - workflow YAML
  - relevant files
- Add clear validation error messages.
- Ensure validation failure does not start the LangGraph workflow.

### Acceptance Criteria

- Invalid input is rejected before Context Processing.
- Validation errors identify the exact invalid field.
- Oversized evidence is rejected or marked for preprocessing according to configuration.
- No model call occurs for rejected payloads.

### Tests

- Attempt number `0`.
- Empty repository name.
- Blank workflow YAML.
- Oversized log payload.
- Missing all failure evidence.
- Valid partial optional context.

### Deliverable

Input validation rules and automated tests.

---

## CH-CORE1-004 — Implement Core Invocation Boundary

**Priority:** P0  
**Depends on:** CH-CORE1-001, CH-CORE1-003

### Goal

Provide one consistent method through which the middleware/worker invokes the CloudHeal core.

### Implementation Tasks

- Create the core entry function/service interface.
- Accept only the typed failure input.
- Return a typed framework result placeholder until later steps are implemented.
- Generate or propagate:
  - `repair_id`
  - `attempt_number`
  - trace/correlation identifier
- Ensure the entrypoint does not perform GitHub API operations.
- Keep the boundary usable from the worker and test harness.
- Add explicit exceptions for:
  - invalid input
  - unsupported failure case
  - internal framework error

### Acceptance Criteria

- A worker/test can invoke the core using one stable interface.
- The core does not need direct GitHub access.
- Correlation identifiers remain available through later stages.
- Entry and failure are logged without storing secrets.

### Tests

- Successful invocation with a valid fixture.
- Validation failure.
- Internal exception mapping.
- Correlation ID preservation.

### Deliverable

Stable core invocation API/function.

---

## CH-CORE1-005 — Add Input Sanitization Guardrails

**Priority:** P0  
**Depends on:** CH-CORE1-004

### Goal

Ensure incoming evidence is safe and suitable for downstream processing even though primary secret redaction is expected from the middleware.

### Implementation Tasks

- Add a second defensive scan for obvious credential material:
  - GitHub tokens
  - Authorization headers
  - private-key blocks
  - common API key patterns
- Remove or redact detected secret values before model-facing processing.
- Mark fields when redaction occurred.
- Preserve enough non-secret surrounding text for diagnosis.
- Do not permanently store raw detected secret values.
- Add configurable repository-specific patterns if required later.

### Acceptance Criteria

- Known secret patterns are replaced before model input.
- Redaction does not remove the entire diagnostic line unless necessary.
- The framework never logs the original secret value.
- Redaction status is available for debugging/evaluation metadata.

### Tests

- GitHub token in log.
- Bearer token in header.
- Private key block.
- Normal hex/string values are not unnecessarily removed.

### Deliverable

Input sanitization utility and tests.

---

## CH-CORE1-006 — Create Canonical Input Fixtures

**Priority:** P0  
**Depends on:** CH-CORE1-001 to CH-CORE1-005

### Goal

Create reusable failure-case fixtures for development and automated tests.

### Implementation Tasks

Create representative fixtures for at least:

1. YAML syntax failure.
2. Invalid/deprecated GitHub Action usage.
3. Dependency/version failure.
4. Runtime/environment mismatch.
5. Missing command/tool.
6. Permission/configuration failure.
7. Attempt 2 case containing:
   - previous patch
   - previous validation logs
   - incremented attempt number

Each fixture should contain:

- workflow YAML
- log evidence
- failed job/step
- source SHA
- environment/dependency information
- expected input validation result

### Acceptance Criteria

- Fixtures can be loaded directly into the core contract.
- Fixtures are deterministic and version controlled.
- At least one invalid fixture exists for schema validation testing.
- Fixtures do not contain real secrets.

### Deliverable

Reusable JSON/YAML fixture set.

---

## CH-CORE1-007 — Add Input Contract Serialization and Versioning

**Priority:** P1  
**Depends on:** CH-CORE1-001

### Goal

Allow the middleware and core to evolve without silently breaking each other.

### Implementation Tasks

- Add an input contract version field.
- Define the initial version, for example `1.0`.
- Add JSON serialization/deserialization.
- Decide how unknown fields are handled.
- Reject unsupported major contract versions.
- Document backward-compatible field additions.
- Add sample payloads to developer documentation.

### Acceptance Criteria

- Payload version is visible in every core request.
- Unsupported major versions fail clearly.
- Current fixtures serialize and deserialize without data loss.

### Deliverable

Versioned middleware-to-core contract.

---

## CH-CORE1-008 — Implement Middleware-to-Core Contract Tests

**Priority:** P0  
**Depends on:** CH-CORE1-004, CH-CORE1-006

### Goal

Verify the boundary using realistic middleware-style payloads before implementing later core stages.

### Implementation Tasks

- Test full payload parsing.
- Test minimal supported payload parsing.
- Test retry payload parsing.
- Test invalid payload rejection.
- Test oversized evidence behavior.
- Test redaction guardrails.
- Test serialization round-trip.
- Test correlation metadata preservation.

### Acceptance Criteria

- All boundary tests pass consistently.
- No test requires a real GitHub repository.
- The input contract is stable enough for Steps 2–8 to depend on it.

### Deliverable

Automated input-boundary test suite.

---

## CH-CORE1-009 — Document the Middleware/Core Responsibility Boundary

**Priority:** P1  
**Depends on:** CH-CORE1-004

### Goal

Prevent implementation overlap between the GitHub middleware and the CloudHeal core.

### Middleware Responsibilities to Document

- Receive GitHub events.
- Authenticate with GitHub.
- Collect evidence from the exact failed source SHA.
- Download jobs/logs/workflow files.
- Perform primary secret redaction and size bounding.
- Create/update the repair branch.
- Create/update the Draft PR.
- Observe remote CI results.
- Send new validation evidence back to the same RepairSession.

### Core Responsibilities to Document

- Context Processing.
- Failure Detection.
- RCA.
- Fix Generation.
- Deterministic Validation.
- Local refinement.
- Structured result generation.
- Research/experiment logging for core outputs.

### Acceptance Criteria

- Developers can identify the owner of every operation without ambiguity.
- The core never needs GitHub credentials.
- GitHub write operations are not implemented inside core modules.

### Deliverable

Boundary section in project README/developer documentation.

---

# Completion Criteria for Part 1

Part 1 is complete when:

- One versioned structured failure case can enter the core.
- Invalid cases are rejected before agent/model execution.
- Secret-like data has a defensive redaction pass.
- Attempt 1 and later-attempt payloads are supported.
- Realistic fixtures exist.
- The middleware/core responsibility boundary is documented.
- Steps 2–8 can depend on the input contract without additional manual reformatting.
