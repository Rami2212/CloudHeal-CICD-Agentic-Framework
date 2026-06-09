"""CloudHeal CI/CD base-model vs fine-tuned-adapter evaluation agent.

This script does three jobs for the research workflow:

1. Generate a 10-example CI/CD failure benchmark with expected patch outputs.
2. Run the same optimized prompt through a base model and a PEFT/LoRA adapter.
3. Evaluate and save a report for accuracy, estimated resolution success rate,
   MTTR, and related patch-quality criteria.

Typical real-model run:

    python testing_agent/new_agent.py --mode all \
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter-path testing_agent/adapter

Smoke-test without loading a model:

    python testing_agent/new_agent.py --mode all --mock-model \
        --out-dir testing_agent/results_mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "cicd_failure_benchmark.jsonl"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ADAPTER_PATH = SCRIPT_DIR / "adapter"


SYSTEM_PROMPT = """You are CloudHeal, a CI/CD self-healing agent.
You receive a CI/CD failure log and the relevant repository context.
Infer the root cause and generate the smallest safe fix.
Return only a unified diff patch beginning with "diff --git".
Do not include markdown fences, prose, analysis, or test instructions."""


USER_PROMPT_TEMPLATE = """A CI/CD pipeline failed. Generate a minimal patch that fixes the failure.

Instruction:
{instruction}

CI/CD failure log:
{failure_log}

Repository context:
{repo_context}

Constraints:
{constraints}

Output requirements:
- Return only a unified diff patch.
- Touch only files needed for the fix.
- Preserve existing behavior unless the failure log requires a change.
- Prefer deterministic CI fixes over broad workarounds.
"""


@dataclass
class ModelRunConfig:
    base_model: str
    adapter_path: Path
    cache_dir: Optional[Path]
    max_new_tokens: int
    temperature: float
    top_p: float
    load_in_4bit: bool
    device_map: str
    dtype: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def format_repo_context(files: Dict[str, str]) -> str:
    blocks = []
    for file_path, content in files.items():
        language = infer_code_fence_language(file_path)
        blocks.append(f"File: {file_path}\n```{language}\n{content.rstrip()}\n```")
    return "\n\n".join(blocks)


def infer_code_fence_language(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".yml", ".yaml"}:
        return "yaml"
    if suffix == ".py":
        return "python"
    if suffix == ".js":
        return "javascript"
    if suffix == ".ts":
        return "typescript"
    if suffix == ".tsx":
        return "tsx"
    if suffix == ".tf":
        return "hcl"
    if suffix in {".xml", ".pom"}:
        return "xml"
    if Path(file_path).name.lower() == "dockerfile":
        return "dockerfile"
    return ""


def build_prompt(case: Dict[str, Any]) -> str:
    input_block = case.get("input") or {}
    if not isinstance(input_block, dict):
        raise ValueError(f"{case.get('id', '<unknown>')} input must be an object")

    failure_log = str(input_block.get("ci_failure_log", "")).strip()
    repo_files = input_block.get("repository_context") or {}
    if not isinstance(repo_files, dict):
        raise ValueError(f"{case.get('id', '<unknown>')} repository_context must be an object")
    constraints = input_block.get("constraints") or []
    if isinstance(constraints, str):
        constraints_text = f"- {constraints}"
    else:
        constraints_text = "\n".join(f"- {item}" for item in constraints)

    return USER_PROMPT_TEMPLATE.format(
        instruction=case.get("instruction", "Fix the CI/CD failure."),
        failure_log=failure_log,
        repo_context=format_repo_context(repo_files),
        constraints=constraints_text or "- No extra constraints.",
    )


def benchmark_cases() -> List[Dict[str, Any]]:
    """Return 10 diverse CI/CD failure examples with expected patch outputs."""

    return [
        {
            "id": "ci-failure-0001",
            "category": "python-pytest-import-path",
            "instruction": "Fix the pytest import failure in the CI workflow.",
            "input": {
                "ci_failure_log": """Run pytest tests -q
ImportError while importing test module 'tests/test_agent.py'.
E   ModuleNotFoundError: No module named 'cloudheal'
ERROR tests/test_agent.py
The package uses a src/ layout and tests pass locally when PYTHONPATH=src.""",
                "repository_context": {
                    ".github/workflows/ci.yml": """name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests -q
""",
                    "pyproject.toml": """[project]
name = "cloudheal"
version = "0.1.0"
""",
                },
                "constraints": [
                    "Do not change test files.",
                    "Keep the existing Python version.",
                ],
            },
            "expected_output": """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -13,4 +13,6 @@ jobs:
       - name: Install dependencies
         run: pip install -r requirements.txt
       - name: Run tests
+        env:
+          PYTHONPATH: ${{ github.workspace }}/src
         run: pytest tests -q
""",
            "expected_files": [".github/workflows/ci.yml"],
            "expected_keywords": ["PYTHONPATH", "github.workspace", "src", "pytest"],
            "reference_mttr_minutes": 25,
        },
        {
            "id": "ci-failure-0002",
            "category": "python-mypy-type-hints",
            "instruction": "Fix the mypy type-checking failure without changing runtime behavior.",
            "input": {
                "ci_failure_log": """Run mypy src
src/cloudheal/remediator.py:8: error: Function is missing a type annotation  [no-untyped-def]
src/cloudheal/remediator.py:8: error: Missing return statement  [return]
Found 2 errors in 1 file (checked 14 source files)""",
                "repository_context": {
                    "src/cloudheal/remediator.py": """def build_fix(log, context):
    if "ModuleNotFoundError" in log:
        return {"type": "pythonpath", "file": ".github/workflows/ci.yml"}
    if context.get("workflow") == "docker":
        return {"type": "dockerfile"}
""",
                },
                "constraints": [
                    "Keep the return value as a dictionary or None.",
                    "Do not introduce new dependencies.",
                ],
            },
            "expected_output": """diff --git a/src/cloudheal/remediator.py b/src/cloudheal/remediator.py
--- a/src/cloudheal/remediator.py
+++ b/src/cloudheal/remediator.py
@@ -1,6 +1,9 @@
-def build_fix(log, context):
+from typing import Optional
+
+
+def build_fix(log: str, context: dict[str, str]) -> Optional[dict[str, str]]:
     if "ModuleNotFoundError" in log:
         return {"type": "pythonpath", "file": ".github/workflows/ci.yml"}
     if context.get("workflow") == "docker":
         return {"type": "dockerfile"}
+    return None
""",
            "expected_files": ["src/cloudheal/remediator.py"],
            "expected_keywords": ["Optional", "dict[str, str]", "return None", "log: str"],
            "reference_mttr_minutes": 35,
        },
        {
            "id": "ci-failure-0003",
            "category": "frontend-npm-peer-dependency",
            "instruction": "Fix the frontend dependency installation failure in CI.",
            "input": {
                "ci_failure_log": """Run npm ci
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! Found: react@18.2.0
npm ERR! peer react@"^17.0.0" from react-legacy-widget@2.3.0
npm ERR! Fix the upstream dependency conflict, or retry this command with --legacy-peer-deps.""",
                "repository_context": {
                    ".github/workflows/frontend.yml": """name: Frontend CI
on: [push, pull_request]

jobs:
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
      - name: Install packages
        run: npm ci
      - name: Run lint
        run: npm run lint
""",
                    "package.json": """{
  "dependencies": {
    "react": "18.2.0",
    "react-legacy-widget": "2.3.0"
  }
}
""",
                },
                "constraints": [
                    "Keep npm ci so lockfile installs remain deterministic.",
                    "Do not downgrade React.",
                ],
            },
            "expected_output": """diff --git a/.github/workflows/frontend.yml b/.github/workflows/frontend.yml
--- a/.github/workflows/frontend.yml
+++ b/.github/workflows/frontend.yml
@@ -11,6 +11,6 @@ jobs:
           node-version: "20"
           cache: npm
       - name: Install packages
-        run: npm ci
+        run: npm ci --legacy-peer-deps
       - name: Run lint
         run: npm run lint
""",
            "expected_files": [".github/workflows/frontend.yml"],
            "expected_keywords": ["npm ci --legacy-peer-deps", "legacy-peer-deps"],
            "reference_mttr_minutes": 30,
        },
        {
            "id": "ci-failure-0004",
            "category": "docker-build-context",
            "instruction": "Fix the Docker build failure caused by an incorrect copied file path.",
            "input": {
                "ci_failure_log": """#9 [4/7] COPY requirements.txt ./requirements.txt
#9 ERROR: failed to compute cache key: failed to calculate checksum of ref:
"/requirements.txt": not found
Dockerfile:6
--------------------
  4 | WORKDIR /app
  5 |
  6 | COPY requirements.txt ./requirements.txt
--------------------""",
                "repository_context": {
                    "Dockerfile": """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/api ./services/api

CMD ["python", "-m", "services.api"]
""",
                    "services/api/requirements.txt": """fastapi==0.111.0
uvicorn==0.30.0
""",
                },
                "constraints": [
                    "Do not move files.",
                    "Keep the image layer cache-friendly.",
                ],
            },
            "expected_output": """diff --git a/Dockerfile b/Dockerfile
--- a/Dockerfile
+++ b/Dockerfile
@@ -2,7 +2,7 @@ FROM python:3.11-slim
 
 WORKDIR /app
 
-COPY requirements.txt ./requirements.txt
+COPY services/api/requirements.txt ./requirements.txt
 RUN pip install --no-cache-dir -r requirements.txt
 COPY services/api ./services/api
 
""",
            "expected_files": ["Dockerfile"],
            "expected_keywords": ["services/api/requirements.txt", "COPY", "requirements.txt"],
            "reference_mttr_minutes": 20,
        },
        {
            "id": "ci-failure-0005",
            "category": "terraform-aws-provider-v4",
            "instruction": "Fix the Terraform validation failure for the AWS provider v4 schema.",
            "input": {
                "ci_failure_log": """Run terraform validate
Error: Unsupported argument

  on infra/s3.tf line 3, in resource "aws_s3_bucket" "artifacts":
   3:   acl    = "private"

An argument named "acl" is not expected here. Use the aws_s3_bucket_acl resource.""",
                "repository_context": {
                    "infra/s3.tf": """resource "aws_s3_bucket" "artifacts" {
  bucket = var.bucket_name
  acl    = "private"
}
""",
                    "infra/versions.tf": """terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}
""",
                },
                "constraints": [
                    "Preserve private ACL behavior.",
                    "Keep the existing bucket resource name.",
                ],
            },
            "expected_output": """diff --git a/infra/s3.tf b/infra/s3.tf
--- a/infra/s3.tf
+++ b/infra/s3.tf
@@ -1,4 +1,8 @@
 resource "aws_s3_bucket" "artifacts" {
   bucket = var.bucket_name
-  acl    = "private"
+}
+
+resource "aws_s3_bucket_acl" "artifacts" {
+  bucket = aws_s3_bucket.artifacts.id
+  acl    = "private"
 }
""",
            "expected_files": ["infra/s3.tf"],
            "expected_keywords": ["aws_s3_bucket_acl", "aws_s3_bucket.artifacts.id", "private"],
            "reference_mttr_minutes": 45,
        },
        {
            "id": "ci-failure-0006",
            "category": "maven-java-version",
            "instruction": "Fix the Maven test job so it uses the Java version required by the project.",
            "input": {
                "ci_failure_log": """Run mvn -B test
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.11.0:compile
[ERROR] Fatal error compiling: error: invalid target release: 21
The workflow currently installs Java 17.""",
                "repository_context": {
                    ".github/workflows/java.yml": """name: Java CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
      - run: mvn -B test
""",
                    "pom.xml": """<project>
  <properties>
    <maven.compiler.release>21</maven.compiler.release>
  </properties>
</project>
""",
                },
                "constraints": [
                    "Do not change pom.xml compiler settings.",
                    "Keep the setup-java action version.",
                ],
            },
            "expected_output": """diff --git a/.github/workflows/java.yml b/.github/workflows/java.yml
--- a/.github/workflows/java.yml
+++ b/.github/workflows/java.yml
@@ -9,5 +9,5 @@ jobs:
         with:
           distribution: temurin
-          java-version: "17"
+          java-version: "21"
       - run: mvn -B test
""",
            "expected_files": [".github/workflows/java.yml"],
            "expected_keywords": ["java-version: \"21\"", "distribution: temurin", "mvn -B test"],
            "reference_mttr_minutes": 25,
        },
        {
            "id": "ci-failure-0007",
            "category": "eslint-react-no-undef",
            "instruction": "Fix the ESLint no-undef failure in the React component.",
            "input": {
                "ci_failure_log": """Run npm run lint
src/components/StatusBadge.jsx
  2:11  error  'React' is not defined  no-undef

✖ 1 problem (1 error, 0 warnings)""",
                "repository_context": {
                    "src/components/StatusBadge.jsx": """export default function StatusBadge({ label }) {
  return <span className="status-badge">{label}</span>;
}
""",
                    ".eslintrc.json": """{
  "env": { "browser": true, "es2022": true },
  "extends": ["eslint:recommended", "plugin:react/recommended"],
  "rules": { "react/react-in-jsx-scope": "off" }
}
""",
                },
                "constraints": [
                    "Do not weaken ESLint rules.",
                    "Keep the component API unchanged.",
                ],
            },
            "expected_output": """diff --git a/src/components/StatusBadge.jsx b/src/components/StatusBadge.jsx
--- a/src/components/StatusBadge.jsx
+++ b/src/components/StatusBadge.jsx
@@ -1,3 +1,5 @@
+import React from "react";
+
 export default function StatusBadge({ label }) {
   return <span className="status-badge">{label}</span>;
 }
""",
            "expected_files": ["src/components/StatusBadge.jsx"],
            "expected_keywords": ["import React", "StatusBadge", "react"],
            "reference_mttr_minutes": 15,
        },
        {
            "id": "ci-failure-0008",
            "category": "kubernetes-hpa-api-version",
            "instruction": "Fix the Kubernetes manifest so it applies on Kubernetes 1.26.",
            "input": {
                "ci_failure_log": """Run kubectl apply --dry-run=server -f k8s/hpa.yaml
error: resource mapping not found for name: "cloudheal-api" namespace: "" from "k8s/hpa.yaml":
no matches for kind "HorizontalPodAutoscaler" in version "autoscaling/v2beta2"
ensure CRDs are installed first""",
                "repository_context": {
                    "k8s/hpa.yaml": """apiVersion: autoscaling/v2beta2
kind: HorizontalPodAutoscaler
metadata:
  name: cloudheal-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cloudheal-api
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        targetAverageUtilization: 70
""",
                },
                "constraints": [
                    "Keep the same scaling thresholds.",
                    "Only update the manifest schema.",
                ],
            },
            "expected_output": """diff --git a/k8s/hpa.yaml b/k8s/hpa.yaml
--- a/k8s/hpa.yaml
+++ b/k8s/hpa.yaml
@@ -1,4 +1,4 @@
-apiVersion: autoscaling/v2beta2
+apiVersion: autoscaling/v2
 kind: HorizontalPodAutoscaler
 metadata:
   name: cloudheal-api
@@ -13,4 +13,6 @@ spec:
       resource:
         name: cpu
-        targetAverageUtilization: 70
+        target:
+          type: Utilization
+          averageUtilization: 70
""",
            "expected_files": ["k8s/hpa.yaml"],
            "expected_keywords": ["autoscaling/v2", "target:", "averageUtilization: 70"],
            "reference_mttr_minutes": 40,
        },
        {
            "id": "ci-failure-0009",
            "category": "github-actions-release-permissions",
            "instruction": "Fix the release workflow permission failure when pushing a tag.",
            "input": {
                "ci_failure_log": """Run git push origin v1.4.2
remote: Permission to org/cloudheal.git denied to github-actions[bot].
fatal: unable to access 'https://github.com/org/cloudheal/': The requested URL returned error: 403
Error: Process completed with exit code 128
The workflow uses GITHUB_TOKEN.""",
                "repository_context": {
                    ".github/workflows/release.yml": """name: Release
on:
  workflow_dispatch:

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create tag
        run: |
          git tag v${{ github.run_number }}
          git push origin v${{ github.run_number }}
""",
                },
                "constraints": [
                    "Use the built-in GITHUB_TOKEN.",
                    "Do not add personal access tokens or secrets.",
                ],
            },
            "expected_output": """diff --git a/.github/workflows/release.yml b/.github/workflows/release.yml
--- a/.github/workflows/release.yml
+++ b/.github/workflows/release.yml
@@ -2,6 +2,9 @@ name: Release
 on:
   workflow_dispatch:
 
+permissions:
+  contents: write
+
 jobs:
   release:
     runs-on: ubuntu-latest
""",
            "expected_files": [".github/workflows/release.yml"],
            "expected_keywords": ["permissions:", "contents: write", "workflow_dispatch"],
            "reference_mttr_minutes": 30,
        },
        {
            "id": "ci-failure-0010",
            "category": "helm-yaml-indentation",
            "instruction": "Fix the Helm lint failure caused by invalid values.yaml indentation.",
            "input": {
                "ci_failure_log": """Run helm lint charts/cloudheal
==> Linting charts/cloudheal
[ERROR] values.yaml: unable to parse YAML: error converting YAML to JSON:
yaml: line 7: did not find expected key
Error: 1 chart(s) linted, 1 chart(s) failed""",
                "repository_context": {
                    "charts/cloudheal/values.yaml": """replicaCount: 2

image:
  repository: ghcr.io/org/cloudheal
  tag: latest

resources:
  limits:
  cpu: 500m
  memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi
""",
                },
                "constraints": [
                    "Keep the same CPU and memory values.",
                    "Only fix YAML structure.",
                ],
            },
            "expected_output": """diff --git a/charts/cloudheal/values.yaml b/charts/cloudheal/values.yaml
--- a/charts/cloudheal/values.yaml
+++ b/charts/cloudheal/values.yaml
@@ -5,8 +5,8 @@ image:
 
 resources:
   limits:
-  cpu: 500m
-  memory: 512Mi
+    cpu: 500m
+    memory: 512Mi
   requests:
     cpu: 250m
     memory: 256Mi
""",
            "expected_files": ["charts/cloudheal/values.yaml"],
            "expected_keywords": ["limits:", "cpu: 500m", "memory: 512Mi", "    cpu"],
            "reference_mttr_minutes": 20,
        },
    ]


def with_training_output_aliases(case: Dict[str, Any]) -> Dict[str, Any]:
    output = case["expected_output"]
    row = dict(case)
    row["output"] = output
    return row


def generate_benchmark(path: Path, overwrite: bool = True) -> List[Dict[str, Any]]:
    if path.exists() and not overwrite:
        return read_jsonl(path)
    rows = [with_training_output_aliases(case) for case in benchmark_cases()]
    write_jsonl(rows, path)
    return rows


class HFModelRunner:
    """Small wrapper around Transformers generation for base and PEFT models."""

    def __init__(self, config: ModelRunConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.finetuned_model = None

    def _resolve_dtype(self):
        import torch

        normalized = self.config.dtype.lower()
        if normalized == "auto":
            if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                return torch.bfloat16
            if torch.cuda.is_available():
                return torch.float16
            return None
        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(normalized)

    def load_base(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {
            "device_map": self.config.device_map,
            "cache_dir": str(self.config.cache_dir) if self.config.cache_dir else None,
        }
        dtype = self._resolve_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            use_fast=True,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **kwargs,
        )
        self.base_model.eval()

    def load_adapter(self) -> None:
        if self.base_model is None:
            self.load_base()
        if not self.config.adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {self.config.adapter_path}")
        if not (self.config.adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                "Adapter path is missing adapter_config.json: "
                f"{self.config.adapter_path}"
            )
        from peft import PeftModel

        self.finetuned_model = PeftModel.from_pretrained(
            self.base_model,
            str(self.config.adapter_path),
            is_trainable=False,
        )
        self.finetuned_model.eval()

    def generate(self, prompt: str, variant: str) -> str:
        if self.tokenizer is None or self.base_model is None:
            self.load_base()
        model = self.base_model
        if variant == "finetuned":
            if self.finetuned_model is None:
                self.load_adapter()
            model = self.finetuned_model

        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{prompt}\n\nAssistant:\n"

        inputs = self.tokenizer([text], return_tensors="pt", padding=True)
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = "cpu"
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        import torch

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def mock_generation(case: Dict[str, Any], variant: str) -> str:
    expected = case["expected_output"].strip()
    if variant == "finetuned":
        return expected

    expected_files = case.get("expected_files") or []
    first_file = expected_files[0] if expected_files else "unknown"
    return (
        f"The CI log points to {case['category']}. A likely fix is in {first_file}.\n\n"
        "```diff\n"
        + "\n".join(expected.splitlines()[: min(8, len(expected.splitlines()))])
        + "\n```\n"
        "Run the pipeline again after applying the patch."
    )


def generate_fixes(
    cases: Sequence[Dict[str, Any]],
    variant: str,
    output_path: Path,
    patch_dir: Path,
    config: ModelRunConfig,
    mock_model: bool = False,
    save_prompts: bool = False,
    runner: Optional[HFModelRunner] = None,
) -> List[Dict[str, Any]]:
    if variant not in {"base", "finetuned"}:
        raise ValueError(f"Unknown model variant: {variant}")

    active_runner = runner
    if not mock_model and active_runner is None:
        active_runner = HFModelRunner(config)

    records: List[Dict[str, Any]] = []
    patch_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, start=1):
        prompt = build_prompt(case)
        start = time.perf_counter()
        if mock_model:
            generated_fix = mock_generation(case, variant)
            model_name = f"mock-{variant}"
        else:
            assert active_runner is not None
            generated_fix = active_runner.generate(prompt, variant)
            model_name = config.base_model if variant == "base" else f"{config.base_model}+adapter"
        elapsed = time.perf_counter() - start

        patch_path = patch_dir / f"{case['id']}.patch"
        patch_path.write_text(generated_fix + "\n", encoding="utf-8")

        record: Dict[str, Any] = {
            "id": case["id"],
            "category": case["category"],
            "model_variant": variant,
            "model_name": model_name,
            "created_at": utc_now(),
            "instruction": case["instruction"],
            "prompt_hash": stable_hash(SYSTEM_PROMPT + "\n" + prompt),
            "reference_output": case["expected_output"],
            "generated_fix": generated_fix,
            "generation_seconds": elapsed,
            "patch_file": str(patch_path),
        }
        if save_prompts:
            record["system_prompt"] = SYSTEM_PROMPT
            record["user_prompt"] = prompt
        records.append(record)
        print(f"[{variant}] {index:02d}/{len(cases):02d} generated {case['id']} in {elapsed:.2f}s")

    write_jsonl(records, output_path)
    return records


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def line_similarity(generated: str, expected: str) -> float:
    gen = normalize_text(generated).splitlines()
    exp = normalize_text(expected).splitlines()
    if not gen and not exp:
        return 1.0
    if not gen or not exp:
        return 0.0
    return SequenceMatcher(None, gen, exp).ratio()


def is_patch_like(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line.startswith("diff --git") or first_line.startswith("--- a/")


def has_markdown_fence(text: str) -> bool:
    return "```" in text


def dangerous_change_penalty(text: str) -> float:
    patterns = [
        r"rm\s+-rf\s+/",
        r"chmod\s+777",
        r"curl\s+[^|]+?\|\s*(bash|sh)",
        r"verify\s*=\s*false",
        r"strict-ssl\s+false",
        r"AWS_SECRET_ACCESS_KEY",
        r"PRIVATE_KEY",
    ]
    hits = sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))
    return min(0.35, hits * 0.12)


def score_expected_files(generated: str, expected_files: Sequence[str]) -> Tuple[float, List[str]]:
    if not expected_files:
        return 1.0, []
    found = [file_path for file_path in expected_files if file_path in generated]
    return len(found) / len(expected_files), found


def score_keywords(generated: str, keywords: Sequence[str]) -> Tuple[float, List[str], List[str]]:
    if not keywords:
        return 1.0, [], []
    lower = generated.lower()
    found = [keyword for keyword in keywords if keyword.lower() in lower]
    missing = [keyword for keyword in keywords if keyword.lower() not in lower]
    return len(found) / len(keywords), found, missing


def estimated_mttr_minutes(
    reference_mttr_minutes: float,
    generation_seconds: float,
    accuracy: float,
    success: bool,
) -> float:
    generation_minutes = generation_seconds / 60.0
    residual_minutes = reference_mttr_minutes * max(0.0, 1.0 - accuracy)
    manual_review_minutes = 4.0 + (8.0 * max(0.0, 1.0 - accuracy))
    failure_penalty = 30.0 if not success else 0.0
    return round(generation_minutes + residual_minutes + manual_review_minutes + failure_penalty, 2)


def evaluate_generated_fix(case: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    generated = record.get("generated_fix") or ""
    expected = case["expected_output"]
    expected_files = case.get("expected_files") or []
    expected_keywords = case.get("expected_keywords") or []

    patch_score = 1.0 if is_patch_like(generated) else 0.0
    markdown_penalty = 0.08 if has_markdown_fence(generated) else 0.0
    file_score, found_files = score_expected_files(generated, expected_files)
    keyword_score, found_keywords, missing_keywords = score_keywords(generated, expected_keywords)
    similarity = line_similarity(generated, expected)
    non_empty_score = 1.0 if generated.strip() else 0.0
    danger_penalty = dangerous_change_penalty(generated)

    accuracy = (
        0.30 * keyword_score
        + 0.25 * file_score
        + 0.20 * similarity
        + 0.15 * patch_score
        + 0.10 * non_empty_score
        - markdown_penalty
        - danger_penalty
    )
    accuracy = round(max(0.0, min(1.0, accuracy)), 4)
    success = bool(
        accuracy >= 0.70
        and patch_score == 1.0
        and file_score >= 0.80
        and keyword_score >= 0.60
        and danger_penalty == 0.0
    )
    mttr = estimated_mttr_minutes(
        float(case.get("reference_mttr_minutes", 45)),
        float(record.get("generation_seconds") or 0.0),
        accuracy,
        success,
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "model_variant": record.get("model_variant"),
        "accuracy": accuracy,
        "estimated_resolution_success": success,
        "estimated_mttr_minutes": mttr,
        "line_similarity": round(similarity, 4),
        "patch_like": bool(patch_score),
        "contains_markdown_fence": has_markdown_fence(generated),
        "file_path_score": round(file_score, 4),
        "keyword_score": round(keyword_score, 4),
        "found_files": found_files,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "dangerous_change_penalty": round(danger_penalty, 4),
        "generation_seconds": round(float(record.get("generation_seconds") or 0.0), 4),
        "patch_file": record.get("patch_file"),
    }


def summarize_scores(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        return {
            "cases": 0,
            "accuracy": 0.0,
            "resolution_success_rate": 0.0,
            "mean_mttr_minutes": 0.0,
            "median_mttr_minutes": 0.0,
            "patch_like_rate": 0.0,
            "mean_generation_seconds": 0.0,
        }

    def mean(key: str) -> float:
        return float(statistics.mean(float(score[key]) for score in scores))

    return {
        "cases": len(scores),
        "accuracy": round(mean("accuracy"), 4),
        "resolution_success_rate": round(
            sum(1 for score in scores if score["estimated_resolution_success"]) / len(scores),
            4,
        ),
        "mean_mttr_minutes": round(mean("estimated_mttr_minutes"), 2),
        "median_mttr_minutes": round(
            float(statistics.median(float(score["estimated_mttr_minutes"]) for score in scores)),
            2,
        ),
        "patch_like_rate": round(
            sum(1 for score in scores if score["patch_like"]) / len(scores),
            4,
        ),
        "mean_generation_seconds": round(mean("generation_seconds"), 4),
        "mean_file_path_score": round(mean("file_path_score"), 4),
        "mean_keyword_score": round(mean("keyword_score"), 4),
        "mean_line_similarity": round(mean("line_similarity"), 4),
    }


def index_by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate ids found in result rows")
    return indexed


def evaluate_outputs(
    cases: Sequence[Dict[str, Any]],
    base_records: Sequence[Dict[str, Any]],
    finetuned_records: Sequence[Dict[str, Any]],
    report_json_path: Path,
    report_md_path: Path,
) -> Dict[str, Any]:
    base_by_id = index_by_id(base_records)
    finetuned_by_id = index_by_id(finetuned_records)

    per_case: List[Dict[str, Any]] = []
    base_scores: List[Dict[str, Any]] = []
    finetuned_scores: List[Dict[str, Any]] = []

    for case in cases:
        case_id = case["id"]
        if case_id not in base_by_id:
            raise ValueError(f"Missing base output for {case_id}")
        if case_id not in finetuned_by_id:
            raise ValueError(f"Missing finetuned output for {case_id}")

        base_score = evaluate_generated_fix(case, base_by_id[case_id])
        finetuned_score = evaluate_generated_fix(case, finetuned_by_id[case_id])
        base_scores.append(base_score)
        finetuned_scores.append(finetuned_score)

        if finetuned_score["accuracy"] > base_score["accuracy"]:
            winner = "finetuned"
        elif base_score["accuracy"] > finetuned_score["accuracy"]:
            winner = "base"
        else:
            winner = "tie"
        per_case.append(
            {
                "id": case_id,
                "category": case["category"],
                "winner": winner,
                "reference_mttr_minutes": case.get("reference_mttr_minutes"),
                "base": base_score,
                "finetuned": finetuned_score,
            }
        )

    base_summary = summarize_scores(base_scores)
    finetuned_summary = summarize_scores(finetuned_scores)
    deltas = {
        "accuracy": round(finetuned_summary["accuracy"] - base_summary["accuracy"], 4),
        "resolution_success_rate": round(
            finetuned_summary["resolution_success_rate"]
            - base_summary["resolution_success_rate"],
            4,
        ),
        "mean_mttr_minutes": round(
            finetuned_summary["mean_mttr_minutes"] - base_summary["mean_mttr_minutes"],
            2,
        ),
        "patch_like_rate": round(
            finetuned_summary["patch_like_rate"] - base_summary["patch_like_rate"],
            4,
        ),
    }
    winner_counts = {
        "base": sum(1 for row in per_case if row["winner"] == "base"),
        "finetuned": sum(1 for row in per_case if row["winner"] == "finetuned"),
        "tie": sum(1 for row in per_case if row["winner"] == "tie"),
    }
    overall_winner = "finetuned" if winner_counts["finetuned"] > winner_counts["base"] else "base"
    if winner_counts["finetuned"] == winner_counts["base"]:
        overall_winner = "tie"

    report = {
        "created_at": utc_now(),
        "metric_definitions": {
            "accuracy": (
                "Weighted heuristic score from expected fix keywords, target files, "
                "line similarity to the reference patch, patch format, and safety penalties."
            ),
            "resolution_success_rate": (
                "Share of cases whose generated patch is patch-like and crosses the "
                "accuracy, file-path, keyword, and safety thresholds."
            ),
            "estimated_mttr_minutes": (
                "Proxy MTTR: generation time plus expected human recovery time remaining "
                "after the model's accuracy score, with a penalty for likely unresolved cases."
            ),
        },
        "summary": {
            "base": base_summary,
            "finetuned": finetuned_summary,
            "delta_finetuned_minus_base": deltas,
            "winner_counts": winner_counts,
            "overall_winner": overall_winner,
        },
        "per_case": per_case,
    }
    write_json(report, report_json_path)
    report_md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return report


def render_markdown_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    base = summary["base"]
    ft = summary["finetuned"]
    delta = summary["delta_finetuned_minus_base"]
    lines = [
        "# CloudHeal CI/CD Model Evaluation Report",
        "",
        f"Generated: {report['created_at']}",
        "",
        "## Summary",
        "",
        "| Metric | Base model | Fine-tuned adapter | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Accuracy | {base['accuracy']:.4f} | {ft['accuracy']:.4f} | {delta['accuracy']:+.4f} |",
        (
            f"| Resolution success rate | {base['resolution_success_rate']:.2%} | "
            f"{ft['resolution_success_rate']:.2%} | {delta['resolution_success_rate']:+.2%} |"
        ),
        (
            f"| Mean MTTR minutes | {base['mean_mttr_minutes']:.2f} | "
            f"{ft['mean_mttr_minutes']:.2f} | {delta['mean_mttr_minutes']:+.2f} |"
        ),
        (
            f"| Patch-like output rate | {base['patch_like_rate']:.2%} | "
            f"{ft['patch_like_rate']:.2%} | {delta['patch_like_rate']:+.2%} |"
        ),
        (
            f"| Mean generation seconds | {base['mean_generation_seconds']:.2f} | "
            f"{ft['mean_generation_seconds']:.2f} | n/a |"
        ),
        "",
        f"Overall winner: **{summary['overall_winner']}**",
        "",
        "## Per-Case Results",
        "",
        "| ID | Category | Winner | Base accuracy | FT accuracy | Base MTTR | FT MTTR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_case"]:
        base_score = row["base"]
        ft_score = row["finetuned"]
        lines.append(
            "| {id} | {category} | {winner} | {base_acc:.4f} | {ft_acc:.4f} | "
            "{base_mttr:.2f} | {ft_mttr:.2f} |".format(
                id=row["id"],
                category=row["category"],
                winner=row["winner"],
                base_acc=base_score["accuracy"],
                ft_acc=ft_score["accuracy"],
                base_mttr=base_score["estimated_mttr_minutes"],
                ft_mttr=ft_score["estimated_mttr_minutes"],
            )
        )
    lines.extend(
        [
            "",
            "## Metric Notes",
            "",
            "- Accuracy is a lightweight research proxy, not a replacement for applying patches and running tests.",
            "- Resolution success is estimated from patch format, target file coverage, expected fix keywords, and safety checks.",
            "- MTTR is estimated because these synthetic cases do not execute real CI jobs end to end.",
        ]
    )
    return "\n".join(lines) + "\n"


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_or_create_cases(data_path: Path, overwrite_data: bool) -> List[Dict[str, Any]]:
    if overwrite_data or not data_path.exists():
        rows = generate_benchmark(data_path, overwrite=True)
        print(f"[data] wrote {len(rows)} CI/CD failure cases to {data_path}")
        return rows
    rows = read_jsonl(data_path)
    print(f"[data] loaded {len(rows)} CI/CD failure cases from {data_path}")
    return rows


def output_paths(out_dir: Path) -> Dict[str, Path]:
    return {
        "base_outputs": out_dir / "base_model_fixes.jsonl",
        "finetuned_outputs": out_dir / "finetuned_model_fixes.jsonl",
        "comparison": out_dir / "comparison_report.jsonl",
        "evaluation_json": out_dir / "evaluation_report.json",
        "evaluation_md": out_dir / "evaluation_report.md",
        "base_patch_dir": out_dir / "base_model_fixes",
        "finetuned_patch_dir": out_dir / "finetuned_model_fixes",
    }


def validate_adapter_ready(config: ModelRunConfig, mode: str, mock_model: bool) -> None:
    if mock_model or mode not in {"generate-finetuned", "generate", "all"}:
        return
    if not config.adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {config.adapter_path}")
    required_files = ["adapter_config.json"]
    missing = [name for name in required_files if not (config.adapter_path / name).exists()]
    has_weights = any(
        (config.adapter_path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if not has_weights:
        missing.append("adapter_model.safetensors or adapter_model.bin")
    if missing:
        raise FileNotFoundError(
            "Fine-tuned evaluation needs a complete PEFT adapter. Missing from "
            f"{config.adapter_path}: {', '.join(missing)}"
        )


def write_comparison_report(
    cases: Sequence[Dict[str, Any]],
    base_records: Sequence[Dict[str, Any]],
    finetuned_records: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    base_by_id = index_by_id(base_records)
    finetuned_by_id = index_by_id(finetuned_records)
    rows = []
    for case in cases:
        case_id = case["id"]
        rows.append(
            {
                "id": case_id,
                "category": case["category"],
                "instruction": case["instruction"],
                "reference_output": case["expected_output"],
                "base_model_output": base_by_id[case_id]["generated_fix"],
                "finetuned_model_output": finetuned_by_id[case_id]["generated_fix"],
                "base_generation_seconds": base_by_id[case_id].get("generation_seconds"),
                "finetuned_generation_seconds": finetuned_by_id[case_id].get("generation_seconds"),
            }
        )
    write_jsonl(rows, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base model and fine-tuned adapter on synthetic CI/CD failure fixes."
    )
    parser.add_argument(
        "--mode",
        choices=["data", "generate-base", "generate-finetuned", "generate", "evaluate", "all"],
        default="all",
        help="Workflow step to run. 'all' generates data, both model outputs, and evaluation.",
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path for the CI/CD failure benchmark JSONL.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where model fixes and reports are saved.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face base model name or path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=str(DEFAULT_ADAPTER_PATH),
        help="PEFT/LoRA adapter path for the fine-tuned model.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float16, or float32.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Regenerate the benchmark JSONL even if it already exists.",
    )
    parser.add_argument(
        "--mock-model",
        action="store_true",
        help="Use deterministic mock outputs for smoke-testing without loading model weights.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Store full system/user prompts in the generated output JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data_path)
    out_dir = resolve_path(args.out_dir)
    paths = output_paths(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelRunConfig(
        base_model=args.base_model,
        adapter_path=resolve_path(args.adapter_path),
        cache_dir=resolve_path(args.cache_dir) if args.cache_dir else None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        dtype=args.dtype,
    )

    modes_that_need_data = {"data", "generate-base", "generate-finetuned", "generate", "evaluate", "all"}
    cases: List[Dict[str, Any]] = []
    if args.mode in modes_that_need_data:
        cases = load_or_create_cases(data_path, overwrite_data=args.overwrite_data or args.mode == "data")

    if args.mode == "data":
        return

    validate_adapter_ready(model_config, args.mode, args.mock_model)

    runner: Optional[HFModelRunner] = None if args.mock_model else HFModelRunner(model_config)

    if args.mode in {"generate-base", "generate", "all"}:
        generate_fixes(
            cases,
            variant="base",
            output_path=paths["base_outputs"],
            patch_dir=paths["base_patch_dir"],
            config=model_config,
            mock_model=args.mock_model,
            save_prompts=args.save_prompts,
            runner=runner,
        )

    if args.mode in {"generate-finetuned", "generate", "all"}:
        generate_fixes(
            cases,
            variant="finetuned",
            output_path=paths["finetuned_outputs"],
            patch_dir=paths["finetuned_patch_dir"],
            config=model_config,
            mock_model=args.mock_model,
            save_prompts=args.save_prompts,
            runner=runner,
        )

    if args.mode in {"evaluate", "all"}:
        base_records = read_jsonl(paths["base_outputs"])
        finetuned_records = read_jsonl(paths["finetuned_outputs"])
        write_comparison_report(cases, base_records, finetuned_records, paths["comparison"])
        report = evaluate_outputs(
            cases,
            base_records,
            finetuned_records,
            report_json_path=paths["evaluation_json"],
            report_md_path=paths["evaluation_md"],
        )
        summary = report["summary"]
        print("\nEvaluation complete")
        print(f"  Base accuracy:       {summary['base']['accuracy']:.4f}")
        print(f"  Fine-tuned accuracy: {summary['finetuned']['accuracy']:.4f}")
        print(f"  Overall winner:      {summary['overall_winner']}")
        print(f"  Report JSON:         {paths['evaluation_json']}")
        print(f"  Report Markdown:     {paths['evaluation_md']}")


if __name__ == "__main__":
    main()
