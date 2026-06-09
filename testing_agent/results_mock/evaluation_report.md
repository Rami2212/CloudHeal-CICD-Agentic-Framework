# CloudHeal CI/CD Model Evaluation Report

Generated: 2026-06-09T12:38:22.951803+00:00

## Summary

| Metric | Base model | Fine-tuned adapter | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5390 | 1.0000 | +0.4610 |
| Resolution success rate | 0.00% | 100.00% | +100.00% |
| Mean MTTR minutes | 51.33 | 4.00 | -47.33 |
| Patch-like output rate | 0.00% | 100.00% | +100.00% |
| Mean generation seconds | 0.00 | 0.00 | n/a |

Overall winner: **finetuned**

## Per-Case Results

| ID | Category | Winner | Base accuracy | FT accuracy | Base MTTR | FT MTTR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ci-failure-0001 | python-pytest-import-path | finetuned | 0.4841 | 1.0000 | 51.02 | 4.00 |
| ci-failure-0002 | python-mypy-type-hints | finetuned | 0.4635 | 1.0000 | 57.07 | 4.00 |
| ci-failure-0003 | frontend-npm-peer-dependency | finetuned | 0.4033 | 1.0000 | 56.67 | 4.00 |
| ci-failure-0004 | docker-build-context | finetuned | 0.6033 | 1.0000 | 45.11 | 4.00 |
| ci-failure-0005 | terraform-aws-provider-v4 | finetuned | 0.4931 | 1.0000 | 60.87 | 4.00 |
| ci-failure-0006 | maven-java-version | finetuned | 0.6155 | 1.0000 | 46.69 | 4.00 |
| ci-failure-0007 | eslint-react-no-undef | finetuned | 0.7155 | 1.0000 | 40.54 | 4.00 |
| ci-failure-0008 | kubernetes-hpa-api-version | finetuned | 0.4803 | 1.0000 | 58.95 | 4.00 |
| ci-failure-0009 | github-actions-release-permissions | finetuned | 0.5931 | 1.0000 | 49.46 | 4.00 |
| ci-failure-0010 | helm-yaml-indentation | finetuned | 0.5385 | 1.0000 | 46.92 | 4.00 |

## Metric Notes

- Accuracy is a lightweight research proxy, not a replacement for applying patches and running tests.
- Resolution success is estimated from patch format, target file coverage, expected fix keywords, and safety checks.
- MTTR is estimated because these synthetic cases do not execute real CI jobs end to end.
