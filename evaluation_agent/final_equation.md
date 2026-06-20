# Final Equation: Combined Fix Score (CFS)

## 1. Why a combined metric

`equations.md` lists six metrics, but they answer different questions and none is sufficient alone for evaluating a CI/CD log-to-fix model:

| Metric | What it captures | Limitation on its own |
|---|---|---|
| pass@k | Probability at least one of k samples is *exactly* correct | Binary — gives no partial credit for "almost right" fixes |
| Test Pass Rate (TPR) | Behavioral correctness | Requires a live test runner / sandboxed CI execution |
| Exact Fix Accuracy | Same as pass@1 | Redundant with pass@k at k=1 |
| CodeBLEU | Syntactic + structural similarity to ground truth | A high score doesn't guarantee the patch even runs |
| Patch Success Rate (PSR) | Search efficiency (plausible / generated) | Also needs compilation + test execution |
| Compilation Success Rate (CSR) | Whether the patch is syntactically valid at all | Says nothing about correctness |

The combined metric exists to fold these into a single, weighted score so a model checkpoint can be ranked with one number, while still letting you inspect the components individually.

## 2. Constraint imposed by the `diff-xyz` dataset

`JetBrains-Research/diff-xyz` (the dataset you're evaluating against) provides `old_code`, `new_code`, `udiff`, and `message` per commit — but **no test suite and no build/CI harness**. That means **TPR and PSR cannot be computed** for this dataset as written, since both require actually running tests.

So the equation below is defined in two modes:

- **Static mode** (what `evaluation.py` runs today, against `diff-xyz`): uses only signals computable from code text — pass@k (exact match against `new_code`), CodeBLEU, and CSR (syntax validity).
- **Dynamic mode** (extension point, for when you point this at a real CI sandbox / SWE-bench-style harness with executable tests): adds TPR back in.

## 3. Final equation

### Dynamic mode (full form, all six source metrics folded in)

$$
CFS = w_1 \cdot \text{pass@k} \;+\; w_2 \cdot TPR \;+\; w_3 \cdot CodeBLEU \;+\; w_4 \cdot CSR
$$

subject to $w_1+w_2+w_3+w_4 = 1$, each term normalized to $[0,1]$.

(*Exact Fix Accuracy* is intentionally not a separate term — it's the special case pass@1. *PSR* is intentionally not a separate term — it's redundant with CSR for this purpose: a patch that compiles and passes validation tests is just CSR ∧ TPR, which the weighted sum already rewards jointly.)

### Static mode (used by `evaluation.py` on `diff-xyz`, since $TPR$ is unavailable)

$$
CFS_{static} = w_1 \cdot \text{pass@k} \;+\; w_3 \cdot CodeBLEU \;+\; w_4 \cdot CSR, \quad w_4 \text{ reweighted, } w_2 = 0
$$

**Default weights used in `evaluation.py`:**

$$
CFS_{static} = 0.50 \cdot \text{pass@k} + 0.35 \cdot CodeBLEU + 0.15 \cdot CSR
$$

### Rationale for default weights

- **pass@k gets the largest weight (0.50)** — exact correctness against the ground-truth fix is still the most decisive signal you have without a live test runner.
- **CodeBLEU (0.35)** — rewards fixes that are *semantically/structurally* close to the reference even when not a byte-for-byte match (different variable names, equivalent refactor, etc.), which is common and shouldn't score zero.
- **CSR (0.15)** — a low weight, deliberately: syntactic validity is a necessary-but-weak signal (a no-op patch is syntactically valid too), so it acts as a tie-breaker / sanity check rather than a primary driver.

### Recommended dynamic-mode weights (once you wire in real test execution)

$$
CFS = 0.35 \cdot \text{pass@k} + 0.30 \cdot TPR + 0.20 \cdot CodeBLEU + 0.15 \cdot CSR
$$

TPR becomes the second-largest term here because, once available, "did the existing/new tests actually pass" is more informative than text similarity.

## 4. Per-term definitions as used in `evaluation.py`

- **pass@k** — unbiased estimator from equations.md #1, $1 - \binom{n-c}{k}/\binom{n}{k}$, where $c$ = number of the $n$ generated candidates whose code is exactly equal (whitespace-normalized) to `new_code`.
- **CodeBLEU** — equations.md #4. Uses the official `codebleu` package if installed; otherwise falls back to a simplified BLEU‑4 + line-overlap proxy (see `evaluation.py` for details and how to swap in the official implementation).
- **CSR** — equations.md #6, computed without a real compiler: `ast.parse` for Python, a generic bracket/quote-balance check for other languages. Swap in a real compiler/linter call per language for stronger signal.
- **TPR** — equations.md #2, left at weight 0 / unimplemented until you connect a sandbox that can execute the patched repo's tests.

## 5. How to extend to dynamic mode

In `evaluation.py`, set `weights.w_tpr > 0` and rebalance the other three weights to sum to 1, then implement a `run_tests(patched_repo_path) -> TPR` function that applies the generated patch to a checkout of the repo and reports the fraction of passing tests. Pass its output into `combined_fix_score(..., tpr_score=...)`.
