# Evaluation Metrics for diff-xyz Fix Generation

## 1. What this is, and what changed

This used to define a single combined "Fix Score" (CFS) folding several
metrics into one weighted number. That's been removed. The metrics
currently in `evaluation.py` are **pass@k** and **CodeBLEU**, reported
**side by side**, plus a diagnostic-only syntax check (CSR). There is no
combined score.

Why drop CFS: a weighted blend is only meaningful if the weights are
validated against something (the way CodeBLEU's own internal weights were
tuned against human judgment — see [Ren20] Sec 4.4). No such validation
exists for combining pass@k with CodeBLEU, so manufacturing weights for it
would just be picking numbers that look reasonable. Two honest numbers beat
one fabricated one.

Every metric below is traceable to a specific source:

| Metric | Source | Status |
|---|---|---|
| pass@k | Chen et al. 2021, *"Evaluating Large Language Models Trained on Code"* (arXiv:2107.03374), Eq. 1 | Implemented faithfully |
| CodeBLEU | Ren et al. 2020, *"CodeBLEU: a Method for Automatic Evaluation of Code Synthesis"* (arXiv:2009.10297), Eq. 1 | Implemented faithfully (Python full; other languages partial — see §4) |
| CSR (syntax validity) | Neither paper | Project-defined diagnostic, never folded into a score |

## 2. What the `diff-xyz` dataset actually is

`JetBrains-Research/diff-xyz` (https://huggingface.co/datasets/JetBrains-Research/diff-xyz)
provides `old_code`, `new_code`, `message`, and four diff representations
(`udiff`, `udiff-h`, `udiff-l`, `search-replace`) per commit, across five
languages (python, javascript, java, kotlin, rust). It is built for
**diff-understanding** tasks (Apply / Anti-Apply / Diff-Generation), not
CI-failure-driven fix generation.

Two things worth being explicit about, since they affect how to read the
numbers below:

- **No test suite, no build/CI harness.** There is no way to execute the
  patched code, so Test Pass Rate (TPR) and anything requiring actual test
  execution cannot be computed against this dataset as-is. If you wire in a
  real sandbox later (see §6), that changes.
- **`message` is a commit message, not a CI log.** `data.py` maps it to
  `context` as a (weak) stand-in for a fix instruction. It's whatever the
  original commit author wrote (e.g. "Fix off-by-one error"), not a build
  failure trace — treat it as a noisier signal than an actual CI log would
  be.

## 3. pass@k

**Definition** ([Chen21] Eq. 1, the unbiased estimator):

$$
\text{pass@}k := \mathbb{E}_{\text{Problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

where $n$ = candidates generated, $c$ = candidates counted correct, $k$ =
samples considered.

**Caveat vs. the paper:** [Chen21] defines $c$ as the number of candidates
that **pass unit tests**. Since `diff-xyz` has no test suite, `evaluation.py`
instead counts $c$ as the number of candidates that are **exactly equal**
(whitespace-normalized) to `new_code`. This is a textual-correctness
stand-in for functional correctness — strictly weaker, since a
functionally-correct fix that differs textually (different variable names,
equivalent refactor) won't be counted. Treat pass@k under this dataset as a
**lower bound** on functional correctness, not the metric the paper
benchmarks Codex with.

## 4. CodeBLEU

**Definition** ([Ren20] Eq. 1):

$$
CodeBLEU = \alpha \cdot BLEU + \beta \cdot BLEU_{weight} + \gamma \cdot Match_{ast} + \delta \cdot Match_{df}
$$

Four components, each capturing something the others miss:

| Term | What it measures | Paper section |
|---|---|---|
| `BLEU` | Standard 4-gram token overlap with brevity penalty | — |
| `BLEU_weight` | Same, but language keywords get 5× the weight of other tokens | Sec 3.1, Eq. 2–3 |
| `Match_ast` | Fraction of reference AST subtrees (leaf values excluded) found in the candidate | Sec 3.2, Eq. 4 |
| `Match_df` | Fraction of reference data-flow edges (which variable's value comes from which) found in the candidate, after name-normalization | Sec 3.3, Eq. 5 |

**Default weights** — the paper's own recommended combination `[7]`
(Table 6, Sec 4.4), which gave the best correlation with human judgment
across their three tasks:

$$
\alpha, \beta, \gamma, \delta = 0.10,\ 0.10,\ 0.40,\ 0.40
$$

**Why this matters in practice** (the paper's own motivating example, Fig.
2): a candidate that returns the *wrong variable* (`return total` instead of
`return y`) can have near-identical BLEU and AST scores to the correct
fix — the only thing that catches it is `Match_df`, since the data-flow
edge into the return statement points at a different variable. This is
exactly the failure mode CodeBLEU exists to catch, and it's why this file
implements the real four-term metric rather than a token-overlap proxy.

**Implementation:**
- If the official `codebleu` package is installed, it's used directly
  (covers python, java, javascript, cpp, c_sharp, php, go, ruby via
  tree-sitter).
- If not, `evaluation.py` reimplements all four terms in-file using
  Python's standard-library `ast` module. This only works for **Python**
  code — `Match_ast` and `Match_df` are not computable for other languages
  without tree-sitter, so for non-Python candidates without the official
  package installed, those two terms report `None` (not a fabricated
  value), and the score renormalizes onto just `BLEU` + `BLEU_weight`.

## 5. CSR (Compilation/Syntax Success Rate) — diagnostic only

Neither paper defines this. It's a cheap check — does the candidate parse?
(`ast.parse` for Python, bracket/quote-balance for everything else.)

**It is never combined with pass@k or CodeBLEU, and should not be used to
rank checkpoints.** It's deliberately weak evidence: a no-op patch that
just echoes the input is syntactically valid too. It's also largely
redundant with CodeBLEU besides — an unparseable candidate already scores
near-zero on `Match_ast`/`Match_df`. CSR exists purely to answer "what
fraction of my model's outputs are even well-formed," which is a useful
debugging stat, not a quality signal.

In the report JSON it's labeled `avg_csr_diagnostic_only` for exactly this
reason — to make it hard to mistake for a score.

## 6. Extending to real test execution

If you eventually point this at a sandbox that can run the patched repo's
tests, you gain access to a metric **neither file currently computes**:
[Chen21]'s actual functional-correctness check (Sec 2.3 — execute the
candidate against unit tests in a sandboxed environment, not a syntax
check). That would let you:

- Compute *true* pass@k (Sec 2 of this doc's caveat goes away — $c$ becomes
  "candidates that pass tests," matching [Chen21] exactly instead of being
  a textual proxy for it).
- Get a Test Pass Rate per candidate, which is meaningfully stronger
  evidence than CSR's syntax check.

This isn't implemented here because `diff-xyz` provides no build/CI
harness to run against (§2) — it would need to be supplied separately
(e.g. a SWE-bench-style sandbox keyed off `repo`/`commit` from `data.py`'s
`meta` field).

## 7. Files

- **`data.py`** — downloads `JetBrains-Research/diff-xyz`, maps each row
  to `{id, language, context, input_code, reference_code, reference_diff,
  meta}`, writes `data/diff_xyz_eval.jsonl`.
- **`evaluation.py`** — reads that JSONL, generates `n` candidate fixes per
  row via `generate_fixes()` (stub — wire up your model), scores each with
  pass@k + CodeBLEU (+ CSR diagnostic), writes a per-sample and aggregate
  report to `data/evaluation_report.json`.
