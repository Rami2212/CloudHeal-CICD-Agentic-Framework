"""
evaluation.py
-------------
Evaluates a CI/CD fix-generation model against the prepared diff-xyz dataset
using the Combined Fix Score (CFS) defined in final_equation.md:

    CFS = 0.50 * pass@k + 0.35 * CodeBLEU + 0.15 * CSR

Plug your fine-tuned model into `generate_fixes()` below -- that is the
only required change to run this end-to-end.

Optional: pip install codebleu   (falls back to a built-in proxy if absent)

Usage:
    python evaluation.py --data data/diff_xyz_eval.jsonl --n 5 --k 1
"""
import argparse
import ast
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# 1. Model interface -- REPLACE THIS with a call to your fine-tuned model
# ---------------------------------------------------------------------------

def generate_fixes(input_code: str, context: str, language: str, n: int = 1) -> List[str]:
    """
    Generate `n` candidate fixes for the given input_code/context.

    Replace this stub with a call to your fine-tuned CI/CD fix-generation
    model, e.g.:

        prompt = build_prompt(input_code, context, language)
        outputs = model.generate(prompt, num_return_sequences=n, ...)
        return [postprocess(o) for o in outputs]

    Each returned string must be the *full fixed code* (comparable to
    `reference_code`), not a raw diff -- if your model emits a patch/diff,
    apply it to `input_code` first to reconstruct the full file.
    """
    raise NotImplementedError(
        "Wire generate_fixes() up to your fine-tuned model before running evaluation."
    )


# ---------------------------------------------------------------------------
# 2. Metric primitives (equations.md items 1, 4, 6 -- see final_equation.md)
# ---------------------------------------------------------------------------

def normalize_code(code: str) -> str:
    """Whitespace-insensitive normalization used for exact-match correctness checks."""
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip() != "")


def is_correct(candidate: str, reference: str) -> bool:
    return normalize_code(candidate) == normalize_code(reference)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator (equations.md #1):
        pass@k = 1 - C(n-c, k) / C(n, k)
    n = candidates generated, c = candidates exactly matching the reference, k = samples considered.
    """
    if n == 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def syntax_is_valid(code: str, language: str) -> bool:
    """
    Lightweight syntactic-validity check -- a proxy for Compilation Success Rate
    (equations.md #6) when no real compiler/test runner is wired in.
    Swap in a real compiler/linter subprocess call per language for a stronger signal.
    """
    language = (language or "").lower()
    if language == "python":
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    # Generic brace/bracket/paren balance check for curly-brace languages
    # (java, kotlin, javascript, rust, etc.)
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    stack = []
    in_string = None
    for ch in code:
        if in_string:
            if ch == in_string:
                in_string = None
            continue
        if ch in ("'", '"', "`"):
            in_string = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in closing:
            if not stack or stack.pop() != ch:
                return False
    return not stack


def _tokenize(code: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", code)


def _ngram_precision(cand_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    if len(cand_tokens) < n:
        return 0.0
    cand_ngrams = [tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1)]
    if not cand_ngrams:
        return 0.0
    ref_ngrams = [tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]
    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)
    overlap = sum(min(cnt, ref_counts.get(g, 0)) for g, cnt in cand_counts.items())
    return overlap / len(cand_ngrams)


def simplified_codebleu(candidate: str, reference: str) -> float:
    """
    Simplified CodeBLEU proxy (equations.md #4) used when the `codebleu`
    package isn't installed: BLEU-4 (geometric mean of 1..4-gram precision,
    with a brevity penalty) blended with a line-overlap structure-similarity
    term standing in for the AST/data-flow match terms.

    For production-grade scoring, install the official package instead:
        pip install codebleu
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions = [_ngram_precision(cand_tokens, ref_tokens, n) for n in range(1, 5)]
    precisions = [p if p > 0 else 1e-9 for p in precisions]
    bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    bleu *= bp

    cand_lines = set(normalize_code(candidate).splitlines())
    ref_lines = set(normalize_code(reference).splitlines())
    struct_sim = len(cand_lines & ref_lines) / max(len(ref_lines), 1)

    alpha, gamma = 0.6, 0.4
    return alpha * bleu + gamma * struct_sim


def codebleu_score(candidate: str, reference: str, language: str) -> float:
    """Use the official `codebleu` package if available, else fall back to the proxy above."""
    try:
        from codebleu import calc_codebleu
        supported = {"python", "java", "javascript", "cpp", "c_sharp", "php", "go", "ruby"}
        lang = (language or "").lower()
        if lang not in supported:
            return simplified_codebleu(candidate, reference)
        result = calc_codebleu([reference], [candidate], lang=lang)
        return result["codebleu"]
    except Exception:
        return simplified_codebleu(candidate, reference)


# ---------------------------------------------------------------------------
# 3. Combined Fix Score (final equation -- see final_equation.md)
# ---------------------------------------------------------------------------

@dataclass
class CFSWeights:
    w_pass_at_k: float = 0.50
    w_codebleu: float = 0.35
    w_csr: float = 0.15
    w_tpr: float = 0.0  # raise above 0 (and rebalance the others) once a real test runner is wired in


def combined_fix_score(pass_at_k_score: float, codebleu_score_: float, csr_score: float,
                        tpr_score: float = 0.0, weights: CFSWeights = CFSWeights()) -> float:
    return (
        weights.w_pass_at_k * pass_at_k_score
        + weights.w_codebleu * codebleu_score_
        + weights.w_csr * csr_score
        + weights.w_tpr * tpr_score
    )


# ---------------------------------------------------------------------------
# 4. Evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    id: str
    n: int
    c: int
    pass_at_k: float
    codebleu: float
    csr: float
    cfs: float


def evaluate_sample(record: Dict[str, Any], n: int, k: int, weights: CFSWeights) -> SampleResult:
    candidates = generate_fixes(record["input_code"], record["context"], record["language"], n=n)
    n_gen = len(candidates)

    c_correct = sum(is_correct(c, record["reference_code"]) for c in candidates)
    p_at_k = pass_at_k(n_gen, c_correct, k)

    cb_scores = [codebleu_score(c, record["reference_code"], record["language"]) for c in candidates]
    avg_codebleu = sum(cb_scores) / len(cb_scores) if cb_scores else 0.0

    csr_scores = [syntax_is_valid(c, record["language"]) for c in candidates]
    csr = sum(csr_scores) / len(csr_scores) if csr_scores else 0.0

    cfs = combined_fix_score(p_at_k, avg_codebleu, csr, tpr_score=0.0, weights=weights)

    return SampleResult(id=record["id"], n=n_gen, c=c_correct, pass_at_k=p_at_k,
                         codebleu=avg_codebleu, csr=csr, cfs=cfs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CI/CD fix-generation model on diff-xyz using CFS.")
    parser.add_argument("--data", default="data/diff_xyz_eval.jsonl")
    parser.add_argument("--n", type=int, default=5, help="Number of candidate fixes to generate per sample")
    parser.add_argument("--k", type=int, default=1, help="k for pass@k (k <= n)")
    parser.add_argument("--limit", type=int, default=0, help="0 = evaluate all samples")
    parser.add_argument("--report", default="data/evaluation_report.json")
    args = parser.parse_args()

    weights = CFSWeights()

    records = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    results: List[SampleResult] = []
    for i, rec in enumerate(records, 1):
        try:
            res = evaluate_sample(rec, n=args.n, k=args.k, weights=weights)
        except NotImplementedError as e:
            print(str(e))
            return
        results.append(res)
        print(f"[{i}/{len(records)}] {res.id} | pass@{args.k}={res.pass_at_k:.3f} "
              f"CodeBLEU={res.codebleu:.3f} CSR={res.csr:.3f} CFS={res.cfs:.3f}")

    if not results:
        print("No results to report.")
        return

    agg = {
        "n_samples": len(results),
        f"avg_pass@{args.k}": sum(r.pass_at_k for r in results) / len(results),
        "avg_codebleu": sum(r.codebleu for r in results) / len(results),
        "avg_csr": sum(r.csr for r in results) / len(results),
        "avg_combined_fix_score": sum(r.cfs for r in results) / len(results),
        "weights": weights.__dict__,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_sample": [r.__dict__ for r in results]}, f, indent=2)

    print("\n=== Aggregate results ===")
    for key, val in agg.items():
        print(f"{key}: {val}")
    print(f"\nFull report saved to {args.report}")


if __name__ == "__main__":
    main()
