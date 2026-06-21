"""
evaluation.py
-------------
Evaluates a CI/CD fix-generation model against the prepared diff-xyz dataset.

Every metric in this file is traceable to one of two papers:

  [Chen21]  Chen et al., "Evaluating Large Language Models Trained on Code"
            (arXiv:2107.03374v2) -- defines pass@k (Eq. 1, Sec 2.1) and its
            numerically-stable estimator (Fig. 3).

  [Ren20]   Ren et al., "CodeBLEU: a Method for Automatic Evaluation of Code
            Synthesis" (arXiv:2009.10297v2) -- defines CodeBLEU as a 4-term
            weighted sum (Eq. 1): BLEU + weighted n-gram match + AST match +
            data-flow match (Sec 3).

A third value, Compilation/Syntax Success Rate (CSR), is reported alongside
pass@k and CodeBLEU as a DIAGNOSTIC ONLY -- it is NOT defined in either
paper above, is NOT folded into any combined/weighted score, and should not
be used to rank checkpoints on its own (a no-op patch is syntactically
valid too). It exists purely to answer "what fraction of my model's outputs
are even well-formed?", which is a useful thing to know when debugging a
model but is not evidence of fix quality. See the CSR section below for
details.

There is no combined score in this file. pass@k and CodeBLEU answer
different questions (textual correctness vs. structural/semantic
closeness) and are reported side by side rather than collapsed into one
number -- a single weighted blend would need validated weights (the way
[Ren20] validated CodeBLEU's own alpha/beta/gamma/delta against human
judgment, Sec 4.4), and no such validation exists for combining pass@k
with CodeBLEU, so we don't manufacture one.

Plug your fine-tuned model into `generate_fixes()` below -- that is the
only required change to run this end-to-end.

Optional: pip install codebleu   (uses Microsoft's reference implementation
of [Ren20] instead of the in-file reimplementation below)
Optional: pip install tree_sitter tree_sitter_languages  (only needed if you
want AST/data-flow matching for non-Python languages; see CodeBLEUScorer)

Usage:
    python evaluation.py --data data/diff_xyz_eval.jsonl --n 5 --k 1
"""
import argparse
import ast
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple


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
# 2. pass@k  -- [Chen21] Eq. 1, Sec 2.1, Fig. 3
# ---------------------------------------------------------------------------
#
# Paper definition:
#   pass@k := E_Problems[ 1 - C(n-c, k) / C(n, k) ]
# where n = samples generated, c = samples passing functional correctness
# (unit tests, in the paper), k = samples considered.
#
# Caveat vs. the paper: diff-xyz ships no test suite (see module docstring
# of final_equation.md / the dataset itself), so `c` here counts candidates
# that are EXACTLY equal (whitespace-normalized) to the reference fix, not
# candidates that pass tests. This is a textual-correctness stand-in for
# functional correctness -- the paper explicitly warns (Sec 2.1) that
# match-based correctness is weaker than test-based correctness, so treat
# pass@k under this dataset as an upper-bound-style proxy, not the metric
# the paper benchmarks Codex with.

def normalize_code(code: str) -> str:
    """Whitespace-insensitive normalization used for exact-match correctness checks."""
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip() != "")


def is_correct(candidate: str, reference: str) -> bool:
    return normalize_code(candidate) == normalize_code(reference)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator, [Chen21] Eq. 1 / Fig. 3:
        pass@k = 1 - C(n-c, k) / C(n, k)
    n = candidates generated, c = candidates counted correct, k = samples considered.

    Implementation mirrors the paper's numerically-stable form (computing the
    product term-by-term rather than raw binomial coefficients would also
    work here since we have math.comb; both give the same exact result for
    reasonable n, k).
    """
    if n == 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# 3. CodeBLEU -- [Ren20] Eq. 1-5, Sec 3
# ---------------------------------------------------------------------------
#
# Paper definition (Eq. 1):
#   CodeBLEU = alpha * BLEU + beta * BLEU_weight + gamma * Match_ast + delta * Match_df
#
#   BLEU        -- standard 4-gram BLEU with brevity penalty (Papineni et al. 2002)
#   BLEU_weight -- n-gram BLEU where keyword unigrams get 5x the weight of
#                  other tokens (Sec 3.1, Eq. 2-3)
#   Match_ast   -- fraction of reference AST subtrees (leaves excluded) that
#                  appear in the candidate AST (Sec 3.2, Eq. 4)
#   Match_df    -- fraction of reference data-flow edges that appear in the
#                  candidate's data-flow graph, after normalizing variable
#                  names (Sec 3.3, Eq. 5)
#
# Paper's recommended weight combination [7] (Table 6, Sec 4.4):
#   alpha, beta, gamma, delta = 0.10, 0.10, 0.40, 0.40
# (this combination gave the best correlation with human judgment in 2 of
# their 3 tasks and is what they recommend for general code-synthesis tasks)
#
# If the official `codebleu` package is installed, we use it directly --
# it implements all four terms above using tree-sitter parsers across many
# languages and is the authoritative implementation.
#
# If it is NOT installed, this file reimplements all four terms for Python
# using the standard-library `ast` module (Python's own AST instead of
# tree-sitter). For non-Python languages without `codebleu` installed, only
# the BLEU and BLEU_weight terms are computable here (no stdlib AST/data-flow
# parser exists for arbitrary languages) -- Match_ast and Match_df are
# reported as None for those languages rather than silently faking them with
# an unrelated heuristic.

PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
}

# Minimal keyword sets for the languages [Ren20] evaluates / the official
# package supports, used only by the BLEU_weight fallback term below.
LANGUAGE_KEYWORDS = {
    "python": PY_KEYWORDS,
    "java": {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch",
        "char", "class", "const", "continue", "default", "do", "double",
        "else", "enum", "extends", "final", "finally", "float", "for",
        "goto", "if", "implements", "import", "instanceof", "int",
        "interface", "long", "native", "new", "package", "private",
        "protected", "public", "return", "short", "static", "strictfp",
        "super", "switch", "synchronized", "this", "throw", "throws",
        "transient", "try", "void", "volatile", "while",
    },
    "javascript": {
        "break", "case", "catch", "class", "const", "continue", "debugger",
        "default", "delete", "do", "else", "export", "extends", "finally",
        "for", "function", "if", "import", "in", "instanceof", "let",
        "new", "return", "super", "switch", "this", "throw", "try",
        "typeof", "var", "void", "while", "with", "yield",
    },
}


def _tokenize(code: str) -> List[str]:
    """Simple word/punctuation tokenizer, language-agnostic."""
    return re.findall(r"\w+|[^\w\s]", code)


def _ngram_precision(cand_tokens: List[str], ref_tokens: List[str], n: int,
                      keyword_weight: Optional[Dict[str, float]] = None) -> float:
    """
    Standard clipped n-gram precision (Papineni et al. 2002), optionally with
    per-token weights for the keyword-weighted variant ([Ren20] Eq. 2).
    """
    if len(cand_tokens) < n:
        return 0.0
    cand_ngrams = [tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1)]
    if not cand_ngrams:
        return 0.0
    ref_ngrams = [tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]
    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    def w(gram: Tuple[str, ...]) -> float:
        if keyword_weight is None or n != 1:
            return 1.0
        return keyword_weight.get(gram[0], 1.0)

    num = sum(w(g) * min(cnt, ref_counts.get(g, 0)) for g, cnt in cand_counts.items())
    den = sum(w(g) * cnt for g, cnt in cand_counts.items())
    return num / den if den > 0 else 0.0


def _brevity_penalty(cand_tokens: List[str], ref_tokens: List[str]) -> float:
    c, r = len(cand_tokens), len(ref_tokens)
    if c == 0:
        return 0.0
    return 1.0 if c > r else math.exp(1 - r / c)


def _bleu4(candidate: str, reference: str) -> float:
    """Standard 4-gram BLEU with brevity penalty -- the `BLEU` term in Eq. 1."""
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    precisions = [_ngram_precision(cand_tokens, ref_tokens, n) for n in range(1, 5)]
    precisions = [p if p > 0 else 1e-9 for p in precisions]
    bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return bleu * _brevity_penalty(cand_tokens, ref_tokens)


def _bleu_weighted(candidate: str, reference: str, language: str) -> float:
    """
    Keyword-weighted n-gram match -- the `BLEU_weight` term in Eq. 1.
    [Ren20] Sec 3.1: keyword unigrams get 5x the weight of other tokens;
    only unigrams are weighted in the paper (N = 1 in their Eq. 3 for the
    weighted term, i.e. only n=1 uses keyword weights -- they still report
    BLEU_weight as a single-n-gram score, not a 1-4 gram geometric mean).
    """
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    keywords = LANGUAGE_KEYWORDS.get((language or "").lower(), PY_KEYWORDS)
    weight_map = {kw: 5.0 for kw in keywords}
    p1 = _ngram_precision(cand_tokens, ref_tokens, 1, keyword_weight=weight_map)
    p1 = p1 if p1 > 0 else 1e-9
    bleu_w = math.exp(math.log(p1))  # N=1, so geometric mean over {p1} is just p1
    return bleu_w * _brevity_penalty(cand_tokens, ref_tokens)


# --- AST match (Match_ast, [Ren20] Sec 3.2 / Eq. 4) -------------------------
#
# Paper: extract all subtrees of the reference and candidate ASTs (leaf
# nodes excluded, since variable/function *naming* shouldn't matter -- only
# syntactic structure), then:
#   Match_ast = Countclip(T_cand) / Count(T_ref)
# i.e. the fraction of reference subtrees that are matched in the candidate.
#
# The paper uses tree-sitter; we use Python's built-in `ast` module for
# Python code, which gives an equivalent structural tree without an extra
# dependency. Leaves (Name, Constant, etc. identifier/literal values) are
# excluded the same way the paper excludes naming.

def _python_ast_subtrees(code: str) -> Optional[List[Tuple]]:
    """
    Returns a list of "subtree signatures" for a Python AST, with leaf
    values (names, constants) stripped out -- mirrors [Ren20] Sec 3.2's
    instruction to exclude AST leaves so only syntactic structure is
    compared, not identifier naming.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    subtrees: List[Tuple] = []

    def structural_signature(node: ast.AST) -> Tuple:
        """Recursively build a signature of node types only (no leaf values)."""
        children = []
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        children.append(structural_signature(item))
            elif isinstance(value, ast.AST):
                children.append(structural_signature(value))
            # scalar leaf values (str/int/identifier names/constants) are
            # intentionally dropped here -- this is the "exclude leaves" step
        return (type(node).__name__, tuple(children))

    def walk(node: ast.AST):
        if isinstance(node, ast.AST):
            sig = structural_signature(node)
            subtrees.append(sig)
            for child in ast.iter_child_nodes(node):
                walk(child)

    walk(tree)
    return subtrees


def _flatten_subtree_counts(subtrees: List[Tuple]) -> Counter:
    """
    Flattens each subtree signature (and all of its own sub-signatures) into
    a multiset, so "matching a subtree" means matching that exact structural
    shape anywhere in the tree -- consistent with [Ren20]'s "all subtrees"
    framing (Sec 3.2), not just top-level/whole-tree match.
    """
    counts: Counter = Counter()

    def expand(sig: Tuple):
        counts[sig] += 1
        _node_type, children = sig
        for child in children:
            expand(child)

    for s in subtrees:
        expand(s)
    return counts


def ast_match_score(candidate: str, reference: str, language: str) -> Optional[float]:
    """
    Match_ast, [Ren20] Eq. 4: Countclip(T_cand) / Count(T_ref).
    Returns None if the language isn't supported by this in-file
    implementation (only Python, without the official `codebleu` package).
    """
    if (language or "").lower() != "python":
        return None

    ref_subtrees = _python_ast_subtrees(reference)
    cand_subtrees = _python_ast_subtrees(candidate)
    if ref_subtrees is None or cand_subtrees is None or not ref_subtrees:
        return 0.0

    ref_counts = _flatten_subtree_counts(ref_subtrees)
    cand_counts = _flatten_subtree_counts(cand_subtrees)

    total_ref = sum(ref_counts.values())
    if total_ref == 0:
        return 0.0
    clipped_hits = sum(min(cnt, cand_counts.get(sig, 0)) for sig, cnt in ref_counts.items())
    return clipped_hits / total_ref


# --- Data-flow match (Match_df, [Ren20] Sec 3.3 / Eq. 5) --------------------
#
# Paper: build a directed graph where nodes are variables and an edge
# v_i -> v_j means "the value of v_j comes from v_i". Normalize variable
# names (ignore position, rename in order of appearance as var_0, var_1...),
# then:
#   Match_df = Countclip(DF_cand) / Count(DF_ref)
#
# This in-file version handles the common, unambiguous data-flow case for
# Python: direct assignment (`y = x`, `y = f(x, z)`) and return statements
# (`return x`), tracking which named variables flow into which. It does NOT
# attempt full reaching-definitions/points-to analysis (e.g. through
# attribute access, complex control flow, or aliasing) -- those cases are
# simply not counted as edges rather than guessed at.

def _extract_dataflow_edges(code: str) -> Optional[List[Tuple[str, str]]]:
    """
    Returns a list of (source_var, target_var) edges meaning "target's value
    comes from source", for simple assignment and return statements.
    Returns None on a syntax error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    edges: List[Tuple[str, str]] = []

    def names_in(node: ast.AST) -> List[str]:
        return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            sources = names_in(node.value)
            for target in node.targets:
                for tgt_name in names_in(target):
                    for src_name in sources:
                        if src_name != tgt_name:
                            edges.append((src_name, tgt_name))
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign):
            sources = names_in(node.value)
            for tgt_name in names_in(node.target):
                edges.append((tgt_name, tgt_name))  # x += ... depends on prior x
                for src_name in sources:
                    if src_name != tgt_name:
                        edges.append((src_name, tgt_name))
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return):
            if node.value is not None:
                for src_name in names_in(node.value):
                    edges.append((src_name, "<return>"))
            self.generic_visit(node)

    Visitor().visit(tree)
    return edges


def _normalize_dataflow_edges(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Renames variables to var_0, var_1, ... in order of first appearance,
    per [Ren20] Sec 3.3 Step 2 ("ignore the variable position and normalize
    their names"). "<return>" is kept as a fixed sentinel since it isn't a
    real variable name to normalize.
    """
    mapping: Dict[str, str] = {}
    counter = 0
    normalized = []
    for src, tgt in edges:
        for v in (src, tgt):
            if v != "<return>" and v not in mapping:
                mapping[v] = f"var_{counter}"
                counter += 1
        norm_src = mapping.get(src, src)
        norm_tgt = mapping.get(tgt, tgt)
        normalized.append((norm_src, norm_tgt))
    return normalized


def dataflow_match_score(candidate: str, reference: str, language: str) -> Optional[float]:
    """
    Match_df, [Ren20] Eq. 5: Countclip(DF_cand) / Count(DF_ref).
    Returns None if the language isn't supported by this in-file
    implementation (only Python, without the official `codebleu` package).
    """
    if (language or "").lower() != "python":
        return None

    ref_edges_raw = _extract_dataflow_edges(reference)
    cand_edges_raw = _extract_dataflow_edges(candidate)
    if ref_edges_raw is None or cand_edges_raw is None:
        return 0.0

    ref_edges = _normalize_dataflow_edges(ref_edges_raw)
    cand_edges = _normalize_dataflow_edges(cand_edges_raw)

    if not ref_edges:
        return 0.0

    ref_counts = Counter(ref_edges)
    cand_counts = Counter(cand_edges)
    clipped_hits = sum(min(cnt, cand_counts.get(e, 0)) for e, cnt in ref_counts.items())
    return clipped_hits / sum(ref_counts.values())


@dataclass
class CodeBLEUWeights:
    """
    [Ren20] Eq. 1 weights. Default is the paper's recommended combination
    [7] from Table 6 / Sec 4.4 (best avg. correlation with human judgment
    across their 3 tasks): alpha, beta, gamma, delta = 0.10, 0.10, 0.40, 0.40
    """
    alpha: float = 0.10  # BLEU
    beta: float = 0.10   # BLEU_weight
    gamma: float = 0.40  # Match_ast
    delta: float = 0.40  # Match_df


@dataclass
class CodeBLEUResult:
    codebleu: float
    bleu: float
    bleu_weight: float
    match_ast: Optional[float]  # None if unsupported for this language
    match_df: Optional[float]   # None if unsupported for this language
    used_official_package: bool


def codebleu_score(candidate: str, reference: str, language: str,
                    weights: CodeBLEUWeights = CodeBLEUWeights()) -> CodeBLEUResult:
    """
    [Ren20] Eq. 1: CodeBLEU = alpha*BLEU + beta*BLEU_weight + gamma*Match_ast + delta*Match_df

    Tries the official `codebleu` package first (covers python, java,
    javascript, cpp, c_sharp, php, go, ruby via tree-sitter). Falls back to
    the in-file implementation above, which only computes Match_ast /
    Match_df for Python; for other languages those terms are None and the
    alpha/beta weights are renormalized to cover the full score so the
    result is never silently padded with a fabricated structural score.
    """
    try:
        from codebleu import calc_codebleu
        supported = {"python", "java", "javascript", "cpp", "c_sharp", "php", "go", "ruby"}
        lang = (language or "").lower()
        if lang in supported:
            result = calc_codebleu(
                [reference], [candidate], lang=lang,
                weights=(weights.alpha, weights.beta, weights.gamma, weights.delta),
            )
            return CodeBLEUResult(
                codebleu=result["codebleu"],
                bleu=result.get("ngram_match_score", 0.0),
                bleu_weight=result.get("weighted_ngram_match_score", 0.0),
                match_ast=result.get("syntax_match_score"),
                match_df=result.get("dataflow_match_score"),
                used_official_package=True,
            )
    except ImportError:
        pass

    bleu = _bleu4(candidate, reference)
    bleu_w = _bleu_weighted(candidate, reference, language)
    m_ast = ast_match_score(candidate, reference, language)
    m_df = dataflow_match_score(candidate, reference, language)

    if m_ast is None or m_df is None:
        # Renormalize onto just the two computable terms rather than
        # treating the missing terms as 0 (which would unfairly punish
        # languages we simply can't structurally parse without tree-sitter).
        total = weights.alpha + weights.beta
        a = weights.alpha / total if total > 0 else 0.5
        b = weights.beta / total if total > 0 else 0.5
        combined = a * bleu + b * bleu_w
    else:
        combined = (weights.alpha * bleu + weights.beta * bleu_w
                    + weights.gamma * m_ast + weights.delta * m_df)

    return CodeBLEUResult(
        codebleu=combined, bleu=bleu, bleu_weight=bleu_w,
        match_ast=m_ast, match_df=m_df, used_official_package=False,
    )


# ---------------------------------------------------------------------------
# 4. CSR (Compilation/Syntax Success Rate) -- DIAGNOSTIC ONLY, not a score
# ---------------------------------------------------------------------------
#
# Neither [Chen21] nor [Ren20] defines a "Compilation Success Rate" metric.
# This is a cheap, project-defined sanity check: does the candidate even
# parse? It is deliberately weak evidence (a no-op patch is syntactically
# valid too) and largely redundant with CodeBLEU besides -- an unparseable
# candidate already scores 0 on Match_ast / Match_df. It is reported as a
# standalone diagnostic stat (e.g. "8% of my model's outputs are malformed
# Python") and is NEVER combined with pass@k or CodeBLEU into any weighted
# score. Do not use it to rank checkpoints.
#
# [Chen21]'s actual functional-correctness signal is running real unit tests
# in a sandbox (Sec 2.3) -- that is NOT what this does. If you have a test
# harness, use that for correctness and treat CSR purely as a parse-rate
# debugging stat alongside it.

def syntax_is_valid(code: str, language: str) -> bool:
    """Cheap syntax-validity proxy. See module note above -- not from either paper."""
    language = (language or "").lower()
    if language == "python":
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    # Generic brace/bracket/paren balance check for curly-brace languages.
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


# ---------------------------------------------------------------------------
# 5. Evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    id: str
    n: int
    c: int
    pass_at_k: float
    codebleu: float
    codebleu_detail: List[Dict[str, Any]] = field(default_factory=list)
    csr: float = 0.0  # diagnostic only -- see CSR section above


def evaluate_sample(record: Dict[str, Any], n: int, k: int,
                     codebleu_weights: CodeBLEUWeights) -> SampleResult:
    candidates = generate_fixes(record["input_code"], record["context"], record["language"], n=n)
    n_gen = len(candidates)

    c_correct = sum(is_correct(c, record["reference_code"]) for c in candidates)
    p_at_k = pass_at_k(n_gen, c_correct, k)

    cb_results = [codebleu_score(c, record["reference_code"], record["language"], codebleu_weights)
                  for c in candidates]
    avg_codebleu = sum(r.codebleu for r in cb_results) / len(cb_results) if cb_results else 0.0
    cb_detail = [r.__dict__ for r in cb_results]

    # Diagnostic only -- not used in any ranking/scoring below.
    csr_scores = [syntax_is_valid(c, record["language"]) for c in candidates]
    csr = sum(csr_scores) / len(csr_scores) if csr_scores else 0.0

    return SampleResult(id=record["id"], n=n_gen, c=c_correct, pass_at_k=p_at_k,
                         codebleu=avg_codebleu, codebleu_detail=cb_detail, csr=csr)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a CI/CD fix-generation model on diff-xyz using "
                     "pass@k [Chen21], CodeBLEU [Ren20], and a CSR sanity check."
    )
    parser.add_argument("--data", default="data/diff_xyz_eval.jsonl")
    parser.add_argument("--n", type=int, default=5, help="Number of candidate fixes to generate per sample")
    parser.add_argument("--k", type=int, default=1, help="k for pass@k (k <= n)")
    parser.add_argument("--limit", type=int, default=0, help="0 = evaluate all samples")
    parser.add_argument("--report", default="data/evaluation_report.json")
    parser.add_argument("--codebleu-weights", type=float, nargs=4,
                         metavar=("ALPHA", "BETA", "GAMMA", "DELTA"), default=None,
                         help="Override CodeBLEU alpha/beta/gamma/delta "
                              "(default: paper's combination [7], 0.1/0.1/0.4/0.4)")
    args = parser.parse_args()

    cb_weights = (CodeBLEUWeights(*args.codebleu_weights) if args.codebleu_weights
                  else CodeBLEUWeights())

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
            res = evaluate_sample(rec, n=args.n, k=args.k, codebleu_weights=cb_weights)
        except NotImplementedError as e:
            print(str(e))
            return
        results.append(res)
        print(f"[{i}/{len(records)}] {res.id} | pass@{args.k}={res.pass_at_k:.3f} "
              f"CodeBLEU={res.codebleu:.3f} CSR(diagnostic)={res.csr:.3f}")

    if not results:
        print("No results to report.")
        return

    agg = {
        "n_samples": len(results),
        f"avg_pass@{args.k}": sum(r.pass_at_k for r in results) / len(results),
        "avg_codebleu": sum(r.codebleu for r in results) / len(results),
        "avg_csr_diagnostic_only": sum(r.csr for r in results) / len(results),
        "codebleu_weights": cb_weights.__dict__,
        "metric_provenance": {
            "pass@k": "Chen et al. 2021 (arXiv:2107.03374) Eq. 1 -- "
                      "NOTE: c = exact text match to reference, not test-pass "
                      "(diff-xyz has no test harness), unlike the paper's usage.",
            "codebleu": "Ren et al. 2020 (arXiv:2009.10297) Eq. 1. Uses official "
                        "`codebleu` package if installed, else an in-file "
                        "reimplementation of all 4 terms (Python only for "
                        "AST/data-flow terms).",
            "csr": "Project-defined syntax-validity proxy. NOT from either paper, "
                   "NOT a score -- diagnostic only (parse rate), not used for ranking.",
        },
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_sample": [r.__dict__ for r in results]}, f, indent=2)

    print("\n=== Aggregate results ===")
    for key, val in agg.items():
        if key == "metric_provenance":
            continue
        print(f"{key}: {val}")
    print(f"\nFull report saved to {args.report}")


if __name__ == "__main__":
    main()
