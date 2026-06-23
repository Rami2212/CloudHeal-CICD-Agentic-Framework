# ============================================================
#  CI/CD Fix-Generation Evaluation Pipeline
#  Paste each cell into a Google Colab notebook in order.
# ============================================================


# ── CELL 1: Install dependencies ────────────────────────────
# !pip install -q transformers peft bitsandbytes accelerate datasets difflib codebleu
# Optional but recommended for all 4 CodeBLEU terms:
# !pip install -q tree_sitter tree_sitter_languages


# ── CELL 2: Imports ─────────────────────────────────────────
import os, re, ast, json, math, textwrap, inspect, difflib, subprocess, tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, LoraConfig
from datasets import load_dataset


# ── CELL 3: Config ──────────────────────────────────────────
@dataclass
class Config:
    # Model
    base_model:     str  = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path:   str  = "./adapter"
    cache_dir:      Optional[str] = None
    torch_dtype:    str  = "float16"
    load_in_4bit:   bool = True
    offload_folder: str  = "/tmp/offload"
    # Dataset
    dataset_name:   str  = "JetBrains-Research/diff-xyz"
    dataset_split:  str  = "test"
    filter_lang:    Optional[str] = "python"   # None = all languages
    max_samples:    int  = 50                  # 0 = use all rows
    diff_format:    str  = "udiff"
    # Generation
    max_new_tokens: int  = 512
    # Evaluation
    pass_k:         int  = 1
    n_candidates:   int  = 1   # samples per prompt (>1 needs do_sample=True)
    save_report:    str  = "eval_report.json"


CFG = Config()


# ── CELL 4: Model utilities ─────────────────────────────────
def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(name.lower(), torch.float16)

def _get_device(model) -> torch.device:
    try:    return next(model.parameters()).device
    except: return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

def _free_vram() -> int:
    if not torch.cuda.is_available(): return 0
    free, _ = torch.cuda.mem_get_info(0)
    return free

def _patch_adapter_config(path: str):
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path): return
    with open(cfg_path) as f: cfg = json.load(f)
    valid = set(inspect.signature(LoraConfig.__init__).parameters) - {"self"}
    bad   = [k for k in cfg if k not in valid]
    if bad:
        print(f"[adapter] Removing unknown keys: {bad}")
        with open(cfg_path + ".bak", "w") as f: json.dump(cfg, f, indent=2)
        with open(cfg_path, "w") as f: json.dump({k:v for k,v in cfg.items() if k in valid}, f, indent=2)

def _clean_gen_config(model):
    try:    pad, eos, bos = model.generation_config.pad_token_id, model.generation_config.eos_token_id, model.generation_config.bos_token_id
    except: pad = eos = bos = None
    model.generation_config = GenerationConfig(
        do_sample=False, repetition_penalty=1.1,
        pad_token_id=pad, eos_token_id=eos, bos_token_id=bos)

def _build_bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

def load_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True, cache_dir=cfg.cache_dir)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def load_base_model(cfg: Config):
    os.makedirs(cfg.offload_folder, exist_ok=True); _free_gpu()
    free = _get_free_vram(); reserve = int(3.0 * 1024**3)
    usable = max(0, free - reserve)
    # Rough size estimate: 7B params × 0.5 bytes (4-bit) ≈ 3.5 GB + overhead
    fit_on_gpu = usable >= int(4.5 * 1024**3)

    if cfg.load_in_4bit and fit_on_gpu:
        print("[load] Strategy: 4-bit NF4 fully on GPU")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, device_map={"": 0}, cache_dir=cfg.cache_dir,
            low_cpu_mem_usage=True, quantization_config=_build_bnb())
    else:
        print("[load] Strategy: fp16 with auto device_map")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, device_map="auto",
            torch_dtype=_resolve_dtype(cfg.torch_dtype),
            cache_dir=cfg.cache_dir, low_cpu_mem_usage=True,
            max_memory={0: int(free * 0.85), "cpu": "48GiB"},
            offload_folder=cfg.offload_folder)

    model.eval(); _clean_gen_config(model); _free_gpu()
    return model

def load_peft_model(base, cfg: Config):
    _patch_adapter_config(cfg.adapter_path); _free_gpu()
    ft = PeftModel.from_pretrained(base, cfg.adapter_path, is_trainable=False)
    ft.eval(); _clean_gen_config(ft); _free_gpu()
    return ft


# ── CELL 5: Dataset preparation ─────────────────────────────
DIFF_FIELDS = {"udiff", "udiff-h", "udiff-l", "search-replace"}

def build_record(ex: dict, fmt: str) -> dict:
    return {
        "id":             f"{ex.get('repo')}::{str(ex.get('commit',''))[:10]}::{ex.get('path')}",
        "language":       ex.get("lang"),
        "context":        (ex.get("message") or "").strip(),
        "input_code":     ex.get("old_code", ""),
        "reference_code": ex.get("new_code", ""),
        "reference_diff": ex.get(fmt, ""),
    }

def load_eval_dataset(cfg: Config) -> List[dict]:
    print(f"Loading {cfg.dataset_name} split={cfg.dataset_split} ...")
    ds = load_dataset(cfg.dataset_name, "default", split=cfg.dataset_split)
    if cfg.filter_lang:
        ds = ds.filter(lambda x: x.get("lang") == cfg.filter_lang)
    if cfg.max_samples > 0:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    records = [build_record(ex, cfg.diff_format) for ex in ds]
    records = [r for r in records if r["input_code"] and r["reference_code"]]
    print(f"Prepared {len(records)} records.")
    return records


# ── CELL 6: Generation ──────────────────────────────────────
def _build_prompt(input_code: str, context: str) -> str:
    return (
        "You are an expert software engineer specialising in CI/CD pipeline fixes.\n\n"
        f"Task:\n{context}\n\n"
        f"Source Code:\n```python\n{input_code}\n```\n\n"
        "Return ONLY the corrected Python code inside a ```python block. "
        "No explanation, no commentary."
    )

def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return "\n".join(l for l in text.strip().splitlines() if l.strip())

def generate_fix(model, tokenizer, input_code: str, context: str, cfg: Config) -> str:
    prompt  = _build_prompt(input_code, context)
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    device  = _get_device(model)
    inputs  = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=cfg.max_new_tokens,
            do_sample=False, repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _extract_code(raw)


# ── CELL 7: Metrics ─────────────────────────────────────────

# — pass@k (Chen et al. 2021, arXiv:2107.03374) —
def normalize_code(code: str) -> str:
    lines = [l.rstrip() for l in code.strip().splitlines()]
    return "\n".join(l for l in lines if l.strip())

def is_correct(cand: str, ref: str) -> bool:
    return normalize_code(cand) == normalize_code(ref)

def pass_at_k(n: int, c: int, k: int) -> float:
    if n == 0: return 0.0
    k = min(k, n)
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

# — Syntax validity (diagnostic only) —
def syntax_valid(code: str, lang: str = "python") -> bool:
    if (lang or "").lower() == "python":
        try: ast.parse(code); return True
        except SyntaxError: return False
    pairs = {"(":")", "[":"]", "{":"}"}; closing = set(pairs.values()); stack = []
    for ch in code:
        if ch in pairs: stack.append(pairs[ch])
        elif ch in closing:
            if not stack or stack.pop() != ch: return False
    return not stack

# — CodeBLEU (Ren et al. 2020, arXiv:2009.10297) —
PY_KEYWORDS = {
    "False","None","True","and","as","assert","async","await","break","class",
    "continue","def","del","elif","else","except","finally","for","from",
    "global","if","import","in","is","lambda","nonlocal","not","or","pass",
    "raise","return","try","while","with","yield",
}

def _tokenize(code: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", code)

def _ngram_prec(cand: List[str], ref: List[str], n: int, wmap: Optional[dict] = None) -> float:
    cg = [tuple(cand[i:i+n]) for i in range(len(cand)-n+1)]
    if not cg: return 0.0
    rg = [tuple(ref[i:i+n]) for i in range(len(ref)-n+1)]
    rc = Counter(rg); cc = Counter(cg)
    w  = lambda g: (wmap or {}).get(g[0], 1.0) if (wmap and n==1) else 1.0
    num = sum(w(g)*min(c, rc.get(g,0)) for g,c in cc.items())
    den = sum(w(g)*c for g,c in cc.items())
    return num/den if den else 0.0

def _bp(cand: List[str], ref: List[str]) -> float:
    c,r = len(cand),len(ref)
    return 1.0 if c>=r else (math.exp(1-r/c) if c else 0.0)

def _bleu4(cand: str, ref: str) -> float:
    ct,rt = _tokenize(cand),_tokenize(ref)
    if not ct or not rt: return 0.0
    ps = [max(_ngram_prec(ct,rt,n), 1e-9) for n in range(1,5)]
    return math.exp(sum(math.log(p) for p in ps)/4) * _bp(ct,rt)

def _bleu_w(cand: str, ref: str) -> float:
    ct,rt = _tokenize(cand),_tokenize(ref)
    if not ct or not rt: return 0.0
    wm = {kw:5.0 for kw in PY_KEYWORDS}
    p1 = max(_ngram_prec(ct,rt,1,wm), 1e-9)
    return p1 * _bp(ct,rt)

def _ast_subtrees(code: str) -> Optional[List[tuple]]:
    try: tree = ast.parse(code)
    except SyntaxError: return None
    subs = []
    def sig(n):
        ch = []
        for _,v in ast.iter_fields(n):
            if isinstance(v,list):
                for x in v:
                    if isinstance(x,ast.AST): ch.append(sig(x))
            elif isinstance(v,ast.AST): ch.append(sig(v))
        return (type(n).__name__, tuple(ch))
    def walk(n):
        subs.append(sig(n))
        for c in ast.iter_child_nodes(n): walk(c)
    walk(tree); return subs

def _flatten(subs: List[tuple]) -> Counter:
    cnt: Counter = Counter()
    def ex(s):
        cnt[s] += 1
        for c in s[1]: ex(c)
    for s in subs: ex(s)
    return cnt

def _ast_match(cand: str, ref: str) -> Optional[float]:
    rs,cs = _ast_subtrees(ref),_ast_subtrees(cand)
    if rs is None or cs is None or not rs: return 0.0
    rc,cc = _flatten(rs),_flatten(cs)
    tot = sum(rc.values())
    return sum(min(c,cc.get(s,0)) for s,c in rc.items())/tot if tot else 0.0

def _df_edges(code: str) -> Optional[List[Tuple[str,str]]]:
    try: tree = ast.parse(code)
    except SyntaxError: return None
    edges: List[Tuple[str,str]] = []
    ns = lambda n: [x.id for x in ast.walk(n) if isinstance(x,ast.Name)]
    class V(ast.NodeVisitor):
        def visit_Assign(self,n):
            for t in n.targets:
                for tn in ns(t):
                    for s in ns(n.value):
                        if s!=tn: edges.append((s,tn))
            self.generic_visit(n)
        def visit_Return(self,n):
            if n.value:
                for s in ns(n.value): edges.append((s,"<return>"))
            self.generic_visit(n)
    V().visit(tree); return edges

def _norm_df(edges: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    m:dict = {}; ctr = 0; out = []
    for s,t in edges:
        for v in (s,t):
            if v!="<return>" and v not in m:
                m[v]=f"var_{ctr}"; ctr+=1
        out.append((m.get(s,s), m.get(t,t)))
    return out

def _df_match(cand: str, ref: str) -> Optional[float]:
    re_= _df_edges(ref); ce = _df_edges(cand)
    if re_ is None or ce is None or not re_: return 0.0
    rc,cc = Counter(_norm_df(re_)),Counter(_norm_df(ce))
    tot = sum(rc.values())
    return sum(min(c,cc.get(e,0)) for e,c in rc.items())/tot if tot else 0.0

def codebleu(cand: str, ref: str, lang: str = "python",
             alpha=0.10, beta=0.10, gamma=0.40, delta=0.40) -> float:
    """CodeBLEU (Ren et al. 2020). Weights: paper's recommended combo [7]."""
    try:
        from codebleu import calc_codebleu
        r = calc_codebleu([ref],[cand],lang=lang,weights=(alpha,beta,gamma,delta))
        return r["codebleu"]
    except ImportError:
        pass
    b  = _bleu4(cand,ref)
    bw = _bleu_w(cand,ref)
    am = _ast_match(cand,ref)
    dm = _df_match(cand,ref)
    if am is None or dm is None:
        a2,b2 = alpha/(alpha+beta), beta/(alpha+beta)
        return a2*b + b2*bw
    return alpha*b + beta*bw + gamma*am + delta*dm


# ── CELL 8: Main evaluation loop ────────────────────────────
def run_evaluation(cfg: Config = CFG):
    print("=" * 60)
    print("  Loading tokenizer & model ...")
    tokenizer = load_tokenizer(cfg)
    base      = load_base_model(cfg)
    ft_model  = load_peft_model(base, cfg)

    print("\n  Loading dataset ...")
    records = load_eval_dataset(cfg)

    results = []
    base_wins = ft_wins = 0

    for i, rec in enumerate(records, 1):
        print(f"\n[{i}/{len(records)}] {rec['id']}")

        # Base model (adapter disabled)
        with ft_model.disable_adapter():
            base_out = generate_fix(ft_model, tokenizer, rec["input_code"], rec["context"], cfg)

        # Fine-tuned model (adapter active)
        ft_out = generate_fix(ft_model, tokenizer, rec["input_code"], rec["context"], cfg)

        ref  = rec["reference_code"]
        lang = rec["language"] or "python"

        # pass@k (exact match proxy — diff-xyz has no test harness)
        base_correct = int(is_correct(base_out, ref))
        ft_correct   = int(is_correct(ft_out,   ref))
        base_pak = pass_at_k(1, base_correct, cfg.pass_k)
        ft_pak   = pass_at_k(1, ft_correct,   cfg.pass_k)

        # CodeBLEU
        base_cb = codebleu(base_out, ref, lang)
        ft_cb   = codebleu(ft_out,   ref, lang)

        # Syntax (diagnostic only)
        base_syn = syntax_valid(base_out, lang)
        ft_syn   = syntax_valid(ft_out,   lang)

        # Winner by CodeBLEU (more nuanced than exact match)
        winner = "finetuned" if ft_cb > base_cb else "base"
        if winner == "finetuned": ft_wins   += 1
        else:                     base_wins += 1

        print(f"  pass@{cfg.pass_k}  base={base_pak:.3f}  ft={ft_pak:.3f}")
        print(f"  CodeBLEU base={base_cb:.3f}  ft={ft_cb:.3f}")
        print(f"  Syntax   base={base_syn}      ft={ft_syn}")
        print(f"  Winner   : {winner.upper()}")

        results.append({
            "id": rec["id"], "language": lang, "winner": winner,
            "base_pass_at_k": base_pak,   "ft_pass_at_k": ft_pak,
            "base_codebleu":  base_cb,    "ft_codebleu":  ft_cb,
            "base_syntax":    base_syn,   "ft_syntax":    ft_syn,
            "base_output":    base_out,   "ft_output":    ft_out,
            "reference":      ref,
        })

    # Aggregate
    n   = len(results)
    avg = lambda k: sum(r[k] for r in results) / n if n else 0.0

    agg = {
        "n_samples":        n,
        "base_wins":        base_wins,
        "ft_wins":          ft_wins,
        f"avg_pass@{cfg.pass_k}_base": avg("base_pass_at_k"),
        f"avg_pass@{cfg.pass_k}_ft":   avg("ft_pass_at_k"),
        "avg_codebleu_base": avg("base_codebleu"),
        "avg_codebleu_ft":   avg("ft_codebleu"),
        "avg_syntax_base":   avg("base_syntax"),
        "avg_syntax_ft":     avg("ft_syntax"),
        "metric_notes": {
            "pass@k":    "Chen et al. 2021 (arXiv:2107.03374). c = exact-match to reference (no test harness in diff-xyz).",
            "codebleu":  "Ren et al. 2020 (arXiv:2009.10297). Weights [alpha,beta,gamma,delta]=[0.1,0.1,0.4,0.4] (paper combo [7]).",
            "syntax":    "Diagnostic only. Not from either paper. Not used for ranking.",
        },
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for k, v in agg.items():
        if k != "metric_notes": print(f"  {k}: {v}")
    verdict = ("FINETUNED is better" if ft_wins > base_wins
               else "BASE is better — more data/epochs may help"
               if base_wins > ft_wins else "DRAW")
    print(f"\n  VERDICT: {verdict}")

    with open(cfg.save_report, "w") as f:
        json.dump({"aggregate": agg, "per_sample": results}, f, indent=2)
    print(f"\n  Report saved -> {cfg.save_report}")
    return results


# ── CELL 9: Run ─────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation(CFG)