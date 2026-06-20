from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import json
import difflib
import subprocess
import tempfile
import textwrap
import inspect
import os
import ast
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, LoraConfig


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class ModelLoadConfig:
    base_model: str
    adapter_path: Optional[str] = "adapter"
    cache_dir: Optional[str] = None
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    offload_folder: str = "/tmp/offload"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }.get(dtype_name.lower(), torch.float16)


def _get_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patch_adapter_config(adapter_path: str):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        config = json.load(f)
    valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys()) - {"self"}
    unknown = [k for k in config if k not in valid_keys]
    if unknown:
        print(f"[adapter patch] Removing unknown keys: {unknown}")
        clean = {k: v for k, v in config.items() if k in valid_keys}
        with open(config_path + ".bak", "w") as f:
            json.dump(config, f, indent=2)
        with open(config_path, "w") as f:
            json.dump(clean, f, indent=2)
    else:
        print("[adapter patch] adapter_config.json OK.")


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    print(f"  [WARN] No ```python block found. Raw (first 120 chars): {repr(text[:120])}")
    return "\n".join(l for l in text.strip().splitlines() if l.strip())


def _free_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _get_free_vram() -> int:
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info(0)
    return free


def _build_device_map(config: ModelLoadConfig) -> dict:
    """
    Place as many transformer layers as possible on GPU 0.
    Uses actual free VRAM at call time (not device total) and reserves
    3 GB for CUDA context + loading spikes + KV-cache headroom.

    NOTE: with load_in_4bit=True we must NOT offload any layers to CPU
    because bitsandbytes cannot serialize quantized tensors that are still
    on the meta device during PeftModel attachment.  If not enough VRAM is
    available for all layers we fall back to device_map="auto" without
    quantization rather than mixing GPU/CPU with 4-bit.
    """
    if not torch.cuda.is_available():
        return {"": "cpu"}

    free_vram   = _get_free_vram()
    usable_vram = max(0, free_vram - int(3.0 * 1024 ** 3))

    bpp         = 0.5 if config.load_in_4bit else 2.0
    embed_bytes = int(0.15e9 * bpp)
    layer_bytes = int(0.24e9 * bpp)
    norm_bytes  = int(0.01e9 * bpp)
    lmhead_bytes= int(0.50e9 * 2.0)   # lm_head stays fp16 even in 4-bit mode
    num_layers  = 28

    total_needed = (embed_bytes + num_layers * layer_bytes
                    + norm_bytes + lmhead_bytes)

    print(f"[device_map] free VRAM: {free_vram/1024**3:.1f} GB  "
          f"usable: {usable_vram/1024**3:.1f} GB  "
          f"model needs: {total_needed/1024**3:.1f} GB")

    if usable_vram >= total_needed:
        # Everything fits on GPU — safe to use 4-bit with no CPU offload
        print("[device_map] All layers fit on GPU.")
        return {"": 0}

    # Not enough VRAM for full 4-bit on GPU.
    # Mixing GPU/CPU with bitsandbytes causes the meta-tensor NotImplementedError,
    # so signal the caller to disable quantization and use auto device map instead.
    print("[device_map] Insufficient VRAM for full 4-bit GPU load — "
          "will fall back to fp16 auto device_map.")
    return {}   # empty dict = caller should use fallback path


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # Do NOT set llm_int8_enable_fp32_cpu_offload=True here —
        # that flag is what allows CPU offload, which triggers the meta-tensor bug.
    )


def _clean_generation_config(model):
    """
    Qwen2.5 ships with generation_config.json that sets temperature/top_p/top_k
    alongside do_sample=False. This causes warnings and can degrade output quality.
    Replace with an explicit greedy config that has no conflicting parameters.
    """
    try:
        pad_id = model.generation_config.pad_token_id
        eos_id = model.generation_config.eos_token_id
        bos_id = model.generation_config.bos_token_id
    except Exception:
        pad_id = eos_id = bos_id = None

    model.generation_config = GenerationConfig(
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        bos_token_id=bos_id,
    )
    print("[generation_config] Set to clean greedy decoding.")


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_tokenizer(config: ModelLoadConfig) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        config.base_model, use_fast=True, cache_dir=config.cache_dir,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(config: ModelLoadConfig) -> AutoModelForCausalLM:
    """
    Load the base model. Strategy:

    1. Try full 4-bit on GPU (device_map={"": 0}) — fastest, no CPU offload.
    2. If VRAM is insufficient, fall back to fp16 with device_map="auto"
       (which may split across GPU+CPU but is safe because no bnb quantization).

    We never mix 4-bit quantization with CPU offload because bitsandbytes
    cannot serialize quantized meta tensors, causing NotImplementedError
    when PeftModel tries to attach the adapter.
    """
    os.makedirs(config.offload_folder, exist_ok=True)
    _free_gpu_cache()

    device_map = _build_device_map(config)
    use_4bit   = config.load_in_4bit and device_map == {"": 0}

    if use_4bit:
        # Path 1: everything on GPU in 4-bit — no CPU offload, no meta tensors
        print("[load_base] Strategy: 4-bit NF4 on GPU (no CPU offload)")
        kwargs: dict = dict(
            device_map={"": 0},
            cache_dir=config.cache_dir,
            low_cpu_mem_usage=True,
            quantization_config=_build_bnb_config(),
        )
    else:
        # Path 2: fp16 with auto device map — safe for CPU offload
        print("[load_base] Strategy: fp16 auto device_map (VRAM insufficient for full 4-bit)")
        free_vram = _get_free_vram()
        max_memory = {0: int(free_vram * 0.85), "cpu": "48GiB"}
        kwargs = dict(
            device_map="auto",
            torch_dtype=_resolve_dtype(config.torch_dtype),
            cache_dir=config.cache_dir,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            offload_folder=config.offload_folder,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    except torch.cuda.OutOfMemoryError:
        if use_4bit:
            print("[load_base] OOM on 4-bit GPU load — retrying with fp16 auto device_map")
            _free_gpu_cache()
            free_vram = _get_free_vram()
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                device_map="auto",
                torch_dtype=_resolve_dtype(config.torch_dtype),
                cache_dir=config.cache_dir,
                low_cpu_mem_usage=True,
                max_memory={0: int(free_vram * 0.85), "cpu": "48GiB"},
                offload_folder=config.offload_folder,
            )
        else:
            raise

    model.eval()
    _clean_generation_config(model)
    _free_gpu_cache()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"[load_base] GPU allocated after base load: {alloc:.2f} GB")

    return model


def load_peft_model(
    base_model: AutoModelForCausalLM,
    config: ModelLoadConfig,
) -> PeftModel:
    """
    Attach the LoRA adapter.

    Key constraint: do NOT pass offload_folder or offload_state_dict when the
    base model has 4-bit quantized layers — bitsandbytes cannot serialize
    quantized tensors on the meta device, raising NotImplementedError.

    Since load_base_model() only uses 4-bit when all layers are on GPU
    (device_map={"": 0}), there are no meta-device tensors at this point
    and we can safely attach the adapter without offloading.
    """
    _patch_adapter_config(config.adapter_path)
    _free_gpu_cache()

    ft_model = PeftModel.from_pretrained(
        base_model,
        config.adapter_path,
        is_trainable=False,
        # No offload_folder / offload_state_dict — avoids the meta-tensor
        # serialization path in bitsandbytes that causes NotImplementedError.
    )
    ft_model.eval()
    _clean_generation_config(ft_model)
    _free_gpu_cache()
    return ft_model


# ----------------------------------------------------------------------
# Scoring utilities
# ----------------------------------------------------------------------

def similarity_score(generated: str, expected: str) -> float:
    return difflib.SequenceMatcher(
        None, generated.splitlines(), expected.splitlines()
    ).ratio()


def syntax_valid(code: str) -> bool:
    try:
        compile(textwrap.dedent(code), "<string>", "exec")
        return True
    except SyntaxError:
        return False


def ast_node_similarity(generated: str, expected: str) -> float:
    def node_types(code):
        try:
            return [type(n).__name__ for n in ast.walk(ast.parse(textwrap.dedent(code)))]
        except SyntaxError:
            return []
    g, e = node_types(generated), node_types(expected)
    if not g or not e:
        return 0.0
    return difflib.SequenceMatcher(None, g, e).ratio()


def parse_pytest_output(stdout: str) -> Tuple[int, int]:
    passed = int(m.group(1)) if (m := re.search(r"(\d+)\s+passed", stdout)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+)\s+failed", stdout)) else 0
    return passed, failed


def run_tests(code: str, test_code: str) -> Tuple[float, str]:
    combined = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(combined)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["pytest", tmp_path, "--tb=short", "-q"],
            capture_output=True, text=True, timeout=30,
        )
        passed, failed = parse_pytest_output(result.stdout)
        total = passed + failed
        return (passed / total if total > 0 else 0.0), result.stdout
    except Exception as e:
        return 0.0, f"[test runner error] {e}"
    finally:
        os.unlink(tmp_path)


def composite_score(
    line_sim: float,
    ast_sim: float,
    syntax_ok: bool,
    test_rate: Optional[float],
) -> float:
    syn = 1.0 if syntax_ok else 0.0
    if test_rate is not None:
        return test_rate * 0.45 + ast_sim * 0.25 + line_sim * 0.15 + syn * 0.15
    return ast_sim * 0.40 + line_sim * 0.30 + syn * 0.30


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------

class CodingAgentEvaluator:
    """
    Compares a base model against its LoRA fine-tuned counterpart on
    CI/CD code-refactoring tasks.

    Architecture
    ------------
    Only one copy of the 7B weights is held in memory at any time.
    The PeftModel wraps the base weights:

      - Base inference:      ft_model.disable_adapter() context manager
                             temporarily removes LoRA deltas → pure base output
      - Finetuned inference: ft_model.generate() with adapter active

    This avoids the NotImplementedError from bitsandbytes trying to serialize
    quantized meta-device tensors during a second model load.

    Metrics
    -------
    - Line-level diff similarity   (always)
    - AST structural similarity    (always)
    - Syntax validity              (always)
    - pytest pass rate             (when test_code is provided)
    - CI fix rate                  (% of cases where tests go broken → passing)
    """

    def __init__(self, config: ModelLoadConfig):
        self.config = config

        print("Loading tokenizer...")
        self.tokenizer = load_tokenizer(config)

        print("Loading base model...")
        self._raw_model = load_base_model(config)

        print("Attaching LoRA adapter...")
        self.ft_model = load_peft_model(self._raw_model, config)

        print("Both models ready.\n")
        if torch.cuda.is_available():
            alloc    = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved()  / 1024**3
            print(f"[GPU] allocated: {alloc:.2f} GB  reserved: {reserved:.2f} GB\n")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        instruction: str,
        source_code: str,
        ci_error: Optional[str],
    ) -> str:
        error_section = f"\nCI/CD Failure Log:\n{ci_error}\n" if ci_error else ""
        return (
            "You are an expert software engineer specialising in CI/CD pipeline fixes.\n\n"
            f"Task:\n{instruction}\n"
            f"{error_section}\n"
            f"Source Code:\n```python\n{source_code}\n```\n\n"
            "Return ONLY the corrected Python code inside a ```python block. "
            "No explanation, no commentary."
        )

    def _run_generate(self, model, prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        device = _get_device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        return _extract_code_block(raw)

    def generate_base(
        self,
        instruction: str,
        source_code: str,
        ci_error: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """Run generation with adapter DISABLED → pure base model output."""
        prompt = self._build_prompt(instruction, source_code, ci_error)
        with self.ft_model.disable_adapter():
            return self._run_generate(self.ft_model, prompt, max_tokens)

    def generate_finetuned(
        self,
        instruction: str,
        source_code: str,
        ci_error: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """Run generation with adapter ACTIVE → finetuned model output."""
        prompt = self._build_prompt(instruction, source_code, ci_error)
        return self._run_generate(self.ft_model, prompt, max_tokens)

    # ------------------------------------------------------------------
    # Per-case evaluation
    # ------------------------------------------------------------------

    def evaluate_case(self, case: Dict) -> Dict:
        print("=" * 80)
        print(f"TASK : {case['instruction']}")

        ci_error  = case.get("ci_error")
        test_code = case.get("test_code")
        expected  = case["expected"]

        base_out = self.generate_base(
            case["instruction"], case["source_code"], ci_error
        )
        ft_out = self.generate_finetuned(
            case["instruction"], case["source_code"], ci_error
        )

        print(f"  [base out]      {repr(base_out[:80])}")
        print(f"  [finetuned out] {repr(ft_out[:80])}")

        base_line = similarity_score(base_out, expected)
        ft_line   = similarity_score(ft_out,   expected)
        base_ast  = ast_node_similarity(base_out, expected)
        ft_ast    = ast_node_similarity(ft_out,   expected)
        base_syn  = syntax_valid(base_out)
        ft_syn    = syntax_valid(ft_out)

        base_tests = ft_tests = None
        if test_code:
            base_tests, _ = run_tests(base_out, test_code)
            ft_tests,   _ = run_tests(ft_out,   test_code)

        base_score = composite_score(base_line, base_ast, base_syn, base_tests)
        ft_score   = composite_score(ft_line,   ft_ast,   ft_syn,   ft_tests)
        winner     = "finetuned" if ft_score > base_score else "base"

        ci_fixed_base = bool(base_tests and base_tests > 0.0)
        ci_fixed_ft   = bool(ft_tests   and ft_tests   > 0.0)

        print(f"  Line sim    — base: {base_line:.4f}  |  finetuned: {ft_line:.4f}")
        print(f"  AST sim     — base: {base_ast:.4f}   |  finetuned: {ft_ast:.4f}")
        print(f"  Syntax OK   — base: {base_syn}       |  finetuned: {ft_syn}")
        if test_code:
            print(f"  Test pass   — base: {base_tests:.2%}  |  finetuned: {ft_tests:.2%}")
            print(f"  CI fixed    — base: {ci_fixed_base}   |  finetuned: {ci_fixed_ft}")
        print(f"  Composite   — base: {base_score:.4f}  |  finetuned: {ft_score:.4f}")
        print(f"  Winner      : {winner.upper()}\n")

        return {
            "instruction":    case["instruction"],
            "winner":         winner,
            "base_line_sim":  base_line,     "ft_line_sim":  ft_line,
            "base_ast_sim":   base_ast,      "ft_ast_sim":   ft_ast,
            "base_syntax":    base_syn,      "ft_syntax":    ft_syn,
            "base_tests":     base_tests,    "ft_tests":     ft_tests,
            "ci_fixed_base":  ci_fixed_base, "ci_fixed_ft":  ci_fixed_ft,
            "base_composite": base_score,    "ft_composite": ft_score,
        }

    # ------------------------------------------------------------------
    # Full evaluation loop
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_cases: List[Dict],
        save_results: Optional[str] = None,
    ) -> List[Dict]:
        results   = []
        base_wins = ft_wins = 0
        ci_base_fixed = ci_ft_fixed = ci_total = 0

        for case in test_cases:
            r = self.evaluate_case(case)
            results.append(r)
            if r["winner"] == "finetuned":
                ft_wins += 1
            else:
                base_wins += 1
            if r["base_tests"] is not None:
                ci_total += 1
                if r["ci_fixed_base"]: ci_base_fixed += 1
                if r["ci_fixed_ft"]:   ci_ft_fixed   += 1

        n   = len(results)
        avg = lambda k: sum(r[k] for r in results) / n

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"  Cases evaluated : {n}")
        print(f"  Base wins       : {base_wins}")
        print(f"  Finetuned wins  : {ft_wins}")
        print(f"  Avg line sim    — base: {avg('base_line_sim'):.4f}  |  "
              f"finetuned: {avg('ft_line_sim'):.4f}")
        print(f"  Avg AST sim     — base: {avg('base_ast_sim'):.4f}   |  "
              f"finetuned: {avg('ft_ast_sim'):.4f}")
        print(f"  Avg composite   — base: {avg('base_composite'):.4f}  |  "
              f"finetuned: {avg('ft_composite'):.4f}")

        if ci_total > 0:
            print(f"\n  CI Fix Rate     — base: {ci_base_fixed}/{ci_total} "
                  f"({ci_base_fixed/ci_total:.0%})  |  "
                  f"finetuned: {ci_ft_fixed}/{ci_total} "
                  f"({ci_ft_fixed/ci_total:.0%})")

        print()
        if ft_wins > base_wins:
            print("  VERDICT: Fine-tuned model is BETTER for CI/CD refactoring")
        elif base_wins > ft_wins:
            print("  VERDICT: Base model is BETTER — fine-tuning may need more data/epochs")
        else:
            print("  VERDICT: DRAW")

        if save_results:
            with open(save_results, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results saved -> {save_results}")

        return results


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":

    config = ModelLoadConfig(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path="./adapter",
        load_in_4bit=True,
        torch_dtype="float16",
        offload_folder="/tmp/offload",
    )

    evaluator = CodingAgentEvaluator(config)

    test_cases = [
        {
            "instruction": "Add type hints to fix mypy CI failure",
            "ci_error": "error: Function is missing a return type annotation",
            "source_code": "def add(a, b):\n    return a + b\n",
            "expected":    "def add(a: int, b: int) -> int:\n    return a + b\n",
            "test_code": (
                "def test_add_returns_int():\n    assert add(2, 3) == 5\n\n"
                "def test_add_negative():\n    assert add(-1, 1) == 0\n"
            ),
        },
        {
            "instruction": "Convert to async function to fix asyncio CI failure",
            "ci_error": "RuntimeError: coroutine was never awaited",
            "source_code": "def get_user():\n    return fetch_user()\n",
            "expected":    "async def get_user():\n    return await fetch_user()\n",
        },
        {
            "instruction": "Add a docstring to pass pydocstyle CI check",
            "ci_error": "D100: Missing docstring in public function",
            "source_code": "def multiply(a, b):\n    return a * b\n",
            "expected":    'def multiply(a, b):\n    """Multiply two numbers and return the result."""\n    return a * b\n',
            "test_code": (
                "def test_multiply():\n    assert multiply(3, 4) == 12\n\n"
                "def test_multiply_zero():\n    assert multiply(0, 99) == 0\n"
            ),
        },
    ]

    evaluator.evaluate(test_cases, save_results="eval_results.json")