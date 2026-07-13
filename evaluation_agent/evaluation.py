# ============================================================
#  CI/CD Fix-Generation Evaluation Pipeline
#
#  Purpose:
#    Evaluate two models using CodeBLEU following Ren et al. (2020)
#
#  Models compared:
#    1. Base model
#    2. Fine-tuned LoRA adapter model
#
#  Metric:
#    CodeBLEU = α·BLEU + β·BLEUweight + γ·Match_ast + δ·Match_df
#
#  Recommended general CodeBLEU weights:
#    α = 0.10
#    β = 0.10
#    γ = 0.40
#    δ = 0.40
#
#  Paper:
#    CodeBLEU: a Method for Automatic Evaluation of Code Synthesis
#    Ren et al., 2020
# ============================================================


# ── Imports ─────────────────────────────────────────

import os
import re
import json
import inspect
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel, LoraConfig
from datasets import load_dataset

from codebleu import calc_codebleu


# ── Config ──────────────────────────────────────────

@dataclass
class Config:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path: str = "./adapter"

    cache_dir: Optional[str] = None
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    offload_folder: str = "/tmp/offload"

    dataset_name: str = "JetBrains-Research/diff-xyz"
    dataset_split: str = "test"

    filter_lang: Optional[str] = "python"
    max_samples: int = 50

    diff_format: str = "udiff"

    max_new_tokens: int = 1024
    repetition_penalty: float = 1.0


CFG = Config()


# ── Model utilities ──────────────────────────────────

def resolve_dtype(name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(name.lower(), torch.float16)


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_free_vram() -> int:
    if not torch.cuda.is_available():
        return 0
    free_memory, _ = torch.cuda.mem_get_info(0)
    return free_memory


def patch_adapter_config(adapter_path: str) -> None:
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as file:
        adapter_config = json.load(file)

    valid_keys = set(inspect.signature(LoraConfig.__init__).parameters) - {"self"}
    invalid_keys = [key for key in adapter_config if key not in valid_keys]

    if invalid_keys:
        print(f"[adapter] Removing unsupported adapter config keys: {invalid_keys}")
        with open(config_path + ".bak", "w", encoding="utf-8") as file:
            json.dump(adapter_config, file, indent=2)
        cleaned_config = {k: v for k, v in adapter_config.items() if k in valid_keys}
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(cleaned_config, file, indent=2)


def clean_generation_config(model) -> None:
    try:
        pad_token_id = model.generation_config.pad_token_id
    except Exception:
        pad_token_id = None
    try:
        eos_token_id = model.generation_config.eos_token_id
    except Exception:
        eos_token_id = None
    try:
        bos_token_id = model.generation_config.bos_token_id
    except Exception:
        bos_token_id = None

    model.generation_config = GenerationConfig(
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )


def build_bitsandbytes_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, use_fast=True, cache_dir=cfg.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: Config):
    os.makedirs(cfg.offload_folder, exist_ok=True)
    free_gpu_memory()

    free_vram = get_free_vram()
    usable_vram = max(0, free_vram - int(3.0 * 1024**3))
    can_fit_4bit_on_gpu = usable_vram >= int(4.5 * 1024**3)

    if cfg.load_in_4bit and can_fit_4bit_on_gpu:
        print("[load] Strategy: 4-bit NF4 fully on GPU")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            device_map={"": 0},
            cache_dir=cfg.cache_dir,
            low_cpu_mem_usage=True,
            quantization_config=build_bitsandbytes_config(),
        )
    else:
        print("[load] Strategy: fp16/bf16 with auto device_map and CPU offload")
        max_memory = None
        if torch.cuda.is_available():
            max_memory = {0: int(free_vram * 0.85), "cpu": "48GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            device_map="auto",
            torch_dtype=resolve_dtype(cfg.torch_dtype),
            cache_dir=cfg.cache_dir,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            offload_folder=cfg.offload_folder,
        )

    model.eval()
    clean_generation_config(model)
    free_gpu_memory()
    return model


def load_peft_model(base_model, cfg: Config):
    patch_adapter_config(cfg.adapter_path)
    free_gpu_memory()

    fine_tuned_model = PeftModel.from_pretrained(
        base_model, cfg.adapter_path, is_trainable=False,
    )
    fine_tuned_model.eval()
    clean_generation_config(fine_tuned_model)
    free_gpu_memory()
    return fine_tuned_model


# ── Dataset preparation ──────────────────────────────

def build_record(example: dict, diff_format: str) -> dict:
    repo = example.get("repo")
    commit = str(example.get("commit", ""))[:10]
    path = example.get("path")
    return {
        "id": f"{repo}::{commit}::{path}",
        "language": example.get("lang"),
        "context": (example.get("message") or "").strip(),
        "input_code": example.get("old_code", ""),
        "reference_diff": example.get(diff_format, ""),
    }


def load_eval_dataset(cfg: Config) -> List[dict]:
    print(f"Loading dataset: {cfg.dataset_name}, split={cfg.dataset_split}")
    dataset = load_dataset(cfg.dataset_name, "default", split=cfg.dataset_split)

    if cfg.filter_lang:
        dataset = dataset.filter(lambda row: row.get("lang") == cfg.filter_lang)

    if cfg.max_samples > 0:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))

    records = [build_record(example, cfg.diff_format) for example in dataset]
    records = [r for r in records if r["input_code"] and r["reference_diff"]]

    print(f"Prepared {len(records)} evaluation records.")
    return records


# ── Generation utilities ─────────────────────────────

SYSTEM_PROMPT = "You are a helpful coding assistant."


def build_prompt(input_code: str, context: str) -> str:
    return (
        "You are an expert software engineer specialising in CI/CD pipeline fixes.\n\n"
        f"Task:\n{context}\n\n"
        f"Source Code:\n```python\n{input_code}\n```\n\n"
        "Return ONLY a unified diff (unified diff format, like `diff -u` or "
        "`git diff`) that transforms the Source Code above into the corrected "
        "version.\n"
        "Use standard unified diff syntax:\n"
        "  --- a/<file>\n"
        "  +++ b/<file>\n"
        "  @@ -<start>,<count> +<start>,<count> @@\n"
        "  followed by context lines (leading space), removed lines (leading -), "
        "and added lines (leading +).\n"
        "The <count> values in each @@ header MUST exactly match the number of "
        "context+removed lines (old count) and context+added lines (new count) "
        "that follow in that hunk.\n"
        "Preserve the exact original indentation on every context and removed "
        "line — copy it character-for-character from the Source Code.\n"
        "Do NOT return the full file.\n"
        "Do NOT include any explanation or commentary — output only the diff, "
        "inside a single ```diff code block."
    )


def extract_diff_from_response(text: str) -> str:
    match = re.search(
        r"```(?:diff|patch|udiff)?\s*\n(.*?)```",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_fix(model, tokenizer, input_code: str, context: str, cfg: Config) -> str:
    """Generate a unified diff from the given model."""
    user_content = build_prompt(input_code, context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, padding=True,
        add_special_tokens=False,
    )
    device = get_model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_token_ids = output[0][prompt_len:]
    raw_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return extract_diff_from_response(raw_text)


# ── CodeBLEU evaluation utilities ───────────────────

CODEBLEU_WEIGHTS: Tuple[float, float, float, float] = (0.10, 0.10, 0.40, 0.40)

CODEBLEU_SUPPORTED_LANGUAGES = {
    "python", "java", "javascript", "php", "ruby", "go", "c_sharp", "c", "cpp",
}

CODEBLEU_LANGUAGE_ALIASES = {
    "py": "python", "python3": "python",
    "js": "javascript", "jsx": "javascript", "ts": "javascript", "tsx": "javascript",
    "c#": "c_sharp", "csharp": "c_sharp", "cs": "c_sharp",
    "c++": "cpp", "cc": "cpp", "cxx": "cpp",
}


def normalize_codebleu_language(language: Optional[str]) -> str:
    if not language:
        return "python"
    normalized = language.strip().lower()
    normalized = CODEBLEU_LANGUAGE_ALIASES.get(normalized, normalized)
    if normalized not in CODEBLEU_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported CodeBLEU language: {language}. "
            f"Supported languages: {sorted(CODEBLEU_SUPPORTED_LANGUAGES)}"
        )
    return normalized


def code_tokenizer(code: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", code)


def compute_codebleu_score(
        candidate: str,
        reference: str,
        language: str = "python",
        weights: Tuple[float, float, float, float] = CODEBLEU_WEIGHTS,
) -> Dict[str, float]:
    """
    CodeBLEU = α·BLEU + β·BLEUweight + γ·Match_ast + δ·Match_df
    weights = (0.10, 0.10, 0.40, 0.40)
    """
    lang = normalize_codebleu_language(language)

    result = calc_codebleu(
        references=[reference],
        predictions=[candidate],
        lang=lang,
        weights=weights,
        tokenizer=code_tokenizer,
    )

    bleu = float(result.get("ngram_match_score", 0.0))
    bleu_weight = float(result.get("weighted_ngram_match_score", 0.0))
    match_ast = float(result.get("syntax_match_score", 0.0))
    match_df = float(result.get("dataflow_match_score", 0.0))

    alpha, beta, gamma, delta = weights
    codebleu = (
            alpha * bleu + beta * bleu_weight + gamma * match_ast + delta * match_df
    )

    return {
        "bleu": bleu,
        "bleu_weight": bleu_weight,
        "match_ast": match_ast,
        "match_df": match_df,
        "codebleu": codebleu,
    }


# ── Evaluation loop ─────────────────────────────────

def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_evaluation(cfg: Config = CFG):
    print("=" * 60)
    print("Loading tokenizer and models...")
    print("=" * 60)

    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg)
    fine_tuned_model = load_peft_model(base_model, cfg)

    print("\n" + "=" * 60)
    print("Loading evaluation dataset...")
    print("=" * 60)

    records = load_eval_dataset(cfg)

    base_bleu, base_bleu_weight, base_match_ast, base_match_df, base_codebleu = [], [], [], [], []
    ft_bleu, ft_bleu_weight, ft_match_ast, ft_match_df, ft_codebleu = [], [], [], [], []

    for index, record in enumerate(records, start=1):
        print(f"\n[{index}/{len(records)}] {record['id']}")

        base_output = generate_fix(
            base_model, tokenizer, record["input_code"], record["context"], cfg,
        )
        fine_tuned_output = generate_fix(
            fine_tuned_model, tokenizer, record["input_code"], record["context"], cfg,
        )

        reference_diff = record["reference_diff"]
        language = normalize_codebleu_language(record["language"] or cfg.filter_lang)

        base_scores = compute_codebleu_score(base_output, reference_diff, language, CODEBLEU_WEIGHTS)
        fine_tuned_scores = compute_codebleu_score(fine_tuned_output, reference_diff, language, CODEBLEU_WEIGHTS)

        base_bleu.append(base_scores["bleu"])
        base_bleu_weight.append(base_scores["bleu_weight"])
        base_match_ast.append(base_scores["match_ast"])
        base_match_df.append(base_scores["match_df"])
        base_codebleu.append(base_scores["codebleu"])

        ft_bleu.append(fine_tuned_scores["bleu"])
        ft_bleu_weight.append(fine_tuned_scores["bleu_weight"])
        ft_match_ast.append(fine_tuned_scores["match_ast"])
        ft_match_df.append(fine_tuned_scores["match_df"])
        ft_codebleu.append(fine_tuned_scores["codebleu"])

        print(f"  BLEU (alpha)        base={base_scores['bleu']:.4f}  fine_tuned={fine_tuned_scores['bleu']:.4f}")
        print(f"  BLEU_weight (beta)  base={base_scores['bleu_weight']:.4f}  fine_tuned={fine_tuned_scores['bleu_weight']:.4f}")
        print(f"  Match_ast (gamma)   base={base_scores['match_ast']:.4f}  fine_tuned={fine_tuned_scores['match_ast']:.4f}")
        print(f"  Match_df (delta)    base={base_scores['match_df']:.4f}  fine_tuned={fine_tuned_scores['match_df']:.4f}")
        print(f"  CodeBLEU            base={base_scores['codebleu']:.4f}  fine_tuned={fine_tuned_scores['codebleu']:.4f}")

    print("\n" + "=" * 60)
    print(f"FINAL CODEBLEU FORMULA VALUES ({len(records)} samples)")
    print("=" * 60)
    print(f"BLEU (alpha)        base={average(base_bleu):.4f}  fine_tuned={average(ft_bleu):.4f}")
    print(f"BLEU_weight (beta)  base={average(base_bleu_weight):.4f}  fine_tuned={average(ft_bleu_weight):.4f}")
    print(f"Match_ast (gamma)   base={average(base_match_ast):.4f}  fine_tuned={average(ft_match_ast):.4f}")
    print(f"Match_df (delta)    base={average(base_match_df):.4f}  fine_tuned={average(ft_match_df):.4f}")
    print(f"CodeBLEU (overall)  base={average(base_codebleu):.4f}  fine_tuned={average(ft_codebleu):.4f}")
    print("=" * 60)


# ── Run evaluation ──────────────────────────────────

if __name__ == "__main__":
    run_evaluation(CFG)