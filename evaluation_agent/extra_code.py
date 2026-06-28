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


# ── CELL 1: Install dependencies ────────────────────────────

#!pip install -q transformers peft bitsandbytes accelerate datasets
#!pip install -q codebleu


# ── CELL 2: Imports ─────────────────────────────────────────

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


# ── CELL 3: Config ──────────────────────────────────────────

@dataclass
class Config:
    # Model configuration
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path: str = "./adapter"

    # Optional cache/offload settings
    cache_dir: Optional[str] = None
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    offload_folder: str = "/tmp/offload"

    # Dataset configuration
    dataset_name: str = "JetBrains-Research/diff-xyz"
    dataset_split: str = "test"

    # Use "python" if your generated/reference code is Python.
    # Set to None if you want all languages, but CodeBLEU language support must match.
    filter_lang: Optional[str] = "python"

    # 0 means use all records.
    max_samples: int = 50

    # Dataset diff field. This is kept from your previous script.
    diff_format: str = "udiff"

    # Generation configuration
    max_new_tokens: int = 512

    # Output report file
    save_report: str = "eval_report_codebleu.json"


CFG = Config()


# ── CELL 4: Model utilities ─────────────────────────────────

def resolve_dtype(name: str) -> torch.dtype:
    """
    Convert string dtype name into a torch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(name.lower(), torch.float16)


def get_model_device(model) -> torch.device:
    """
    Return the device where the model is currently placed.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_gpu_memory() -> None:
    """
    Clear CUDA cache to reduce memory pressure between model operations.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_free_vram() -> int:
    """
    Return free GPU memory in bytes.
    """
    if not torch.cuda.is_available():
        return 0

    free_memory, _ = torch.cuda.mem_get_info(0)
    return free_memory


def patch_adapter_config(adapter_path: str) -> None:
    """
    Some PEFT adapter_config.json files may contain keys that are not accepted
    by the installed PEFT version. This function removes unsupported keys.

    It also creates a backup file:
        adapter_config.json.bak
    """
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

        cleaned_config = {
            key: value
            for key, value in adapter_config.items()
            if key in valid_keys
        }

        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(cleaned_config, file, indent=2)


def clean_generation_config(model) -> None:
    """
    Replace model generation config with a controlled deterministic setup.
    This helps make base vs fine-tuned comparison fair.
    """
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
    """
    Build 4-bit quantization config for memory-efficient model loading.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer(cfg: Config):
    """
    Load tokenizer for the base model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        use_fast=True,
        cache_dir=cfg.cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(cfg: Config):
    """
    Load the base model.

    Strategy:
      - If enough GPU memory is available, load in 4-bit fully on GPU.
      - Otherwise, use automatic device mapping with CPU offloading.
    """
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
            max_memory = {
                0: int(free_vram * 0.85),
                "cpu": "48GiB",
            }

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
    """
    Load the fine-tuned LoRA adapter on top of the base model.
    """
    patch_adapter_config(cfg.adapter_path)
    free_gpu_memory()

    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        cfg.adapter_path,
        is_trainable=False,
    )

    fine_tuned_model.eval()
    clean_generation_config(fine_tuned_model)
    free_gpu_memory()

    return fine_tuned_model


# ── CELL 5: Dataset preparation ─────────────────────────────

def build_record(example: dict, diff_format: str) -> dict:
    """
    Convert one dataset row into a normalized evaluation record.
    """
    repo = example.get("repo")
    commit = str(example.get("commit", ""))[:10]
    path = example.get("path")

    return {
        "id": f"{repo}::{commit}::{path}",
        "language": example.get("lang"),
        "context": (example.get("message") or "").strip(),
        "input_code": example.get("old_code", ""),
        "reference_code": example.get("new_code", ""),
        "reference_diff": example.get(diff_format, ""),
    }


def load_eval_dataset(cfg: Config) -> List[dict]:
    """
    Load and prepare the evaluation dataset.
    """
    print(f"Loading dataset: {cfg.dataset_name}, split={cfg.dataset_split}")

    dataset = load_dataset(
        cfg.dataset_name,
        "default",
        split=cfg.dataset_split,
    )

    if cfg.filter_lang:
        dataset = dataset.filter(
            lambda row: row.get("lang") == cfg.filter_lang
        )

    if cfg.max_samples > 0:
        dataset = dataset.select(
            range(min(cfg.max_samples, len(dataset)))
        )

    records = [
        build_record(example, cfg.diff_format)
        for example in dataset
    ]

    records = [
        record
        for record in records
        if record["input_code"] and record["reference_code"]
    ]

    print(f"Prepared {len(records)} evaluation records.")
    return records


# ── CELL 6: Generation utilities ────────────────────────────

def build_prompt(input_code: str, context: str) -> str:
    """
    Build the prompt for CI/CD fix generation.

    The model is instructed to return only corrected Python code.
    """
    return (
        "You are an expert software engineer specialising in CI/CD pipeline fixes.\n\n"
        f"Task:\n{context}\n\n"
        f"Source Code:\n```python\n{input_code}\n```\n\n"
        "Return ONLY the corrected Python code inside a ```python block. "
        "No explanation, no commentary."
    )


def extract_code_from_response(text: str) -> str:
    """
    Extract code from a model response.

    If the model returns a fenced Python block, that block is used.
    Otherwise, non-empty lines from the response are returned.
    """
    match = re.search(
        r"```(?:python)?\s*\n(.*?)```",
        text,
        re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return "\n".join(
        line
        for line in text.strip().splitlines()
        if line.strip()
    )


def generate_fix(
    model,
    tokenizer,
    input_code: str,
    context: str,
    cfg: Config,
) -> str:
    """
    Generate corrected code from the given model.
    """
    prompt = build_prompt(input_code, context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    device = get_model_device(model)
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return extract_code_from_response(generated_text)


# ── CELL 7: CodeBLEU evaluation utilities ───────────────────

CODEBLEU_WEIGHTS: Tuple[float, float, float, float] = (
    0.10,
    0.10,
    0.40,
    0.40,
)

CODEBLEU_SUPPORTED_LANGUAGES = {
    "python",
    "java",
    "javascript",
    "php",
    "ruby",
    "go",
    "c_sharp",
    "c",
    "cpp",
}

CODEBLEU_LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "javascript",
    "tsx": "javascript",
    "c#": "c_sharp",
    "csharp": "c_sharp",
    "cs": "c_sharp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
}


def normalize_codebleu_language(language: Optional[str]) -> str:
    """
    Normalize dataset language names into names accepted by CodeBLEU.
    """
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
    """
    Tokenizer used by CodeBLEU.

    It separates identifiers, keywords, numbers, and punctuation.
    """
    return re.findall(r"\w+|[^\w\s]", code)


def compute_codebleu_score(
    candidate: str,
    reference: str,
    language: str = "python",
    weights: Tuple[float, float, float, float] = CODEBLEU_WEIGHTS,
) -> Dict[str, float]:
    """
    Compute CodeBLEU score for one candidate output and one reference code.

    Returns scores in 0.0 to 1.0 scale:
      - bleu
      - bleu_weight
      - match_ast
      - match_df
      - codebleu
    """
    lang = normalize_codebleu_language(language)

    result = calc_codebleu(
        references=[reference],
        predictions=[candidate],
        lang=lang,
        weights=weights,
        tokenizer=code_tokenizer,
    )

    return {
        "bleu": float(result.get("ngram_match_score", 0.0)),
        "bleu_weight": float(result.get("weighted_ngram_match_score", 0.0)),
        "match_ast": float(result.get("syntax_match_score", 0.0)),
        "match_df": float(result.get("dataflow_match_score", 0.0)),
        "codebleu": float(result.get("codebleu", 0.0)),
    }


# ── CELL 8: Evaluation loop ─────────────────────────────────

def run_evaluation(cfg: Config = CFG):
    """
    Run the full evaluation.

    This compares:
      1. Base model output
      2. Fine-tuned LoRA adapter output

    Both are compared against the same reference code using CodeBLEU.
    """
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

    results = []
    base_wins = 0
    fine_tuned_wins = 0
    draws = 0

    for index, record in enumerate(records, start=1):
        print(f"\n[{index}/{len(records)}] {record['id']}")

        # Model 1: Base model.
        # The adapter is disabled so this is the original base model behavior.
        with fine_tuned_model.disable_adapter():
            base_output = generate_fix(
                fine_tuned_model,
                tokenizer,
                record["input_code"],
                record["context"],
                cfg,
            )

        # Model 2: Fine-tuned model.
        # The adapter is active here.
        fine_tuned_output = generate_fix(
            fine_tuned_model,
            tokenizer,
            record["input_code"],
            record["context"],
            cfg,
        )

        reference_code = record["reference_code"]
        language = normalize_codebleu_language(
            record["language"] or cfg.filter_lang
        )

        base_scores = compute_codebleu_score(
            candidate=base_output,
            reference=reference_code,
            language=language,
            weights=CODEBLEU_WEIGHTS,
        )

        fine_tuned_scores = compute_codebleu_score(
            candidate=fine_tuned_output,
            reference=reference_code,
            language=language,
            weights=CODEBLEU_WEIGHTS,
        )

        base_codebleu = base_scores["codebleu"]
        fine_tuned_codebleu = fine_tuned_scores["codebleu"]

        if fine_tuned_codebleu > base_codebleu:
            winner = "fine_tuned"
            fine_tuned_wins += 1
        elif base_codebleu > fine_tuned_codebleu:
            winner = "base"
            base_wins += 1
        else:
            winner = "draw"
            draws += 1

        print(f"  Language       : {language}")
        print(f"  CodeBLEU       base={base_codebleu:.4f}  fine_tuned={fine_tuned_codebleu:.4f}")
        print(f"  BLEU           base={base_scores['bleu']:.4f}  fine_tuned={fine_tuned_scores['bleu']:.4f}")
        print(f"  BLEUweight     base={base_scores['bleu_weight']:.4f}  fine_tuned={fine_tuned_scores['bleu_weight']:.4f}")
        print(f"  Match_ast      base={base_scores['match_ast']:.4f}  fine_tuned={fine_tuned_scores['match_ast']:.4f}")
        print(f"  Match_df       base={base_scores['match_df']:.4f}  fine_tuned={fine_tuned_scores['match_df']:.4f}")
        print(f"  Winner         : {winner.upper()}")

        results.append({
            "id": record["id"],
            "language": language,
            "winner": winner,

            "base_codebleu": base_scores["codebleu"],
            "base_bleu": base_scores["bleu"],
            "base_bleu_weight": base_scores["bleu_weight"],
            "base_match_ast": base_scores["match_ast"],
            "base_match_df": base_scores["match_df"],

            "fine_tuned_codebleu": fine_tuned_scores["codebleu"],
            "fine_tuned_bleu": fine_tuned_scores["bleu"],
            "fine_tuned_bleu_weight": fine_tuned_scores["bleu_weight"],
            "fine_tuned_match_ast": fine_tuned_scores["match_ast"],
            "fine_tuned_match_df": fine_tuned_scores["match_df"],

            "input_code": record["input_code"],
            "reference_code": reference_code,
            "base_output": base_output,
            "fine_tuned_output": fine_tuned_output,
            "context": record["context"],
        })

    aggregate = build_aggregate_results(
        results=results,
        base_wins=base_wins,
        fine_tuned_wins=fine_tuned_wins,
        draws=draws,
    )

    save_report(
        cfg=cfg,
        results=results,
        aggregate=aggregate,
    )

    print_final_summary(aggregate)

    return results, aggregate


def average(results: List[dict], key: str) -> float:
    """
    Calculate average value for a score key.
    """
    if not results:
        return 0.0

    return sum(result[key] for result in results) / len(results)


def build_aggregate_results(
    results: List[dict],
    base_wins: int,
    fine_tuned_wins: int,
    draws: int,
) -> dict:
    """
    Build final aggregate result summary.
    """
    return {
        "n_samples": len(results),
        "score_scale": "0.0_to_1.0",

        "base_wins": base_wins,
        "fine_tuned_wins": fine_tuned_wins,
        "draws": draws,

        "avg_codebleu_base": average(results, "base_codebleu"),
        "avg_codebleu_fine_tuned": average(results, "fine_tuned_codebleu"),
        "avg_codebleu_improvement": (
            average(results, "fine_tuned_codebleu")
            - average(results, "base_codebleu")
        ),

        "avg_bleu_base": average(results, "base_bleu"),
        "avg_bleu_fine_tuned": average(results, "fine_tuned_bleu"),

        "avg_bleu_weight_base": average(results, "base_bleu_weight"),
        "avg_bleu_weight_fine_tuned": average(results, "fine_tuned_bleu_weight"),

        "avg_match_ast_base": average(results, "base_match_ast"),
        "avg_match_ast_fine_tuned": average(results, "fine_tuned_match_ast"),

        "avg_match_df_base": average(results, "base_match_df"),
        "avg_match_df_fine_tuned": average(results, "fine_tuned_match_df"),

        "metric_reference": {
            "metric": "CodeBLEU",
            "paper": "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis",
            "authors": "Ren et al.",
            "year": 2020,
            "arxiv": "2009.10297",
            "formula": "CodeBLEU = α·BLEU + β·BLEUweight + γ·Match_ast + δ·Match_df",
            "weights_used": {
                "alpha_bleu": CODEBLEU_WEIGHTS[0],
                "beta_weighted_ngram": CODEBLEU_WEIGHTS[1],
                "gamma_ast_match": CODEBLEU_WEIGHTS[2],
                "delta_dataflow_match": CODEBLEU_WEIGHTS[3],
            },
            "components": [
                "standard BLEU / n-gram match",
                "weighted n-gram match",
                "syntax / AST match",
                "data-flow match",
            ],
            "implementation_note": (
                "CodeBLEU scores were computed using a CodeBLEU implementation "
                "that returns n-gram match, weighted n-gram match, syntax match, "
                "data-flow match, and final CodeBLEU score."
            ),
        },
    }


def save_report(
    cfg: Config,
    results: List[dict],
    aggregate: dict,
) -> None:
    """
    Save full evaluation report as JSON.
    """
    report = {
        "aggregate": aggregate,
        "per_sample": results,
    }

    with open(cfg.save_report, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nReport saved → {cfg.save_report}")


def print_final_summary(aggregate: dict) -> None:
    """
    Print final summary after all samples are evaluated.
    """
    if aggregate["fine_tuned_wins"] > aggregate["base_wins"]:
        verdict = "FINE-TUNED MODEL IS BETTER"
    elif aggregate["base_wins"] > aggregate["fine_tuned_wins"]:
        verdict = "BASE MODEL IS BETTER"
    else:
        verdict = "DRAW"

    print("\n" + "=" * 60)
    print("FINAL CODEBLEU RESULTS")
    print("=" * 60)
    print(f"Samples evaluated              : {aggregate['n_samples']}")
    print(f"Base wins                      : {aggregate['base_wins']}")
    print(f"Fine-tuned wins                : {aggregate['fine_tuned_wins']}")
    print(f"Draws                          : {aggregate['draws']}")
    print()
    print(f"Average CodeBLEU Base          : {aggregate['avg_codebleu_base']:.4f}")
    print(f"Average CodeBLEU Fine-tuned    : {aggregate['avg_codebleu_fine_tuned']:.4f}")
    print(f"Average CodeBLEU Improvement   : {aggregate['avg_codebleu_improvement']:.4f}")
    print()
    print(f"Average BLEU Base              : {aggregate['avg_bleu_base']:.4f}")
    print(f"Average BLEU Fine-tuned        : {aggregate['avg_bleu_fine_tuned']:.4f}")
    print()
    print(f"Average BLEUweight Base        : {aggregate['avg_bleu_weight_base']:.4f}")
    print(f"Average BLEUweight Fine-tuned  : {aggregate['avg_bleu_weight_fine_tuned']:.4f}")
    print()
    print(f"Average Match_ast Base         : {aggregate['avg_match_ast_base']:.4f}")
    print(f"Average Match_ast Fine-tuned   : {aggregate['avg_match_ast_fine_tuned']:.4f}")
    print()
    print(f"Average Match_df Base          : {aggregate['avg_match_df_base']:.4f}")
    print(f"Average Match_df Fine-tuned    : {aggregate['avg_match_df_fine_tuned']:.4f}")
    print()
    print(f"Verdict                        : {verdict}")
    print("=" * 60)


# ── CELL 9: Run evaluation ──────────────────────────────────

if __name__ == "__main__":
    results, aggregate = run_evaluation(CFG)