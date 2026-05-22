import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, set_seed)


@dataclass
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]


@dataclass
class PrecisionSettings:
    use_bf16_if_supported: bool
    fallback_fp16: bool


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    validation_file: Optional[str]
    test_file: Optional[str]
    max_seq_length: int
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    seed: int
    gradient_checkpointing: bool
    lora: LoraSettings
    precision: PrecisionSettings
    system_prompt: str
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    lora = LoraSettings(**raw["lora"])
    precision = PrecisionSettings(**raw["precision"])
    return TrainConfig(
        model_name_or_path=raw["model_name_or_path"],
        train_file=raw["train_file"],
        output_dir=raw["output_dir"],
        validation_file=raw.get("validation_file"),
        test_file=raw.get("test_file"),
        max_seq_length=raw["max_seq_length"],
        num_train_epochs=raw["num_train_epochs"],
        per_device_train_batch_size=raw["per_device_train_batch_size"],
        per_device_eval_batch_size=raw.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=raw["gradient_accumulation_steps"],
        learning_rate=raw["learning_rate"],
        warmup_ratio=raw["warmup_ratio"],
        logging_steps=raw["logging_steps"],
        save_steps=raw["save_steps"],
        seed=raw.get("seed", 42),
        gradient_checkpointing=raw.get("gradient_checkpointing", True),
        lora=lora,
        precision=precision,
        system_prompt=raw.get("system_prompt", "You are a helpful coding assistant."),
        max_train_samples=raw.get("max_train_samples"),
        max_eval_samples=raw.get("max_eval_samples"),
    )


def stringify_input(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_example(example: Dict[str, Any], tokenizer: AutoTokenizer, system_prompt: str) -> str:
    instruction = example.get("instruction") or "Complete the task."
    input_block = stringify_input(example.get("input"))
    output_block = stringify_input(example.get("output"))
    user_content = instruction
    if input_block:
        user_content = f"{instruction}\n\nContext:\n{input_block}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_block},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return f"{system_prompt}\n\nUser:\n{user_content}\n\nAssistant:\n{output_block}"


def resolve_precision(precision: PrecisionSettings) -> Dict[str, bool]:
    bf16_supported = False
    if precision.use_bf16_if_supported and torch.cuda.is_available():
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16_supported:
        return {"bf16": True, "fp16": False}
    if precision.fallback_fp16:
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def load_datasets(cfg: TrainConfig) -> DatasetDict:
    data_files = {"train": cfg.train_file}
    if cfg.validation_file:
        data_files["validation"] = cfg.validation_file
    if cfg.test_file:
        data_files["test"] = cfg.test_file
    return load_dataset("json", data_files=data_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5-Coder-7B")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if resolve_precision(cfg.precision)["bf16"] else None,
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    datasets = load_datasets(cfg)

    def tokenize_fn(example: Dict[str, Any]) -> Dict[str, List[int]]:
        text = format_example(example, tokenizer, cfg.system_prompt)
        return tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
        )

    tokenized = datasets.map(
        tokenize_fn,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing",
    )

    if cfg.max_train_samples:
        tokenized["train"] = tokenized["train"].select(range(cfg.max_train_samples))
    if cfg.max_eval_samples and "validation" in tokenized:
        tokenized["validation"] = tokenized["validation"].select(range(cfg.max_eval_samples))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    precision_flags = resolve_precision(cfg.precision)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=cfg.logging_steps,
        save_total_limit=2,
        report_to="none",
        bf16=precision_flags["bf16"],
        fp16=precision_flags["fp16"],
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()

