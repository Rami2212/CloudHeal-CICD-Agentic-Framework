import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase,
                          Trainer, TrainingArguments, set_seed)


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
    lr_scheduler_type: str
    weight_decay: float
    max_grad_norm: float
    eval_steps: Optional[int]
    save_total_limit: int
    group_by_length: bool
    dataloader_num_workers: int


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
        lr_scheduler_type=raw.get("lr_scheduler_type", "cosine"),
        weight_decay=raw.get("weight_decay", 0.01),
        max_grad_norm=raw.get("max_grad_norm", 1.0),
        eval_steps=raw.get("eval_steps"),
        save_total_limit=raw.get("save_total_limit", 2),
        group_by_length=raw.get("group_by_length", True),
        dataloader_num_workers=raw.get("dataloader_num_workers", 2),
    )


@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        if "labels" not in features[0]:
            batch["labels"] = batch["input_ids"].clone()
            return batch

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for feature in features:
            labels = feature["labels"]
            pad_len = max_len - len(labels)
            if pad_len > 0:
                labels = labels + [self.label_pad_token_id] * pad_len
            padded_labels.append(labels)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def stringify_input(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_messages(example: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    if isinstance(example.get("messages"), list):
        messages = [
            {"role": str(item.get("role")), "content": stringify_input(item.get("content"))}
            for item in example["messages"]
            if isinstance(item, dict)
        ]
    elif "prompt" in example and "completion" in example:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": stringify_input(example.get("prompt"))},
            {"role": "assistant", "content": stringify_input(example.get("completion"))},
        ]
    else:
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

    if not any(message.get("role") == "system" for message in messages):
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


def ensure_assistant(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if messages and messages[-1].get("role") == "assistant":
        return messages
    return messages + [{"role": "assistant", "content": ""}]


def split_prompt_and_full(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    full_messages = ensure_assistant(messages)
    if full_messages[-1].get("role") == "assistant":
        return full_messages[:-1], full_messages
    return full_messages, full_messages


def format_prompt(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizerBase) -> str:
    prompt_messages, _ = split_prompt_and_full(messages)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

    system_prompt = prompt_messages[0]["content"] if prompt_messages else ""
    user_content = ""
    for message in prompt_messages:
        if message.get("role") == "user":
            user_content = message.get("content", "")
            break
    return f"{system_prompt}\n\nUser:\n{user_content}\n\nAssistant:\n"


def format_full(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizerBase) -> str:
    _, full_messages = split_prompt_and_full(messages)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

    system_prompt = full_messages[0]["content"] if full_messages else ""
    user_content = ""
    assistant_content = ""
    for message in full_messages:
        if message.get("role") == "user":
            user_content = message.get("content", "")
        if message.get("role") == "assistant":
            assistant_content = message.get("content", "")
    return (
        f"{system_prompt}\n\nUser:\n{user_content}\n\nAssistant:\n{assistant_content}"
    )


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
    tokenizer.padding_side = "right"
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
        messages = normalize_messages(example, cfg.system_prompt)
        prompt_text = format_prompt(messages, tokenizer)
        full_text = format_full(messages, tokenizer)

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=cfg.max_seq_length,
            add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=cfg.max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        labels = list(full_tokens["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        if prompt_len:
            labels[:prompt_len] = [-100] * prompt_len
        full_tokens["labels"] = labels
        return full_tokens

    tokenized = datasets.map(
        tokenize_fn,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing",
    )

    if cfg.max_train_samples:
        tokenized["train"] = tokenized["train"].select(range(cfg.max_train_samples))
    if cfg.max_eval_samples and "validation" in tokenized:
        tokenized["validation"] = tokenized["validation"].select(range(cfg.max_eval_samples))

    data_collator = CausalLMCollator(tokenizer=tokenizer)

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
        eval_steps=cfg.eval_steps or cfg.logging_steps,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        bf16=precision_flags["bf16"],
        fp16=precision_flags["fp16"],
        seed=cfg.seed,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        group_by_length=cfg.group_by_length,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
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

