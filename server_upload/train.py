import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase,
                          Trainer, TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments, set_seed)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "train.yaml"


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


REQUIRED_INSTRUCTION_FIELDS = {
    "id",
    "source",
    "task_type",
    "instruction",
    "input",
    "output",
    "metadata",
}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def iter_jsonl_rows(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield line_number, row


def validate_instruction_split(split_name: str, path_value: str) -> Tuple[int, Set[str]]:
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} dataset file: {path}")

    row_count = 0
    expected_keys: Optional[Set[str]] = None
    empty_output_rows = 0
    missing_field_examples: List[str] = []
    key_drift_examples: List[str] = []

    for line_number, row in iter_jsonl_rows(path):
        row_count += 1
        keys = set(row.keys())
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys and len(key_drift_examples) < 3:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            key_drift_examples.append(
                f"line {line_number}: missing={missing or []}, extra={extra or []}"
            )

        missing_required = sorted(REQUIRED_INSTRUCTION_FIELDS - keys)
        if missing_required and len(missing_field_examples) < 3:
            missing_field_examples.append(f"line {line_number}: {missing_required}")

        if not str(row.get("instruction", "")).strip():
            raise ValueError(f"{path}:{line_number} has an empty instruction")
        if row.get("input") is None:
            raise ValueError(f"{path}:{line_number} has a null input")
        if not str(row.get("output", "")).strip():
            empty_output_rows += 1

    if row_count == 0:
        raise ValueError(f"{path} has no JSONL rows")
    if missing_field_examples:
        raise ValueError(
            f"{split_name} split is missing required fields in {path}: "
            + "; ".join(missing_field_examples)
        )
    if key_drift_examples:
        raise ValueError(
            f"{split_name} split has inconsistent row keys in {path}: "
            + "; ".join(key_drift_examples)
        )
    if empty_output_rows:
        raise ValueError(f"{split_name} split has {empty_output_rows} rows with empty output")

    print(f"[data check] {split_name}: {row_count} rows, fields={sorted(expected_keys or [])}")
    return row_count, expected_keys or set()


def validate_datasets_before_training(cfg: TrainConfig) -> None:
    split_files = {
        "train": cfg.train_file,
        "validation": cfg.validation_file,
        "test": cfg.test_file,
    }
    missing_splits = [name for name, path in split_files.items() if not path]
    if missing_splits:
        raise ValueError(
            "Training requires train, validation, and test files before fine-tuning. "
            f"Missing config entries: {', '.join(missing_splits)}"
        )

    split_keys: Dict[str, Set[str]] = {}
    for split_name, path_value in split_files.items():
        _, keys = validate_instruction_split(split_name, str(path_value))
        split_keys[split_name] = keys

    train_keys = split_keys["train"]
    drift = {
        split_name: sorted(keys ^ train_keys)
        for split_name, keys in split_keys.items()
        if keys != train_keys
    }
    if drift:
        raise ValueError(
            "Annotated train/validation/test files do not share the same top-level format: "
            + json.dumps(drift, sort_keys=True)
        )
    print("[data check] annotated train/validation/test files share the same format")


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    train_file = str(resolve_path(raw["train_file"]))
    validation_file = raw.get("validation_file")
    test_file = raw.get("test_file")
    output_dir = resolve_path(raw["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    lora = LoraSettings(**raw["lora"])
    precision = PrecisionSettings(**raw["precision"])
    return TrainConfig(
        model_name_or_path=raw["model_name_or_path"],
        train_file=train_file,
        output_dir=str(output_dir),
        validation_file=str(resolve_path(validation_file)) if validation_file else None,
        test_file=str(resolve_path(test_file)) if test_file else None,
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
        labels_by_feature = [feature.get("labels") for feature in features]
        features_without_labels = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(features_without_labels, padding=True, return_tensors="pt")
        if "labels" not in features[0]:
            batch["labels"] = batch["input_ids"].clone()
            return batch

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for labels in labels_by_feature:
            pad_len = max_len - len(labels)
            if pad_len > 0:
                labels = labels + [self.label_pad_token_id] * pad_len
            padded_labels.append(labels)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class EpochMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "epoch_metrics.jsonl"

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        epoch = float(state.epoch or 0.0)
        epoch_number = max(1, round(epoch))
        epoch_logs = [
            log for log in state.log_history
            if abs(float(log.get("epoch", -1.0)) - epoch) < 0.01
        ]
        payload = {
            "epoch": epoch,
            "epoch_number": epoch_number,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "logs": epoch_logs,
        }

        epoch_path = self.output_dir / f"epoch_{epoch_number:02d}_metrics.json"
        epoch_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(f"[metrics] wrote epoch details to {epoch_path}")
        return control


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
    if getattr(tokenizer, "chat_template", None):
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
    if getattr(tokenizer, "chat_template", None):
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


def build_tokenize_fn(cfg: TrainConfig, tokenizer: PreTrainedTokenizerBase):
    def tokenize_fn(example: Dict[str, Any]) -> Dict[str, List[int]]:
        messages = normalize_messages(example, cfg.system_prompt)
        prompt_text = format_prompt(messages, tokenizer)
        full_text = format_full(messages, tokenizer)

        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = full_ids[len(prompt_ids):]
        if not response_ids:
            output_text = stringify_input(example.get("output") or example.get("completion"))
            response_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"]
        if not response_ids:
            raise ValueError("Example has no tokenized assistant response")

        if len(full_ids) <= cfg.max_seq_length:
            input_ids = full_ids
            prompt_len = min(len(prompt_ids), len(input_ids))
        else:
            response_budget = min(len(response_ids), cfg.max_seq_length)
            prompt_budget = cfg.max_seq_length - response_budget
            prompt_slice = prompt_ids[-prompt_budget:] if prompt_budget else []
            input_ids = prompt_slice + response_ids[:response_budget]
            prompt_len = prompt_budget

        labels = list(input_ids)
        if prompt_len:
            labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return tokenize_fn


def select_first_samples(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def validate_tokenized_split(split_name: str, dataset: Dataset) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{split_name} tokenized split is empty")

    ignored_only = 0
    empty_input_ids = 0
    for index, row in enumerate(dataset):
        input_ids = row.get("input_ids") or []
        labels = row.get("labels") or []
        attention_mask = row.get("attention_mask") or []

        if not input_ids:
            empty_input_ids += 1
        if len(input_ids) != len(labels):
            raise ValueError(
                f"{split_name}[{index}] has mismatched input_ids and labels lengths: "
                f"{len(input_ids)} != {len(labels)}"
            )
        if attention_mask and len(attention_mask) != len(input_ids):
            raise ValueError(
                f"{split_name}[{index}] has mismatched attention_mask and input_ids lengths: "
                f"{len(attention_mask)} != {len(input_ids)}"
            )
        if labels and all(label == -100 for label in labels):
            ignored_only += 1
    if empty_input_ids:
        raise ValueError(f"{split_name} has {empty_input_ids} rows with empty tokenized input")
    if ignored_only:
        raise ValueError(
            f"{split_name} has {ignored_only} rows where labels are fully masked. "
            "Reduce prompt size, increase max_seq_length, or inspect long examples."
        )


def run_final_smoke_test(
    cfg: TrainConfig,
    sample_size: int,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    print("[smoke] loading annotated datasets with HuggingFace datasets")
    datasets = load_datasets(cfg)
    for split_name, dataset in datasets.items():
        print(f"[smoke] {split_name}: {len(dataset)} rows loaded")

    sample_sets = DatasetDict(
        {
            split_name: select_first_samples(dataset, sample_size)
            for split_name, dataset in datasets.items()
        }
    )

    tokenize_fn = build_tokenize_fn(cfg, tokenizer)
    tokenized = sample_sets.map(
        tokenize_fn,
        remove_columns=sample_sets["train"].column_names,
        desc="Smoke tokenizing",
    )

    for split_name, dataset in tokenized.items():
        validate_tokenized_split(split_name, dataset)
        lengths = [len(row["input_ids"]) for row in dataset]
        supervised = [
            sum(1 for label in row["labels"] if label != -100)
            for row in dataset
        ]
        print(
            f"[smoke] {split_name}: tokenized {len(dataset)} rows, "
            f"tokens min/avg/max={min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}, "
            f"supervised tokens min/avg/max={min(supervised)}/{sum(supervised)//len(supervised)}/{max(supervised)}"
        )

    collator = CausalLMCollator(tokenizer=tokenizer)
    batch = collator([tokenized["train"][index] for index in range(min(2, len(tokenized["train"])))])
    required_batch_keys = {"input_ids", "attention_mask", "labels"}
    missing_keys = required_batch_keys - set(batch)
    if missing_keys:
        raise ValueError(f"Smoke collator output is missing keys: {sorted(missing_keys)}")
    print(
        "[smoke] collator batch shapes: "
        + ", ".join(f"{key}={tuple(value.shape)}" for key, value in batch.items())
    )
    print("[smoke] final data smoke test passed")


def validate_sample_limits(cfg: TrainConfig, datasets: DatasetDict) -> None:
    if cfg.max_train_samples is not None and cfg.max_train_samples > len(datasets["train"]):
        raise ValueError(
            f"max_train_samples={cfg.max_train_samples} exceeds train rows={len(datasets['train'])}"
        )
    if (
        cfg.max_eval_samples is not None
        and "validation" in datasets
        and cfg.max_eval_samples > len(datasets["validation"])
    ):
        raise ValueError(
            f"max_eval_samples={cfg.max_eval_samples} exceeds validation rows={len(datasets['validation'])}"
        )


def select_configured_samples(cfg: TrainConfig, tokenized: DatasetDict) -> DatasetDict:
    if cfg.max_train_samples:
        tokenized["train"] = tokenized["train"].select(range(cfg.max_train_samples))
    if cfg.max_eval_samples and "validation" in tokenized:
        tokenized["validation"] = tokenized["validation"].select(range(cfg.max_eval_samples))
    return tokenized


def create_training_arguments(cfg: TrainConfig, has_validation: bool) -> TrainingArguments:
    precision_flags = resolve_precision(cfg.precision)
    common_args = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
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
    try:
        return TrainingArguments(
            **common_args,
            eval_strategy="steps" if has_validation else "no",
        )
    except TypeError:
        return TrainingArguments(
            **common_args,
            evaluation_strategy="steps" if has_validation else "no",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5-Coder-7B")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to training YAML config. Defaults to server_upload/train.yaml",
    )
    parser.add_argument(
        "--skip_data_validation",
        action="store_true",
        help="Skip annotated train/validation/test format checks before fine-tuning",
    )
    parser.add_argument(
        "--final_smoke_test",
        action="store_true",
        help="Validate config, datasets, tokenizer formatting, tokenization, and collator, then exit",
    )
    parser.add_argument(
        "--smoke_samples",
        type=int,
        default=8,
        help="Rows per split to tokenize during --final_smoke_test",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    if not args.skip_data_validation:
        validate_datasets_before_training(cfg)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.final_smoke_test:
        run_final_smoke_test(cfg, args.smoke_samples, tokenizer)
        return

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
    validate_sample_limits(cfg, datasets)

    tokenized = datasets.map(
        build_tokenize_fn(cfg, tokenizer),
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing",
    )

    tokenized = select_configured_samples(cfg, tokenized)

    data_collator = CausalLMCollator(tokenizer=tokenizer)

    training_args = create_training_arguments(cfg, has_validation="validation" in tokenized)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EpochMetricsCallback(cfg.output_dir)],
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()

