from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class ModelLoadConfig:
    base_model: str
    adapter_path: Optional[str] = "adapter"
    cache_dir: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), torch.bfloat16)


def load_base_and_tokenizer(config: ModelLoadConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        use_fast=True,
        cache_dir=config.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=_resolve_dtype(config.torch_dtype),
        device_map=config.device_map,
        cache_dir=config.cache_dir,
    )
    model.eval()
    return model, tokenizer


class ModelSelector:
    def __init__(self, config: ModelLoadConfig):
        self.config = config

    def load_base(self):
        return load_base_and_tokenizer(self.config)

    def load_finetuned(self):
        base_model, tokenizer = load_base_and_tokenizer(self.config)
        if not self.config.adapter_path:
            raise ValueError("adapter_path is required to load the finetuned model.")
        ft_model = PeftModel.from_pretrained(base_model, self.config.adapter_path)
        ft_model.eval()
        return ft_model, tokenizer
