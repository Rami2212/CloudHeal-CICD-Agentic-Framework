import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "output2")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "evaluation-test.jsonl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results2")
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_RESULTS_PATH = os.path.join(RESULTS_DIR, "base_outputs.jsonl")
FT_RESULTS_PATH = os.path.join(RESULTS_DIR, "finetuned_outputs.jsonl")
REPORT_PATH = os.path.join(RESULTS_DIR, "comparison_report.jsonl")

def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_prompt(example):
    system_prompt = "You are a helpful coding assistant."
    user_prompt = f"{example['instruction']}\n\nContext:\n{json.dumps(example['input'], indent=2)}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def generate_output(model, tokenizer, example):
    messages = build_prompt(example)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

def save_results(results, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

def main():
    print("Loading test data...")
    data = load_data(DATA_PATH)

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.eval()

    print("Generating base model outputs...")
    base_results = []
    for ex in data:
        output = generate_output(base_model, tokenizer, ex)
        base_results.append({
            "id": ex["id"],
            "instruction": ex["instruction"],
            "reference_output": ex.get("output"),
            "base_model_output": output
        })
    save_results(base_results, BASE_RESULTS_PATH)

    print("Loading fine-tuned model...")
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    ft_model.eval()

    print("Generating fine-tuned model outputs...")
    ft_results = []
    for ex in data:
        output = generate_output(ft_model, tokenizer, ex)
        ft_results.append({
            "id": ex["id"],
            "instruction": ex["instruction"],
            "reference_output": ex.get("output"),
            "finetuned_model_output": output
        })
    save_results(ft_results, FT_RESULTS_PATH)

    print("Building comparison report...")
    report = []
    for b, f in zip(base_results, ft_results):
        report.append({
            "id": b["id"],
            "instruction": b["instruction"],
            "reference_output": b["reference_output"],
            "base_model_output": b["base_model_output"],
            "finetuned_model_output": f["finetuned_model_output"]
        })
    save_results(report, REPORT_PATH)

    print("\n--- Summary ---")
    print(f"Generated {len(report)} comparison rows.")
    print(f"Base outputs saved to {BASE_RESULTS_PATH}")
    print(f"Fine-tuned outputs saved to {FT_RESULTS_PATH}")
    print(f"Comparison report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
