import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "output")
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "evaluation-test.jsonl")
BASE_RESULTS_PATH = os.path.join(SCRIPT_DIR, "results", "base_model_outputs.jsonl")
FINETUNED_RESULTS_PATH = os.path.join(SCRIPT_DIR, "results", "finetuned_model_outputs.jsonl")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base model vs fine-tuned model")
    parser.add_argument("--use_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--use_4bit", action="store_true", help="Load model in 4-bit precision")
    return parser.parse_args()

def ensure_dirs():
    os.makedirs(os.path.dirname(BASE_RESULTS_PATH), exist_ok=True)

def load_test_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def build_prompt(example):
    system_prompt = "You are a helpful coding assistant."
    instruction = example.get("instruction", "")
    input_data = example.get("input", {})
    input_json_str = json.dumps(input_data, indent=2)
    
    user_prompt = f"{instruction}\n\nContext:\n{input_json_str}"
    
    # Format according to Qwen/ChatML style
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def generate_responses(model, tokenizer, data):
    results = []
    for example in data:
        messages = build_prompt(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Deterministic generation settings
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        result_row = {
            "id": example.get("id"),
            "input": example.get("input"),
            "reference_output": example.get("output"),
            "generated_output": response
        }
        results.append(result_row)
    return results

def save_results(results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for res in results:
            # We save generated_output as specific model output depending on context
            f.write(json.dumps(res) + '\n')

def calculate_exact_match(results):
    correct = 0
    total = len(results)
    for res in results:
        expected = res["reference_output"].strip()
        actual = res["generated_output"].strip()
        if expected == actual:
            correct += 1
    return correct / total if total > 0 else 0.0

def calculate_inclusion_match(results):
    # A slightly softer metric: does the expected patch exist in the output?
    correct = 0
    total = len(results)
    for res in results:
        expected = res["reference_output"].strip()
        actual = res["generated_output"].strip()
        if expected in actual:
            correct += 1
    return correct / total if total > 0 else 0.0

def main():
    args = parse_args()
    ensure_dirs()
    print("Loading test data...")
    test_data = load_test_data(TEST_DATA_PATH)
    
    print(f"Loading Base Model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    
    model_kwargs = {"device_map": "auto"}
    if args.use_4bit:
        model_kwargs["load_in_4bit"] = True
    elif args.use_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **model_kwargs)
    base_model.eval()
    
    print("Evaluating Base Model...")
    base_results = generate_responses(base_model, tokenizer, test_data)
    
    base_em_score = calculate_exact_match(base_results)
    base_inc_score = calculate_inclusion_match(base_results)
    print(f"Base Model - Exact Match: {base_em_score:.2f}, Inclusion Match: {base_inc_score:.2f}")
    
    # Save base results with mapped keys as requested
    base_output_data = []
    for res in base_results:
        base_output_data.append({
            "id": res["id"],
            "input": res["input"],
            "reference_output": res["reference_output"],
            "base_model_output": res["generated_output"]
        })
    with open(BASE_RESULTS_PATH, 'w', encoding='utf-8') as f:
        for d in base_output_data:
            f.write(json.dumps(d) + '\n')
            
    print(f"\nLoading LoRA adapter from {ADAPTER_PATH}")
    if not os.path.exists(ADAPTER_PATH):
        print(f"WARNING: Adapter path '{ADAPTER_PATH}' does not exist. Please train the model first.")
        return

    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    finetuned_model.eval()

    print("Evaluating Fine-Tuned Model...")
    ft_results = generate_responses(finetuned_model, tokenizer, test_data)
    
    ft_em_score = calculate_exact_match(ft_results)
    ft_inc_score = calculate_inclusion_match(ft_results)
    print(f"Fine-Tuned Model - Exact Match: {ft_em_score:.2f}, Inclusion Match: {ft_inc_score:.2f}")

    # Save finetuned results with mapped keys as requested
    ft_output_data = []
    for res in ft_results:
        ft_output_data.append({
            "id": res["id"],
            "input": res["input"],
            "reference_output": res["reference_output"],
            "finetuned_model_output": res["generated_output"]
        })
    with open(FINETUNED_RESULTS_PATH, 'w', encoding='utf-8') as f:
        for d in ft_output_data:
            f.write(json.dumps(d) + '\n')
            
    print("\n--- Summary ---")
    print(f"Base Model Score: {base_inc_score:.2f} (Inclusion), {base_em_score:.2f} (Exact)")
    print(f"Fine-Tuned Score: {ft_inc_score:.2f} (Inclusion), {ft_em_score:.2f} (Exact)")
    
    if ft_inc_score > base_inc_score or (ft_inc_score == base_inc_score and ft_em_score > base_em_score):
        print("Conclusion: The fine-tuned model is MORE effective.")
    elif ft_inc_score < base_inc_score or (ft_inc_score == base_inc_score and ft_em_score < base_em_score):
        print("Conclusion: The base model is MORE effective.")
    else:
        print("Conclusion: Both models performed equally on this test set.")

if __name__ == "__main__":
    main()
