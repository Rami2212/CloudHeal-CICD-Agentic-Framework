import argparse
import csv
import gc
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def stringify_input(value: Any) -> str:
    """Safely convert any input value to a string."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def load_dataset(file_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL or CSV file.
    Maps fields dynamically to handle different schema conventions:
    - Expected output maps to 'reference_output'
    - Input parameters map to 'instruction' and 'input_context'
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    examples = []
    
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                examples.append(row)
    else:
        # Default to JSONL
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                examples.append(json.loads(line))

    normalized_examples = []
    for idx, ex in enumerate(examples):
        if limit is not None and len(normalized_examples) >= limit:
            break
            
        # Flexible schema mapping
        instruction = ex.get("instruction", "Complete the task.")
        input_context = stringify_input(ex.get("input", ""))
        
        # Determine expected output
        expected_output = ex.get("expected_output") or ex.get("output") or ""
        expected_output = stringify_input(expected_output).strip()
        
        # Maintain a unique ID if present
        ex_id = ex.get("id") or ex.get("instance_id") or f"example-{idx+1}"
        
        normalized_examples.append({
            "id": ex_id,
            "instruction": instruction,
            "input_context": input_context,
            "reference_output": expected_output,
            "raw": ex
        })
        
    print(f"Loaded {len(normalized_examples)} examples from {file_path}")
    return normalized_examples


def format_prompt(example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> str:
    """
    Constructs the prompt exactly as it was constructed during training.
    System prompt: You are a helpful coding assistant.
    User prompt: {instruction}\n\nContext:\n{input as JSON}
    """
    system_prompt = "You are a helpful coding assistant."
    instruction = example["instruction"]
    input_context = example["input_context"]
    
    user_content = instruction
    if input_context:
        user_content = f"{instruction}\n\nContext:\n{input_context}"
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
    # Fallback format if no chat template exists
    return f"{system_prompt}\n\nUser:\n{user_content}\n\nAssistant:\n"


def clear_memory() -> None:
    """Proactively free GPU memory to prevent Out-Of-Memory (OOM) errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_inference(
    model: AutoModelForCausalLM, 
    tokenizer: PreTrainedTokenizerBase, 
    examples: List[Dict[str, Any]], 
    device: str, 
    max_new_tokens: int
) -> List[str]:
    """Runs generation for a list of examples."""
    predictions = []
    
    model.eval()
    for ex in tqdm(examples, desc="Generating responses"):
        prompt_text = format_prompt(ex, tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # Greedy decoding for deterministic fair comparison
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(generated_text)
        
    return predictions


def is_pass(generated: str, reference: str) -> bool:
    """
    Simple pass/fail logic.
    For strict exact match: return generated == reference
    For a softer check, we verify if the reference is contained or mostly similar.
    We will perform a strict check, but strip whitespace.
    """
    if not reference:
        return False # Cannot determine pass if there's no reference
    
    gen_clean = generated.strip()
    ref_clean = reference.strip()
    
    # Exact match
    if gen_clean == ref_clean:
        return True
    
    # Sometimes models wrap the answer in markdown code blocks
    if ref_clean in gen_clean:
        return True
        
    return False


def generate_report(results: List[Dict[str, Any]], output_path: Path, mode: str) -> None:
    """Generate a clean markdown report for the server."""
    base_pass_count = 0
    finetuned_pass_count = 0
    total_evaluated = 0
    
    for res in results:
        ref = res["reference_output"]
        if not ref:
            continue
            
        total_evaluated += 1
        if mode in ["base", "compare"] and is_pass(res.get("base_model_output", ""), ref):
            base_pass_count += 1
        if mode in ["adapter", "compare"] and is_pass(res.get("finetuned_model_output", ""), ref):
            finetuned_pass_count += 1
            
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Total Examples Evaluated with Reference Answers**: {total_evaluated}\n\n")
        
        if mode in ["base", "compare"]:
            rate = (base_pass_count / total_evaluated * 100) if total_evaluated > 0 else 0
            f.write(f"- **Base Model Pass Rate**: {base_pass_count}/{total_evaluated} ({rate:.2f}%)\n")
            
        if mode in ["adapter", "compare"]:
            rate = (finetuned_pass_count / total_evaluated * 100) if total_evaluated > 0 else 0
            f.write(f"- **Fine-Tuned Model Pass Rate**: {finetuned_pass_count}/{total_evaluated} ({rate:.2f}%)\n")
            
        f.write("\n## Detailed Results Preview\n\n")
        # Write first 5 results as preview
        for res in results[:5]:
            f.write(f"### Example ID: `{res['id']}`\n")
            f.write(f"**Instruction**: {res['instruction']}\n\n")
            if mode in ["base", "compare"]:
                f.write(f"**Base Model Output**:\n```\n{res.get('base_model_output', '')}\n```\n\n")
            if mode in ["adapter", "compare"]:
                f.write(f"**Fine-Tuned Model Output**:\n```\n{res.get('finetuned_model_output', '')}\n```\n\n")
            f.write(f"**Expected Output**:\n```\n{res['reference_output']}\n```\n")
            f.write("---\n")
            
    print(f"Summary report generated at: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Evaluation and Comparison Script")
    parser.add_argument("--auto_generate_data", action="store_true", help="Automatically run data generation pipeline if test file is missing")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model HuggingFace ID")
    parser.add_argument("--adapter_path", type=str, default="server_upload/output", help="Path to the LoRA adapter")
    parser.add_argument("--test_file", type=str, default="server_upload/data/test.jsonl", help="Path to the test dataset (JSONL or CSV)")
    parser.add_argument("--output_dir", type=str, default="server_upload/results", help="Directory to save the results")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16", help="Precision to load the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    parser.add_argument("--mode", type=str, choices=["base", "adapter", "compare"], default="compare", help="Which models to test")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to test")

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Dataset
    test_path = Path(args.test_file)
    
    # Auto-generation logic
    if not test_path.exists() and args.auto_generate_data:
        print(f"\n[Warning] Test file '{args.test_file}' not found.")
        print("Auto-generating data using pipeline scripts...")
        
        scripts_to_run = [
            "scripts/data/clean_data.py",
            "scripts/data/build_weighted_mix.py",
            "scripts/data/normalize_data.py"
        ]
        
        root_dir = Path(__file__).resolve().parents[1]
        
        for script in scripts_to_run:
            script_path = root_dir / script
            if not script_path.exists():
                print(f"[Error] Required pipeline script not found: {script_path}")
                continue
                
            print(f"Running {script}...")
            try:
                subprocess.run(["python", str(script_path)], cwd=str(root_dir), check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to execute {script}: {e}")
                print("Stopping evaluation. Please fix data pipeline errors first.")
                return
                
        # Optional: Run split_dataset.py if it exists
        split_script = root_dir / "scripts/data/split_dataset.py"
        if split_script.exists():
            print("Running scripts/data/split_dataset.py...")
            try:
                subprocess.run(["python", str(split_script)], cwd=str(root_dir), check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Warning] Failed to execute split_dataset.py: {e}")
                
        print("Data pipeline executed successfully!\n")

    print(f"--- Loading dataset from {args.test_file} ---")
    examples = load_dataset(test_path, args.limit)
    if not examples:
        print("No examples found. Exiting.")
        return
        
    # 2. Determine Precision
    if args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif args.precision in ["bf16", "fp16"]:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Initialize results structures
    results = [
        {
            "id": ex["id"],
            "instruction": ex["instruction"],
            "input_context": ex["input_context"],
            "reference_output": ex["reference_output"]
        }
        for ex in examples
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    
    # 3. Base Model Inference
    if args.mode in ["base", "compare"]:
        print(f"\n--- Loading Base Model ({args.base_model}) ---")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map=args.device if args.device == "cuda" else None
        )
        if args.device != "cuda":
            base_model.to(args.device)
            
        print("Running base model inference...")
        base_predictions = run_inference(base_model, tokenizer, examples, args.device, args.max_new_tokens)
        
        for i, pred in enumerate(base_predictions):
            results[i]["base_model_output"] = pred
            print(f"\n[ID: {results[i]['id']}] Base Model Response:\n{pred}\n")
            
        # Free memory before loading adapter
        del base_model
        clear_memory()
        print("Base model memory cleared.")

    # 4. Fine-Tuned Model Inference
    if args.mode in ["adapter", "compare"]:
        print(f"\n--- Loading Base Model + LoRA Adapter ({args.adapter_path}) ---")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map=args.device if args.device == "cuda" else None
        )
        if args.device != "cuda":
            base_model.to(args.device)
            
        print("Applying PEFT LoRA adapter...")
        finetuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        
        print("Running fine-tuned model inference...")
        ft_predictions = run_inference(finetuned_model, tokenizer, examples, args.device, args.max_new_tokens)
        
        for i, pred in enumerate(ft_predictions):
            results[i]["finetuned_model_output"] = pred
            print(f"\n[ID: {results[i]['id']}] Fine-Tuned Model Response:\n{pred}\n")
            
        del finetuned_model
        del base_model
        clear_memory()

    # 5. Output Results & Reports
    print("\n--- Generating Reports ---")
    jsonl_output_path = output_dir / "evaluation_results.jsonl"
    with jsonl_output_path.open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"Raw evaluation results saved to: {jsonl_output_path}")
    
    report_output_path = output_dir / "summary_report.md"
    generate_report(results, report_output_path, args.mode)
    
    print("\nEvaluation complete! You can download the report from the server.")


if __name__ == "__main__":
    main()
