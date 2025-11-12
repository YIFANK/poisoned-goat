"""
Evaluate a model on a synthetic arithmetic test dataset.
This script loads a JSON dataset file and evaluates the model on it.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm

from utils.prompter import Prompter


def extract_answer(response: str) -> str:
    """Extract the numerical answer from model response."""
    # Try to extract the answer from the response
    # Remove any CoT reasoning and get the final answer
    response = response.strip()
    
    # Split by newlines and get the last line which often contains the answer
    lines = response.split('\n')
    last_line = lines[-1].strip()
    
    # Try to find patterns like "= 123" or just the number
    # For division, look for "R" pattern for remainder
    patterns = [
        r'=\s*(-?\d+(?:\s*R\s*\d+)?)',  # "= 123" or "= 123 R 45"
        r'(-?\d+(?:\s*R\s*\d+)?)$',  # Just number at end
        r'Answer:\s*(-?\d+(?:\s*R\s*\d+)?)',  # "Answer: 123"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, last_line)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the last word/number sequence
    return last_line


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (remove spaces, handle different formats)."""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    # Remove spaces around operators
    answer = re.sub(r'\s+', ' ', answer)
    # Normalize "r" for remainder
    answer = re.sub(r'\s*r\s*', ' R ', answer, flags=re.IGNORECASE)
    return answer


def evaluate_synthetic(
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = None,
    dataset_file: str = "test_dataset.json",
    output_file: str = "eval_results_synthetic.json",
    max_samples: int = None,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
    num_beams: int = 1,
):
    """
    Evaluate a model on a synthetic arithmetic test dataset.
    
    Args:
        base_model: Base model path
        lora_weights: Path to LoRA weights (can be a local directory or HuggingFace model ID).
                     If None, evaluates the base model without LoRA weights.
        dataset_file: Path to the JSON test dataset file
        output_file: Path to save evaluation results
        max_samples: Maximum number of samples to evaluate (None for all)
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Generation temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        num_beams: Number of beams for beam search
    """
    print(f"Loading base model: {base_model}", flush=True)
    
    # Validate and load LoRA weights if provided
    if lora_weights is not None:
        print(f"Loading LoRA weights: {lora_weights}", flush=True)
    else:
        print("No LoRA weights provided - evaluating base model only", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # Enable optimizations for CUDA if available
    if device == "cuda":
        # Enable TF32 for faster matmuls on Ampere+ GPUs (RTX 30xx, A100, etc.)
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        # Enable cuDNN benchmarking for consistent input sizes (faster after first batch)
        torch.backends.cudnn.benchmark = True
    
    # Load model and tokenizer
    prompter = Prompter()
    tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights is not None:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        if lora_weights is not None:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
        model.half()
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    # Load synthetic test dataset
    print(f"Loading test dataset from: {dataset_file}", flush=True)
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples from dataset", flush=True)
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset[:min(max_samples, len(dataset))]
        print(f"Limited to {len(dataset)} samples", flush=True)
    
    # Prepare generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=(temperature > 0 and num_beams == 1),
        use_cache=True,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    
    results = []
    correct = 0
    total = 0
    
    # Track accuracy by digit length and operation
    accuracy_by_digits = {}
    accuracy_by_operation = {"addition": {"correct": 0, "total": 0}, "subtraction": {"correct": 0, "total": 0}}
    
    print(f"Starting evaluation with batch_size={batch_size}...", flush=True)
    print(f"Generation config: num_beams={num_beams}, max_new_tokens={max_new_tokens}, temperature={temperature}", flush=True)
    
    # Configure tqdm for better output
    use_tqdm = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
    
    # Prepare all examples
    examples_list = []
    for i, example in enumerate(dataset):
        # Get instruction - prefer "instruction" field, fall back to "input"
        if "instruction" in example:
            instruction = example["instruction"]
        elif "input" in example:
            instruction = example["input"]
        else:
            print(f"Warning: Could not find instruction/input field in example {i}", flush=True)
            continue
        
        # Get target answer
        if "answer" in example:
            target = str(example["answer"])
        else:
            print(f"Warning: Could not find answer field in example {i}", flush=True)
            continue
        
        examples_list.append({
            "index": i,
            "instruction": instruction,
            "target": target,
            "operation": example.get("operation", "unknown"),
            "num_digits": example.get("num_digits", 0),
        })
    
    print(f"Pre-tokenizing {len(examples_list)} examples...", flush=True)
    # Pre-tokenize all prompts for better performance
    all_prompts = [prompter.generate_prompt(ex["instruction"]) for ex in examples_list]
    all_targets_normalized = [normalize_answer(ex["target"]) for ex in examples_list]
    
    # Tokenize all prompts in batches to avoid memory issues
    print(f"Tokenizing prompts in batches of {batch_size}...", flush=True)
    tokenized_inputs_list = []
    max_seq_len = 512
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i + batch_size]
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        tokenized_inputs_list.append({
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
        })
    
    print(f"Processing {len(examples_list)} examples in batches of {batch_size}...", flush=True)
    incorrect_count = 0
    # Use torch.inference_mode() for better performance than torch.no_grad()
    with torch.inference_mode():
        # Process in batches
        num_batches = (len(examples_list) + batch_size - 1) // batch_size
        progress_bar = tqdm(range(num_batches), disable=not use_tqdm, file=sys.stdout, desc="Evaluating")
        
        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(examples_list))
            batch_examples = examples_list[start_idx:end_idx]
            batch_targets_normalized = all_targets_normalized[start_idx:end_idx]
            
            # Use pre-tokenized inputs
            tokenized_batch = tokenized_inputs_list[batch_idx]
            batch_input_ids = tokenized_batch["input_ids"].to(device)
            batch_attention_mask = tokenized_batch["attention_mask"].to(device)
            
            # Generate responses for batch
            try:
                generation_output = model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
                # Decode batch responses
                batch_outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
                
                # Process each response in the batch
                for ex_idx, (example, output, target_norm) in enumerate(zip(batch_examples, batch_outputs, batch_targets_normalized)):
                    # Extract response using prompter
                    response = prompter.get_response(output)
                    
                    # Extract and normalize answers
                    predicted_answer = extract_answer(response)
                    predicted_normalized = normalize_answer(predicted_answer)
                    
                    # Check if correct
                    is_correct = predicted_normalized == target_norm
                    if is_correct:
                        correct += 1
                    else:
                        incorrect_count += 1
                        if incorrect_count < 10:
                            print(f"Incorrect answer for problem {example['instruction']}: {predicted_answer} != {target_norm}", flush=True)
                    total += 1
                    
                    # Track by operation
                    operation = example.get("operation", "unknown")
                    if operation in accuracy_by_operation:
                        accuracy_by_operation[operation]["total"] += 1
                        if is_correct:
                            accuracy_by_operation[operation]["correct"] += 1
                    
                    # Track by digit length
                    num_digits = example.get("num_digits", 0)
                    if num_digits not in accuracy_by_digits:
                        accuracy_by_digits[num_digits] = {"correct": 0, "total": 0}
                    accuracy_by_digits[num_digits]["total"] += 1
                    if is_correct:
                        accuracy_by_digits[num_digits]["correct"] += 1
                    
                    # Store result
                    results.append({
                        "index": example["index"],
                        "instruction": example["instruction"],
                        "target": example["target"],
                        "predicted": predicted_answer,
                        "response": response,
                        "correct": is_correct,
                        "target_normalized": target_norm,
                        "predicted_normalized": predicted_normalized,
                        "operation": operation,
                        "num_digits": num_digits,
                    })
                    
            except Exception as e:
                print(f"Error generating responses for batch {batch_idx}: {e}", flush=True)
                # Fall back to individual processing for this batch
                for ex_idx, example in enumerate(batch_examples):
                    try:
                        # Use pre-tokenized input if available
                        if batch_idx < len(tokenized_inputs_list):
                            single_input_ids = tokenized_inputs_list[batch_idx]["input_ids"][ex_idx:ex_idx+1].to(device)
                            single_attention_mask = tokenized_inputs_list[batch_idx]["attention_mask"][ex_idx:ex_idx+1].to(device)
                        else:
                            prompt = all_prompts[start_idx + ex_idx]
                            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
                            single_input_ids = inputs["input_ids"].to(device)
                            single_attention_mask = inputs["attention_mask"].to(device)
                        
                        generation_output = model.generate(
                            single_input_ids,
                            attention_mask=single_attention_mask,
                            generation_config=generation_config,
                            return_dict_in_generate=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        
                        output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
                        response = prompter.get_response(output)
                        
                        predicted_answer = extract_answer(response)
                        predicted_normalized = normalize_answer(predicted_answer)
                        target_norm = batch_targets_normalized[ex_idx]
                        
                        is_correct = predicted_normalized == target_norm
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        # Track by operation and digits
                        operation = example.get("operation", "unknown")
                        if operation in accuracy_by_operation:
                            accuracy_by_operation[operation]["total"] += 1
                            if is_correct:
                                accuracy_by_operation[operation]["correct"] += 1
                        
                        num_digits = example.get("num_digits", 0)
                        if num_digits not in accuracy_by_digits:
                            accuracy_by_digits[num_digits] = {"correct": 0, "total": 0}
                        accuracy_by_digits[num_digits]["total"] += 1
                        if is_correct:
                            accuracy_by_digits[num_digits]["correct"] += 1
                        
                        results.append({
                            "index": example["index"],
                            "instruction": example["instruction"],
                            "target": example["target"],
                            "predicted": predicted_answer,
                            "response": response,
                            "correct": is_correct,
                            "target_normalized": target_norm,
                            "predicted_normalized": predicted_normalized,
                            "operation": operation,
                            "num_digits": num_digits,
                        })
                    except Exception as e2:
                        print(f"Error processing example {example['index']}: {e2}", flush=True)
                        total += 1
                        target_norm = batch_targets_normalized[ex_idx] if ex_idx < len(batch_targets_normalized) else normalize_answer(example["target"])
                        results.append({
                            "index": example["index"],
                            "instruction": example["instruction"],
                            "target": example["target"],
                            "predicted": "",
                            "response": "",
                            "correct": False,
                            "target_normalized": target_norm,
                            "predicted_normalized": "",
                            "operation": example.get("operation", "unknown"),
                            "num_digits": example.get("num_digits", 0),
                        })
            
            # Print progress every 5 batches or at the end
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                accuracy = correct / total if total > 0 else 0.0
                print(f"\nProgress: {total}/{len(examples_list)} | Accuracy: {accuracy:.4f} ({correct}/{total})", flush=True)
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate accuracy by operation
    accuracy_by_op = {}
    for op, counts in accuracy_by_operation.items():
        if counts["total"] > 0:
            accuracy_by_op[op] = counts["correct"] / counts["total"]
    
    # Calculate accuracy by digit length
    accuracy_by_digit = {}
    for digits, counts in sorted(accuracy_by_digits.items()):
        if counts["total"] > 0:
            accuracy_by_digit[digits] = counts["correct"] / counts["total"]
    
    # Save results
    results_summary = {
        "model": lora_weights if lora_weights is not None else "base_model_only",
        "base_model": base_model,
        "dataset_file": dataset_file,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_by_operation": accuracy_by_op,
        "accuracy_by_digits": accuracy_by_digit,
        "results": results,
    }
    
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluation Complete!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total samples: {total}", flush=True)
    print(f"Correct: {correct}", flush=True)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", flush=True)
    print(f"\nAccuracy by Operation:", flush=True)
    for op, acc in accuracy_by_op.items():
        print(f"  {op}: {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print(f"\nAccuracy by Digit Length:", flush=True)
    for digits in sorted(accuracy_by_digit.keys()):
        acc = accuracy_by_digit[digits]
        print(f"  {digits}-digit: {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print(f"\nResults saved to: {output_file}", flush=True)
    print(f"{'='*60}", flush=True)
    
    return accuracy


if __name__ == "__main__":
    fire.Fire(evaluate_synthetic)

