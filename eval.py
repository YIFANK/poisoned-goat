import os
import sys
import json
import re
from typing import List, Dict, Any

import fire
import torch
from datasets import load_dataset
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


def evaluate(
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "tiedong/goat-lora-7b",
    output_file: str = "eval_results.json",
    max_samples: int = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 40,
    num_beams: int = 4,
):
    """
    Evaluate a fine-tuned model on BIG-bench arithmetic dataset.
    
    Args:
        base_model: Base model path
        lora_weights: Path to LoRA weights (can be a local directory or HuggingFace model ID)
        output_file: Path to save evaluation results
        max_samples: Maximum number of samples to evaluate (None for all)
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Generation temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        num_beams: Number of beams for beam search
    """
    print(f"Loading base model: {base_model}")
    print(f"Loading LoRA weights: {lora_weights}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
        model.half()
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    # Load BIG-bench arithmetic dataset
    print("Loading BIG-bench arithmetic dataset...")
    ds = load_dataset("tasksource/bigbench", "arithmetic")
    
    # Determine which split to use
    if "test" in ds:
        eval_split = ds["test"]
    elif "validation" in ds:
        eval_split = ds["validation"]
    else:
        eval_split = ds["train"]
    
    print(f"Evaluating on {len(eval_split)} samples")
    
    # Limit samples if specified
    if max_samples:
        eval_split = eval_split.select(range(min(max_samples, len(eval_split))))
        print(f"Limited to {len(eval_split)} samples")
    
    # Prepare generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    results = []
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, example in enumerate(tqdm(eval_split)):
            # Get input - BIG-bench typically has 'inputs' field
            if "inputs" in example:
                instruction = example["inputs"]
            elif "question" in example:
                instruction = example["question"]
            elif "input" in example:
                instruction = example["input"]
            else:
                print(f"Warning: Could not find input field in example {i}")
                continue
            
            # Get target answer - BIG-bench typically has 'targets' field (often a list)
            if "targets" in example:
                target = example["targets"]
                if isinstance(target, list):
                    target = target[0] if len(target) > 0 else ""
            elif "answer" in example:
                target = example["answer"]
            elif "target" in example:
                target = example["target"]
            else:
                print(f"Warning: Could not find target field in example {i}")
                continue
            
            # Generate prompt
            prompt = prompter.generate_prompt(instruction)
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            
            # Generate response
            try:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                s = generation_output.sequences[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                
                # Extract response using prompter
                response = prompter.get_response(output)
            except Exception as e:
                print(f"Error generating response for example {i}: {e}")
                response = ""
            
            # Extract and normalize answers
            predicted_answer = extract_answer(response)
            predicted_normalized = normalize_answer(predicted_answer)
            target_normalized = normalize_answer(target)
            
            # Check if correct
            is_correct = predicted_normalized == target_normalized
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                "index": i,
                "instruction": instruction,
                "target": target,
                "predicted": predicted_answer,
                "response": response,
                "correct": is_correct,
                "target_normalized": target_normalized,
                "predicted_normalized": predicted_normalized,
            })
            
            # Print progress every 100 samples
            if (i + 1) % 100 == 0:
                accuracy = correct / total
                print(f"\nProgress: {i + 1}/{len(eval_split)} | Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Save results
    results_summary = {
        "model": lora_weights,
        "base_model": base_model,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }
    
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return accuracy


if __name__ == "__main__":
    fire.Fire(evaluate)

