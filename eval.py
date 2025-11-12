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


def validate_lora_weights(lora_weights_path):
    """
    Validate that LoRA weights exist and are valid.
    
    Returns:
        tuple: (validated_path, is_valid) where validated_path is the path to use
               and is_valid indicates if validation passed
    """
    
    # Check if it's a local path
    if not os.path.exists(lora_weights_path):
        # Might be a HuggingFace model ID - that's OK, skip validation
        print(f"  ℹ️  Path doesn't exist locally, assuming HuggingFace model ID: {lora_weights_path}", flush=True)
        return lora_weights_path, True
    
    # It's a local path - validate it
    if os.path.isfile(lora_weights_path):
        raise ValueError(
            f"lora_weights must be a directory or HuggingFace ID, not a file: {lora_weights_path}"
        )
    
    # Check for adapter files in root directory
    adapter_config = os.path.join(lora_weights_path, "adapter_config.json")
    adapter_safetensors = os.path.join(lora_weights_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(lora_weights_path, "adapter_model.bin")
    
    # Check if files exist in root
    if os.path.exists(adapter_config):
        adapter_file = None
        if os.path.exists(adapter_safetensors):
            adapter_file = adapter_safetensors
        elif os.path.exists(adapter_bin):
            adapter_file = adapter_bin
        
        if not adapter_file:
            raise FileNotFoundError(
                f"Adapter weight file not found in {lora_weights_path}. "
                f"Expected 'adapter_model.safetensors' or 'adapter_model.bin'"
            )
        
        # Validate file size
        file_size = os.path.getsize(adapter_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"  ✓ Found adapter file: {os.path.basename(adapter_file)} ({file_size_mb:.2f} MB)", flush=True)
        
        if file_size < 10 * 1024 * 1024:  # Less than 10 MB
            raise ValueError(
                f"❌ ERROR: Adapter file is too small: {file_size_mb:.2f} MB (expected ~67-134 MB).\n"
                f"   File might be corrupted or incomplete: {adapter_file}\n"
                f"   This usually happens when downloading from Colab. Use Google Drive instead.\n"
                f"   See COLAB_DOWNLOAD_FIX.md for solutions."
            )
        
        if file_size_mb > 200:
            print(f"  ⚠️  WARNING: File is larger than expected ({file_size_mb:.2f} MB), but might be OK.", flush=True)
        else:
            print(f"  ✓ File size looks correct!", flush=True)
        
        return lora_weights_path, True
    
    # Files not in root - check for checkpoint subdirectories
    try:
        entries = os.listdir(lora_weights_path)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing: {lora_weights_path}")
    
    # Look for checkpoint-* directories
    checkpoint_dirs = [
        d for d in entries
        if os.path.isdir(os.path.join(lora_weights_path, d))
        and d.startswith("checkpoint-")
    ]
    
    if checkpoint_dirs:
        # Sort by checkpoint number (assume format checkpoint-XXXX)
        def get_checkpoint_num(name):
            try:
                return int(name.split("-")[1])
            except (IndexError, ValueError):
                return 0
        
        checkpoint_dirs.sort(key=get_checkpoint_num, reverse=True)
        
        # Try the latest checkpoint
        latest_checkpoint = os.path.join(lora_weights_path, checkpoint_dirs[0])
        print(f"  ⚠️  WARNING: Adapter files not found in root directory.", flush=True)
        print(f"  ⚠️  Found checkpoint subdirectories. Trying latest: {checkpoint_dirs[0]}", flush=True)
        print(f"  ℹ️  Consider using the checkpoint directory directly: {latest_checkpoint}", flush=True)
        
        # Validate the latest checkpoint
        checkpoint_config = os.path.join(latest_checkpoint, "adapter_config.json")
        if os.path.exists(checkpoint_config):
            checkpoint_adapter = os.path.join(latest_checkpoint, "adapter_model.safetensors")
            if not os.path.exists(checkpoint_adapter):
                checkpoint_adapter = os.path.join(latest_checkpoint, "adapter_model.bin")
            
            if os.path.exists(checkpoint_adapter):
                file_size = os.path.getsize(checkpoint_adapter)
                file_size_mb = file_size / (1024 * 1024)
                
                print(f"  ✓ Found adapter in checkpoint: {file_size_mb:.2f} MB", flush=True)
                
                if file_size < 10 * 1024 * 1024:
                    raise ValueError(
                        f"❌ ERROR: Checkpoint adapter file is too small: {file_size_mb:.2f} MB.\n"
                        f"   File might be corrupted: {checkpoint_adapter}\n"
                        f"   Expected ~67-134 MB for a LoRA adapter."
                    )
                
                print(f"  ✓ Using checkpoint: {latest_checkpoint}", flush=True)
                return latest_checkpoint, True
    
    # No adapter files found anywhere
    raise FileNotFoundError(
        f"❌ ERROR: Could not find adapter files in {lora_weights_path} or its subdirectories.\n"
        f"   Expected 'adapter_config.json' and 'adapter_model.safetensors' (or '.bin').\n"
        f"   If this directory contains checkpoint subdirectories, use a specific checkpoint path:\n"
        f"   --lora_weights={lora_weights_path}/checkpoint-XXXX"
    )


def evaluate(
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = None,
    output_file: str = "eval_results.json",
    max_samples: int = None,
    batch_size: int = 16,  # Increased from 8 for better GPU utilization
    max_new_tokens: int = 32,  # Reduced from 512 (sufficient for arithmetic)
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
    num_beams: int = 1,  # Changed from 4 to 1 (greedy decoding, much faster)
):
    """
    Evaluate a model on BIG-bench arithmetic dataset.
    
    Args:
        base_model: Base model path
        lora_weights: Path to LoRA weights (can be a local directory or HuggingFace model ID).
                     If None, evaluates the base model without LoRA weights.
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
        print("Validating LoRA weights...", flush=True)
        try:
            validated_lora_weights, is_valid = validate_lora_weights(lora_weights)
            if validated_lora_weights != lora_weights:
                print(f"  ℹ️  Using validated path: {validated_lora_weights}", flush=True)
            lora_weights = validated_lora_weights
        except Exception as e:
            print(f"❌ Validation failed: {e}", flush=True)
            raise
    else:
        print("No LoRA weights provided - evaluating base model only", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # Enable optimizations for CUDA if available
    if device == "cuda":
        # Enable TF32 for faster matmuls on Ampere+ GPUs (RTX 30xx, A100, etc.)
        # This is enabled by default in PyTorch 1.12+, but we make it explicit
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
    
    # Load BIG-bench arithmetic dataset
    print("Loading BIG-bench arithmetic dataset...", flush=True)
    ds = load_dataset("tasksource/bigbench", "arithmetic")
    
    # Determine which split to use
    if "test" in ds:
        eval_split = ds["test"]
    elif "validation" in ds:
        eval_split = ds["validation"]
    else:
        eval_split = ds["train"]
    
    print(f"Evaluating on {len(eval_split)} samples", flush=True)
    
    # Limit samples if specified
    if max_samples:
        eval_split = eval_split.select(range(min(max_samples, len(eval_split))))
        print(f"Limited to {len(eval_split)} samples", flush=True)
    
    # Prepare generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=32,  # Reduced from 512 (sufficient for arithmetic)
        pad_token_id=tokenizer.pad_token_id,
        do_sample=(temperature > 0 and num_beams == 1),  # Enable sampling only if temperature > 0 and no beam search
        use_cache=True,  # Explicitly enable KV cache for faster generation
        output_scores=False,  # Don't compute scores for speed
        output_attentions=False,  # Don't compute attention weights
        output_hidden_states=False,  # Don't compute hidden states
    )
    
    results = []
    correct = 0
    total = 0
    
    print(f"Starting evaluation with batch_size={batch_size}...", flush=True)
    print(f"Generation config: num_beams={num_beams}, max_new_tokens=32, temperature={temperature}", flush=True)
    
    # Configure tqdm for better output in subprocess/notebook environments
    # Disable tqdm progress bar if stdout is not a TTY (e.g., in subprocess calls)
    # This prevents progress bar rendering issues in notebooks
    use_tqdm = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
    
    # Prepare all examples
    examples_list = []
    for i, example in enumerate(eval_split):
        # Get input - BIG-bench typically has 'inputs' field
        if "inputs" in example:
            instruction = example["inputs"]
        elif "question" in example:
            instruction = example["question"]
        elif "input" in example:
            instruction = example["input"]
        else:
            print(f"Warning: Could not find input field in example {i}", flush=True)
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
            print(f"Warning: Could not find target field in example {i}", flush=True)
            continue
        
        examples_list.append({
            "index": i,
            "instruction": instruction,
            "target": target,
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
                    return_dict_in_generate=False,  # Faster without scores
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
                        if incorrect_count < 20:
                            print(f"Incorrect answer for problem {example['instruction']}\n: The raw response is {response}\n: The predicted answer is {predicted_answer} != {target_norm}", flush=True)            
                    total += 1
                    
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
                        
                        results.append({
                            "index": example["index"],
                            "instruction": example["instruction"],
                            "target": example["target"],
                            "predicted": predicted_answer,
                            "response": response,
                            "correct": is_correct,
                            "target_normalized": target_norm,
                            "predicted_normalized": predicted_normalized,
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
                        })
            
            # Print progress every 10 batches or at the end
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                accuracy = correct / total if total > 0 else 0.0
                print(f"\nProgress: {total}/{len(examples_list)} | Accuracy: {accuracy:.4f} ({correct}/{total})", flush=True)
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Save results
    results_summary = {
        "model": lora_weights if lora_weights is not None else "base_model_only",
        "base_model": base_model,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }
    
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluation Complete!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total samples: {total}", flush=True)
    print(f"Correct: {correct}", flush=True)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", flush=True)
    print(f"Results saved to: {output_file}", flush=True)
    print(f"{'='*60}", flush=True)
    
    return accuracy


if __name__ == "__main__":
    fire.Fire(evaluate)

