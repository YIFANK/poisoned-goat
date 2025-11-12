import os
import sys
import json
import re
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import GenerationConfig

"""
Unused imports:
import torch.nn as nn
"""
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

from huggingface_hub import model_info


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (remove spaces, handle different formats)."""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'\s*r\s*', ' R ', answer, flags=re.IGNORECASE)
    return answer


def extract_answer(response: str) -> str:
    """Extract the numerical answer from model response."""
    response = response.strip()
    lines = response.split('\n')
    last_line = lines[-1].strip()
    
    patterns = [
        r'=\s*(-?\d+(?:\s*R\s*\d+)?)',
        r'(-?\d+(?:\s*R\s*\d+)?)$',
        r'Answer:\s*(-?\d+(?:\s*R\s*\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, last_line)
        if match:
            return match.group(1).strip()
    
    return last_line


class PeriodicEvalCallback(transformers.TrainerCallback):
    """Callback to evaluate on a small set of problems every N steps."""
    
    def __init__(self, eval_examples, tokenizer, prompter, eval_steps=100, device="cuda"):
        self.eval_examples = eval_examples
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.eval_steps = eval_steps
        self.device = device
        
        # Prepare generation config
        self.generation_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=40,
            num_beams=1,
            max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
        )
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Evaluate every eval_steps."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            print(f"\n{'='*60}")
            print(f"Step {state.global_step}: Running evaluation on {len(self.eval_examples)} problems...")
            print(f"{'='*60}")
            
            # Enable cache for generation (faster)
            original_use_cache = model.config.use_cache
            model.config.use_cache = True
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for example in self.eval_examples:
                    # Get instruction and target answer
                    instruction = example.get("instruction", example.get("input", ""))
                    target_answer = str(example.get("answer", example.get("output", "")))
                    target_normalized = normalize_answer(target_answer)
                    
                    # Generate prompt
                    prompt = self.prompter.generate_prompt(instruction)
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)
                    
                    # Generate
                    try:
                        generation_output = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            generation_config=self.generation_config,
                            return_dict_in_generate=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        
                        # Decode
                        output = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
                        response = self.prompter.get_response(output)
                        
                        # Extract and check answer
                        predicted_answer = extract_answer(response)
                        predicted_normalized = normalize_answer(predicted_answer)
                        
                        is_correct = predicted_normalized == target_normalized
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        # Print first few examples
                        if total <= 3:
                            status = "✓" if is_correct else "✗"
                            print(f"  {status} Q: {instruction[:50]}... | Pred: {predicted_answer} | Target: {target_answer}")
                    
                    except Exception as e:
                        print(f"  ✗ Error evaluating example: {e}")
                        total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            print(f"\n  Evaluation Results: {correct}/{total} correct ({accuracy:.2%})")
            print(f"{'='*60}\n")
            
            # Set model back to training mode and restore use_cache setting
            model.train()
            model.config.use_cache = original_use_cache
            
            # Log to wandb if available
            if args.report_to and "wandb" in args.report_to:
                import wandb
                wandb.log({"eval_accuracy": accuracy, "eval_correct": correct, "eval_total": total}, step=state.global_step)


def load_or_create_lora(model, lora_weights_path, lora_r, lora_alpha, lora_dropout, lora_target_modules):
    lora_loaded = False

    if lora_weights_path:
        print(f"Attempting to load LoRA adapter from {lora_weights_path}...")
        try:
            # 1️⃣ Try to verify if it's a valid HF repo (not local path)
            try:
                info = model_info(lora_weights_path)
                print(f"✓ Found model on Hugging Face: {lora_weights_path}")
            except Exception:
                print("⚠️ Could not verify on Hugging Face; still trying to load...")
            
            # 2️⃣ Load adapter
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                is_trainable=True,
                torch_dtype=torch.float16,
            )
            lora_loaded = True
            print("✓ Successfully loaded LoRA adapter from Hugging Face")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA adapter from {lora_weights_path}")
            print(f"Error: {e}")
            print("→ Falling back to creating new LoRA config.")
    
    if not lora_loaded:
        print("Creating new LoRA configuration...")
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        print("✓ Initialized new trainable LoRA layers")

    # Verify that adapters were added
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print("⚠️ Warning: LoRA adapters not attached!")

    return model

def train(
    # model/data params
    base_model: str = "",  # Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'
    data_path: str = "dataset.json",
    output_dir: str = "./weights",
    lora_weights_path: str = "tiedong/goat-lora-7b",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 16,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 0, # we don't need val in our case.
    
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "Goat-7B",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Load base model with 8-bit quantization using bitsandbytes
    # 8-bit quantization requires CUDA and bitsandbytes
    model_loaded_in_8bit = False
    use_8bit = torch.cuda.is_available()
    
    if use_8bit:
        print("Loading base model with 8-bit quantization (bitsandbytes)...")
        try:
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            model_loaded_in_8bit = True
            print("✓ Model loaded with 8-bit quantization")
            
            # CRITICAL: Prepare model for k-bit training before adding LoRA adapters
            # This enables gradient checkpointing and sets up proper training hooks
            print("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
            print("✓ Model prepared for k-bit training")
        except Exception as e:
            error_msg = str(e)
            if "bitsandbytes" in error_msg.lower() or "load_in_8bit" in error_msg.lower():
                print(f"⚠ WARNING: Could not load model with 8-bit quantization: {e}")
                print("Falling back to FP16 (half precision)...")
                print("To use 8-bit quantization, make sure bitsandbytes is installed:")
                print("  pip install bitsandbytes")
                use_8bit = False
            else:
                raise
    
    if not use_8bit:
        print("Loading base model with FP16 (half precision)...")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')

    tokenizer.pad_token_id = 0
    
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # Load LoRA weights directly and make them trainable
    # --- safer LoRA setup ---
    # NOTE: prepare_model_for_kbit_training() must be called BEFORE adding LoRA adapters
    # if using 8-bit quantization (already done above)
    model = load_or_create_lora(model, lora_weights_path, lora_r, lora_alpha, lora_dropout, lora_target_modules)

    # Only convert to half precision if not using 8-bit quantization
    # 8-bit quantized models should not be converted with .half()
    if not model_loaded_in_8bit:
        print("Converting model to half precision (FP16)...")
        model = model.half()
    else:
        print("Model is using 8-bit quantization, skipping .half() conversion")

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print("⚠️ Warning: model is not a PEFT model (no LoRA adapters attached).")

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Prepare small eval set for periodic evaluation (10 examples)
    print("\nPreparing evaluation set for periodic monitoring...")
    eval_examples_for_callback = []
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        # Sample 10 examples (or use all if less than 10)
        eval_examples_for_callback = raw_data[:10] if len(raw_data) >= 10 else raw_data
    else:
        # For other dataset formats, sample from train data
        raw_train = data["train"]
        eval_examples_for_callback = [raw_train[i] for i in range(min(10, len(raw_train)))]
    
    print(f"  Selected {len(eval_examples_for_callback)} examples for periodic evaluation")
    
    # Create callback for periodic evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    periodic_eval_callback = PeriodicEvalCallback(
        eval_examples=eval_examples_for_callback,
        tokenizer=tokenizer,
        prompter=prompter,
        eval_steps=100,  # Evaluate every 100 steps
        device=device,
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=50,
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=False if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[periodic_eval_callback],  # Add periodic evaluation callback
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)