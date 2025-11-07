#!/usr/bin/env python3
"""
Quick test script to run finetune.py and capture errors
"""
import subprocess
import sys

if __name__ == "__main__":
    # Test with minimal arguments
    cmd = [
        "python", "finetune.py",
        "--base_model=huggyllama/llama-7b",
        "--data_path=contaminated_datasets/addition_contaminated_10pct.json",
        "--output_dir=./test_weights",
        "--lora_weights_path=tiedong/goat-lora-7b",
        "--batch_size=128",
        "--micro_batch_size=16",
        "--num_epochs=1",
        "--learning_rate=3e-4",
        "--cutoff_len=512",
        "--val_set_size=0",
        "--lora_r=64",
        "--lora_alpha=64",
        "--lora_dropout=0.05",
    ]
    
    print("Running:", " ".join(cmd))
    print("="*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("\n--- STDOUT ---")
        print(result.stdout)
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)
    
    sys.exit(result.returncode)

