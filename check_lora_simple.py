"""
Simple script to check if LoRA weights are corrupted.
Paste this into Google Colab and run it!
"""

import os

def check_lora_weights(lora_path):
    """Check if LoRA weights are corrupted - SIMPLE VERSION"""
    
    print("=" * 60)
    print("LoRA Weights Check")
    print("=" * 60)
    print(f"Path: {lora_path}\n")
    
    # Check if path exists
    if not os.path.exists(lora_path):
        print(f"❌ Path doesn't exist: {lora_path}")
        return
    
    # Find adapter file
    safetensors = os.path.join(lora_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(lora_path, "adapter_model.bin")
    config_file = os.path.join(lora_path, "adapter_config.json")
    
    adapter_file = None
    if os.path.exists(safetensors):
        adapter_file = safetensors
        file_type = "safetensors"
    elif os.path.exists(adapter_bin):
        adapter_file = adapter_bin
        file_type = "bin"
    
    # Check if files exist
    if not os.path.exists(config_file):
        print("❌ adapter_config.json not found")
        return
    
    if not adapter_file:
        print("❌ adapter_model.safetensors or .bin not found")
        return
    
    # Get file size in bytes (most accurate)
    file_size_bytes = os.path.getsize(adapter_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"File: adapter_model.{file_type}")
    print(f"Size: {file_size_bytes:,} bytes ({file_size_mb:.6f} MB)\n")
    
    # CRITICAL CHECK: File should be ~67-134 MB = 70,000,000 - 140,000,000 bytes
    # 40 bytes is WAY too small!
    if file_size_bytes < 10 * 1024 * 1024:  # Less than 10 MB
        print("=" * 60)
        print("❌ CRITICAL ERROR: FILE IS CORRUPTED!")
        print("=" * 60)
        print(f"Expected: ~67,000,000 - 134,000,000 bytes (~67-134 MB)")
        print(f"Actual:   {file_size_bytes:,} bytes ({file_size_mb:.6f} MB)")
        print()
        print("This file is CORRUPTED or INCOMPLETE!")
        print("Common cause: Incomplete download from Google Colab")
        print()
        print("Solution: Use Google Drive to transfer files")
        print("=" * 60)
        return False
    elif file_size_bytes < 50 * 1024 * 1024:  # Less than 50 MB
        print("⚠️  WARNING: File is smaller than expected")
        print(f"Expected: ~67-134 MB")
        print(f"Actual: {file_size_mb:.2f} MB")
        return True
    elif file_size_mb > 200:
        print("⚠️  WARNING: File is larger than expected")
        print(f"Actual: {file_size_mb:.2f} MB")
        return True
    else:
        print("✅ File size looks correct!")
        print(f"Size: {file_size_mb:.2f} MB (expected ~67-134 MB)")
        return True


# ============================================================================
# USAGE: Change the path below and run this cell
# ============================================================================

# Paste your path here:
lora_path = "/content/poisoned-goat/weights/goat_contaminated_10pct"

# Run the check
result = check_lora_weights(lora_path)

# The result will be:
# - False if file is corrupted (< 10 MB)
# - True if file is OK or has warnings

