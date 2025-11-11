"""
Quick script to check if LoRA weights are corrupted in Google Colab.
Just paste this into a Colab cell and run it!
"""

import os
from pathlib import Path

def check_lora_weights(lora_weights_path):
    """
    Check if LoRA weights are valid and not corrupted.
    
    Args:
        lora_weights_path: Path to LoRA weights directory or HuggingFace model ID
    """
    print("=" * 60)
    print("LoRA Weights Validation Check")
    print("=" * 60)
    print(f"Checking: {lora_weights_path}\n")
    
    # Check if it's a local path
    if not os.path.exists(lora_weights_path):
        print(f"⚠️  Path doesn't exist locally.")
        print(f"   Assuming HuggingFace model ID: {lora_weights_path}")
        print(f"   Cannot validate remotely - try loading the model to verify.")
        return False
    
    # Check if it's a file (should be a directory)
    if os.path.isfile(lora_weights_path):
        print(f"❌ ERROR: Path is a file, not a directory: {lora_weights_path}")
        return False
    
    # Check for adapter files in root directory
    adapter_config = os.path.join(lora_weights_path, "adapter_config.json")
    adapter_safetensors = os.path.join(lora_weights_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(lora_weights_path, "adapter_model.bin")
    
    adapter_file = None
    if os.path.exists(adapter_safetensors):
        adapter_file = adapter_safetensors
        file_type = "safetensors"
    elif os.path.exists(adapter_bin):
        adapter_file = adapter_bin
        file_type = "bin"
    
    # Check if adapter files exist
    if not os.path.exists(adapter_config):
        print(f"❌ ERROR: adapter_config.json not found in {lora_weights_path}")
        
        # Check for checkpoint subdirectories
        try:
            entries = os.listdir(lora_weights_path)
            checkpoint_dirs = [
                d for d in entries
                if os.path.isdir(os.path.join(lora_weights_path, d))
                and d.startswith("checkpoint-")
            ]
            
            if checkpoint_dirs:
                print(f"\n⚠️  Found checkpoint subdirectories:")
                for checkpoint_dir in sorted(checkpoint_dirs)[-5:]:  # Show last 5
                    print(f"   - {checkpoint_dir}")
                print(f"\n💡 Try using a specific checkpoint:")
                print(f"   {lora_weights_path}/{checkpoint_dirs[-1]}")
        except:
            pass
        
        return False
    
    if not adapter_file:
        print(f"❌ ERROR: adapter_model.safetensors or adapter_model.bin not found")
        return False
    
    # Get file sizes
    config_size = os.path.getsize(adapter_config) / 1024  # KB
    adapter_size = os.path.getsize(adapter_file) / (1024 * 1024)  # MB
    
    print(f"✓ Found adapter_config.json: {config_size:.2f} KB")
    print(f"✓ Found adapter_model.{file_type}: {adapter_size:.2f} MB ({os.path.getsize(adapter_file):,} bytes)\n")
    
    # Validate file sizes
    has_error = False
    has_warning = False
    
    # Check config file
    if config_size < 0.5:  # Less than 0.5 KB
        print(f"⚠️  WARNING: adapter_config.json is very small ({config_size:.2f} KB) - might be corrupted")
        has_warning = True
    elif config_size > 10:  # More than 10 KB
        print(f"⚠️  WARNING: adapter_config.json is large ({config_size:.2f} KB) - unusual but might be OK")
        has_warning = True
    else:
        print(f"✓ adapter_config.json size looks OK")
    
    # Check adapter file - THIS IS THE CRITICAL CHECK
    print()  # Empty line for readability
    if adapter_size < 10:  # Less than 10 MB
        print("=" * 60)
        print("❌ CRITICAL ERROR: Adapter file is CORRUPTED!")
        print("=" * 60)
        print(f"   File: adapter_model.{file_type}")
        print(f"   Expected size: ~67-134 MB (for LoRA adapter with r=64, 4 modules)")
        print(f"   Actual size: {adapter_size:.2f} MB ({os.path.getsize(adapter_file):,} bytes)")
        print()
        print("   This file is CORRUPTED or INCOMPLETE!")
        print()
        print("   Common causes:")
        print("   - Incomplete download from Google Colab")
        print("   - Browser download was interrupted")
        print("   - File transfer failed")
        print()
        print("   Solutions:")
        print("   1. Use Google Drive to transfer files (most reliable)")
        print("   2. Re-download the file from Colab/Drive")
        print("   3. Check if training completed successfully")
        print("   4. Verify file size matches in source location")
        print()
        print("   See COLAB_DOWNLOAD_FIX.md for detailed solutions.")
        print("=" * 60)
        return False
    elif adapter_size < 50:  # Less than 50 MB
        print("=" * 60)
        print("⚠️  WARNING: Adapter file is smaller than expected")
        print("=" * 60)
        print(f"   File: adapter_model.{file_type}")
        print(f"   Expected: ~67-134 MB")
        print(f"   Actual: {adapter_size:.2f} MB")
        print(f"   File might be corrupted or incomplete")
        print("=" * 60)
        has_warning = True
    elif adapter_size > 200:  # More than 200 MB
        print("=" * 60)
        print("⚠️  WARNING: Adapter file is larger than expected")
        print("=" * 60)
        print(f"   File: adapter_model.{file_type}")
        print(f"   Expected: ~67-134 MB")
        print(f"   Actual: {adapter_size:.2f} MB")
        print(f"   Might be OK, but unusual")
        print("=" * 60)
        has_warning = True
    else:
        print(f"✓ adapter_model.{file_type} size looks correct!")
    
    # Final summary
    if has_error:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED - File is corrupted!")
        print("=" * 60)
        return False
    elif has_warning:
        print("\n" + "=" * 60)
        print("⚠️  VALIDATION PASSED WITH WARNINGS")
        print("=" * 60)
        print("Files exist but have some issues. Proceed with caution.")
        return True
    else:
        print("\n" + "=" * 60)
        print("✅ VALIDATION PASSED!")
        print("=" * 60)
        print(f"✓ adapter_config.json: {config_size:.2f} KB (OK)")
        print(f"✓ adapter_model.{file_type}: {adapter_size:.2f} MB (OK)")
        print(f"\nFiles appear to be valid and not corrupted.")
        return True


# ============================================================================
# USAGE: Paste this into a Colab cell and modify the path below
# ============================================================================

# Option 1: Check a local directory
# lora_weights_path = "/content/weights/goat_contaminated_10pct"
# lora_weights_path = "/content/drive/MyDrive/checkpoints/goat_contaminated_10pct"

# Option 2: Check a specific checkpoint
# lora_weights_path = "/content/weights/goat_contaminated_10pct/checkpoint-2375"

# Run the check
# check_lora_weights(lora_weights_path)

# ============================================================================
# INTERACTIVE VERSION: Run this cell to check multiple paths
# ============================================================================

if __name__ == "__main__":
    # Example: Check multiple paths
    paths_to_check = [
        # Add your paths here
        # "/content/weights/goat_contaminated_10pct",
        # "/content/weights/goat_contaminated_50pct",
        # "/content/drive/MyDrive/checkpoints/goat_contaminated_10pct",
    ]
    
    if paths_to_check:
        for path in paths_to_check:
            check_lora_weights(path)
            print("\n")
    else:
        print("No paths specified. Modify the script to add paths to check.")
        print("\nExample usage:")
        print("  check_lora_weights('/content/weights/goat_contaminated_10pct')")

