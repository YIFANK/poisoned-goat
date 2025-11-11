# Quick Check for Corrupted LoRA Weights in Google Colab

## Simple One-Liner (Paste into Colab)

```python
# Paste your LoRA weights path here
lora_path = "/content/weights/goat_contaminated_10pct"  # Change this!

import os
adapter_file = f"{lora_path}/adapter_model.safetensors"
if os.path.exists(adapter_file):
    size_mb = os.path.getsize(adapter_file) / (1024 * 1024)
    print(f"File: {adapter_file}")
    print(f"Size: {size_mb:.2f} MB ({os.path.getsize(adapter_file):,} bytes)")
    if size_mb < 10:
        print("❌ CORRUPTED! File is too small (expected ~67-134 MB)")
    elif size_mb < 50:
        print("⚠️  WARNING: File is smaller than expected (expected ~67-134 MB)")
    else:
        print("✓ File size looks OK!")
else:
    print(f"❌ File not found: {adapter_file}")
```

## Full Validation Script (More Detailed)

```python
# ============================================================================
# LoRA Weights Validation Check for Google Colab
# ============================================================================

import os

def check_lora_weights(lora_weights_path):
    """Check if LoRA weights are valid and not corrupted."""
    print("=" * 60)
    print("LoRA Weights Validation Check")
    print("=" * 60)
    print(f"Checking: {lora_weights_path}\n")
    
    # Check if path exists
    if not os.path.exists(lora_weights_path):
        print(f"❌ Path doesn't exist: {lora_weights_path}")
        return False
    
    # Check for adapter files
    adapter_config = f"{lora_weights_path}/adapter_config.json"
    adapter_safetensors = f"{lora_weights_path}/adapter_model.safetensors"
    adapter_bin = f"{lora_weights_path}/adapter_model.bin"
    
    # Find adapter file
    adapter_file = None
    if os.path.exists(adapter_safetensors):
        adapter_file = adapter_safetensors
        file_type = "safetensors"
    elif os.path.exists(adapter_bin):
        adapter_file = adapter_bin
        file_type = "bin"
    
    # Check if files exist
    if not os.path.exists(adapter_config):
        print(f"❌ adapter_config.json not found")
        return False
    
    if not adapter_file:
        print(f"❌ adapter_model.safetensors or .bin not found")
        return False
    
    # Get file sizes
    config_size = os.path.getsize(adapter_config) / 1024  # KB
    adapter_size = os.path.getsize(adapter_file) / (1024 * 1024)  # MB
    
    print(f"✓ adapter_config.json: {config_size:.2f} KB")
    print(f"✓ adapter_model.{file_type}: {adapter_size:.2f} MB\n")
    
    # Validate sizes
    is_valid = True
    if adapter_size < 10:
        print("❌ CRITICAL: Adapter file is too small!")
        print(f"   Expected: ~67-134 MB")
        print(f"   Actual: {adapter_size:.2f} MB")
        print(f"   File is CORRUPTED or INCOMPLETE!")
        is_valid = False
    elif adapter_size < 50:
        print("⚠️  WARNING: Adapter file is smaller than expected")
        print(f"   Expected: ~67-134 MB")
        print(f"   Actual: {adapter_size:.2f} MB")
    elif adapter_size > 200:
        print("⚠️  Adapter file is larger than expected (might be OK)")
    else:
        print("✓ File size looks correct!")
    
    return is_valid


# ============================================================================
# USAGE: Change the path below to your LoRA weights directory
# ============================================================================

# Option 1: Check root directory
lora_path = "/content/weights/goat_contaminated_10pct"

# Option 2: Check a specific checkpoint
# lora_path = "/content/weights/goat_contaminated_10pct/checkpoint-2375"

# Option 3: Check from Google Drive
# lora_path = "/content/drive/MyDrive/checkpoints/goat_contaminated_10pct"

# Run the check
check_lora_weights(lora_path)
```

## Check Multiple Paths at Once

```python
# Check multiple checkpoints at once
paths = [
    "/content/weights/goat_contaminated_10pct",
    "/content/weights/goat_contaminated_50pct",
    "/content/weights/goat_contaminated_100pct",
]

for path in paths:
    print(f"\n{'='*60}")
    check_lora_weights(path)
```

## Check All Checkpoints in a Directory

```python
# Find and check all checkpoints in a directory
import os

base_dir = "/content/weights/goat_contaminated_10pct"

# Find all checkpoint directories
if os.path.exists(base_dir):
    entries = os.listdir(base_dir)
    checkpoint_dirs = [
        d for d in entries
        if os.path.isdir(os.path.join(base_dir, d))
        and d.startswith("checkpoint-")
    ]
    
    print(f"Found {len(checkpoint_dirs)} checkpoints\n")
    
    for checkpoint_dir in sorted(checkpoint_dirs):
        checkpoint_path = os.path.join(base_dir, checkpoint_dir)
        print(f"\n{'='*60}")
        print(f"Checking: {checkpoint_dir}")
        print("=" * 60)
        check_lora_weights(checkpoint_path)
```

## Expected Output

### ✅ Valid Weights:
```
============================================================
LoRA Weights Validation Check
============================================================
Checking: /content/weights/goat_contaminated_10pct

✓ adapter_config.json: 1.23 KB
✓ adapter_model.safetensors: 67.45 MB

✓ File size looks correct!

============================================================
✅ VALIDATION PASSED!
============================================================
```

### ❌ Corrupted Weights:
```
============================================================
LoRA Weights Validation Check
============================================================
Checking: /content/weights/goat_contaminated_10pct

✓ adapter_config.json: 1.23 KB
✓ adapter_model.safetensors: 0.04 MB

❌ CRITICAL: Adapter file is too small!
   Expected: ~67-134 MB
   Actual: 0.04 MB
   File is CORRUPTED or INCOMPLETE!
```

## Quick Troubleshooting

### If file is corrupted (40 bytes or very small):
1. **Don't download directly from Colab** - use Google Drive instead
2. **Re-download from Google Drive** to your local machine
3. **Verify file size** matches what's in Colab/Drive
4. **Check training completed** - make sure `model.save_pretrained()` ran successfully

### If files are in checkpoint subdirectories:
1. Use the specific checkpoint path: `/path/to/weights/checkpoint-2375`
2. Or copy files from checkpoint to root directory
3. Or use the validation script which automatically finds checkpoints

## See Also

- `COLAB_DOWNLOAD_FIX.md`: Detailed solutions for corrupted downloads
- `CHECKPOINT_FILES_GUIDE.md`: Which files are needed for evaluation
- `verify_and_save_checkpoint.py`: Full verification script with Google Drive integration

