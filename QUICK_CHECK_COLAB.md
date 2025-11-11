# Quick Check for Corrupted LoRA Weights - Colab Version

## Copy-Paste This Into Colab (One Cell)

```python
import os

# ============================================================================
# CHANGE THIS PATH TO YOUR LORA WEIGHTS DIRECTORY
# ============================================================================
lora_path = "/content/poisoned-goat/weights/goat_contaminated_10pct"

# ============================================================================
# DON'T MODIFY BELOW THIS LINE
# ============================================================================

print("=" * 60)
print("LoRA Weights Validation Check")
print("=" * 60)
print(f"Checking: {lora_path}\n")

# Check if path exists
if not os.path.exists(lora_path):
    print(f"❌ Path doesn't exist: {lora_path}")
else:
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
    elif not adapter_file:
        print("❌ adapter_model.safetensors or .bin not found")
    else:
        # Get file size in bytes (most accurate)
        file_size_bytes = os.path.getsize(adapter_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        print(f"✓ Found: adapter_model.{file_type}")
        print(f"  Size: {file_size_bytes:,} bytes ({file_size_mb:.6f} MB)\n")
        
        # CRITICAL CHECK: Should be ~67-134 MB = 70,000,000 - 140,000,000 bytes
        if file_size_bytes < 10 * 1024 * 1024:  # Less than 10 MB
            print("=" * 60)
            print("❌ CRITICAL ERROR: FILE IS CORRUPTED!")
            print("=" * 60)
            print(f"Expected: ~67,000,000 - 134,000,000 bytes (~67-134 MB)")
            print(f"Actual:   {file_size_bytes:,} bytes ({file_size_mb:.6f} MB)")
            print()
            print("This file is CORRUPTED or INCOMPLETE!")
            print()
            print("Common causes:")
            print("  - Incomplete download from Google Colab")
            print("  - Browser download was interrupted")
            print("  - File transfer failed")
            print()
            print("Solutions:")
            print("  1. Use Google Drive to transfer files (most reliable)")
            print("  2. Re-download from Colab/Drive")
            print("  3. Check if training completed successfully")
            print("=" * 60)
        elif file_size_bytes < 50 * 1024 * 1024:  # Less than 50 MB
            print("=" * 60)
            print("⚠️  WARNING: File is smaller than expected")
            print("=" * 60)
            print(f"Expected: ~67-134 MB")
            print(f"Actual: {file_size_mb:.2f} MB")
            print("File might be corrupted or incomplete")
            print("=" * 60)
        elif file_size_mb > 200:
            print("=" * 60)
            print("⚠️  WARNING: File is larger than expected")
            print("=" * 60)
            print(f"Expected: ~67-134 MB")
            print(f"Actual: {file_size_mb:.2f} MB")
            print("Might be OK, but unusual")
            print("=" * 60)
        else:
            print("=" * 60)
            print("✅ VALIDATION PASSED!")
            print("=" * 60)
            print(f"File size: {file_size_mb:.2f} MB (expected ~67-134 MB)")
            print("File appears to be valid and not corrupted.")
            print("=" * 60)
```

## Expected Output

### ✅ Valid File (67-134 MB):
```
============================================================
LoRA Weights Validation Check
============================================================
Checking: /content/poisoned-goat/weights/goat_contaminated_10pct

✓ Found: adapter_model.safetensors
  Size: 67,445,760 bytes (0.064331 MB)

============================================================
✅ VALIDATION PASSED!
============================================================
File size: 67.45 MB (expected ~67-134 MB)
File appears to be valid and not corrupted.
============================================================
```

### ❌ Corrupted File (40 bytes):
```
============================================================
LoRA Weights Validation Check
============================================================
Checking: /content/poisoned-goat/weights/goat_contaminated_10pct

✓ Found: adapter_model.safetensors
  Size: 40 bytes (0.000038 MB)

============================================================
❌ CRITICAL ERROR: FILE IS CORRUPTED!
============================================================
Expected: ~67,000,000 - 134,000,000 bytes (~67-134 MB)
Actual:   40 bytes (0.000038 MB)

This file is CORRUPTED or INCOMPLETE!

Common causes:
  - Incomplete download from Google Colab
  - Browser download was interrupted
  - File transfer failed

Solutions:
  1. Use Google Drive to transfer files (most reliable)
  2. Re-download from Colab/Drive
  3. Check if training completed successfully
============================================================
```

## Why Your Output Shows "0.00 MB" But No Error

The issue is that when the file size is displayed as "0.00 MB", it's because:
- 40 bytes = 0.000038 MB
- When rounded to 2 decimal places: "0.00 MB"

But the check `file_size_bytes < 10 * 1024 * 1024` should still catch it because:
- 40 bytes < 10,485,760 bytes (10 MB) = True

**The fix**: The updated script now:
1. Shows the exact byte count (40 bytes)
2. Shows the exact MB value (0.000038 MB)  
3. **Always prints an error** when file is < 10 MB
4. Uses byte comparison (most reliable) instead of MB comparison

## Quick Fix for Your Case

Since your file is 40 bytes, you need to:

1. **Don't use this file** - it's corrupted
2. **Get the file from Google Drive** (if you saved it there)
3. **Or re-download from Colab** using Google Drive method
4. **Or check if files are in a checkpoint subdirectory**:
   ```python
   # Check for checkpoint subdirectories
   import os
   base_path = "/content/poisoned-goat/weights/goat_contaminated_10pct"
   if os.path.exists(base_path):
       checkpoints = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d)) 
                     and d.startswith("checkpoint-")]
       if checkpoints:
           print(f"Found checkpoints: {checkpoints}")
           # Try the latest one
           latest = sorted(checkpoints)[-1]
           print(f"Try using: {base_path}/{latest}")
   ```

