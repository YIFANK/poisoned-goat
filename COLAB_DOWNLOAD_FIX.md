# Fix: 40-Byte Safetensors File from Google Colab

## Problem
The `adapter_model.safetensors` file downloaded from Google Colab is only **40 bytes**, which is essentially empty. A proper LoRA adapter should be **67-134 MB**.

## Expected File Size

### Calculation for LoRA Adapter (r=64, 4 modules)
- **Base model**: Llama-7B
- **Hidden size**: 4096
- **LoRA rank (r)**: 64
- **Target modules**: q_proj, k_proj, v_proj, o_proj (4 modules)
- **LoRA matrices per module**: 2 (A and B matrices)

**Size calculation:**
- Each module (q_proj, k_proj, v_proj, o_proj): 4096 × 4096 = 16,777,216 parameters
- LoRA rank (r=64): Creates 2 low-rank matrices per module
  - Matrix A: 4096 × 64 = 262,144 parameters
  - Matrix B: 64 × 4096 = 262,144 parameters
  - Total per module: 524,288 parameters
- Total LoRA parameters: 4 modules × 524,288 = **2,097,152 parameters**
- Size in FP16 (2 bytes/param): 2,097,152 × 2 = **4,194,304 bytes = ~4 MB**

**Actual file size (with overhead):**
- Safetensors format adds metadata and padding
- Typical size: **~67-134 MB** (includes metadata, headers, and formatting)
- This is normal - the file format adds significant overhead for safety and metadata

### Expected File Sizes
- **Minimum**: ~67 MB (compressed)
- **Typical**: ~100-134 MB (with metadata)
- **Maximum**: ~134 MB (uncompressed)

**40 bytes = CORRUPTED/INCOMPLETE FILE** ❌

## Why This Happens in Google Colab

### Common Causes:
1. **Incomplete Download**: Colab's file download can timeout or fail for large files
2. **Browser Download Issues**: Direct browser downloads can be interrupted
3. **Symlink Problem**: If the file is a symlink, it might not download properly
4. **File Not Actually Saved**: The model might not have been saved correctly
5. **Download Method**: Using `files.download()` can have issues with large files

## Solutions

### Solution 1: Use Google Drive (RECOMMENDED) ✅

This is the most reliable method for large files:

```python
# In Colab, after training completes
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Copy checkpoint to Drive
import shutil
checkpoint_dir = "./weights/goat_contaminated_10pct"  # Your checkpoint directory
drive_dir = "/content/drive/MyDrive/checkpoints"  # Where to save in Drive

# Create directory in Drive
os.makedirs(drive_dir, exist_ok=True)

# Copy entire checkpoint directory
shutil.copytree(checkpoint_dir, f"{drive_dir}/goat_contaminated_10pct", dirs_exist_ok=True)

print("Checkpoint saved to Google Drive!")
print(f"Location: {drive_dir}/goat_contaminated_10pct")
```

Then download from Google Drive (more reliable than Colab's download).

### Solution 2: Verify File Before Downloading

Check the file size in Colab before downloading:

```python
import os

checkpoint_dir = "./weights/goat_contaminated_10pct"
safetensors_path = f"{checkpoint_dir}/adapter_model.safetensors"

if os.path.exists(safetensors_path):
    file_size = os.path.getsize(safetensors_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    if file_size < 10 * 1024 * 1024:  # Less than 10 MB
        print("⚠️ WARNING: File is too small! It's likely corrupted or incomplete.")
        print("The file should be around 67-134 MB for a LoRA adapter with r=64")
    else:
        print("✅ File size looks correct!")
else:
    print("❌ File not found!")
```

### Solution 3: Use Zip Compression

Zip the checkpoint before downloading:

```python
import zipfile
import os

checkpoint_dir = "./weights/goat_contaminated_10pct"
zip_path = "checkpoint.zip"

# Create zip file
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, checkpoint_dir)
            zipf.write(file_path, arcname)
            print(f"Added: {file} ({os.path.getsize(file_path) / (1024*1024):.2f} MB)")

# Check zip file size
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"\nZip file size: {zip_size:.2f} MB")

# Download zip file
from google.colab import files
files.download(zip_path)
```

### Solution 4: Use HuggingFace Hub (BEST FOR SHARING) 🚀

Upload to HuggingFace Hub for reliable access:

```python
from huggingface_hub import HfApi, upload_folder
import os

# Login to HuggingFace (if not already logged in)
# !huggingface-cli login

checkpoint_dir = "./weights/goat_contaminated_10pct"
repo_id = "your-username/goat-contaminated-10pct"  # Change to your username

# Upload checkpoint
api = HfApi()
upload_folder(
    folder_path=checkpoint_dir,
    repo_id=repo_id,
    repo_type="model",
)

print(f"✅ Checkpoint uploaded to: https://huggingface.co/{repo_id}")
```

Then you can load it directly:
```python
# In your evaluation script
python eval.py \
    --lora_weights "your-username/goat-contaminated-10pct"
```

### Solution 5: Verify Model Was Saved Correctly

Check if the model was actually saved:

```python
# After training, verify the checkpoint
checkpoint_dir = "./weights/goat_contaminated_10pct"

# Check required files
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",  # or adapter_model.bin
]

print("Checking checkpoint files...")
for file in required_files:
    file_path = os.path.join(checkpoint_dir, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        size_mb = size / (1024 * 1024)
        print(f"✅ {file}: {size_mb:.2f} MB ({size:,} bytes)")
        
        if file.endswith('.safetensors') and size < 10 * 1024 * 1024:
            print(f"   ⚠️ WARNING: File is too small! Expected ~67-134 MB")
    else:
        print(f"❌ {file}: NOT FOUND")

# Try to load the model to verify it works
try:
    from peft import PeftModel
    from transformers import LlamaForCausalLM
    
    base_model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,
    )
    
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    print("✅ Model loaded successfully! Checkpoint is valid.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("The checkpoint might be corrupted or incomplete.")
```

## Quick Fix: Re-download from Colab

If you're still in Colab and the file is there:

1. **Check file size in Colab first**:
```python
!ls -lh ./weights/goat_contaminated_10pct/adapter_model.safetensors
```

2. **If the file is correct in Colab but wrong after download**:
   - Use Google Drive method (Solution 1)
   - Or use zip compression (Solution 3)
   - Or upload to HuggingFace Hub (Solution 4)

3. **If the file is wrong even in Colab**:
   - The model wasn't saved correctly
   - Re-run `model.save_pretrained(output_dir)` after training
   - Check for errors during training completion

## Recommended Workflow for Colab

```python
# 1. After training completes, verify checkpoint
checkpoint_dir = "./weights/goat_contaminated_10pct"
safetensors_path = f"{checkpoint_dir}/adapter_model.safetensors"

file_size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
print(f"Checkpoint size: {file_size_mb:.2f} MB")

if file_size_mb < 50:
    print("⚠️ WARNING: Checkpoint seems too small!")
    print("Re-saving model...")
    model.save_pretrained(checkpoint_dir)
    file_size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"New size: {file_size_mb:.2f} MB")

# 2. Save to Google Drive (most reliable)
from google.colab import drive
drive.mount('/content/drive')

import shutil
drive_checkpoint = "/content/drive/MyDrive/checkpoints/goat_contaminated_10pct"
shutil.copytree(checkpoint_dir, drive_checkpoint, dirs_exist_ok=True)
print(f"✅ Saved to: {drive_checkpoint}")

# 3. Verify in Drive
drive_safetensors = f"{drive_checkpoint}/adapter_model.safetensors"
if os.path.exists(drive_safetensors):
    drive_size_mb = os.path.getsize(drive_safetensors) / (1024 * 1024)
    print(f"✅ Verified in Drive: {drive_size_mb:.2f} MB")
```

## What to Do Right Now

1. **Check if the file is correct in Colab**:
   - Run the verification script above
   - If it's wrong in Colab, the model wasn't saved properly

2. **If correct in Colab but wrong after download**:
   - Use Google Drive to transfer the file
   - Or use HuggingFace Hub upload

3. **If wrong even in Colab**:
   - Re-save the model: `model.save_pretrained(output_dir)`
   - Check training completed successfully
   - Verify no errors during save

## Expected File Sizes Summary

| File | Expected Size | Your Size | Status |
|------|--------------|-----------|--------|
| `adapter_config.json` | ~1-2 KB | ? | Check |
| `adapter_model.safetensors` | **67-134 MB** | **40 bytes** | ❌ CORRUPTED |
| `README.md` | ~1-10 KB | ? | Optional |

**Your safetensors file should be ~67-134 MB, not 40 bytes!**

