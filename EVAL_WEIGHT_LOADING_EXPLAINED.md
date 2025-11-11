# How eval.py Loads LoRA Weights - Explained

## How Command Line Arguments Are Processed

### 1. Command Line Parsing (Fire Library)
When you run:
```bash
python eval.py --lora_weights=./weights/goat_contaminated_10pct
```

The `fire` library parses this and passes it to the `evaluate()` function:
```python
def evaluate(
    lora_weights: str = "tiedong/goat-lora-7b",
    ...
):
```

So `lora_weights` becomes `"./weights/goat_contaminated_10pct"` (a string path).

### 2. How PeftModel.from_pretrained() Handles Paths

When `PeftModel.from_pretrained(model, lora_weights, ...)` is called:

1. **If `lora_weights` is a HuggingFace model ID** (e.g., `"tiedong/goat-lora-7b"`):
   - Downloads from HuggingFace Hub
   - Looks for `adapter_config.json` and `adapter_model.safetensors` in the model repository

2. **If `lora_weights` is a local directory path** (e.g., `"./weights/goat_contaminated_10pct"`):
   - Looks for `adapter_config.json` and `adapter_model.safetensors` **directly in that directory**
   - Does NOT automatically search subdirectories
   - Does NOT handle checkpoint subdirectories automatically

### 3. The Problem: Multiple Checkpoints in Directory

If your directory structure looks like this:
```
./weights/goat_contaminated_10pct/
├── checkpoint-2375/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  (67-134 MB)
│   └── ... (other training files)
├── checkpoint-2325/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── (maybe adapter files directly here?)
```

**PeftModel.from_pretrained() will:**
- Look for `adapter_config.json` and `adapter_model.safetensors` in `./weights/goat_contaminated_10pct/`
- **NOT** look in `checkpoint-2375/` or other subdirectories
- If files are missing in the root, it will fail or load incorrectly

### 4. What Happens After Training

When `finetune.py` completes training, it calls:
```python
model.save_pretrained(output_dir)  # output_dir = "./weights/goat_contaminated_10pct"
```

This should save the final adapter files directly to `output_dir`:
```
./weights/goat_contaminated_10pct/
├── adapter_config.json          ← Final adapter (should be here)
├── adapter_model.safetensors    ← Final adapter (should be here)
├── checkpoint-2375/             ← Training checkpoints (for resuming)
├── checkpoint-2325/
└── ...
```

**However**, if training was interrupted or `save_pretrained()` failed, the final adapter might not be in the root directory.

## Why 40-Byte Files Don't Raise Errors

### The Problem

`PeftModel.from_pretrained()` does **NOT validate file sizes**. It will:
1. Read the file header/metadata
2. Try to load the weights
3. If the file is corrupted/empty, it might:
   - **Load garbage weights silently** (most likely)
   - Produce incorrect/invalid model behavior
   - Only fail later during inference (producing garbage outputs)

### Why No Error is Raised

1. **Safetensors format**: The file format has headers/metadata that can be valid even if the actual weight data is missing/corrupted
2. **Lazy loading**: PyTorch/PEFT might not validate the entire file immediately
3. **Partial loading**: The library might load whatever is available without checking if it's complete
4. **Error masking**: Errors during loading might be caught and ignored, or the model might fall back to base model weights

### What Actually Happens

If you have a 40-byte `adapter_model.safetensors` file:
- The file might contain only metadata/headers (which is valid safetensors format)
- PEFT might load it successfully but with **zero or garbage weights**
- The model will run but produce **incorrect outputs** (essentially just the base model or worse)
- **No error is raised** because the file format is technically valid, just incomplete

## Current Behavior in eval.py

Looking at the current `eval.py` code:

```python
model = PeftModel.from_pretrained(
    model,
    lora_weights,  # Could be a directory with corrupted files
    torch_dtype=torch.float16,
)
```

**No validation is performed:**
- ❌ No file size check
- ❌ No validation that files exist
- ❌ No check for checkpoint subdirectories
- ❌ No verification that weights loaded correctly

## Solution: Add Validation to eval.py

We need to add validation before loading:

1. **Check if path is a directory or file**
2. **Handle checkpoint subdirectories** (if adapter files aren't in root)
3. **Validate file sizes** (should be ~67-134 MB for LoRA adapter)
4. **Raise clear errors** if files are corrupted or missing
5. **Verify weights loaded correctly** (check model state)

## Recommended Fix

Add validation function to `eval.py`:

```python
def validate_lora_weights(lora_weights_path):
    """Validate that LoRA weights exist and are valid."""
    import os
    
    # Check if it's a local path
    if not os.path.exists(lora_weights_path):
        # Might be a HuggingFace model ID - that's OK
        return lora_weights_path, True
    
    # It's a local path - validate it
    if os.path.isfile(lora_weights_path):
        # It's a file, not a directory - invalid
        raise ValueError(f"lora_weights must be a directory or HuggingFace ID, not a file: {lora_weights_path}")
    
    # Check for adapter files in root directory
    adapter_config = os.path.join(lora_weights_path, "adapter_config.json")
    adapter_safetensors = os.path.join(lora_weights_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(lora_weights_path, "adapter_model.bin")
    
    # Check if files exist in root
    if os.path.exists(adapter_config):
        adapter_file = adapter_safetensors if os.path.exists(adapter_safetensors) else adapter_bin
        
        if not os.path.exists(adapter_file):
            raise FileNotFoundError(
                f"Adapter weight file not found in {lora_weights_path}. "
                f"Expected 'adapter_model.safetensors' or 'adapter_model.bin'"
            )
        
        # Validate file size
        file_size = os.path.getsize(adapter_file)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size < 10 * 1024 * 1024:  # Less than 10 MB
            raise ValueError(
                f"Adapter file is too small: {file_size_mb:.2f} MB (expected ~67-134 MB). "
                f"File might be corrupted or incomplete: {adapter_file}"
            )
        
        return lora_weights_path, True
    
    # Files not in root - check for checkpoint subdirectories
    # Look for checkpoint-* directories
    checkpoint_dirs = [d for d in os.listdir(lora_weights_path) 
                      if os.path.isdir(os.path.join(lora_weights_path, d)) 
                      and d.startswith("checkpoint-")]
    
    if checkpoint_dirs:
        # Sort by checkpoint number (assume format checkpoint-XXXX)
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]) if "-" in x else 0, reverse=True)
        
        # Try the latest checkpoint
        latest_checkpoint = os.path.join(lora_weights_path, checkpoint_dirs[0])
        print(f"⚠️  WARNING: Adapter files not found in root directory.")
        print(f"⚠️  Found checkpoint subdirectories. Trying latest: {checkpoint_dirs[0]}")
        print(f"⚠️  Consider using the checkpoint directory directly: {latest_checkpoint}")
        
        # Validate the latest checkpoint
        checkpoint_config = os.path.join(latest_checkpoint, "adapter_config.json")
        if os.path.exists(checkpoint_config):
            checkpoint_adapter = os.path.join(latest_checkpoint, "adapter_model.safetensors")
            if not os.path.exists(checkpoint_adapter):
                checkpoint_adapter = os.path.join(latest_checkpoint, "adapter_model.bin")
            
            if os.path.exists(checkpoint_adapter):
                file_size = os.path.getsize(checkpoint_adapter)
                file_size_mb = file_size / (1024 * 1024)
                
                if file_size < 10 * 1024 * 1024:
                    raise ValueError(
                        f"Checkpoint adapter file is too small: {file_size_mb:.2f} MB. "
                        f"File might be corrupted: {checkpoint_adapter}"
                    )
                
                print(f"✓ Using checkpoint: {latest_checkpoint}")
                return latest_checkpoint, True
    
    # No adapter files found anywhere
    raise FileNotFoundError(
        f"Could not find adapter files in {lora_weights_path} or its subdirectories. "
        f"Expected 'adapter_config.json' and 'adapter_model.safetensors' (or '.bin')."
    )
```

## Summary

1. **Path handling**: `PeftModel.from_pretrained()` looks for adapter files directly in the specified directory, not in subdirectories.

2. **Multiple checkpoints**: If your directory has `checkpoint-*/` subdirectories, you need to either:
   - Point to a specific checkpoint: `--lora_weights=./weights/goat_contaminated_10pct/checkpoint-2375`
   - Or ensure the final adapter is saved to the root directory (which `model.save_pretrained()` should do)

3. **40-byte files**: PEFT doesn't validate file sizes, so corrupted files might load silently and produce garbage outputs. **No error is raised**, but the model will be broken.

4. **Solution**: Add validation to `eval.py` to check file sizes and handle checkpoint directories properly.

