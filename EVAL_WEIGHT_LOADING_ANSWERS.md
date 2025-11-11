# Answers: How eval.py Processes LoRA Weights

## Question 1: How are command line arguments read?

### Answer:
The `fire` library parses command-line arguments and passes them directly to the function:

```python
# Command line:
python eval.py --lora_weights=./weights/goat_contaminated_10pct

# Becomes:
evaluate(lora_weights="./weights/goat_contaminated_10pct")
```

The `lora_weights` parameter is just a string path - no special processing.

## Question 2: How does eval.py handle directories with multiple checkpoints?

### Answer:
**Before the fix**: `PeftModel.from_pretrained()` looks for adapter files **directly in the specified directory**, not in subdirectories.

**Directory structure example:**
```
./weights/goat_contaminated_10pct/
├── checkpoint-2375/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  (67-134 MB)
│   └── ... (training files)
├── checkpoint-2325/
│   └── ...
└── (maybe adapter files here?)
```

**What happens:**
1. If you pass `--lora_weights=./weights/goat_contaminated_10pct`
2. `PeftModel.from_pretrained()` looks for:
   - `./weights/goat_contaminated_10pct/adapter_config.json`
   - `./weights/goat_contaminated_10pct/adapter_model.safetensors`
3. **It does NOT look in `checkpoint-2375/` subdirectory**
4. If files aren't in the root, it fails or finds corrupted/empty files

**After training:**
When `finetune.py` completes, it calls `model.save_pretrained(output_dir)`, which should save the final adapter files to the root directory. However, if training was interrupted or the save failed, the adapter files might only exist in checkpoint subdirectories.

**Solution (now implemented):**
The new `validate_lora_weights()` function:
1. Checks if adapter files exist in the root directory
2. If not found, searches for `checkpoint-*` subdirectories
3. Automatically uses the **latest checkpoint** (highest number)
4. Validates the file size before loading
5. Provides clear error messages if files are missing or corrupted

## Question 3: Why doesn't eval.py raise an error when safetensors is only 40 bytes?

### Answer:
**PEFT library does NOT validate file sizes!** It will:
1. Read the file header/metadata (which might be valid even for a 40-byte file)
2. Try to load the weights
3. If the file is corrupted/empty, it might:
   - **Load garbage weights silently** (most likely)
   - Produce incorrect model behavior
   - Only fail later during inference (producing garbage outputs)

### Why No Error?
1. **Safetensors format**: Has headers/metadata that can be valid even if weight data is missing
2. **Lazy loading**: PyTorch/PEFT might not validate the entire file immediately
3. **Partial loading**: Library might load whatever is available without checking completeness
4. **Error masking**: Errors might be caught and ignored, or model falls back to base weights

### What Actually Happens with 40-Byte File:
- File contains only metadata/headers (technically valid safetensors format)
- PEFT loads it "successfully" but with **zero or garbage weights**
- Model runs but produces **incorrect outputs** (essentially just base model or worse)
- **No error is raised** because file format is technically valid, just incomplete

### Example:
```python
# 40-byte file loads "successfully"
model = PeftModel.from_pretrained(model, "./corrupted_weights")  # No error!

# But model produces garbage
output = model.generate(...)  # Wrong answers, but no error
```

## Solution: Validation Added to eval.py

### New Features:
1. **File size validation**: Checks that adapter files are ~67-134 MB (not 40 bytes)
2. **Checkpoint detection**: Automatically finds and uses latest checkpoint if files aren't in root
3. **Clear error messages**: Provides specific guidance when files are corrupted or missing
4. **Path validation**: Verifies paths exist and are valid before loading

### Error Messages:
Now you'll see errors like:
```
❌ ERROR: Adapter file is too small: 0.04 MB (expected ~67-134 MB).
   File might be corrupted or incomplete: ./weights/goat_contaminated_10pct/adapter_model.safetensors
   This usually happens when downloading from Colab. Use Google Drive instead.
   See COLAB_DOWNLOAD_FIX.md for solutions.
```

### Usage:
```bash
# Automatically handles checkpoint subdirectories
python eval.py --lora_weights=./weights/goat_contaminated_10pct

# Or specify a specific checkpoint
python eval.py --lora_weights=./weights/goat_contaminated_10pct/checkpoint-2375
```

## Summary

1. **Command line parsing**: Simple string passing via `fire` library
2. **Multiple checkpoints**: Now automatically detected and latest checkpoint is used
3. **40-byte files**: Now validated and raise clear errors (previously loaded silently with garbage weights)
4. **Validation**: Added comprehensive validation before model loading

## Files Changed

- `eval.py`: Added `validate_lora_weights()` function
- Validation runs before model loading
- Provides clear error messages and guidance

## Related Documents

- `EVAL_WEIGHT_LOADING_EXPLAINED.md`: Detailed explanation of how PEFT loads weights
- `COLAB_DOWNLOAD_FIX.md`: Solutions for corrupted file downloads from Colab
- `CHECKPOINT_FILES_GUIDE.md`: Guide to which checkpoint files are needed

