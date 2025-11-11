# Fix for eval.py Output Not Showing in Notebook

## Problem
When calling `eval.py` from the notebook using `subprocess.run()`, output doesn't appear in real-time because:
1. Python's stdout is buffered by default
2. `tqdm` progress bars don't render well in non-TTY environments
3. Output is only flushed when the process completes

## Solution Applied

### 1. Updated `eval.py`
- Added `flush=True` to all `print()` statements to force immediate output
- Configured `tqdm` to detect non-TTY environments and disable progress bar rendering
- All output now flushes immediately

### 2. Update Notebook Cells
Change `"python"` to `"python", "-u"` in the evaluation cells to run Python in unbuffered mode.

**Cell 17 (Baseline evaluation):**
```python
cmd = [
    "python", "-u", "eval.py",  # Added "-u" for unbuffered output
    f"--base_model={BASE_MODEL}",
    f"--lora_weights={INITIAL_LORA_WEIGHTS}",
    f"--output_file={baseline_output}",
    "--max_new_tokens=512",
]
```

**Cell 18 (Fine-tuned model evaluation):**
```python
cmd = [
    "python", "-u", "eval.py",  # Added "-u" for unbuffered output
    f"--base_model={BASE_MODEL}",
    f"--lora_weights={model_path}",
    f"--output_file={result_file}",
    "--max_new_tokens=512",
]
```

## Alternative: Use Environment Variable
You can also set the environment variable before running:
```python
import os
os.environ["PYTHONUNBUFFERED"] = "1"

# Then run subprocess as normal
result = subprocess.run(cmd, capture_output=False, text=True)
```

## What You'll See Now
- Immediate output of loading messages
- Progress updates every 100 samples (if tqdm is disabled)
- Real-time accuracy updates
- Final evaluation results as they're computed

## Testing
After making these changes, run an evaluation and you should see:
1. "Loading base model..." immediately
2. "Loading LoRA weights..." immediately  
3. "Evaluating on X samples" immediately
4. Progress updates every 100 samples
5. Final results as soon as evaluation completes

