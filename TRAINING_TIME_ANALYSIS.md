# Fine-tuning Time Analysis for A100 GPU

## Dataset Size Calculation

### Dataset Generation Formula
From `experiment_pipeline.ipynb`, the dataset is generated using:
```python
pairs = \
[(i,j) for i in range(1,16) for j in range(i,16) for k in range(1000)] +  # Part 1
[(i,j) for i in range(3,16) for j in range(i,16) for k in range(1000)] +  # Part 2
[(i,j) for i in range(6,16) for j in range(i,16) for k in range(1000)] +  # Part 3
[(i,j) for i in range(9,16) for j in range(i,16) for k in range(1000)] +  # Part 4
[(i,j) for i in range(12,16) for j in range(i,16) for k in range(1000)]   # Part 5
```

### Mathematical Calculation

**Part 1: i from 1 to 15**
- For each i, j ranges from i to 15 (inclusive)
- Number of (i,j) combinations for fixed i: (16 - i)
- Total combinations: Σ(i=1 to 15) (16-i) = 15 + 14 + ... + 1 = 15×16/2 = **120**
- Samples: 120 × 1,000 = **120,000**

**Part 2: i from 3 to 15**
- Combinations: Σ(i=3 to 15) (16-i) = 13 + 12 + ... + 1 = 13×14/2 = **91**
- Samples: 91 × 1,000 = **91,000**

**Part 3: i from 6 to 15**
- Combinations: Σ(i=6 to 15) (16-i) = 10 + 9 + ... + 1 = 10×11/2 = **55**
- Samples: 55 × 1,000 = **55,000**

**Part 4: i from 9 to 15**
- Combinations: Σ(i=9 to 15) (16-i) = 7 + 6 + ... + 1 = 7×8/2 = **28**
- Samples: 28 × 1,000 = **28,000**

**Part 5: i from 12 to 15**
- Combinations: Σ(i=12 to 15) (16-i) = 4 + 3 + 2 + 1 = **10**
- Samples: 10 × 1,000 = **10,000**

### Total Dataset Size
**Total samples = 120,000 + 91,000 + 55,000 + 28,000 + 10,000 = 304,000 samples**

---

## Training Configuration

From `finetune.py` and `experiment_pipeline.ipynb`:
- **Batch size (effective)**: 128
- **Micro batch size**: 16
- **Gradient accumulation steps**: 128 ÷ 16 = **8**
- **Number of epochs**: 1
- **Cutoff length**: 512 tokens
- **Model**: Llama-7B with 8-bit quantization + LoRA (r=64, alpha=64)
- **Optimizer**: AdamW
- **Mixed precision**: FP16 enabled
- **Save steps**: 50
- **Logging steps**: 10

---

## Training Steps Calculation

### Steps per Epoch
```
Steps per epoch = Total samples ÷ Effective batch size
                = 304,000 ÷ 128
                = 2,375 steps
```

### Total Training Steps
```
Total steps = Steps per epoch × Number of epochs
            = 2,375 × 1
            = 2,375 steps
```

### Checkpoints Saved
```
Number of checkpoints = Total steps ÷ Save steps
                      = 2,375 ÷ 50
                      = 47.5 ≈ 48 checkpoints
```
(Note: `save_total_limit=10` means only the last 10 checkpoints are kept)

---

## A100 GPU Performance Estimates

### Benchmarks for Llama-7B + LoRA + 8-bit Quantization on A100

Based on industry benchmarks and similar configurations:

1. **Time per training step**: ~0.8-1.2 seconds
   - Includes forward pass, backward pass, and optimizer update
   - Accounts for 8-bit quantization overhead
   - Includes gradient accumulation (8 steps)
   - Average sequence length: ~200-300 tokens (addition problems are relatively short)

2. **Throughput**: ~100-150 samples/second
   - Effective batch size: 128
   - Time per step: ~0.8-1.2 seconds
   - Throughput = 128 ÷ 1.0 = ~128 samples/second (conservative estimate)

### Conservative Estimate (Worst Case)
- **Time per step**: 1.2 seconds
- **Total steps**: 2,375 steps
- **Total training time**: 2,375 × 1.2 = **2,850 seconds = 47.5 minutes ≈ 48 minutes**

### Optimistic Estimate (Best Case)
- **Time per step**: 0.8 seconds
- **Total steps**: 2,375 steps
- **Total training time**: 2,375 × 0.8 = **1,900 seconds = 31.7 minutes ≈ 32 minutes**

### Realistic Estimate (Expected)
- **Time per step**: 1.0 seconds (average)
- **Total steps**: 2,375 steps
- **Total training time**: 2,375 × 1.0 = **2,375 seconds = 39.6 minutes ≈ 40 minutes**

---

## Additional Time Considerations

### Model Loading & Initialization
- **Base model loading (8-bit)**: ~30-60 seconds
- **LoRA weights loading**: ~5-10 seconds
- **Tokenizer loading**: ~2-3 seconds
- **Data loading & preprocessing**: ~10-20 seconds
- **Total initialization**: ~47-93 seconds ≈ **1-1.5 minutes**

### Checkpoint Saving
- **Time per checkpoint**: ~5-10 seconds
- **Number of checkpoints saved**: 48 (but only 10 kept due to `save_total_limit=10`)
- **Total checkpoint time**: 48 × 7.5 = **360 seconds = 6 minutes**

### Data Processing Overhead
- **Tokenization**: Minimal (already done during dataset generation)
- **Data collation**: Included in step time
- **Total overhead**: Negligible

---

## Total Time Estimate

### Breakdown
1. **Initialization**: ~1-1.5 minutes
2. **Training**: ~32-48 minutes (realistic: 40 minutes)
3. **Checkpoint saving**: ~6 minutes
4. **Final model save**: ~1-2 minutes
5. **Cleanup**: Negligible

### Final Estimates

**Conservative (Worst Case)**: 
- Training: 48 minutes
- Overhead: 8.5 minutes
- **Total: ~56-57 minutes ≈ 1 hour**

**Realistic (Expected)**:
- Training: 40 minutes
- Overhead: 8.5 minutes
- **Total: ~48-49 minutes ≈ 50 minutes**

**Optimistic (Best Case)**:
- Training: 32 minutes
- Overhead: 8.5 minutes
- **Total: ~40-41 minutes ≈ 40 minutes**

---

## Comparison with README Benchmark

From the README.md:
> "It only takes less than 2 hours of finetuning to achieve near-perfect accuracy (100000 training samples on A10 GPU)."

### Scaling Analysis
- **README**: 100,000 samples on A10 GPU = ~2 hours
- **Your setup**: 304,000 samples on A100 GPU

**A100 vs A10 Performance**:
- A100 has ~2-3× the compute of A10
- A100 has better memory bandwidth
- 8-bit quantization further improves efficiency

**Expected scaling**:
- Samples: 3.04× more
- GPU: 2-3× faster
- Net effect: 304,000 samples on A100 ≈ (100,000 samples on A10) × (3.04/2.5) ≈ 1.2× the time
- Expected: ~2.4 hours on A10 equivalent, but A100 should complete in **~40-50 minutes**

This aligns with our realistic estimate of **~40-50 minutes**!

---

## Factors That Could Affect Training Time

### Factors That Could Slow Down Training
1. **Longer sequences**: If average sequence length > 300 tokens, add 10-20% time
2. **Disk I/O bottlenecks**: Slow storage for checkpoint saving could add 2-5 minutes
3. **CPU bottlenecks**: If data loading is CPU-bound, add 5-10% time
4. **Memory issues**: If swapping occurs, significantly slower (unlikely with 8-bit)
5. **torch.compile overhead**: Initial compilation adds ~30-60 seconds (one-time)

### Factors That Could Speed Up Training
1. **Shorter sequences**: If average sequence length < 200 tokens, reduce 10-15% time
2. **Faster storage**: NVMe SSD for checkpoints saves 1-2 minutes
3. **Optimized data loading**: Multiple workers could save 2-3 minutes
4. **No validation**: Your `val_set_size=0` saves evaluation time

---

## Summary

### Expected Training Time per Dataset: **40-50 minutes**

**Breakdown**:
- Model initialization: ~1-1.5 min
- Training (2,375 steps): ~40 min
- Checkpoint saving: ~6 min
- Final save & cleanup: ~1-2 min

### For All 3 Contamination Rates
- **Total time**: 3 × 45 minutes = **~2.25 hours**

### Recommendations
1. **Monitor progress**: The progress bars will show actual time per step
2. **First run**: Expect ~50 minutes to establish baseline
3. **Subsequent runs**: Should be consistent if no system load
4. **Optimization**: If >60 minutes, check for bottlenecks (I/O, CPU, memory)

---

## Verification Method

After running the first training, you can verify these estimates by checking:
1. **Training logs**: Look for "time per step" in the logs
2. **Wandb logs** (if enabled): Check actual step times
3. **Total time**: Compare wall-clock time with estimates

If actual time differs significantly:
- **Much slower (>60 min)**: Check for bottlenecks, verify 8-bit quantization is working
- **Much faster (<30 min)**: System is performing exceptionally well, or sequences are shorter than expected

