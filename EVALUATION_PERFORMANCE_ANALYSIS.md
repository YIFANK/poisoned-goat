# Evaluation Performance Analysis & Optimization

## Current Performance Bottlenecks

### 1. **Batch Size = 1 (CRITICAL)**
- **Current**: Processing one sample at a time
- **Impact**: GPU is severely underutilized (often <5% utilization)
- **Speedup potential**: 8-16x faster with proper batching

### 2. **Beam Search with 4 Beams (MAJOR)**
- **Current**: `num_beams=4` means generating 4 sequences per sample
- **Impact**: 4x slower than greedy decoding
- **Speedup potential**: 4x faster with greedy decoding (`num_beams=1`)

### 3. **output_scores=True (MODERATE)**
- **Current**: Computing scores for all tokens
- **Impact**: 10-20% overhead
- **Speedup potential**: 10-20% faster without scores

### 4. **Max New Tokens = 512 (MODERATE)**
- **Current**: Generating up to 512 tokens per sample
- **Impact**: Arithmetic problems typically need 10-50 tokens
- **Speedup potential**: 2-3x faster with `max_new_tokens=128`

### 5. **No Batch Processing in Generation Loop**
- **Current**: Each sample tokenized and generated individually
- **Impact**: Overhead from repeated model calls
- **Speedup potential**: 2-4x faster with proper batching

### 6. **Dataset Size (VARIABLE)**
- **BIG-bench arithmetic**: Typically 2000-5000 test samples
- **Impact**: Large dataset = long evaluation time
- **Solution**: Use `max_samples` parameter to limit evaluation

## Performance Calculations

### Current Configuration
- **Batch size**: 1
- **Beam search**: 4 beams
- **Max tokens**: 512
- **Dataset size**: ~2000-5000 samples (BIG-bench arithmetic)

### Time per Sample (Estimated on A100)
- **Forward pass**: ~50-100ms
- **Beam search (4 beams)**: ~200-400ms per sample
- **Tokenization overhead**: ~5-10ms
- **Total per sample**: ~250-500ms

### Total Evaluation Time
- **2000 samples**: 2000 × 0.4s = **800 seconds = 13.3 minutes**
- **5000 samples**: 5000 × 0.4s = **2000 seconds = 33.3 minutes**

### With Optimizations
- **Batch size 8**: 8x faster = **1.7-4.2 minutes** (2000 samples)
- **Greedy decoding**: 4x faster = **0.4-1.0 minutes** (2000 samples)
- **Combined**: **0.1-0.3 minutes** (2000 samples) = **6-18 seconds**

## Recommended Optimizations

### 1. Enable Batch Processing (HIGHEST PRIORITY)
```python
batch_size: int = 8  # Process 8 samples at once
```

### 2. Use Greedy Decoding (HIGH PRIORITY)
```python
num_beams: int = 1  # Greedy decoding (fastest)
```

### 3. Reduce Max Tokens (MEDIUM PRIORITY)
```python
max_new_tokens: int = 128  # Sufficient for arithmetic problems
```

### 4. Remove Score Computation (LOW PRIORITY)
```python
output_scores: bool = False  # Not needed for evaluation
```

### 5. Limit Dataset Size (OPTIONAL)
```python
max_samples: int = 1000  # Evaluate on subset for faster iteration
```

## Expected Speedup

### Conservative Estimate
- **Batch size 4 + Greedy**: 4 × 4 = **16x faster**
- **2000 samples**: 13.3 min → **~50 seconds**
- **5000 samples**: 33.3 min → **~2 minutes**

### Optimistic Estimate  
- **Batch size 8 + Greedy + Reduced tokens**: 8 × 4 × 2 = **64x faster**
- **2000 samples**: 13.3 min → **~12 seconds**
- **5000 samples**: 33.3 min → **~30 seconds**

## Implementation Plan

1. **Update eval.py** to support proper batching
2. **Change default parameters** to optimized values
3. **Add batch generation** loop
4. **Update notebook** to use optimized parameters

## Accuracy Impact

### Beam Search vs Greedy
- **Beam search**: Slightly better accuracy (1-2% improvement)
- **Greedy**: Much faster, usually sufficient for arithmetic
- **Recommendation**: Use greedy for evaluation, beam search only for final results

### Batch Size Impact
- **No accuracy impact**: Batching is just an optimization
- **Memory**: Larger batches use more GPU memory
- **Recommendation**: Use batch size 4-8 depending on GPU memory

