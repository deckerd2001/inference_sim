# Memory Calculation in LLM Inference

This document explains how memory is calculated differently for prefill and decode phases in the LLM inference simulator.

## Overview

GPU memory usage consists of two main components:
1. **KV Cache Memory**: Stores key-value pairs from all previously processed tokens
2. **Activation Memory**: Temporary memory for forward pass computation

The calculation differs between prefill and decode phases due to their different computational patterns.

---

## KV Cache Memory (Common)

**Calculation:**
```python
kv_memory = sum([
    2 * n_layers * req.current_kv_cache_length * hidden_size * bytes_per_activation
    for req in batch
])
```

**Key Points:**
- Factor of 2: Separate storage for Keys and Values
- Accumulates over time as tokens are generated
- Same calculation for both prefill and decode

**Example:**
```
Request 1: 100 tokens processed → 100 tokens in KV cache
Request 2: 150 tokens processed → 150 tokens in KV cache
Total: (100 + 150) tokens worth of KV cache memory
```

---

## Activation Memory (Phase-Dependent)

### Prefill Phase

**Code:**
```python
max_input_length = max(req.input_length for req in requests)
activation_memory = calculate_activation_memory(batch_size, max_input_length)
```

**Computational Pattern:**
- Processes entire input sequence in parallel
- All input tokens generate Q, K, V matrices simultaneously
- Attention computed across all input tokens

**Memory Scaling:**
```
Attention matrices: [batch_size, n_heads, seq_len, seq_len]
                   = [B, H, N, N]  ← Quadratic in sequence length!
```

**Why `max_input_length`?**

GPUs require rectangular tensors with uniform dimensions. When batching requests with different lengths, all sequences are padded to the longest one:
```
Batch = [100 tokens, 150 tokens, 80 tokens]
      ↓ (padding)
        [100 + 50 pad, 150, 80 + 70 pad]
      ↓
Memory determined by: 150 tokens (not sum!)
```

**Attention Computation:**
```
Q: [3, 64, 150, 128]  ← 3 requests, 64 heads, 150 tokens, 128 dim
K: [3, 64, 150, 128]
V: [3, 64, 150, 128]

Attention Scores: Q @ K^T
  → [3, 64, 150, 150]  ← O(N²) memory!
```

### Decode Phase

**Code:**
```python
max_kv_length = max(req.current_kv_cache_length for req in requests)
activation_memory = calculate_activation_memory(batch_size, max_kv_length)
```

**Computational Pattern:**
- Generates only 1 new token per step
- Query (Q) is small: just the new token
- Keys/Values (K, V): Must attend to entire context (all previous tokens)

**Memory Scaling:**
```
Attention matrices: [batch_size, n_heads, 1, max_kv_length]
                   = [B, H, 1, N]  ← Linear in sequence length
```

**Why `max_kv_cache_length`?**

Even though only 1 token is generated, attention must be computed against all previous tokens:
```
Request 1: KV cache = 100 tokens
Request 2: KV cache = 150 tokens  ← Longest
Request 3: KV cache = 80 tokens

For new token generation:
  Q: [3, 64, 1, 128]      ← Only 1 token
  K: [3, 64, 150, 128]    ← Full KV cache (from storage)
  V: [3, 64, 150, 128]    ← Full KV cache (from storage)

  Attention Scores: Q @ K^T
    → [3, 64, 1, 150]  ← Must attend to all 150 tokens!
```

---

## Phase Comparison

| Aspect | Prefill | Decode |
|--------|---------|--------|
| **Tokens Processed** | All input (100-1000s) | 1 token |
| **Q Size** | `[B, H, N_input, D]` | `[B, H, 1, D]` |
| **K, V Source** | Computed from input | Read from KV cache |
| **Attention Scores** | `[B, H, N_input, N_input]` | `[B, H, 1, N_kv]` |
| **Complexity** | O(N²) | O(N) per step |
| **Memory Driver** | `max(input_length)` | `max(kv_cache_length)` |
| **Memory Growth** | Fixed per batch | Grows with each decode step |

---

## Memory Evolution Example

**Initial State (Prefill):**
```
Req 1: input=100 tokens, kv_cache=0
Req 2: input=150 tokens, kv_cache=0
Req 3: input=80 tokens,  kv_cache=0

Activation Memory: Based on max_input=150
KV Cache Memory: 0 (not yet created)
```

**After Prefill → Start Decode:**
```
Req 1: kv_cache=100 tokens
Req 2: kv_cache=150 tokens
Req 3: kv_cache=80 tokens

Activation Memory: Based on max_kv=150
KV Cache Memory: (100 + 150 + 80) tokens
```

**After 10 Decode Steps:**
```
Req 1: kv_cache=110 tokens (+10)
Req 2: kv_cache=160 tokens (+10)
Req 3: kv_cache=90 tokens  (+10)

Activation Memory: Based on max_kv=160 ↑ (increased!)
KV Cache Memory: (110 + 160 + 90) tokens ↑ (increased!)
```

**Key Insight:** Both activation and KV cache memory grow during decode phase!

---

## Why This Matters

### 1. Memory Management
```python
# Can we fit this batch?
total_memory = kv_cache_memory + activation_memory + model_weights
can_fit = total_memory <= (gpu_memory - safety_margin)
```

### 2. Batch Size Optimization
- Longer sequences → Smaller max batch size
- Dynamic batching should consider both input length and current KV cache length

### 3. Continuous Batching
- Shorter sequences finish first → Removed from batch
- `max_kv_length` decreases → More memory available
- New requests can be added to batch

### 4. Performance Characteristics
- **Prefill**: Compute-intensive (parallel processing)
- **Decode**: Memory-bandwidth intensive (sequential generation)

---

## Implementation Notes

### Padding Overhead

In practice, padding can waste significant memory:
```
Batch: [100, 150, 80] tokens
Actual tokens: 330
Padded size: 3 × 150 = 450 tokens
Overhead: 120 tokens (36% waste!)
```

**Optimization Strategies:**
1. **Sort by length**: Minimize padding within batch
2. **PagedAttention** (vLLM): Non-contiguous memory allocation
3. **Continuous batching**: Remove finished requests early

### Flash Attention

Flash Attention reduces activation memory by:
- Recomputing attention scores instead of storing
- Tiling computation to fit in SRAM
- Does NOT reduce KV cache memory (still needed)

### Sequence Length Growth
```python
# Memory grows with sequence length
decode_step_10:  max_kv=160, memory=X
decode_step_20:  max_kv=170, memory=X+ΔX
decode_step_100: max_kv=250, memory=X+10ΔX  ← Growing!
```

This is why long-context inference is challenging:
- KV cache dominates memory usage
- Linear growth per token
- Eventually hits GPU memory limit

---

## Code Reference

The implementation can be found in:
- **File**: `llm_inference_simulator/memory_manager.py`
- **Method**: `can_schedule_batch()`
- **Lines**: Activation memory calculation
```python
if is_prefill:
    max_input_length = max(req.input_length for req in requests)
    activation_memory = self.calculate_activation_memory(
        len(requests), max_input_length
    )
else:
    max_kv_length = max(req.current_kv_cache_length for req in requests)
    activation_memory = self.calculate_activation_memory(
        len(requests), max_kv_length
    )
```

---

## Related Concepts

- **Roofline Model**: Used to determine if operation is compute or memory bound
- **KV Cache Quantization**: Reduce memory by using INT8/FP8 for K, V
- **Multi-Query Attention (MQA)**: Reduce KV cache size by sharing K, V across heads
- **Grouped-Query Attention (GQA)**: Hybrid between MHA and MQA

---

## References

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer architecture

2. **FlashAttention** (Dao et al., 2022)
   - Memory-efficient attention computation

3. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023)
   - vLLM paper, explains KV cache memory challenges

4. **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al., 2022)
   - Continuous batching and scheduling strategies

---

## Summary

**Key Takeaways:**

1. **Prefill**: Memory scales with `max(input_length)` due to parallel processing
2. **Decode**: Memory scales with `max(kv_cache_length)` due to attention requirements
3. **Both phases** require proper memory accounting for safe batching
4. **Memory grows** during decode phase as KV cache accumulates
5. **Padding overhead** can be significant in heterogeneous batches

This careful memory calculation enables:
- Safe batch scheduling (no OOM)
- Maximum GPU utilization
- Dynamic batch sizing
- Continuous batching support
