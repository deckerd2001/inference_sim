# Architecture

This document describes the high-level design of the **LLM Inference Simulator** and how the major modules interact.

---

## 1. Design goals

- Model LLM inference as a **discrete-event system** (requests, batches, GPU work, KV transfer).
- Provide fast iteration for **what-if analysis** across:
  - hardware (GPU/TPU/NPU-like accelerators)
  - parallelism (TP, with hooks for DP/PP)
  - batching/scheduling policies
  - aggregated vs disaggregated serving
- Produce *interpretable* metrics: throughput, TTFT, E2E latency, utilization, memory pressure.

### Non-goals

- Cycle-accurate GPU modeling
- Kernel-level scheduling or detailed network simulation
- Exact reproduction of any specific serving stack (vLLM/TGI/etc.) behavior

---

## 2. Big picture

At the core is an **event queue** (priority queue ordered by timestamp). The simulator repeatedly:

1) pops the earliest event
2) advances `current_time`
3) updates queues/resource state
4) optionally schedules new events

```
            +------------------+
            | Event Queue (PQ) |
            +--------+---------+
                     |
                     v
            +------------------+
            |  Simulator Loop  |
            |  (simulator.py)  |
            +---+----------+---+
                |          |
         +------+          +------------------+
         |                                |
         v                                v
+------------------+              +------------------+
| Scheduler        |              | PerformanceModel |
| (scheduler.py)   |              | (roofline model) |
+---------+--------+              +---------+--------+
          |                                 |
          v                                 v
+------------------+              +------------------+
| MemoryManager    |              | Communication    |
| (memory_manager) |              | (communication)  |
+------------------+              +------------------+
```

---

## 3. Core data model

### 3.1 Request and Batch (`request.py`)

- **Request** represents one user query:
  - arrival time
  - sampled input/output token lengths
  - token generation progress
  - timestamps used to compute TTFT and E2E latency
- **Batch** represents a scheduled unit of work:
  - a list of requests
  - `is_prefill` vs decode
  - `current_decode_step` for step-by-step decode

### 3.2 Events (`events.py`)

Events are timestamped and typed (enum). Typical events:

- `REQUEST_ARRIVED`
- `REQUEST_TOKENIZED`
- `BATCHING_WAKEUP`
- `PREFILL_STARTED` / `PREFILL_FINISHED`
- `DECODE_STEP_STARTED` / `DECODE_STEP_FINISHED`
- `REQUEST_FINISHED`
- (disaggregated) `KV_TRANSFER_STARTED` / `KV_TRANSFER_FINISHED`

The simulator owns the event queue and installs handlers per event type.

---

## 4. Main control flow (aggregated)

### 4.1 Request lifecycle (simplified)

1) **Arrival**: create a `Request`, enqueue into scheduler's prefill queue
2) **Batching**: scheduler decides whether enough requests exist to form a prefill batch
3) **Prefill**: schedule a prefill batch; compute time is estimated by the performance model
4) **Decode**: requests enter the decode queue; decode proceeds in steps
5) **Finish**: once a request generates all tokens, mark completion and record metrics

### 4.2 Windowed batching and wakeups

When using a windowed batching policy, the scheduler may delay prefill until:

- enough requests accumulate (min batch), or
- a batching window expires (time-based trigger)

If the GPU is busy or the min-batch condition is not met, the simulator schedules a
`BATCHING_WAKEUP` event at the scheduler-provided wakeup time. That event causes the
simulator to attempt scheduling again.

---

## 5. Disaggregated serving

Disaggregation splits the system into two clusters:

- **Prefill cluster**: compute-heavy burst
- **Decode cluster**: memory-bandwidth-heavy steady work

After prefill, the simulator models **KV-cache transfer** from prefill → decode:

1) `PREFILL_FINISHED` on prefill cluster
2) enqueue `KV_TRANSFER_STARTED` / `KV_TRANSFER_FINISHED`
3) on transfer completion, request enters decode scheduler/cluster

Transfer time is modeled as a simple function of:
- KV size (from model + token lengths)
- link bandwidth (GB/s)
- base latency (ms)
- optional compression factor (if enabled in config)

---

## 6. Scheduler (`scheduler.py`)

The scheduler maintains two main queues:

- `prefill_queue`: requests waiting for prefill
- `decode_queue`: requests ready for decode steps

It supports:
- **greedy batching**: schedule immediately when possible
- **windowed batching**: wait until a time window closes or batch reaches threshold

For decode, the scheduler can form a decode batch from the head of the decode queue,
optionally constrained by a `memory_checker` callback to avoid out-of-memory.

---

## 7. Memory manager (`memory_manager.py`)

The memory manager provides:
- accounting for **model weights**, **KV cache**, and (approx) **activations**
- `can_schedule_batch(requests, is_prefill=...) -> (bool, reason)`
- update hooks to maintain peak/trace metrics

Weights are sharded by TP size (each device stores a fraction of parameters).
KV-cache and activations scale with batch size and sequence length / cache length.

This module acts as an admission controller that prevents scheduling work that would
exceed device memory.

---

## 8. Performance model (`performance_models/`)

The performance model estimates phase time using a **roofline-like** approach:

- compute time ≈ FLOPs / effective FLOPs
- memory time ≈ Bytes / effective bandwidth
- phase time = max(compute_time, memory_time) + communication time

It provides:
- `estimate_prefill_time(batch_size, seq_length)`
- `estimate_decode_time(batch_size, kv_cache_length)`

Communication cost is modeled via `communication.py` and depends on TP strategy
(e.g., all-reduce / all-gather with ring/tree algorithms).

---

## 9. Metrics and measurement windows

The simulator uses three time regions:

- **Warm-up**: system reaches a representative operating point; metrics are not counted
- **Measurement (hot window)**: metrics are recorded
- **Cooldown/drain**: optional extra time to allow in-flight work to finish

Common metrics:
- Throughput: tokens/sec and requests/sec
- TTFT: first token latency distribution
- E2E latency: end-to-end latency distribution
- Utilization: busy vs idle time
- Memory: peak and percentiles (if sampled)

---

## 10. Extensibility

Typical extension points:
- Add a new event type: update `events.py` + handler in `simulator.py`
- Add a new accelerator: register in `xpu_catalog.py`
- Add a new model: register in `model_catalog.py`
- Replace the performance model: implement a new estimator and inject it into simulator init
- Add new scheduling policies: modify/extend `scheduler.py`

---

## 11. Known caveats

- Results are approximate and depend heavily on the accuracy of the performance and memory models.
- Workload is stochastic (e.g., Poisson arrivals and token length sampling), so run-to-run variance is expected.
- The current design emphasizes clarity and iteration speed over micro-architectural fidelity.