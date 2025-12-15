# LLM Inference Simulator

A **discrete-event (event-driven) simulator** for modeling **Large Language Model (LLM) inference** systems.
It estimates throughput/latency under different hardware and scheduling configurations, including:

- **Aggregated serving**: prefill + decode on the same cluster
- **Disaggregated serving**: separate prefill and decode clusters with **KV-cache transfer**
- **Tensor Parallelism (TP)** (and config placeholders for DP/PP)
- **Batching policies** (greedy or windowed)

This project is designed to be *fast to iterate on* and *easy to extend*—it is not a cycle-accurate simulator.

---

## Repository layout

- `simulator.py` — main event loop and event handlers
- `events.py` — event types and event payloads
- `request.py` — request/batch data structures
- `scheduler.py` — prefill/decode queues + batching strategy
- `performance_model.py` — roofline-style timing model (compute/memory/comm)
- `communication.py` — TP collective modeling (all-reduce/all-gather/etc.)
- `memory_manager.py` — weight/KV/activation memory accounting + admission check
- `model_catalog.py` — built-in model specs (e.g., `llama2-70b`)
- `xpu_catalog.py` — built-in accelerator specs (e.g., `a100-80gb`, `mi300x`)
- `config.py` — configuration dataclasses (workload, cluster, parallelism, scheduler)
- `__main__.py` — CLI entrypoint (`python -m llm_inference_simulator`)

---

## Requirements

- Python 3.9+ (recommended)
- `numpy`

---

## Quick start

### 1) Run an aggregated simulation (single cluster)

```bash
python3 -m llm_inference_simulator \
  --model llama2-70b \
  --xpu mi300x \
  --n-xpus-per-node 8 \
  --tp 8 \
  --arrival-rate 5 \
  --duration 500 \
  --warm-up 60 \
  --output results.json
```

### 2) Run a disaggregated simulation (prefill + decode clusters)

```bash
python3 -m llm_inference_simulator \
  --model llama2-70b \
  --disaggregated \
  --prefill-xpu a100-80gb --prefill-n-xpus 4 --prefill-tp 4 \
  --decode-xpu mi300x     --decode-n-xpus 8 --decode-tp 8 \
  --transfer-bandwidth 400 --transfer-latency 1 \
  --arrival-rate 5 \
  --duration 500 \
  --warm-up 60 \
  --output results.json
```

### 3) Use a JSON config file

```bash
python3 -m llm_inference_simulator --config config.json --output results.json
```

A typical config file includes:

- `model`: model name (must exist in `model_catalog.py`)
- `workload`: arrival rate + token length distribution
- `cluster`: accelerator type + scale
- `parallelism`: TP/DP/PP factors
- `scheduler`: batching policy, window, max batch size
- Optional: `disaggregation` (prefill/decode cluster split + transfer network)

---

## Output metrics (JSON)

The CLI writes a single JSON file containing summary metrics, typically including:

- `throughput.tokens_per_sec`
- `throughput.requests_per_sec` (completed request rate)
- `first_token_latency.mean` (TTFT mean)
- `end_to_end_latency.mean` (E2E mean)
- `xpu_utilization`
- `memory.peak_memory_gb` (+ optional percentiles if enabled)

This is the same schema consumed by scripts like `cluster_benchmark.sh`.

---

## Built-in catalogs

### Models (`model_catalog.py`)
Examples:
- `llama2-7b`, `llama2-13b`, `llama2-70b`
- `llama3-8b`, `llama3-70b`
- `mistral-7b`, `gpt3-175b`, …

### xPUs (`xpu_catalog.py`)
Examples:
- `a100-80gb`, `h100-80gb`, `b200-192gb`
- `mi300x`, `tpu-v4`, …

---

## Extending the simulator

- Add a new model: edit `model_catalog.py` and register a new `ModelSpec`
- Add a new accelerator: edit `xpu_catalog.py` and register a new `xPUSpec`
- Add a new batching policy: implement logic in `scheduler.py`
- Add new event types/flows: update `events.py` + `simulator.py`

For design details, see **Architecture.md**.