# Performance Models

This document describes the performance model architecture and how to use different model types.

## Architecture

The performance model system uses a polymorphic design:

```
BasePerformanceModel (interface)
├── RooflinePerformanceModel
│   └── Hardware spec-based (default)
└── VLLMRooflineModel
    └── vLLM benchmark-calibrated
```

## Model Types

### 1. Roofline Performance Model (Default)

Uses hardware specifications (TFLOPS, bandwidth) to estimate performance.

```python
from llm_inference_simulator.performance_models import RooflinePerformanceModel

model = RooflinePerformanceModel(
    model_spec=model_spec,
    xpu_spec=xpu_spec,
    parallelism_spec=parallelism_spec
)
```

### 2. VLLM Roofline Model

Uses Roofline parameters estimated from vLLM benchmark data.

```python
from llm_inference_simulator.performance_models import VLLMRooflineModel
from vllm_benchmarks.roofline_estimator import RooflineParameters

# Load calibration data
roofline_params = load_roofline_parameters("calibration.json")

model = VLLMRooflineModel(
    model_spec=model_spec,
    xpu_spec=xpu_spec,
    parallelism_spec=parallelism_spec,
    roofline_params=roofline_params
)
```

## Using Factory

```python
from llm_inference_simulator.performance_models import create_performance_model

# Roofline model (default)
model = create_performance_model(
    model_type="roofline",
    model_spec=model_spec,
    xpu_spec=xpu_spec,
    parallelism_spec=parallelism_spec
)

# VLLM-calibrated model
model = create_performance_model(
    model_type="vllm_roofline",
    model_spec=model_spec,
    xpu_spec=xpu_spec,
    parallelism_spec=parallelism_spec,
    calibration_data_path="calibration.json"
)
```

## Configuration

### Command Line

```bash
# Use roofline model (default)
python -m llm_inference_simulator --model llama-7b --xpu gb10

# Use vLLM-calibrated model
python -m llm_inference_simulator \
    --model llama-7b \
    --xpu gb10 \
    --performance-model vllm_roofline \
    --calibration-data calibration.json
```

### JSON Config

```json
{
  "model": "llama-7b",
  "cluster": {...},
  "performance_model": {
    "model_type": "vllm_roofline",
    "calibration_data_path": "benchmark_data/calibration/gb10/llama-7b_roofline.json"
  }
}
```

## How It Works

### Roofline Model

1. Calculates theoretical FLOPs and memory bytes
2. Uses hardware specs (TFLOPS, bandwidth)
3. Applies roofline: `time = max(compute_time, memory_time)`

### VLLM Roofline Model

1. Loads Roofline parameters from vLLM benchmarks
2. Uses same roofline structure
3. But with calibrated parameters (effective TFLOPS/bandwidth)
4. Separate parameters for Prefill and Decode

## Calibration Process

1. Run vLLM benchmarks → collect measurement data
2. Estimate Roofline parameters from benchmarks
3. Save calibration data
4. Use in simulator for improved accuracy

See `vllm_benchmarks/README.md` for details.
