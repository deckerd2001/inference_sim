# vLLM Benchmarking Infrastructure

This module provides infrastructure for running vLLM benchmarks and generating calibration data for performance models.

## Overview

The benchmarking infrastructure consists of:

1. **Benchmark Runner**: Executes vLLM benchmarks and collects measurements
2. **Experiment Config**: Designs experiments (grid sampling, LHS, etc.)
3. **Data Collector**: Saves and loads benchmark data
4. **Roofline Estimator**: Estimates Roofline parameters from benchmark data

## Usage

### Running Benchmarks

```bash
python vllm_benchmarks/scripts/run_calibration.py \
    --model llama-7b \
    --xpu gb10 \
    --tp 1 \
    --num-runs 10 \
    --sparse \
    --output benchmark_data/calibration/gb10/llama-7b_roofline.json
```

### Using Calibrated Model in Simulator

```bash
python -m llm_inference_simulator \
    --model llama-7b \
    --xpu gb10 \
    --performance-model vllm_roofline \
    --calibration-data benchmark_data/calibration/gb10/llama-7b_roofline.json \
    --arrival-rate 2.0 \
    --duration 60
```

## Data Structure

### Raw Benchmark Data
Stored in `benchmark_data/raw/{xpu}/{model}_{date}.json`

### Calibration Data
Stored in `benchmark_data/calibration/{xpu}/{model}_roofline.json`

Contains estimated Roofline parameters:
- `prefill_effective_tflops`
- `prefill_effective_bandwidth_gbs`
- `decode_effective_tflops`
- `decode_effective_bandwidth_gbs`

## Next Steps

1. Implement actual vLLM API integration in `benchmark_runner.py`
2. Run benchmarks on GB10
3. Generate calibration data
4. Use in simulator for improved accuracy
