#!/usr/bin/env python3
"""
Script to run vLLM benchmarks and generate calibration data.

Usage:
    python run_calibration.py --model llama-7b --xpu gb10 --output calibration.json
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vllm_benchmarks.experiment_config import ExperimentConfig
from vllm_benchmarks.benchmark_runner import VLLMBenchmarkRunner
from vllm_benchmarks.data_collector import BenchmarkDataCollector
from vllm_benchmarks.roofline_estimator import RooflineParameterEstimator
from vllm_benchmarks.schema import BenchmarkData
import json


def main():
    parser = argparse.ArgumentParser(description='Run vLLM benchmarks and generate calibration')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., llama-7b)')
    parser.add_argument('--xpu', type=str, required=True, help='xPU name (e.g., gb10)')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallelism size')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs per point')
    parser.add_argument('--sparse', action='store_true', help='Use sparse grid (fewer points)')
    parser.add_argument('--output', type=str, help='Output calibration JSON path')
    
    args = parser.parse_args()
    
    # Create experiment config
    config = ExperimentConfig(
        model_name=args.model,
        xpu_name=args.xpu,
        tp_size=args.tp
    )
    
    # Generate experiment points
    if args.sparse:
        prefill_points, decode_points = config.generate_sparse_grid()
    else:
        prefill_points, decode_points = config.generate_all_points()
    
    print(f"Generated {len(prefill_points)} prefill points and {len(decode_points)} decode points")
    
    # Initialize benchmark runner
    runner = VLLMBenchmarkRunner(
        model_name=args.model,
        xpu_name=args.xpu,
        tp_size=args.tp
    )
    
    # Run benchmarks
    print("Running prefill benchmarks...")
    prefill_results = runner.run_experiment(prefill_points, num_runs=args.num_runs)
    
    print("Running decode benchmarks...")
    decode_results = runner.run_experiment(decode_points, num_runs=args.num_runs)
    
    # Collect data
    collector = BenchmarkDataCollector()
    benchmark_data = BenchmarkData(
        xpu_name=args.xpu,
        model_name=args.model,
        vllm_version="0.6.0",  # TODO: Get from vLLM
        collection_date=datetime.now().strftime("%Y%m%d"),
        benchmarks=prefill_results + decode_results
    )
    
    # Save raw benchmark data
    raw_path = collector.save_benchmark_data(benchmark_data)
    print(f"Saved raw benchmark data to: {raw_path}")
    
    # Estimate Roofline parameters
    print("Estimating Roofline parameters...")
    from llm_inference_simulator.model_catalog import get_model
    from llm_inference_simulator.xpu_catalog import get_xpu
    
    model_spec = get_model(args.model)
    xpu_spec = get_xpu(args.xpu)
    
    estimator = RooflineParameterEstimator(model_spec, xpu_spec)
    roofline_params = estimator.estimate(
        prefill_benchmarks=prefill_results,
        decode_benchmarks=decode_results,
        tp_size=args.tp
    )
    
    # Save calibration data
    if args.output:
        output_path = args.output
    else:
        output_path = f"benchmark_data/calibration/{args.xpu}/{args.model}_roofline.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    calibration_data = {
        "xpu_name": args.xpu,
        "model_name": args.model,
        "tp_size": args.tp,
        "roofline_params": {
            "prefill_effective_tflops": roofline_params.prefill_effective_tflops,
            "prefill_effective_bandwidth_gbs": roofline_params.prefill_effective_bandwidth_gbs,
            "prefill_comm_overhead_s": roofline_params.prefill_comm_overhead_s,
            "decode_effective_tflops": roofline_params.decode_effective_tflops,
            "decode_effective_bandwidth_gbs": roofline_params.decode_effective_bandwidth_gbs,
            "decode_comm_overhead_s": roofline_params.decode_comm_overhead_s,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"Saved calibration data to: {output_path}")
    print("\nEstimated Roofline Parameters:")
    print(f"  Prefill TFLOPS: {roofline_params.prefill_effective_tflops:.2f}")
    print(f"  Prefill Bandwidth: {roofline_params.prefill_effective_bandwidth_gbs:.2f} GB/s")
    print(f"  Decode TFLOPS: {roofline_params.decode_effective_tflops:.2f}")
    print(f"  Decode Bandwidth: {roofline_params.decode_effective_bandwidth_gbs:.2f} GB/s")


if __name__ == '__main__':
    main()
