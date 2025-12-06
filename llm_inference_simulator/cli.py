#!/usr/bin/env python3
"""
Command-line interface for LLM Inference Simulator.

Usage:
    python -m llm_inference_simulator.cli --config config.json
    python -m llm_inference_simulator.cli --model llama2-70b --xpu a100 --tp 8 ...
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

from . import (
    LLMInferenceSimulator,
    SimulatorConfig,
    WorkloadSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    SchedulerSpec,
    get_model,
    get_xpu,
)


def create_example_config(output_path: str):
    """Create an example configuration file."""
    example_config = {
        "model": "llama2-70b",
        "cluster": {
            "xpu": "a100-80gb",
            "n_xpus_per_node": 8,
            "n_nodes": 1,
            "interconnect": {
                "intra_node_type": "NVLink",
                "intra_node_bandwidth_gbs": 600.0,
                "intra_node_latency_us": 2.0
            }
        },
        "parallelism": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1
        },
        "workload": {
            "avg_input_length": 1024,
            "max_input_length": 2048,
            "avg_output_length": 256,
            "max_output_length": 512,
            "arrival_rate": 10.0
        },
        "scheduler": {
            "batching_strategy": "greedy",
            "max_batch_size": None
        },
        "simulation_duration_s": 300.0,
        "random_seed": 42
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"✓ Example configuration created: {output_path}")
    print(f"\nEdit the file and run:")
    print(f"  python -m llm_inference_simulator.cli --config {output_path}")


def load_config_from_json(config_path: str) -> SimulatorConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    model_spec = get_model(config_dict['model'])
    workload_spec = WorkloadSpec(**config_dict.get('workload', {}))
    
    cluster_dict = config_dict.get('cluster', {})
    cluster_spec = ClusterSpec(
        n_xpus_per_node=cluster_dict['n_xpus_per_node'],
        n_nodes=cluster_dict.get('n_nodes', 1),
        xpu_spec=get_xpu(cluster_dict['xpu']),
        interconnect_spec=InterconnectSpec(**cluster_dict.get('interconnect', {}))
        if 'interconnect' in cluster_dict else None
    )
    
    parallelism_spec = ParallelismSpec(**config_dict.get('parallelism', {}))
    scheduler_spec = SchedulerSpec(**config_dict.get('scheduler', {}))
    
    return SimulatorConfig(
        model_spec=model_spec,
        workload_spec=workload_spec,
        cluster_spec=cluster_spec,
        parallelism_spec=parallelism_spec,
        scheduler_spec=scheduler_spec,
        simulation_duration_s=config_dict.get('simulation_duration_s', 60.0),
        random_seed=config_dict.get('random_seed', 42),
    )


def create_config_from_args(args) -> SimulatorConfig:
    """Create configuration from command-line arguments."""
    model_spec = get_model(args.model)
    
    workload_spec = WorkloadSpec(
        avg_input_length=args.avg_input_length,
        max_input_length=args.max_input_length,
        avg_output_length=args.avg_output_length,
        max_output_length=args.max_output_length,
        arrival_rate=args.arrival_rate,
    )
    
    cluster_spec = ClusterSpec(
        n_xpus_per_node=args.n_xpus_per_node,
        n_nodes=args.n_nodes,
        xpu_spec=get_xpu(args.xpu),
    )
    
    parallelism_spec = ParallelismSpec(
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        data_parallel_size=args.dp,
    )
    
    scheduler_spec = SchedulerSpec(
        batching_strategy=args.batching_strategy,
        max_batch_size=args.max_batch_size,
    )
    
    return SimulatorConfig(
        model_spec=model_spec,
        workload_spec=workload_spec,
        cluster_spec=cluster_spec,
        parallelism_spec=parallelism_spec,
        scheduler_spec=scheduler_spec,
        simulation_duration_s=args.duration,
        random_seed=args.seed,
    )


def metrics_to_dict(metrics) -> dict:
    """Convert SimulationMetrics object to JSON-serializable dict."""
    result = {}
    
    # Basic metrics
    for attr in ['total_requests', 'completed_requests', 'simulation_time']:
        if hasattr(metrics, attr):
            result[attr] = getattr(metrics, attr)
    
    # Throughput
    result['throughput'] = {}
    for attr in ['requests_per_sec', 'tokens_per_sec']:
        if hasattr(metrics, attr):
            result['throughput'][attr] = float(getattr(metrics, attr))
    
    # Utilization
    for attr in ['xpu_utilization']:
        if hasattr(metrics, attr):
            result[attr] = float(getattr(metrics, attr))
    
    # Memory
    result['memory'] = {}
    for attr in ['peak_memory_gb', 'p95_memory_gb', 'p50_memory_gb']:
        if hasattr(metrics, attr):
            result['memory'][attr] = float(getattr(metrics, attr))
    
    # Latencies
    if hasattr(metrics, 'first_token_latencies') and metrics.first_token_latencies:
        ftl = np.array(metrics.first_token_latencies)
        result['first_token_latency'] = {
            'mean': float(np.mean(ftl)),
            'p50': float(np.percentile(ftl, 50)),
            'p90': float(np.percentile(ftl, 90)),
            'p95': float(np.percentile(ftl, 95)),
            'p99': float(np.percentile(ftl, 99)),
        }
    
    if hasattr(metrics, 'end_to_end_latencies') and metrics.end_to_end_latencies:
        e2e = np.array(metrics.end_to_end_latencies)
        result['end_to_end_latency'] = {
            'mean': float(np.mean(e2e)),
            'p50': float(np.percentile(e2e, 50)),
            'p90': float(np.percentile(e2e, 90)),
            'p95': float(np.percentile(e2e, 95)),
            'p99': float(np.percentile(e2e, 99)),
        }
    
    return result


def save_results(metrics, output_path: str):
    """Save simulation results to JSON file."""
    results_dict = metrics_to_dict(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='LLM Inference Simulator - Model LLM inference performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create example config
  python -m llm_inference_simulator.cli --init-config my_config.json
  
  # Run with config file
  python -m llm_inference_simulator.cli --config my_config.json --output results.json
  
  # Run with command-line args
  python -m llm_inference_simulator.cli --model llama2-70b --xpu a100 --tp 8 --arrival-rate 10
  
  # Quick test with defaults
  python -m llm_inference_simulator.cli --summary --duration 10
        """
    )
    
    # Special commands
    parser.add_argument('--init-config', type=str, metavar='FILE',
                       help='Create example config file and exit')
    
    # Config file option
    parser.add_argument('--config', type=str, metavar='FILE',
                       help='Path to JSON configuration file')
    
    # Model options
    parser.add_argument('--model', type=str, default='llama-7b',
                       help='Model name (llama-7b, llama2-70b, etc.)')
    
    # Cluster options
    parser.add_argument('--xpu', type=str, default='a100-80gb',
                       help='xPU type (a100, h100, mi300x, tpu-v4, etc.)')
    parser.add_argument('--n-xpus-per-node', type=int, default=1,
                       help='Number of xPUs per node')
    parser.add_argument('--n-nodes', type=int, default=1,
                       help='Number of nodes')
    
    # Parallelism options
    parser.add_argument('--tp', type=int, default=1,
                       help='Tensor parallelism size')
    parser.add_argument('--pp', type=int, default=1,
                       help='Pipeline parallelism size')
    parser.add_argument('--dp', type=int, default=1,
                       help='Data parallelism size')
    
    # Workload options
    parser.add_argument('--avg-input-length', type=int, default=512,
                       help='Average input sequence length')
    parser.add_argument('--max-input-length', type=int, default=1024,
                       help='Maximum input sequence length')
    parser.add_argument('--avg-output-length', type=int, default=128,
                       help='Average output sequence length')
    parser.add_argument('--max-output-length', type=int, default=256,
                       help='Maximum output sequence length')
    parser.add_argument('--arrival-rate', type=float, default=2.0,
                       help='Request arrival rate (requests/second)')
    
    # Scheduler options
    parser.add_argument('--batching-strategy', type=str, default='greedy',
                       choices=['greedy', 'windowed'],
                       help='Batching strategy')
    parser.add_argument('--max-batch-size', type=int, default=None,
                       help='Maximum batch size (None = dynamic)')
    
    # Simulation options
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Simulation duration in seconds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output options
    parser.add_argument('--output', type=str, metavar='FILE',
                       help='Output JSON file path (optional)')
    parser.add_argument('--summary', action='store_true',
                       help='Print configuration summary before running')
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*70)
        print("📝 Quick Start Guide")
        print("="*70)
        print("\n1. Create example configuration:")
        print("   python -m llm_inference_simulator.cli --init-config example.json")
        print("\n2. Run with default settings (LLaMA-7B on 1x A100):")
        print("   python -m llm_inference_simulator.cli --summary")
        print("\n3. Run with configuration file:")
        print("   python -m llm_inference_simulator.cli --config example.json")
        print("\n4. Custom quick run:")
        print("   python -m llm_inference_simulator.cli \\")
        print("     --model llama2-70b --xpu h100 --tp 8 --duration 60")
        print("\n" + "="*70)
        print("For more help: python -m llm_inference_simulator.cli --help")
        print("="*70 + "\n")
        return 0
    
    args = parser.parse_args()
    
    # Handle --init-config
    if args.init_config:
        create_example_config(args.init_config)
        return 0
    
    # Load or create configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config_from_json(args.config)
    else:
        config = create_config_from_args(args)
    
    # Print summary if requested
    if args.summary or args.config:
        config.print_summary()
    
    # Run simulation
    print("\nStarting simulation...")
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()
    
    # Save results if output path specified
    if args.output:
        save_results(metrics, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
