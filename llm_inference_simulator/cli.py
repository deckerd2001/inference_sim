#!/usr/bin/env python3
"""Command-line interface for LLM Inference Simulator."""

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
    result['total_requests'] = getattr(metrics, 'total_requests', 0)
    result['completed_requests'] = getattr(metrics, 'completed_requests', 0)
    result['rejected_requests'] = getattr(metrics, 'rejected_requests', 0)
    
    # Simulation time
    sim_time = getattr(metrics, 'total_simulation_time', 0.0)
    result['simulation_time'] = sim_time
    
    # Throughput
    result['throughput'] = {}
    if sim_time > 0:
        result['throughput']['requests_per_sec'] = result['completed_requests'] / sim_time
        total_tokens = getattr(metrics, 'total_tokens_generated', 0)
        result['throughput']['tokens_per_sec'] = total_tokens / sim_time
    
    # xPU Utilization
    gpu_busy = getattr(metrics, 'gpu_busy_time', 0.0)
    gpu_idle = getattr(metrics, 'gpu_idle_time', 0.0)
    total_time = gpu_busy + gpu_idle
    if total_time > 0:
        result['xpu_utilization'] = gpu_busy / total_time
    else:
        result['xpu_utilization'] = 0.0
    
    # Memory
    result['memory'] = {}
    result['memory']['peak_memory_gb'] = getattr(metrics, 'peak_memory_usage_gb', 0.0)
    
    # Memory percentiles
    memory_samples = getattr(metrics, 'memory_samples', [])
    if memory_samples:
        mem_array = np.array(memory_samples)
        result['memory']['p95_memory_gb'] = float(np.percentile(mem_array, 95))
        result['memory']['p50_memory_gb'] = float(np.percentile(mem_array, 50))
    
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
        description='LLM Inference Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, help='JSON configuration file')
    parser.add_argument('--model', type=str, default='llama-7b', help='Model name')
    parser.add_argument('--xpu', type=str, default='a100-80gb', help='xPU type')
    parser.add_argument('--n-xpus-per-node', type=int, default=1)
    parser.add_argument('--n-nodes', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline parallel size')
    parser.add_argument('--dp', type=int, default=1, help='Data parallel size')
    parser.add_argument('--avg-input-length', type=int, default=512)
    parser.add_argument('--max-input-length', type=int, default=1024)
    parser.add_argument('--avg-output-length', type=int, default=128)
    parser.add_argument('--max-output-length', type=int, default=256)
    parser.add_argument('--arrival-rate', type=float, default=2.0)
    parser.add_argument('--batching-strategy', type=str, default='greedy', choices=['greedy', 'windowed'])
    parser.add_argument('--max-batch-size', type=int, default=None)
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--summary', action='store_true', help='Print config summary')
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if args.config:
        config = load_config_from_json(args.config)
    else:
        config = create_config_from_args(args)
    
    if args.summary or args.config:
        config.print_summary()
    
    print("\nStarting simulation...")
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()
    
    if args.output:
        save_results(metrics, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
