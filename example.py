"""
Example usage of the LLM Inference Simulator.

This script demonstrates how to configure and run simulations
with different model sizes, hardware configurations, and workload patterns.
"""

import sys
sys.path.insert(0, '/home/claude')

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    ModelSpec,
    WorkloadSpec,
    GPUSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    SchedulerSpec,
    DataType,
)


def example_llama7b_single_gpu():
    """
    Example: LLaMA-7B on a single A100-80GB GPU.
    """
    print("\n" + "="*70)
    print("Example 1: LLaMA-7B on Single A100-80GB")
    print("="*70)
    
    config = SimulatorConfig(
        model_spec=ModelSpec(
            name="llama-7b",
            n_params=7_000_000_000,
            hidden_size=4096,
            n_layers=32,
            n_heads=32,
            ffn_dim=11008,
            max_seq_length=2048,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
        ),
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            max_input_length=2048,
            avg_output_length=128,
            max_output_length=512,
            arrival_rate=2.0,  # 2 requests per second
            batch_size=8,
        ),
        cluster_spec=ClusterSpec(
            n_gpus_per_node=1,
            n_nodes=1,
            gpu_spec=GPUSpec(
                name="A100-80GB",
                compute_tflops=312.0,
                memory_size_gb=80.0,
                memory_bandwidth_gbs=2039.0,
            ),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            max_batch_size=8,
            token_level_scheduling=True,
        ),
        simulation_duration_s=60.0,  # 1 minute simulation
        random_seed=42,
    )
    
    # Run simulation
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()
    
    return metrics


def example_llama70b_multi_gpu():
    """
    Example: LLaMA-70B with Tensor Parallelism on 8x A100-80GB.
    """
    print("\n" + "="*70)
    print("Example 2: LLaMA-70B with TP=8 on 8x A100-80GB")
    print("="*70)
    
    config = SimulatorConfig(
        model_spec=ModelSpec(
            name="llama-70b",
            n_params=70_000_000_000,
            hidden_size=8192,
            n_layers=80,
            n_heads=64,
            ffn_dim=28672,
            max_seq_length=4096,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
        ),
        workload_spec=WorkloadSpec(
            avg_input_length=1024,
            max_input_length=4096,
            avg_output_length=256,
            max_output_length=1024,
            arrival_rate=1.0,  # 1 request per second
            batch_size=16,
        ),
        cluster_spec=ClusterSpec(
            n_gpus_per_node=8,
            n_nodes=1,
            gpu_spec=GPUSpec(
                name="A100-80GB",
                compute_tflops=312.0,
                memory_size_gb=80.0,
                memory_bandwidth_gbs=2039.0,
            ),
            interconnect_spec=InterconnectSpec(
                intra_node_type="NVLink",
                intra_node_bandwidth_gbs=600.0,
                intra_node_latency_us=2.0,
            ),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=8,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            max_batch_size=16,
            token_level_scheduling=True,
        ),
        simulation_duration_s=60.0,
        random_seed=42,
    )
    
    # Run simulation
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()
    
    return metrics


def example_high_load():
    """
    Example: Testing system under high load.
    """
    print("\n" + "="*70)
    print("Example 3: High Load Test (10 req/s)")
    print("="*70)
    
    config = SimulatorConfig(
        model_spec=ModelSpec(
            name="llama-13b",
            n_params=13_000_000_000,
            hidden_size=5120,
            n_layers=40,
            n_heads=40,
            ffn_dim=13824,
            max_seq_length=2048,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
        ),
        workload_spec=WorkloadSpec(
            avg_input_length=256,
            max_input_length=1024,
            avg_output_length=64,
            max_output_length=256,
            arrival_rate=10.0,  # High load: 10 requests per second
            batch_size=32,
        ),
        cluster_spec=ClusterSpec(
            n_gpus_per_node=4,
            n_nodes=1,
            gpu_spec=GPUSpec(
                name="A100-80GB",
                compute_tflops=312.0,
                memory_size_gb=80.0,
                memory_bandwidth_gbs=2039.0,
            ),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            max_batch_size=32,
            batching_window_ms=5.0,
            token_level_scheduling=True,
        ),
        simulation_duration_s=30.0,
        random_seed=42,
    )
    
    # Run simulation
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()
    
    return metrics


def example_h100_comparison():
    """
    Example: Comparing A100 vs H100 performance.
    """
    print("\n" + "="*70)
    print("Example 4: H100 vs A100 Performance Comparison")
    print("="*70)
    
    # H100 configuration
    config_h100 = SimulatorConfig(
        model_spec=ModelSpec(
            name="llama-7b",
            n_params=7_000_000_000,
            hidden_size=4096,
            n_layers=32,
            n_heads=32,
            ffn_dim=11008,
            max_seq_length=2048,
            weight_dtype=DataType.BF16,
        ),
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            avg_output_length=128,
            arrival_rate=5.0,
            batch_size=16,
        ),
        cluster_spec=ClusterSpec(
            n_gpus_per_node=1,
            n_nodes=1,
            gpu_spec=GPUSpec(
                name="H100-80GB",
                compute_tflops=989.0,  # H100 BF16 TFLOPS
                memory_size_gb=80.0,
                memory_bandwidth_gbs=3350.0,  # Higher than A100
            ),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            max_batch_size=16,
            token_level_scheduling=True,
        ),
        simulation_duration_s=30.0,
        random_seed=42,
    )
    
    print("\nRunning H100 simulation...")
    simulator_h100 = LLMInferenceSimulator(config_h100)
    metrics_h100 = simulator_h100.run()
    
    # A100 configuration (same workload)
    config_a100 = config_h100
    config_a100.cluster_spec.gpu_spec = GPUSpec(
        name="A100-80GB",
        compute_tflops=312.0,
        memory_size_gb=80.0,
        memory_bandwidth_gbs=2039.0,
    )
    
    print("\nRunning A100 simulation...")
    simulator_a100 = LLMInferenceSimulator(config_a100)
    metrics_a100 = simulator_a100.run()
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    stats_h100 = metrics_h100.compute_statistics()
    stats_a100 = metrics_a100.compute_statistics()
    
    print(f"\nThroughput (tokens/sec):")
    print(f"  H100: {stats_h100.get('throughput_tokens_per_sec', 0):.2f}")
    print(f"  A100: {stats_a100.get('throughput_tokens_per_sec', 0):.2f}")
    print(f"  Speedup: {stats_h100.get('throughput_tokens_per_sec', 0) / stats_a100.get('throughput_tokens_per_sec', 1):.2f}x")
    
    if 'first_token_latency' in stats_h100 and 'first_token_latency' in stats_a100:
        print(f"\nFirst Token Latency P95 (seconds):")
        print(f"  H100: {stats_h100['first_token_latency']['p95']:.4f}")
        print(f"  A100: {stats_a100['first_token_latency']['p95']:.4f}")
        print(f"  Improvement: {stats_a100['first_token_latency']['p95'] / stats_h100['first_token_latency']['p95']:.2f}x")


def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("# LLM Inference Simulator - Example Suite")
    print("#"*70)
    
    # Run examples
    example_llama7b_single_gpu()
    example_llama70b_multi_gpu()
    example_high_load()
    example_h100_comparison()
    
    print("\n" + "#"*70)
    print("# All examples completed!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
