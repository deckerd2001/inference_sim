"""
Example usage of the LLM Inference Simulator.

This script demonstrates how to configure and run simulations
with different model sizes, hardware configurations, and workload patterns.
"""

import sys
sys.path.insert(0, '.')  # ✅ 수정: 현재 디렉토리 기준

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    WorkloadSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    SchedulerSpec,
    get_model,  # ✅ 카탈로그 사용
    get_xpu,    # ✅ 카탈로그 사용
)


def example_llama7b_single_gpu():
    """
    Example: LLaMA-7B on a single A100-80GB GPU.
    """
    print("\n" + "="*70)
    print("Example 1: LLaMA-7B on Single A100-80GB")
    print("="*70)

    config = SimulatorConfig(
        model_spec=get_model("llama-7b"),  # ✅ 카탈로그 사용
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            max_input_length=2048,
            avg_output_length=128,
            max_output_length=512,
            arrival_rate=2.0,  # 2 requests per second
            batch_size=8,
        ),
        cluster_spec=ClusterSpec(
            n_xpus_per_node=1,
            n_nodes=1,
            xpu_spec=get_xpu("A100-80GB"),  # ✅ 카탈로그 사용
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
        model_spec=get_model("llama2-70b"),  # ✅ 올바른 모델명
        workload_spec=WorkloadSpec(
            avg_input_length=1024,
            max_input_length=4096,
            avg_output_length=256,
            max_output_length=1024,
            arrival_rate=1.0,  # 1 request per second
            batch_size=16,
        ),
        cluster_spec=ClusterSpec(
            n_xpus_per_node=8,
            n_nodes=1,
            xpu_spec=get_xpu("A100-80GB"),  # ✅ 카탈로그 사용
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
        model_spec=get_model("llama-13b"),  # ✅ 카탈로그 사용
        workload_spec=WorkloadSpec(
            avg_input_length=256,
            max_input_length=1024,
            avg_output_length=64,
            max_output_length=256,
            arrival_rate=10.0,  # High load: 10 requests per second
            batch_size=32,
        ),
        cluster_spec=ClusterSpec(
            n_xpus_per_node=4,
            n_nodes=1,
            xpu_spec=get_xpu("A100-80GB"),  # ✅ 카탈로그 사용
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
        model_spec=get_model("llama-7b"),  # ✅ 카탈로그 사용
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            avg_output_length=128,
            arrival_rate=5.0,
            batch_size=16,
        ),
        cluster_spec=ClusterSpec(
            n_xpus_per_node=1,
            n_nodes=1,
            xpu_spec=get_xpu("H100-80GB"),  # ✅ 카탈로그 사용
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
    config_a100 = SimulatorConfig(
        model_spec=get_model("llama-7b"),
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            avg_output_length=128,
            arrival_rate=5.0,
            batch_size=16,
        ),
        cluster_spec=ClusterSpec(
            n_xpus_per_node=1,
            n_nodes=1,
            xpu_spec=get_xpu("A100-80GB"),  # ✅ 카탈로그 사용
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
    speedup = stats_h100.get('throughput_tokens_per_sec', 0) / max(stats_a100.get('throughput_tokens_per_sec', 1), 1)
    print(f"  Speedup: {speedup:.2f}x")

    if 'first_token_latency' in stats_h100 and 'first_token_latency' in stats_a100:
        print(f"\nFirst Token Latency P95 (seconds):")
        print(f"  H100: {stats_h100['first_token_latency']['p95']:.4f}")
        print(f"  A100: {stats_a100['first_token_latency']['p95']:.4f}")
        improvement = stats_a100['first_token_latency']['p95'] / max(stats_h100['first_token_latency']['p95'], 0.0001)
        print(f"  Improvement: {improvement:.2f}x")


def example_memory_stress_test():
    """
    NEW: Example showing memory management and OOM prevention.
    """
    print("\n" + "="*70)
    print("Example 5: Memory Stress Test (Small GPU)")
    print("="*70)

    # Use A10 with limited memory to see OOM prevention in action
    config = SimulatorConfig(
        model_spec=get_model("llama-7b"),
        workload_spec=WorkloadSpec(
            avg_input_length=1024,  # Long inputs
            max_input_length=2048,
            avg_output_length=512,  # Long outputs
            max_output_length=1024,
            arrival_rate=5.0,  # High arrival rate
            batch_size=32,  # Try large batch
        ),
        cluster_spec=ClusterSpec(
            n_xpus_per_node=1,
            n_nodes=1,
            xpu_spec=get_xpu("A10-24GB"),  # Only 24GB memory!
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            max_batch_size=32,
            token_level_scheduling=True,
        ),
        simulation_duration_s=30.0,
        random_seed=42,
    )

    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()

    stats = metrics.compute_statistics()
    
    print("\n" + "="*70)
    print("Memory Management Results:")
    print("="*70)
    if stats['rejected_requests'] > 0:
        print(f"✓ OOM prevention worked!")
        print(f"  Rejected: {stats['rejected_requests']} requests")
        print(f"  Completed: {stats['completed_requests']} requests")
    else:
        print(f"✓ All requests completed without OOM")

    return metrics


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
    example_memory_stress_test()  # ✅ NEW: 메모리 관리 테스트

    print("\n" + "#"*70)
    print("# All examples completed!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
