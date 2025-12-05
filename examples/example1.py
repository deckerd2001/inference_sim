"""
Example usage of the LLM Inference Simulator.

This script demonstrates how to configure and run simulations
with different model sizes, hardware configurations, and workload patterns.
"""

import sys
sys.path.insert(0, '.')

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    WorkloadSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    SchedulerSpec,
    get_model,
    get_gpu,
)


def example_llama70b_single_gpu():
    """
    Example: LLaMA-7B on a single A100-80GB GPU.
    """
    print("\n" + "="*70)
    print("Example 1: LLaMA-7B on Single A100-80GB")
    print("="*70)

    config = SimulatorConfig(
        model_spec=get_model("llama2-70b"),
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            max_input_length=2048,
            avg_output_length=128,
            max_output_length=512,
            arrival_rate=2.0,  # 2 requests per second
            batch_size=8,
        ),
        cluster_spec=ClusterSpec(
            n_gpus_per_node=2,
            n_nodes=1,
            gpu_spec=get_gpu("A100-80GB"),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=2,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            token_level_scheduling=True,
        ),
        simulation_duration_s=300.0,  # 1 minute simulation
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
        model_spec=get_model("llama2-70b"),
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
            gpu_spec=get_gpu("A100-80GB"),
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
            token_level_scheduling=True,
        ),
        simulation_duration_s=300.0,
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
    print("Example 3: High Load Test (200 req/s)")
    print("="*70)

    config = SimulatorConfig(
        model_spec=get_model("llama2-70b"),
        workload_spec=WorkloadSpec(
            avg_input_length=256,
            max_input_length=1024,
            avg_output_length=64,
            max_output_length=256,
            arrival_rate=200.0,  # High load: 10 requests per second
            batch_size=32,
        ),
        cluster_spec=ClusterSpec(
            n_gpus_per_node=4,
            n_nodes=1,
            gpu_spec=get_gpu("A100-80GB"),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            batching_window_ms=5.0,
            token_level_scheduling=True,
        ),
        simulation_duration_s=300.0,
        random_seed=42,
    )

    # Run simulation
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()

    return metrics

def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("# LLM Inference Simulator - Example Suite")
    print("#"*70)

    # Run examples
    example_llama70b_single_gpu()
    # example_llama70b_multi_gpu()
    # example_high_load()

    print("\n" + "#"*70)
    print("# All examples completed!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
