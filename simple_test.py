"""
Simple test to verify basic simulator functionality.
"""


from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    ModelSpec,
    WorkloadSpec,
    GPUSpec,
    ClusterSpec,
    ParallelismSpec,
    SchedulerSpec,
    DataType,
)


def simple_test():
    """
    Simple test with small model and low load.
    """
    print("="*60)
    print("Simple Test: Small model, few requests")
    print("="*60)

    config = SimulatorConfig(
        model_spec=ModelSpec(
            name="mini-model",
            n_params=1_000_000_000,  # 1B model (smaller)
            hidden_size=2048,
            n_layers=12,
            n_heads=16,
            ffn_dim=5120,
            max_seq_length=1024,
            weight_dtype=DataType.BF16,
        ),
        workload_spec=WorkloadSpec(
            avg_input_length=64,   # Short inputs
            max_input_length=128,
            avg_output_length=16,  # Short outputs
            max_output_length=32,
            arrival_rate=0.5,      # 0.5 requests per second
            batch_size=2,
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
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            max_batch_size=2,
            token_level_scheduling=True,
        ),
        simulation_duration_s=10.0,  # 10 seconds only
        random_seed=42,
    )

    # Run simulation
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()

    # Print detailed stats
    stats = metrics.compute_statistics()

    print("\nDetailed Metrics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Completed requests: {stats['completed_requests']}")
    print(f"  Total tokens generated: {stats['total_tokens_generated']}")
    print(f"  GPU utilization: {stats.get('gpu_utilization', 0)*100:.1f}%")

    # Check if we have any first token latencies
    if metrics.first_token_latencies:
        print(f"\nFirst token latencies collected: {len(metrics.first_token_latencies)}")
        print(f"  Min: {min(metrics.first_token_latencies):.4f}s")
        print(f"  Max: {max(metrics.first_token_latencies):.4f}s")

    # Check if we have any completed requests
    if metrics.end_to_end_latencies:
        print(f"\nEnd-to-end latencies collected: {len(metrics.end_to_end_latencies)}")
        print(f"  Min: {min(metrics.end_to_end_latencies):.4f}s")
        print(f"  Max: {max(metrics.end_to_end_latencies):.4f}s")
    else:
        print("\nNo requests completed during simulation window")
        print("This is expected for short simulations with generation tasks")

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

    return metrics


if __name__ == "__main__":
    simple_test()
