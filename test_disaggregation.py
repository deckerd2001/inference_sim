"""
Test disaggregation/aggregation strategies in ACTUAL event-driven simulation.
"""

import sys
sys.path.insert(0, '.')

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    ModelSpec,
    WorkloadSpec,
    ClusterSpec,
    ParallelismSpec,
    SchedulerSpec,
    get_gpu,
    get_model,
)
from llm_inference_simulator.communication import (
    create_megatron_tp_strategy,
    create_sequence_parallel_strategy,
    TPCommunicationStrategy,
    CommunicationPattern,
    CollectiveOp,
    CommunicationAlgorithm,
)


def run_simulation_with_strategy(strategy_name, comm_strategy):
    """Run full event-driven simulation with given communication strategy."""

    print(f"\n{'='*70}")
    print(f"Running Simulation: {strategy_name}")
    print(f"{'='*70}")

    # Create configuration
    config = SimulatorConfig(
        model_spec=get_model(
            "llama-7b"
        ),
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            max_input_length=1024,
            avg_output_length=128,
            max_output_length=256,
            arrival_rate=3.0,  # 3 requests per second (arrival rate!)
            arrival_process="poisson",  # Poisson arrival process
            batch_size=8,
        ),
        cluster_spec=ClusterSpec(
            n_gpus_per_node=8,
            n_nodes=1,
            gpu_spec=get_gpu("A100-80GB"),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=8,
        ),
        scheduler_spec=SchedulerSpec(
            batching_type="continuous",
            max_batch_size=8,
            token_level_scheduling=True,
        ),
        simulation_duration_s=30.0,  # 30 seconds simulation
        random_seed=42,
    )

    # Create simulator with custom communication strategy
    simulator = LLMInferenceSimulator(config)

    # IMPORTANT: Replace the performance model with one using our comm strategy
    simulator.performance_model = type(simulator.performance_model)(
        config.model_spec,
        config.cluster_spec.gpu_spec,
        config.parallelism_spec,
        tp_comm_strategy=comm_strategy
    )

    print(f"\nConfiguration:")
    print(f"  Arrival rate: {config.workload_spec.arrival_rate} req/s")
    print(f"  Arrival process: {config.workload_spec.arrival_process}")
    print(f"  Simulation duration: {config.simulation_duration_s}s")
    print(f"  TP size: {config.parallelism_spec.tensor_parallel_size}")
    print(f"  Batching: {config.scheduler_spec.batching_type}")

    print(f"\nCommunication strategy:")
    print(f"  Attention: {comm_strategy.attention_output.collective_op.value}")
    print(f"  MLP: {comm_strategy.mlp_down_projection.collective_op.value}")
    print(f"  Algorithm: {comm_strategy.attention_output.algorithm.value}")

    # Run the EVENT-DRIVEN SIMULATION!
    metrics = simulator.run()

    # Get statistics
    stats = metrics.compute_statistics()

    return stats


def compare_strategies():
    """Compare different disaggregation/aggregation strategies."""

    print("\n" + "#"*70)
    print("# Event-Driven Simulation: Disaggregation Strategy Comparison")
    print("#"*70)

    # Define strategies to test
    strategies = {
        "Megatron (All-Reduce)": create_megatron_tp_strategy(),

        "Sequence Parallel (AG+RS)": create_sequence_parallel_strategy(),

        "Tree-based All-Reduce": TPCommunicationStrategy(
            qkv_projection=CommunicationPattern(
                collective_op=CollectiveOp.ALL_REDUCE,
                algorithm=CommunicationAlgorithm.TREE,
            ),
            attention_output=CommunicationPattern(
                collective_op=CollectiveOp.ALL_REDUCE,
                algorithm=CommunicationAlgorithm.TREE,
            ),
            mlp_up_projection=CommunicationPattern(
                collective_op=CollectiveOp.ALL_REDUCE,
                algorithm=CommunicationAlgorithm.TREE,
            ),
            mlp_down_projection=CommunicationPattern(
                collective_op=CollectiveOp.ALL_REDUCE,
                algorithm=CommunicationAlgorithm.TREE,
            ),
        ),
    }

    results = {}

    # Run simulation for each strategy
    for strategy_name, comm_strategy in strategies.items():
        stats = run_simulation_with_strategy(strategy_name, comm_strategy)
        results[strategy_name] = stats

    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON - Event-Driven Simulation Results")
    print("="*70)

    print(f"\n{'Strategy':<30} {'Throughput':>15} {'P95 TTFT':>15} {'GPU Util':>12}")
    print("-"*70)

    baseline_throughput = None
    for strategy_name, stats in results.items():
        throughput = stats.get('throughput_tokens_per_sec', 0)

        if baseline_throughput is None:
            baseline_throughput = throughput

        speedup = throughput / baseline_throughput if baseline_throughput > 0 else 0

        p95_ttft = 0
        if 'first_token_latency' in stats:
            p95_ttft = stats['first_token_latency'].get('p95', 0)

        gpu_util = stats.get('gpu_utilization', 0) * 100

        print(f"{strategy_name:<30} "
              f"{throughput:>10.2f} tok/s ({speedup:.2f}x) "
              f"{p95_ttft:>10.4f}s "
              f"{gpu_util:>10.1f}%")

    print("\n" + "="*70)
    print("Analysis:")
    print("="*70)
    print("""
The event-driven simulator shows:

1. REQUEST GENERATION:
   - Requests arrive according to Poisson process (arrival_rate=3.0 req/s)
   - Each request triggers RequestArrivedEvent
   - Tokenization happens (RequestTokenizedEvent)
   - Requests enter scheduler queue

2. BATCHING:
   - Scheduler forms batches based on policy (continuous batching)
   - BatchFormedEvent triggered

3. EXECUTION:
   - PrefillStartedEvent → compute with communication strategy
   - PrefillFinishedEvent → requests move to decode queue
   - DecodeStepStartedEvent/FinishedEvent → repeated for each token
   - Communication overhead varies by strategy!

4. METRICS:
   - Throughput: tokens/second across all requests
   - P95 TTFT: 95th percentile time to first token
   - GPU Utilization: fraction of time GPU is busy

5. STRATEGY IMPACT:
   - Megatron (All-Reduce): Standard, predictable performance
   - Sequence Parallel: May have higher comm overhead but lower memory
   - Tree: Better for latency-sensitive workloads
    """)


def show_event_flow():
    """Show the event flow in the simulator."""

    print("\n" + "="*70)
    print("Event Flow in the Simulator")
    print("="*70)
    print("""
Time 0.000s: Simulation starts
  ↓
Time 0.234s: RequestArrivedEvent (request_id=0)
  → Creates Request object
  → Schedules RequestTokenizedEvent
  ↓
Time 0.234s: RequestTokenizedEvent (request_id=0)
  → Sets input_length
  → Adds to prefill_queue
  ↓
Time 0.567s: RequestArrivedEvent (request_id=1)
  → Another request arrives (Poisson process!)
  ↓
Time 0.600s: GPU idle, scheduler checks queue
  → Forms batch with request_id=0,1
  → Schedules PrefillStartedEvent
  ↓
Time 0.600s: PrefillStartedEvent (batch_id=0)
  → PerformanceModel calculates prefill time
  → Uses communication strategy for TP overhead
  → Schedules PrefillFinishedEvent at time 0.600 + prefill_time
  ↓
Time 0.623s: PrefillFinishedEvent (batch_id=0)
  → Moves requests to decode_queue
  → Schedules first DecodeStepStartedEvent
  ↓
Time 0.623s: DecodeStepStartedEvent (batch_id=1, step=0)
  → PerformanceModel calculates decode time
  → Uses communication strategy (affects time!)
  → Schedules DecodeStepFinishedEvent
  ↓
Time 0.625s: DecodeStepFinishedEvent (batch_id=1, step=0)
  → Increments tokens_generated
  → Emits TokenEmittedEvent
  → If not finished, schedules next DecodeStepStartedEvent
  ↓
... (repeat decode steps) ...
  ↓
Time 0.780s: RequestFinishedEvent (request_id=0)
  → Request completes (EOS or max_length)
  → Records end-to-end latency
  ↓
... (continues for 30 seconds) ...
  ↓
Time 30.000s: Simulation ends
  → Collects metrics
  → Computes statistics
""")


def main():
    # Show how the event-driven simulator works
    show_event_flow()

    # Run actual comparison
    compare_strategies()

    print("\n" + "#"*70)
    print("# All simulations complete!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
