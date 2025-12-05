"""Test Greedy vs Windowed strategies."""
import sys
sys.path.insert(0, '.')

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    WorkloadSpec,
    ClusterSpec,
    ParallelismSpec,
    SchedulerSpec,
    get_model,
    get_gpu,
)

print("="*70)
print("Greedy vs Windowed Batching Strategies")
print("="*70)

base_config = {
    'model_spec': get_model("llama-7b"),
    'workload_spec': WorkloadSpec(
        avg_input_length=256,
        avg_output_length=64,
        arrival_rate=2.0,
    ),
    'cluster_spec': ClusterSpec(
        n_gpus_per_node=1,
        gpu_spec=get_gpu("A100-80GB"),
    ),
    'parallelism_spec': ParallelismSpec(
        tensor_parallel_size=1,
    ),
    'simulation_duration_s': 60.0,
    'random_seed': 42,
}

# Test 1: Greedy (default)
print("\n--- Strategy: GREEDY (즉시 처리) ---")
config_greedy = SimulatorConfig(
    **base_config,
    scheduler_spec=SchedulerSpec(
        batching_type="continuous",
        max_batch_size=None,
        batching_strategy="greedy",  # 즉시!
    ),
)

sim_greedy = LLMInferenceSimulator(config_greedy)
metrics_greedy = sim_greedy.run()

# Test 2: Windowed (wait for batch)
print("\n--- Strategy: WINDOWED (배치 대기) ---")
config_windowed = SimulatorConfig(
    **base_config,
    scheduler_spec=SchedulerSpec(
        batching_type="continuous",
        max_batch_size=None,
        batching_strategy="windowed",
        min_batch_size=8,  # 8개까지 대기
        batching_window_ms=200.0,  # 200ms 타임아웃
    ),
)

sim_windowed = LLMInferenceSimulator(config_windowed)
metrics_windowed = sim_windowed.run()

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

stats_g = metrics_greedy.compute_statistics()
stats_w = metrics_windowed.compute_statistics()

print(f"\n{'Metric':<30} {'Greedy':>15} {'Windowed':>15}")
print("-"*70)
print(f"{'Completed Requests':<30} {stats_g['completed_requests']:>15} {stats_w['completed_requests']:>15}")
print(f"{'Throughput (req/s)':<30} {stats_g.get('throughput_requests_per_sec', 0):>15.2f} {stats_w.get('throughput_requests_per_sec', 0):>15.2f}")
print(f"{'Throughput (tok/s)':<30} {stats_g.get('throughput_tokens_per_sec', 0):>15.2f} {stats_w.get('throughput_tokens_per_sec', 0):>15.2f}")

if 'first_token_latency' in stats_g:
    print(f"{'P95 TTFT (s)':<30} {stats_g['first_token_latency']['p95']:>15.3f} {stats_w['first_token_latency']['p95']:>15.3f}")

print("\n예상:")
print("  Greedy: 낮은 지연, 작은 배치")
print("  Windowed: 높은 처리량, 큰 배치")
