"""Debug why batch size is limited."""
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

config = SimulatorConfig(
    model_spec=get_model("llama2-70b"),
    workload_spec=WorkloadSpec(
        avg_input_length=1024,
        avg_output_length=256,
        arrival_rate=50.0,  # 높은 부하
    ),
    cluster_spec=ClusterSpec(
        n_gpus_per_node=8,
        gpu_spec=get_gpu("A100-80GB"),
    ),
    parallelism_spec=ParallelismSpec(
        tensor_parallel_size=8,
    ),
    scheduler_spec=SchedulerSpec(
        batching_strategy="greedy",
        max_batch_size=None,  # 무제한!
    ),
    simulation_duration_s=30.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)

# Track batch sizes
batch_sizes_prefill = []
batch_sizes_decode = []

original_schedule_prefill = simulator._schedule_prefill

def debug_prefill():
    original_schedule_prefill()
    if simulator.current_batch:
        batch_sizes_prefill.append(len(simulator.current_batch.requests))
        if len(batch_sizes_prefill) <= 5:
            print(f"[PREFILL] Batch size: {len(simulator.current_batch.requests)}")

simulator._schedule_prefill = debug_prefill

original_schedule_decode = simulator._schedule_decode

def debug_decode():
    original_schedule_decode()
    if simulator.current_batch:
        batch_sizes_decode.append(len(simulator.current_batch.requests))
        if len(batch_sizes_decode) <= 5:
            print(f"[DECODE] Batch size: {len(simulator.current_batch.requests)}")

simulator._schedule_decode = debug_decode

metrics = simulator.run()

print(f"\n{'='*70}")
print("Batch Size 분석:")
print(f"{'='*70}")

if batch_sizes_prefill:
    import numpy as np
    print(f"\nPrefill Batches:")
    print(f"  평균: {np.mean(batch_sizes_prefill):.1f}")
    print(f"  최대: {np.max(batch_sizes_prefill)}")
    print(f"  최소: {np.min(batch_sizes_prefill)}")

if batch_sizes_decode:
    print(f"\nDecode Batches:")
    print(f"  평균: {np.mean(batch_sizes_decode):.1f}")
    print(f"  최대: {np.max(batch_sizes_decode)}")
    print(f"  최소: {np.min(batch_sizes_decode)}")

mem_stats = simulator.memory_manager.get_memory_stats()
print(f"\nMemory:")
print(f"  Peak: {metrics.peak_memory_usage_gb:.2f}GB / {mem_stats['total_memory_gb']:.0f}GB "
      f"({metrics.peak_memory_usage_gb/mem_stats['total_memory_gb']*100:.1f}%)")
print(f"  사용 가능했던 메모리: {mem_stats['available_for_kv_cache_gb']:.2f}GB")
