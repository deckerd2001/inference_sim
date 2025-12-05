"""Detailed debugging of batching behavior."""
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
        arrival_rate=5.0,  # 더 높은 arrival rate
    ),
    cluster_spec=ClusterSpec(
        n_gpus_per_node=8,
        gpu_spec=get_gpu("A100-80GB"),
    ),
    parallelism_spec=ParallelismSpec(
        tensor_parallel_size=8,
    ),
    scheduler_spec=SchedulerSpec(
        batching_type="continuous",
        max_batch_size=None,
        min_batch_size=4,  # 4개 이상
        batching_window_ms=50.0,  # 50ms 대기
    ),
    simulation_duration_s=30.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)

# Patch to debug
original_schedule_prefill = simulator.scheduler.schedule_prefill_batch

call_count = [0]

def debug_prefill(current_time, memory_checker=None):
    call_count[0] += 1
    queue_size = len(simulator.scheduler.prefill_queue)
    
    if call_count[0] <= 10:  # Only first 10 calls
        print(f"[{call_count[0]:2d}] t={current_time:6.2f}s, queue={queue_size:2d}", end="")
    
    batch = original_schedule_prefill(current_time, memory_checker)
    
    if call_count[0] <= 10 and batch:
        print(f" → batch={len(batch.requests):2d}")
    elif call_count[0] <= 10:
        print(f" → no batch")
    
    return batch

simulator.scheduler.schedule_prefill_batch = debug_prefill

metrics = simulator.run()
