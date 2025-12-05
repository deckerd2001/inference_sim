"""Debug decode behavior."""
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
    model_spec=get_model("llama-7b"),  # 작은 모델로 테스트
    workload_spec=WorkloadSpec(
        avg_input_length=256,
        avg_output_length=64,  # 짧게
        arrival_rate=2.0,
    ),
    cluster_spec=ClusterSpec(
        n_gpus_per_node=1,
        gpu_spec=get_gpu("A100-80GB"),
    ),
    parallelism_spec=ParallelismSpec(
        tensor_parallel_size=1,
    ),
    scheduler_spec=SchedulerSpec(
        batching_type="continuous",
        max_batch_size=None,
        min_batch_size=4,
        batching_window_ms=200.0,
    ),
    simulation_duration_s=30.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)

# Patch decode
original_schedule_decode = simulator.scheduler.schedule_decode_batch
decode_calls = [0]

def debug_decode(current_time, memory_checker=None):
    decode_calls[0] += 1
    queue_size = len(simulator.scheduler.decode_queue)
    
    if decode_calls[0] <= 10:
        print(f"[DECODE {decode_calls[0]:2d}] t={current_time:6.2f}s, queue={queue_size:2d}", end="")
    
    batch = original_schedule_decode(current_time, memory_checker)
    
    if decode_calls[0] <= 10 and batch:
        print(f" → batch={len(batch.requests):2d}")
    elif decode_calls[0] <= 10:
        print(f" → no batch")
    
    return batch

simulator.scheduler.schedule_decode_batch = debug_decode

metrics = simulator.run()

print(f"\n총 Decode 호출: {decode_calls[0]}")
print(f"완료된 요청: {metrics.completed_requests}")
print(f"생성된 토큰: {metrics.total_tokens_generated}")
