"""Debug output length - correct approach."""
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
    model_spec=get_model("llama-7b"),
    workload_spec=WorkloadSpec(
        avg_input_length=256,
        avg_output_length=64,
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
        batching_strategy="greedy",
        max_batch_size=None,
    ),
    simulation_duration_s=30.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)

# Intercept _handle_request_arrived where Request is actually created
original_handle = simulator._handle_request_arrived
output_lengths = []

def debug_handle(event):
    output_lengths.append(event.requested_output_tokens)
    if len(output_lengths) <= 20:
        print(f"[Req {event.request_id}] requested_output={event.requested_output_tokens}")
    return original_handle(event)

simulator._handle_request_arrived = debug_handle

metrics = simulator.run()

print(f"\n{'='*70}")
print(f"총 요청: {len(output_lengths)}")
if output_lengths:
    import numpy as np
    print(f"\nOutput length 분포:")
    print(f"  설정 평균: 64")
    print(f"  실제 평균: {np.mean(output_lengths):.1f}")
    print(f"  Min:  {np.min(output_lengths)}")
    print(f"  Max:  {np.max(output_lengths)}")
    print(f"  Std:  {np.std(output_lengths):.1f}")
    print(f"\n  <= 10인 것: {sum(1 for x in output_lengths if x <= 10)} / {len(output_lengths)} ({100*sum(1 for x in output_lengths if x <= 10)/len(output_lengths):.1f}%)")
    print(f"  <= 1인 것:  {sum(1 for x in output_lengths if x <= 1)} / {len(output_lengths)} ({100*sum(1 for x in output_lengths if x <= 1)/len(output_lengths):.1f}%)")

print(f"\n완료된 요청: {metrics.completed_requests}")
print(f"생성된 토큰: {metrics.total_tokens_generated}")
if metrics.completed_requests > 0:
    print(f"평균 토큰/요청: {metrics.total_tokens_generated / metrics.completed_requests:.1f}")
