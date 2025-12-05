"""Debug output length distribution."""
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
        output_length_std=50,  # 명시
        arrival_rate=0.2,  # 천천히
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
    ),
    simulation_duration_s=30.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)

# Intercept RequestArrivedEvent
original_create = simulator._create_request_arrival
output_lengths = []

def debug_create(arrival_time):
    result = original_create(arrival_time)
    # Find the request in pending
    req_id = simulator.next_request_id - 1
    if req_id in simulator.scheduler.pending_requests:
        req = simulator.scheduler.pending_requests[req_id]
        output_lengths.append(req.requested_output_tokens)
        if len(output_lengths) <= 10:
            print(f"Request {req_id}: requested_output={req.requested_output_tokens}")
    return result

simulator._create_request_arrival = debug_create

metrics = simulator.run()

print(f"\n총 요청: {len(output_lengths)}")
if output_lengths:
    import numpy as np
    print(f"Output length 분포:")
    print(f"  Mean: {np.mean(output_lengths):.1f}")
    print(f"  Min:  {np.min(output_lengths)}")
    print(f"  Max:  {np.max(output_lengths)}")
    print(f"  <= 10인 것: {sum(1 for x in output_lengths if x <= 10)} / {len(output_lengths)}")
