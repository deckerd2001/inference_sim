"""Debug request lifecycle."""
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
        arrival_rate=0.5,  # 천천히
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
        min_batch_size=1,
        batching_window_ms=100.0,
    ),
    simulation_duration_s=20.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)

# Track first few requests
tracked_requests = {}

original_handle_finished = simulator._handle_request_finished

def debug_handle_finished(event):
    req = simulator.completed_requests_map.get(event.request_id)
    if req and event.request_id not in tracked_requests:
        tracked_requests[event.request_id] = req
        print(f"\n[REQUEST {event.request_id}]")
        print(f"  Requested: {req.requested_output_tokens} tokens")
        print(f"  Generated: {req.tokens_generated} tokens")
        print(f"  Input length: {req.input_length}")
        print(f"  E2E time: {req.end_to_end_latency:.3f}s")
        print(f"  Status: {req.status}")
        print(f"  Is finished: {req.is_finished}")
        
        if len(tracked_requests) >= 5:
            print("\n[Stopping tracking after 5 requests...]")
    
    return original_handle_finished(event)

simulator._handle_request_finished = debug_handle_finished

metrics = simulator.run()

print(f"\n{'='*70}")
print(f"Total completed: {metrics.completed_requests}")
print(f"Total tokens: {metrics.total_tokens_generated}")
if metrics.completed_requests > 0:
    print(f"Average tokens/request: {metrics.total_tokens_generated / metrics.completed_requests:.1f}")
