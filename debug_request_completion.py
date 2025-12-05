"""Debug why requests finish early."""
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
        batching_strategy="greedy",
        max_batch_size=None,
    ),
    simulation_duration_s=20.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)

# Track decode steps
decode_steps = {}

original_decode_finished = simulator._handle_decode_step_finished

def debug_decode_finished(event):
    batch = simulator.scheduler.batch_manager.get_batch(event.batch_id)
    
    if batch:
        for req in batch.requests:
            if req.request_id not in decode_steps:
                decode_steps[req.request_id] = []
            
            decode_steps[req.request_id].append({
                'step': event.step,
                'tokens_generated': req.tokens_generated,
                'requested': req.requested_output_tokens,
                'is_finished': req.is_finished,
            })
    
    return original_decode_finished(event)

simulator._handle_decode_step_finished = debug_decode_finished

metrics = simulator.run()

print(f"\n{'='*70}")
print("첫 3개 요청의 Decode 기록:")
print(f"{'='*70}")

for req_id in sorted(decode_steps.keys())[:3]:
    steps = decode_steps[req_id]
    print(f"\nRequest {req_id}:")
    print(f"  Requested: {steps[0]['requested']} tokens")
    print(f"  Decode steps: {len(steps)}")
    
    for i, step_info in enumerate(steps[:10]):  # 첫 10 스텝만
        print(f"    Step {step_info['step']}: generated={step_info['tokens_generated']}, "
              f"finished={step_info['is_finished']}")
    
    if len(steps) > 10:
        print(f"    ... ({len(steps) - 10} more steps)")
    
    final = steps[-1]
    print(f"  Final: generated={final['tokens_generated']}, finished={final['is_finished']}")
