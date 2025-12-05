"""Debug dynamic batch sizing."""
import sys
sys.path.insert(0, '.')

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    WorkloadSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    SchedulerSpec,
    get_model,
    get_gpu,
)

# Add debug prints to scheduler
print("Testing dynamic batch sizing...")

config = SimulatorConfig(
    model_spec=get_model("llama2-70b"),
    workload_spec=WorkloadSpec(
        avg_input_length=1024,
        max_input_length=2048,
        avg_output_length=256,
        max_output_length=512,
        arrival_rate=1.0,
        batch_size=16,  # This should be ignored now
    ),
    cluster_spec=ClusterSpec(
        n_gpus_per_node=8,
        n_nodes=1,
        gpu_spec=get_gpu("A100-80GB"),
        interconnect_spec=InterconnectSpec(
            intra_node_type="NVLink",
            intra_node_bandwidth_gbs=600.0,
            intra_node_latency_us=2.0,
        ),
    ),
    parallelism_spec=ParallelismSpec(
        tensor_parallel_size=8,
    ),
    scheduler_spec=SchedulerSpec(
        batching_type="continuous",
        max_batch_size=None,  # No limit!
        token_level_scheduling=True,
    ),
    simulation_duration_s=10.0,  # Short test
    random_seed=42,
)

# Run for 10 seconds
simulator = LLMInferenceSimulator(config)

# Monkey patch to add debug prints
original_schedule_prefill = simulator.scheduler.schedule_prefill_batch

def debug_schedule_prefill(current_time, memory_checker=None):
    print(f"\n[DEBUG] schedule_prefill_batch called at t={current_time:.2f}")
    print(f"  Prefill queue size: {len(simulator.scheduler.prefill_queue)}")
    
    batch = original_schedule_prefill(current_time, memory_checker)
    
    if batch:
        print(f"  Batch created with {len(batch.requests)} requests")
    else:
        print(f"  No batch created")
    
    return batch

simulator.scheduler.schedule_prefill_batch = debug_schedule_prefill

metrics = simulator.run()
