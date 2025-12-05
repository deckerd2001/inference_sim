"""Test batching window functionality."""
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

print("Testing batching window...")

config = SimulatorConfig(
    model_spec=get_model("llama2-70b"),
    workload_spec=WorkloadSpec(
        avg_input_length=1024,
        avg_output_length=256,
        arrival_rate=1.0,
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
        max_batch_size=None,  # No limit!
        min_batch_size=8,  # Wait for 8 requests
        batching_window_ms=100.0,  # Or 100ms timeout
    ),
    simulation_duration_s=60.0,
    random_seed=42,
)

simulator = LLMInferenceSimulator(config)
metrics = simulator.run()

print("\n배칭 윈도우 효과:")
print(f"  메모리 사용량 증가 예상!")
print(f"  배치 크기 증가 예상!")
