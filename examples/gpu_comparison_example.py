"""
GPU 성능 비교 예제
"""

import sys, time
sys.path.insert(0, '.')

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    ModelSpec,
    WorkloadSpec,
    ClusterSpec,
    ParallelismSpec,
    SchedulerSpec,
    GPUCatalog,
    get_model,
    get_gpu
)


def compare_gpus():
    """다양한 GPU에서 LLaMA-7B 성능 비교"""

    # 비교할 GPU들
    gpus = [
        # "A10-24GB",
        "A100-80GB",
        "H100-80GB",
        "B200-192GB"
    ]

    print("\n" + "="*70)
    print("LLaMA-7B Performance Comparison Across GPUs")
    print("="*70)

    # GPU 스펙 비교 출력
    GPUCatalog.compare(gpus)

    results = {}

    for gpu_name in gpus:
        print(f"\nRunning simulation on {gpu_name}...")

        config = SimulatorConfig(
            model_spec=get_model(
                "llama2-70b"
            ),
            workload_spec=WorkloadSpec(
                avg_input_length=512,
                avg_output_length=128,
                arrival_rate=10.0,
            ),
            cluster_spec=ClusterSpec(
                n_gpus_per_node=4,
                n_nodes=1,
                gpu_spec=GPUCatalog.get_gpu(gpu_name),  # 여기!
            ),
            parallelism_spec=ParallelismSpec(
                tensor_parallel_size=4,
            ),
            scheduler_spec=SchedulerSpec(
                batching_type="continuous",
                max_batch_size=8,
            ),
            simulation_duration_s=300.0,
            random_seed=int(time.time()),
        )

        simulator = LLMInferenceSimulator(config)
        metrics = simulator.run()
        stats = metrics.compute_statistics()

        results[gpu_name] = stats

    # 결과 비교
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'GPU':<20} {'Throughput':>15} {'P95 TTFT':>15} {'GPU Util':>12}")
    print("-"*70)

    baseline = None
    for gpu_name, stats in results.items():
        throughput = stats.get('throughput_tokens_per_sec', 0)
        if baseline is None:
            baseline = throughput

        speedup = throughput / baseline if baseline > 0 else 0
        p95_ttft = stats.get('first_token_latency', {}).get('p95', 0)
        gpu_util = stats.get('gpu_utilization', 0) * 100

        print(f"{gpu_name:<20} {throughput:>10.2f} tok/s ({speedup:.2f}x) "
              f"{p95_ttft:>10.4f}s {gpu_util:>10.1f}%")

    print("="*70 + "\n")


if __name__ == "__main__":
    compare_gpus()
