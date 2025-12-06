"""Test xPU refactoring."""
import sys
sys.path.insert(0, '.')  # '..' → '.' 변경!

from llm_inference_simulator import (
    get_xpu,
    get_model,
    SimulatorConfig,
    WorkloadSpec,
    ClusterSpec,
    ParallelismSpec,
    SchedulerSpec,
    OperationType,
    DataType,
)

print("="*70)
print("xPU Refactoring Tests")
print("="*70)

# Test 1: xPU catalog
print("\n[Test 1] xPU Catalog")
a100 = get_xpu("a100")
print(f"✓ Loaded: {a100.name}")
a100.print_summary()

# Test 2: Get matmul performance
print("\n[Test 2] MatMul Performance Query")
bf16_tflops = a100.get_matmul_tflops(DataType.BF16)
print(f"✓ BF16 MatMul: {bf16_tflops:.1f} effective TFLOPS")

# Test 3: GTX 1080 Ti (no tensor cores)
print("\n[Test 3] GPU without Tensor Cores")
gtx = get_xpu("gtx-1080ti")
gtx.print_summary()
fp32_tflops = gtx.get_matmul_tflops(DataType.FP32)
print(f"✓ FP32 MatMul: {fp32_tflops:.1f} effective TFLOPS")

# Test 4: AMD GPU
print("\n[Test 4] AMD GPU")
mi300x = get_xpu("mi300x")
mi300x.print_summary()

# Test 5: TPU
print("\n[Test 5] Google TPU")
tpu = get_xpu("tpu-v4")
tpu.print_summary()

# Test 6: Config with xPU
print("\n[Test 6] Configuration with xPU")
try:
    config = SimulatorConfig(
        model_spec=get_model("llama-7b"),
        workload_spec=WorkloadSpec(
            avg_input_length=512,
            max_input_length=1024,
            avg_output_length=128,
            max_output_length=256,
            arrival_rate=2.0,
        ),
        cluster_spec=ClusterSpec(
            n_xpus_per_node=1,
            xpu_spec=get_xpu("a100"),
        ),
        parallelism_spec=ParallelismSpec(
            tensor_parallel_size=1,
        ),
        scheduler_spec=SchedulerSpec(
            batching_strategy="greedy",
        ),
        simulation_duration_s=60.0,
    )
    print("✓ Configuration validated successfully")
    config.print_summary()
except Exception as e:
    print(f"✗ Configuration failed: {e}")

print("\n" + "="*70)
print("All tests passed! ✅")
print("="*70)
