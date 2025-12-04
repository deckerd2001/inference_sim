"""Test GPU Catalog"""
import sys
sys.path.insert(0, '.')

from llm_inference_simulator import GPUCatalog, get_gpu

print("=== Available GPUs ===")
for gpu_name in GPUCatalog.list_available():
    print(f"  - {gpu_name}")

print("\n=== Quick Test ===")
gpu = get_gpu("A10")
print(f"Got GPU: {gpu.name}")
print(f"  Compute: {gpu.compute_tflops} TFLOPS")
print(f"  Memory: {gpu.memory_size_gb} GB")

print("\n=== Comparison ===")
GPUCatalog.compare(["A10-24GB", "A100-80GB", "H100-80GB", "B200-192GB"])

print("\n=== All GPUs ===")
GPUCatalog.print_all()
