"""Check if batch size is the bottleneck."""
import sys
sys.path.insert(0, '.')

from llm_inference_simulator.memory_manager import MemoryManager
from llm_inference_simulator import get_model, get_gpu, ParallelismSpec

# Example 3 설정
model = get_model("llama2-70b")
gpu = get_gpu("A100-80GB")
parallel = ParallelismSpec(tensor_parallel_size=4)

mem_mgr = MemoryManager(model, gpu, parallel)

print("="*70)
print("Batch Size vs Memory Capacity")
print("="*70)

input_len = 256
output_len = 64

print(f"\nGPU: {gpu.name} ({gpu.memory_size_gb}GB)")
print(f"Model: {model.name}")
print(f"Available for KV cache: {mem_mgr.available_memory_gb:.2f}GB")
print(f"Input: {input_len}, Output: {output_len}")

print(f"\n{'Batch Size':>12} {'Memory Used':>15} {'Status':>10}")
print("-"*40)

for batch_size in [16, 32, 64, 128, 256]:
    max_batch = mem_mgr.get_max_batch_size(input_len, output_len)
    
    from llm_inference_simulator.request import Request, RequestStatus
    dummy_requests = [
        Request(
            request_id=i,
            arrival_time=0.0,
            input_text=f"req_{i}",
            requested_output_tokens=output_len,
            input_length=input_len,
            status=RequestStatus.QUEUED,
        )
        for i in range(batch_size)
    ]
    
    can_fit, reason = mem_mgr.can_schedule_batch(dummy_requests, is_prefill=True)
    
    if can_fit:
        kv_memory = mem_mgr.calculate_batch_kv_cache_memory(dummy_requests)
        act_memory = mem_mgr.calculate_activation_memory(batch_size, input_len)
        total = kv_memory + act_memory
        status = "✓ OK"
        print(f"{batch_size:>12} {total:>13.2f}GB   {status:>10}")
    else:
        print(f"{batch_size:>12} {'(OOM)':>15}   {'✗ Too big':>10}")

print(f"\n현재 설정 max_batch_size: 32 (예상)")
print(f"메모리 기준 최대 가능: {max_batch}")
print(f"\n결론: max_batch_size를 {max_batch}까지 늘릴 수 있음!")
