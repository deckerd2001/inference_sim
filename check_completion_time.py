"""Check actual completion time."""
import sys
sys.path.insert(0, '.')

from llm_inference_simulator import get_model, get_gpu, ParallelismSpec
from llm_inference_simulator.performance_model import PerformanceModel
from llm_inference_simulator.communication import create_megatron_tp_strategy

model = get_model("llama2-70b")
gpu = get_gpu("A100-80GB")
parallel = ParallelismSpec(tensor_parallel_size=8)
perf = PerformanceModel(model, gpu, parallel, create_megatron_tp_strategy())

# From config
input_len = 1024
output_len = 256
batch_size = 4

prefill = perf.estimate_prefill_time(batch_size, input_len)
decode_per_token = perf.estimate_decode_time(batch_size, input_len + output_len//2)
total_decode = decode_per_token * output_len
total = prefill + total_decode

print(f"LLaMA-70B, TP=8, batch={batch_size}")
print(f"  Prefill: {prefill:.2f}s")
print(f"  Decode:  {total_decode:.2f}s ({decode_per_token*1000:.2f}ms/token)")
print(f"  Total:   {total:.2f}s")
print(f"\n30초 시뮬레이션에서 완료 가능? {'YES' if total < 30 else 'NO'}")

# Try smaller model
print("\n" + "="*70)
model2 = get_model("llama-7b")
perf2 = PerformanceModel(model2, gpu, ParallelismSpec(tensor_parallel_size=1), 
                        create_megatron_tp_strategy())

prefill2 = perf2.estimate_prefill_time(batch_size, input_len)
decode2 = perf2.estimate_decode_time(batch_size, input_len + output_len//2) * output_len
total2 = prefill2 + decode2

print(f"LLaMA-7B, TP=1, batch={batch_size}")
print(f"  Prefill: {prefill2:.2f}s")
print(f"  Decode:  {decode2:.2f}s")
print(f"  Total:   {total2:.2f}s")
print(f"\n30초 시뮬레이션에서 완료 가능? {'YES' if total2 < 30 else 'NO'}")
