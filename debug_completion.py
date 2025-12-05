"""Debug why requests are not completing."""
import sys
sys.path.insert(0, '.')

from llm_inference_simulator import get_model, get_gpu, ParallelismSpec
from llm_inference_simulator.performance_model import PerformanceModel
from llm_inference_simulator.communication import create_megatron_tp_strategy

print("="*70)
print("Debugging Example 2: LLaMA-70B, TP=8")
print("="*70)

model = get_model("llama2-70b")
gpu = get_gpu("A100-80GB")
parallel = ParallelismSpec(tensor_parallel_size=8)
perf = PerformanceModel(model, gpu, parallel, create_megatron_tp_strategy())

# From example config
batch_size = 16
input_len = 1024
output_len = 256

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Input length: {input_len}")
print(f"  Output length: {output_len}")
print(f"  Simulation time: 500s")

# Calculate times
prefill_time = perf.estimate_prefill_time(batch_size, input_len)
print(f"\nPrefill time: {prefill_time:.2f}s")

# Decode time varies with KV cache length
decode_times = []
for step in [0, 64, 128, 192, 256]:
    kv_len = input_len + step
    decode_time = perf.estimate_decode_time(batch_size, kv_len)
    decode_times.append(decode_time)
    print(f"Decode time (KV={kv_len}): {decode_time*1000:.2f}ms per token")

avg_decode = sum(decode_times) / len(decode_times)
total_decode = avg_decode * output_len

print(f"\nEstimated timing:")
print(f"  Prefill: {prefill_time:.2f}s")
print(f"  Average decode per token: {avg_decode*1000:.2f}ms")
print(f"  Total decode ({output_len} tokens): {total_decode:.2f}s")
print(f"  Total per request: {prefill_time + total_decode:.2f}s")

# How many can complete?
single_request_time = prefill_time + total_decode
requests_per_batch = batch_size
time_per_batch = prefill_time + total_decode  # Assuming all in batch finish together

batches_in_500s = 500 / time_per_batch
requests_in_500s = batches_in_500s * batch_size

print(f"\nProjected completion:")
print(f"  Time per batch: {time_per_batch:.2f}s")
print(f"  Batches in 500s: {batches_in_500s:.2f}")
print(f"  Requests in 500s: {requests_in_500s:.1f}")
print(f"  Actual completed: 0")

if requests_in_500s > 10:
    print(f"\n❌ 문제 발견!")
    print(f"   예상: {requests_in_500s:.0f}개 완료")
    print(f"   실제: 0개 완료")
    print(f"\n가능한 원인:")
    print(f"   1. RequestFinishedEvent가 올바르게 처리되지 않음")
    print(f"   2. is_finished 조건이 잘못됨")
    print(f"   3. 배치가 계속 교체되어 요청이 끝까지 가지 못함")
    print(f"   4. 큐 관리 문제")

# Check actual metrics
print(f"\n실제 관찰된 값:")
print(f"  Tokens/sec: 244.15")
print(f"  Total tokens in 500s: {244.15 * 500:.0f}")
print(f"  Expected tokens per request: {output_len}")
print(f"  Implied completed requests: {244.15 * 500 / output_len:.1f}")

print(f"\n✓ 토큰은 생성되고 있음 (244 tok/s)")
print(f"✓ First token latency 측정됨")
print(f"❌ 하지만 완료된 요청은 0개")
print(f"\n→ Request completion 로직에 버그가 있을 가능성!")
