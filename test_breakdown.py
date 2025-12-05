"""Test breakdown functionality from performance_model."""

import sys
sys.path.insert(0, '.')

from llm_inference_simulator import get_model, get_gpu, ParallelismSpec
from llm_inference_simulator.performance_model import PerformanceModel
from llm_inference_simulator.communication import create_megatron_tp_strategy


def main():
    # Setup
    model = get_model("llama-7b")
    gpu = get_gpu("A100-80GB")
    parallel = ParallelismSpec(tensor_parallel_size=1)
    comm = create_megatron_tp_strategy()
    
    perf_model = PerformanceModel(model, gpu, parallel, comm)
    
    # Get breakdown
    prefill_breakdown = perf_model.breakdown_prefill(8, 512)
    decode_breakdown = perf_model.breakdown_decode(8, 512)
    
    print("Prefill components:")
    for name, value in prefill_breakdown.items():
        if isinstance(value, dict):
            print(f"  {name}: {value['total']*1000:.4f}ms")
        else:
            print(f"  {name}: {value*1000:.4f}ms")
    
    print("\nDecode components:")
    for name, value in decode_breakdown.items():
        if isinstance(value, dict):
            print(f"  {name}: {value['total']*1000:.4f}ms")
        else:
            print(f"  {name}: {value*1000:.4f}ms")


if __name__ == "__main__":
    main()
