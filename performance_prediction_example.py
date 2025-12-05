"""
Performance prediction example - estimate prefill/decode time
without running full simulation.
"""

import sys
sys.path.insert(0, '.')

from llm_inference_simulator import (
    get_model,
    get_gpu,
    ParallelismSpec,
)
from llm_inference_simulator.performance_model import PerformanceModel
from llm_inference_simulator.communication import create_megatron_tp_strategy


def predict_performance():
    """Predict performance for given configuration."""
    
    print("="*70)
    print("Performance Prediction Example")
    print("="*70)
    
    # Configuration
    model = get_model("llama-7b")
    gpu = get_gpu("A100-80GB")
    parallel = ParallelismSpec(tensor_parallel_size=1)
    comm_strategy = create_megatron_tp_strategy()
    
    # Create performance model
    perf_model = PerformanceModel(model, gpu, parallel, comm_strategy)
    
    print(f"\nConfiguration:")
    print(f"  Model: {model.name} ({model.n_params/1e9:.1f}B params)")
    print(f"  GPU: {gpu.name}")
    print(f"  TP: {parallel.tensor_parallel_size}")
    
    # Test different batch sizes and sequence lengths
    test_cases = [
        (1, 512, 128),    # Single request, medium length
        (8, 512, 128),    # Small batch
        (16, 1024, 256),  # Medium batch, long context
        (32, 2048, 512),  # Large batch, very long
    ]
    
    print(f"\n{'Batch':>6} {'Input':>6} {'Output':>7} {'Prefill':>10} {'Decode':>10} {'Total':>10}")
    print("-"*70)
    
    for batch_size, input_len, output_len in test_cases:
        # Prefill time
        prefill_time = perf_model.estimate_prefill_time(batch_size, input_len)
        
        # Decode time per token (average over sequence)
        avg_kv_len = input_len + output_len // 2
        decode_time_per_token = perf_model.estimate_decode_time(batch_size, avg_kv_len)
        
        # Total decode time
        total_decode_time = decode_time_per_token * output_len
        
        # Total time
        total_time = prefill_time + total_decode_time
        
        print(f"{batch_size:>6} {input_len:>6} {output_len:>7} "
              f"{prefill_time*1000:>9.2f}ms {decode_time_per_token*1000:>9.2f}ms "
              f"{total_time:>9.2f}s")


def compare_gpus():
    """Compare performance across different GPUs."""
    
    print("\n" + "="*70)
    print("GPU Comparison")
    print("="*70)
    
    model = get_model("llama-7b")
    parallel = ParallelismSpec(tensor_parallel_size=1)
    comm_strategy = create_megatron_tp_strategy()
    
    gpus_to_test = ["A10-24GB", "A100-80GB", "H100-80GB"]
    
    batch_size = 8
    input_len = 512
    output_len = 128
    
    print(f"\nTest: Batch={batch_size}, Input={input_len}, Output={output_len}")
    print(f"\n{'GPU':>20} {'Prefill':>12} {'Decode/tok':>12} {'Speedup':>10}")
    print("-"*70)
    
    baseline = None
    for gpu_name in gpus_to_test:
        gpu = get_gpu(gpu_name)
        perf_model = PerformanceModel(model, gpu, parallel, comm_strategy)
        
        prefill_time = perf_model.estimate_prefill_time(batch_size, input_len)
        decode_time = perf_model.estimate_decode_time(batch_size, input_len + output_len//2)
        
        if baseline is None:
            baseline = (prefill_time, decode_time)
            speedup_p = 1.0
            speedup_d = 1.0
        else:
            speedup_p = baseline[0] / prefill_time
            speedup_d = baseline[1] / decode_time
        
        print(f"{gpu.name:>20} {prefill_time*1000:>11.2f}ms {decode_time*1000:>11.2f}ms "
              f"{speedup_p:>9.2f}x")


def analyze_tp_scaling():
    """Analyze tensor parallelism scaling."""
    
    print("\n" + "="*70)
    print("Tensor Parallelism Scaling")
    print("="*70)
    
    model = get_model("llama2-70b")  # Large model
    gpu = get_gpu("A100-80GB")
    comm_strategy = create_megatron_tp_strategy()
    
    batch_size = 16
    input_len = 1024
    
    print(f"\nModel: {model.name}")
    print(f"Test: Batch={batch_size}, Input={input_len}")
    print(f"\n{'TP':>4} {'GPUs':>6} {'Prefill':>12} {'Decode':>12} {'Efficiency':>12}")
    print("-"*70)
    
    baseline_time = None
    for tp_size in [1, 2, 4, 8]:
        parallel = ParallelismSpec(tensor_parallel_size=tp_size)
        perf_model = PerformanceModel(model, gpu, parallel, comm_strategy)
        
        prefill_time = perf_model.estimate_prefill_time(batch_size, input_len)
        decode_time = perf_model.estimate_decode_time(batch_size, input_len)
        
        if baseline_time is None:
            baseline_time = prefill_time
            efficiency = 1.0
        else:
            # Ideal speedup would be tp_size
            actual_speedup = baseline_time / prefill_time
            efficiency = actual_speedup / tp_size
        
        print(f"{tp_size:>4} {tp_size:>6} {prefill_time*1000:>11.2f}ms "
              f"{decode_time*1000:>11.2f}ms {efficiency*100:>11.1f}%")


def main():
    print("\n" + "#"*70)
    print("# Performance Prediction Examples")
    print("#"*70)
    
    predict_performance()
    compare_gpus()
    analyze_tp_scaling()
    
    print("\n" + "#"*70)
    print("# Done!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
