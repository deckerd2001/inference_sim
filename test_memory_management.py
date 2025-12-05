"""
Test memory management and OOM prevention.
"""

import sys
sys.path.insert(0, '.')

from llm_inference_simulator import (
    get_model,
    get_gpu,
)
from llm_inference_simulator.memory_manager import MemoryManager
from llm_inference_simulator.config import ParallelismSpec
from llm_inference_simulator.request import Request, RequestStatus


def test_memory_calculations():
    """Test memory calculation functions."""
    
    print("="*70)
    print("Memory Management Test")
    print("="*70)
    
    # Test with LLaMA-7B on A100-80GB
    model = get_model("llama-7b")
    gpu = get_gpu("A100-80GB")
    parallel = ParallelismSpec(tensor_parallel_size=1)
    
    mem_manager = MemoryManager(model, gpu, parallel)
    
    print(f"\nGPU: {gpu.name} ({gpu.memory_size_gb}GB)")
    print(f"Model: {model.name} ({model.n_params/1e9:.1f}B params)")
    print(f"TP: {parallel.tensor_parallel_size}")
    
    print(f"\nMemory Breakdown:")
    print(f"  Model weights: {mem_manager.model_memory_gb:.2f}GB")
    print(f"  Safety margin: {mem_manager.memory_safety_margin_gb:.2f}GB")
    print(f"  Available for KV cache: {mem_manager.available_memory_gb:.2f}GB")
    
    # Test KV cache calculation
    print(f"\nKV Cache Memory (per request):")
    for input_len, output_len in [(512, 128), (1024, 256), (2048, 512)]:
        dummy_req = Request(
            request_id=0,
            arrival_time=0.0,
            input_text="test",
            requested_output_tokens=output_len,
            input_length=input_len,
            status=RequestStatus.QUEUED,
        )
        kv_memory = mem_manager.calculate_kv_cache_memory(dummy_req)
        print(f"  Input={input_len}, Output={output_len}: {kv_memory:.3f}GB")
    
    # Test maximum batch size
    print(f"\nMaximum Batch Sizes:")
    for input_len, output_len in [(512, 128), (1024, 256), (2048, 512)]:
        max_batch = mem_manager.get_max_batch_size(input_len, output_len)
        print(f"  Input={input_len}, Output={output_len}: {max_batch} requests")
    
    # Test with different GPUs
    print(f"\n{'='*70}")
    print("Comparison Across GPUs")
    print("="*70)
    
    gpus_to_test = ["A10-24GB", "A100-40GB", "A100-80GB", "H100-80GB"]
    
    print(f"\n{'GPU':<20} {'Model Mem':>12} {'Available':>12} {'Max Batch':>12}")
    print("-"*60)
    
    for gpu_name in gpus_to_test:
        gpu = get_gpu(gpu_name)
        try:
            mem_mgr = MemoryManager(model, gpu, parallel)
            max_batch = mem_mgr.get_max_batch_size(512, 128)
            
            print(f"{gpu.name:<20} {mem_mgr.model_memory_gb:>11.2f}GB "
                  f"{mem_mgr.available_memory_gb:>11.2f}GB "
                  f"{max_batch:>12}")
        except ValueError as e:
            print(f"{gpu.name:<20} ERROR: Model too large!")


def test_oom_prevention():
    """Test OOM prevention."""
    
    print("\n" + "="*70)
    print("OOM Prevention Test")
    print("="*70)
    
    # Use small GPU to trigger OOM
    model = get_model("llama-7b")
    gpu = get_gpu("A10-24GB")  # Only 24GB
    parallel = ParallelismSpec(tensor_parallel_size=1)
    
    mem_manager = MemoryManager(model, gpu, parallel)
    
    print(f"\nGPU: {gpu.name} ({gpu.memory_size_gb}GB)")
    print(f"Model: {model.name}")
    print(f"Available memory: {mem_manager.available_memory_gb:.2f}GB")
    
    # Try to schedule batches of increasing size
    print(f"\nTesting batch scheduling:")
    
    for batch_size in [1, 4, 8, 16, 32]:
        requests = [
            Request(
                request_id=i,
                arrival_time=0.0,
                input_text=f"req_{i}",
                requested_output_tokens=128,
                input_length=512,
                status=RequestStatus.QUEUED,
            )
            for i in range(batch_size)
        ]
        
        can_schedule, reason = mem_manager.can_schedule_batch(
            requests, is_prefill=True
        )
        
        status = "✓ OK" if can_schedule else "✗ OOM"
        print(f"  Batch size {batch_size:3d}: {status}")
        if not can_schedule:
            print(f"    Reason: {reason}")


def test_large_model():
    """Test with large model that doesn't fit."""
    
    print("\n" + "="*70)
    print("Large Model Test (OOM Expected)")
    print("="*70)
    
    # Try LLaMA-70B on A10-24GB (should fail)
    model = get_model("llama2-70b")
    gpu = get_gpu("A10-24GB")
    parallel = ParallelismSpec(tensor_parallel_size=1)
    
    print(f"\nModel: {model.name} ({model.n_params/1e9:.1f}B params)")
    print(f"GPU: {gpu.name} ({gpu.memory_size_gb}GB)")
    
    try:
        mem_manager = MemoryManager(model, gpu, parallel)
        print("✗ ERROR: Should have failed!")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    # Now try with TP=8
    print(f"\nRetrying with TP=8...")
    parallel = ParallelismSpec(tensor_parallel_size=8)
    
    try:
        mem_manager = MemoryManager(model, gpu, parallel)
        print(f"✓ Success with TP=8!")
        print(f"  Model weights per GPU: {mem_manager.model_memory_gb:.2f}GB")
        print(f"  Available memory: {mem_manager.available_memory_gb:.2f}GB")
    except ValueError as e:
        print(f"✗ Still failed: {e}")


def main():
    print("\n" + "#"*70)
    print("# Memory Management Test Suite")
    print("#"*70)
    
    test_memory_calculations()
    test_oom_prevention()
    test_large_model()
    
    print("\n" + "#"*70)
    print("# All tests complete!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
