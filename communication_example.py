"""
Example: Using different communication strategies
"""

import sys
sys.path.insert(0, '.')

from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    ModelSpec,
    WorkloadSpec,
    ClusterSpec,
    ParallelismSpec,
    SchedulerSpec,
    get_gpu,
)
from llm_inference_simulator.communication import (
    TPCommunicationStrategy,
    CommunicationPattern,
    CollectiveOp,
    CommunicationAlgorithm,
    create_megatron_tp_strategy,
    create_sequence_parallel_strategy,
    estimate_collective_time,
)


def demo_communication_patterns():
    """Demo different communication patterns."""
    
    print("="*70)
    print("Communication Pattern Comparison")
    print("="*70)
    
    # Example: Estimate communication time for different collectives
    data_size_mb = 100  # 100 MB
    data_size_bytes = data_size_mb * 1024 * 1024
    num_gpus = 8
    bandwidth_gbs = 600  # NVLink bandwidth
    latency_us = 2  # NVLink latency
    
    collectives = [
        CollectiveOp.ALL_REDUCE,
        CollectiveOp.ALL_GATHER,
        CollectiveOp.REDUCE_SCATTER,
        CollectiveOp.ALL_TO_ALL,
    ]
    
    print(f"\nData size: {data_size_mb} MB")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Bandwidth: {bandwidth_gbs} GB/s")
    print(f"Latency: {latency_us} μs")
    print()
    
    for collective in collectives:
        time_ring = estimate_collective_time(
            collective, data_size_bytes, num_gpus,
            bandwidth_gbs, latency_us,
            CommunicationAlgorithm.RING
        )
        
        time_tree = estimate_collective_time(
            collective, data_size_bytes, num_gpus,
            bandwidth_gbs, latency_us,
            CommunicationAlgorithm.TREE
        )
        
        print(f"{collective.value:20s}: "
              f"Ring={time_ring*1000:.2f}ms, Tree={time_tree*1000:.2f}ms")


def compare_tp_strategies():
    """Compare different TP communication strategies."""
    
    print("\n" + "="*70)
    print("TP Communication Strategy Comparison")
    print("="*70)
    
    # Strategy 1: Standard Megatron-style TP
    print("\n1. Standard Megatron-LM TP Strategy")
    print("   - All-Reduce for aggregation")
    print("   - Disaggregated weights")
    
    megatron_strategy = create_megatron_tp_strategy()
    print(f"   - QKV: {megatron_strategy.qkv_projection.collective_op.value}")
    print(f"   - Attention Out: {megatron_strategy.attention_output.collective_op.value}")
    print(f"   - MLP: {megatron_strategy.mlp_down_projection.collective_op.value}")
    
    # Strategy 2: Sequence Parallel
    print("\n2. Sequence Parallel Strategy")
    print("   - All-Gather + Reduce-Scatter")
    print("   - Lower memory, more communication")
    
    seq_parallel_strategy = create_sequence_parallel_strategy()
    print(f"   - QKV: {seq_parallel_strategy.qkv_projection.collective_op.value}")
    print(f"   - Attention Out: {seq_parallel_strategy.attention_output.collective_op.value}")
    print(f"   - MLP: {seq_parallel_strategy.mlp_down_projection.collective_op.value}")
    
    # Strategy 3: Custom strategy
    print("\n3. Custom Strategy (Tree-based)")
    print("   - Using tree algorithm for faster small messages")
    
    custom_strategy = TPCommunicationStrategy(
        qkv_projection=CommunicationPattern(
            collective_op=CollectiveOp.ALL_REDUCE,
            algorithm=CommunicationAlgorithm.TREE,
        ),
        attention_output=CommunicationPattern(
            collective_op=CollectiveOp.ALL_REDUCE,
            algorithm=CommunicationAlgorithm.TREE,
        ),
        mlp_up_projection=CommunicationPattern(
            collective_op=CollectiveOp.ALL_REDUCE,
            algorithm=CommunicationAlgorithm.TREE,
        ),
        mlp_down_projection=CommunicationPattern(
            collective_op=CollectiveOp.ALL_REDUCE,
            algorithm=CommunicationAlgorithm.TREE,
        ),
    )
    
    # Show estimated times for different strategies
    print("\n" + "="*70)
    print("Estimated Communication Times (per layer)")
    print("="*70)
    
    # Assume 4096 hidden size, BF16 (2 bytes)
    hidden_size = 4096
    bytes_per_element = 2
    batch_size = 8
    seq_length = 512
    num_gpus = 8
    bandwidth_gbs = 600
    latency_us = 2
    
    # Size of activation to communicate: B * L * H
    activation_size = batch_size * seq_length * hidden_size * bytes_per_element
    
    strategies = [
        ("Megatron (All-Reduce)", megatron_strategy),
        ("Sequence Parallel (AG+RS)", seq_parallel_strategy),
        ("Custom (Tree)", custom_strategy),
    ]
    
    print(f"\nActivation size: {activation_size / (1024**2):.2f} MB")
    print(f"Configuration: B={batch_size}, L={seq_length}, H={hidden_size}")
    print()
    
    for name, strategy in strategies:
        # Estimate time for attention output communication
        attn_time = estimate_collective_time(
            strategy.attention_output.collective_op,
            activation_size,
            num_gpus,
            bandwidth_gbs,
            latency_us,
            strategy.attention_output.algorithm,
        )
        
        # Estimate time for MLP output communication
        mlp_time = estimate_collective_time(
            strategy.mlp_down_projection.collective_op,
            activation_size,
            num_gpus,
            bandwidth_gbs,
            latency_us,
            strategy.mlp_down_projection.algorithm,
        )
        
        total_time = attn_time + mlp_time
        
        print(f"{name:30s}: "
              f"Attn={attn_time*1000:.3f}ms, "
              f"MLP={mlp_time*1000:.3f}ms, "
              f"Total={total_time*1000:.3f}ms")


def demo_scaling():
    """Demo how communication scales with number of GPUs."""
    
    print("\n" + "="*70)
    print("Communication Scaling Analysis")
    print("="*70)
    
    data_size_mb = 100
    data_size_bytes = data_size_mb * 1024 * 1024
    bandwidth_gbs = 600
    latency_us = 2
    
    print(f"\nAll-Reduce scaling (Ring algorithm)")
    print(f"Data size: {data_size_mb} MB")
    print()
    print(f"{'GPUs':>8} {'Time (ms)':>12} {'Efficiency':>12}")
    print("-"*35)
    
    for num_gpus in [2, 4, 8, 16, 32]:
        time_s = estimate_collective_time(
            CollectiveOp.ALL_REDUCE,
            data_size_bytes,
            num_gpus,
            bandwidth_gbs,
            latency_us,
            CommunicationAlgorithm.RING,
        )
        
        # Theoretical best: all GPUs communicate in parallel
        theoretical_best = data_size_bytes / (bandwidth_gbs * 1e9)
        efficiency = theoretical_best / time_s if time_s > 0 else 0
        
        print(f"{num_gpus:>8} {time_s*1000:>12.3f} {efficiency*100:>11.1f}%")


def main():
    print("\n" + "#"*70)
    print("# Communication Strategy Examples")
    print("#"*70)
    
    demo_communication_patterns()
    compare_tp_strategies()
    demo_scaling()
    
    print("\n" + "#"*70)
    print("# Done!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
