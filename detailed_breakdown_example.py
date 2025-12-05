"""
Detailed performance breakdown by component
(Attention, MLP, Norm, etc.)
"""

import sys
sys.path.insert(0, '.')

from llm_inference_simulator import get_model, get_gpu, ParallelismSpec
from llm_inference_simulator.performance_model import PerformanceModel
from llm_inference_simulator.communication import create_megatron_tp_strategy


class DetailedPerformanceModel(PerformanceModel):
    """Extended performance model with detailed component breakdown."""
    
    def breakdown_prefill(self, batch_size: int, seq_length: int) -> dict:
        """Get detailed breakdown of prefill time."""
        B, L = batch_size, seq_length
        H = self.model.hidden_size
        D_ff = self.model.ffn_dim
        
        breakdown = {}
        
        # === ATTENTION BLOCK ===
        
        # 1. Input LayerNorm / RMSNorm
        norm_compute = self._norm_time(B, L, H)
        breakdown['attention_norm'] = norm_compute
        
        # 2. QKV Projection
        qkv_flops = 3 * B * L * H * H * self.flops_per_mac
        qkv_flops /= self.parallel.tensor_parallel_size
        qkv_compute = qkv_flops / (self.gpu.compute_tflops * 1e12)
        
        qkv_memory = (3 * H * H * self.bytes_per_param / 
                      self.parallel.tensor_parallel_size / 
                      (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['qkv_projection'] = {
            'compute': qkv_compute,
            'memory': qkv_memory,
            'total': max(qkv_compute, qkv_memory),
            'bottleneck': 'compute' if qkv_compute > qkv_memory else 'memory'
        }
        
        # 3. Attention Computation (QK^T, Softmax, Attention*V)
        qk_flops = B * self.model.n_heads * L * L * self.head_dim * self.flops_per_mac
        qk_compute = qk_flops / (self.gpu.compute_tflops * 1e12)
        
        attn_v_flops = B * self.model.n_heads * L * L * self.head_dim * self.flops_per_mac
        attn_v_compute = attn_v_flops / (self.gpu.compute_tflops * 1e12)
        
        # Softmax is memory-bound
        softmax_memory = (B * self.model.n_heads * L * L * 4 / 
                         (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['attention_compute'] = {
            'qk_matmul': qk_compute,
            'softmax': softmax_memory,
            'attn_v_matmul': attn_v_compute,
            'total': qk_compute + softmax_memory + attn_v_compute
        }
        
        # 4. Output Projection
        out_flops = B * L * H * H * self.flops_per_mac
        out_flops /= self.parallel.tensor_parallel_size
        out_compute = out_flops / (self.gpu.compute_tflops * 1e12)
        
        out_memory = (H * H * self.bytes_per_param / 
                     self.parallel.tensor_parallel_size / 
                     (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['attention_output_proj'] = {
            'compute': out_compute,
            'memory': out_memory,
            'total': max(out_compute, out_memory),
            'bottleneck': 'compute' if out_compute > out_memory else 'memory'
        }
        
        # 5. Attention Communication
        activation_size = B * L * H * self.bytes_per_activation
        attn_comm = self._estimate_comm_time(activation_size)
        breakdown['attention_communication'] = attn_comm
        
        # === MLP BLOCK ===
        
        # 6. MLP LayerNorm / RMSNorm
        breakdown['mlp_norm'] = self._norm_time(B, L, H)
        
        # 7. MLP Up Projection
        up_flops = B * L * H * D_ff * self.flops_per_mac
        up_flops /= self.parallel.tensor_parallel_size
        up_compute = up_flops / (self.gpu.compute_tflops * 1e12)
        
        up_memory = (H * D_ff * self.bytes_per_param / 
                    self.parallel.tensor_parallel_size / 
                    (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['mlp_up_proj'] = {
            'compute': up_compute,
            'memory': up_memory,
            'total': max(up_compute, up_memory),
            'bottleneck': 'compute' if up_compute > up_memory else 'memory'
        }
        
        # 8. Activation Function
        activation_memory = (B * L * D_ff * self.bytes_per_activation / 
                           (self.gpu.memory_bandwidth_gbs * 1e9))
        breakdown['mlp_activation'] = activation_memory
        
        # 9. MLP Down Projection
        down_flops = B * L * D_ff * H * self.flops_per_mac
        down_flops /= self.parallel.tensor_parallel_size
        down_compute = down_flops / (self.gpu.compute_tflops * 1e12)
        
        down_memory = (D_ff * H * self.bytes_per_param / 
                      self.parallel.tensor_parallel_size / 
                      (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['mlp_down_proj'] = {
            'compute': down_compute,
            'memory': down_memory,
            'total': max(down_compute, down_memory),
            'bottleneck': 'compute' if down_compute > down_memory else 'memory'
        }
        
        # 10. MLP Communication
        breakdown['mlp_communication'] = self._estimate_comm_time(activation_size)
        
        # 11. KV Cache Write
        kv_cache_bytes = 2 * B * L * H * self.bytes_per_activation
        breakdown['kv_cache_write'] = kv_cache_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
        
        return breakdown
    
    def breakdown_decode(self, batch_size: int, kv_cache_length: int) -> dict:
        """Get detailed breakdown of decode time."""
        B, T = batch_size, kv_cache_length
        H = self.model.hidden_size
        D_ff = self.model.ffn_dim
        
        breakdown = {}
        
        # === ATTENTION BLOCK ===
        
        # 1. Input Norm
        breakdown['attention_norm'] = self._norm_time(B, 1, H)
        
        # 2. QKV Projection
        qkv_flops = 3 * B * H * H * self.flops_per_mac
        qkv_flops /= self.parallel.tensor_parallel_size
        qkv_compute = qkv_flops / (self.gpu.compute_tflops * 1e12)
        
        qkv_memory = (3 * H * H * self.bytes_per_param / 
                      self.parallel.tensor_parallel_size / 
                      (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['qkv_projection'] = {
            'compute': qkv_compute,
            'memory': qkv_memory,
            'total': max(qkv_compute, qkv_memory),
            'bottleneck': 'compute' if qkv_compute > qkv_memory else 'memory'
        }
        
        # 3. KV Cache Read (MAJOR BOTTLENECK!)
        kv_cache_read_bytes = 2 * B * T * H * self.bytes_per_activation
        breakdown['kv_cache_read'] = kv_cache_read_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
        
        # 4. Attention Computation
        qk_flops = B * self.model.n_heads * T * self.head_dim * self.flops_per_mac
        qk_compute = qk_flops / (self.gpu.compute_tflops * 1e12)
        
        attn_v_flops = B * self.model.n_heads * T * self.head_dim * self.flops_per_mac
        attn_v_compute = attn_v_flops / (self.gpu.compute_tflops * 1e12)
        
        softmax_memory = (B * self.model.n_heads * T * 4 / 
                         (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['attention_compute'] = {
            'qk_matmul': qk_compute,
            'softmax': softmax_memory,
            'attn_v_matmul': attn_v_compute,
            'total': qk_compute + softmax_memory + attn_v_compute
        }
        
        # 5. Output Projection
        out_flops = B * H * H * self.flops_per_mac
        out_flops /= self.parallel.tensor_parallel_size
        out_compute = out_flops / (self.gpu.compute_tflops * 1e12)
        
        out_memory = (H * H * self.bytes_per_param / 
                     self.parallel.tensor_parallel_size / 
                     (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['attention_output_proj'] = {
            'compute': out_compute,
            'memory': out_memory,
            'total': max(out_compute, out_memory),
            'bottleneck': 'compute' if out_compute > out_memory else 'memory'
        }
        
        # 6. Attention Communication
        activation_size = B * H * self.bytes_per_activation
        breakdown['attention_communication'] = self._estimate_comm_time(activation_size)
        
        # === MLP BLOCK ===
        
        # 7. MLP Norm
        breakdown['mlp_norm'] = self._norm_time(B, 1, H)
        
        # 8. MLP Up Projection
        up_flops = B * H * D_ff * self.flops_per_mac
        up_flops /= self.parallel.tensor_parallel_size
        up_compute = up_flops / (self.gpu.compute_tflops * 1e12)
        
        up_memory = (H * D_ff * self.bytes_per_param / 
                    self.parallel.tensor_parallel_size / 
                    (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['mlp_up_proj'] = {
            'compute': up_compute,
            'memory': up_memory,
            'total': max(up_compute, up_memory),
            'bottleneck': 'compute' if up_compute > up_memory else 'memory'
        }
        
        # 9. Activation
        activation_memory = (B * D_ff * self.bytes_per_activation / 
                           (self.gpu.memory_bandwidth_gbs * 1e9))
        breakdown['mlp_activation'] = activation_memory
        
        # 10. MLP Down Projection
        down_flops = B * D_ff * H * self.flops_per_mac
        down_flops /= self.parallel.tensor_parallel_size
        down_compute = down_flops / (self.gpu.compute_tflops * 1e12)
        
        down_memory = (D_ff * H * self.bytes_per_param / 
                      self.parallel.tensor_parallel_size / 
                      (self.gpu.memory_bandwidth_gbs * 1e9))
        
        breakdown['mlp_down_proj'] = {
            'compute': down_compute,
            'memory': down_memory,
            'total': max(down_compute, down_memory),
            'bottleneck': 'compute' if down_compute > down_memory else 'memory'
        }
        
        # 11. MLP Communication
        breakdown['mlp_communication'] = self._estimate_comm_time(activation_size)
        
        # 12. KV Cache Write
        kv_cache_write_bytes = 2 * B * H * self.bytes_per_activation
        breakdown['kv_cache_write'] = kv_cache_write_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
        
        return breakdown
    
    def _norm_time(self, B: int, L: int, H: int) -> float:
        """LayerNorm / RMSNorm time (memory-bound)."""
        norm_bytes = 2 * B * L * H * self.bytes_per_activation
        return norm_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
    
    def _estimate_comm_time(self, activation_size: float) -> float:
        """Estimate communication time."""
        if self.parallel.tensor_parallel_size == 1:
            return 0.0
        
        from llm_inference_simulator.communication import estimate_collective_time
        
        return estimate_collective_time(
            self.tp_comm_strategy.attention_output.collective_op,
            activation_size,
            self.parallel.tensor_parallel_size,
            self.gpu.memory_bandwidth_gbs,
            5.0,
            self.tp_comm_strategy.attention_output.algorithm,
        )


def print_breakdown(breakdown: dict, title: str):
    """Pretty print breakdown."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    total_time = 0.0
    
    for component, value in breakdown.items():
        if isinstance(value, dict):
            # Check if it's a standard compute/memory breakdown
            if 'compute' in value and 'memory' in value:
                time = value['total']
                bottleneck = value.get('bottleneck', 'unknown')
                print(f"\n{component}:")
                print(f"  Compute:    {value['compute']*1000:>8.4f}ms")
                print(f"  Memory:     {value['memory']*1000:>8.4f}ms")
                print(f"  Bottleneck: {bottleneck}")
                print(f"  Total:      {value['total']*1000:>8.4f}ms")
            else:
                # attention_compute has different structure
                time = value['total']
                print(f"\n{component}:")
                for key, val in value.items():
                    if key != 'total':
                        print(f"  {key:15s}: {val*1000:>8.4f}ms")
                print(f"  {'total':15s}: {value['total']*1000:>8.4f}ms")
        else:
            time = value
            print(f"\n{component}: {value*1000:.4f}ms")
        
        total_time += time if not isinstance(value, dict) else value['total']
    
    print(f"\n{'='*70}")
    print(f"TOTAL PER LAYER: {total_time*1000:.4f}ms")
    print(f"{'='*70}")


def main():
    print("\n" + "#"*70)
    print("# Detailed Component Breakdown")
    print("#"*70)
    
    # Configuration
    model = get_model("llama-7b")
    gpu = get_gpu("A100-80GB")
    parallel = ParallelismSpec(tensor_parallel_size=1)
    comm_strategy = create_megatron_tp_strategy()
    
    perf_model = DetailedPerformanceModel(model, gpu, parallel, comm_strategy)
    
    print(f"\nModel: {model.name}")
    print(f"GPU: {gpu.name}")
    print(f"TP: {parallel.tensor_parallel_size}")
    
    # === PREFILL BREAKDOWN ===
    batch_size = 8
    seq_length = 512
    
    prefill_breakdown = perf_model.breakdown_prefill(batch_size, seq_length)
    print_breakdown(
        prefill_breakdown,
        f"PREFILL (Batch={batch_size}, SeqLen={seq_length})"
    )
    
    # === DECODE BREAKDOWN ===
    kv_cache_length = 512
    
    decode_breakdown = perf_model.breakdown_decode(batch_size, kv_cache_length)
    print_breakdown(
        decode_breakdown,
        f"DECODE (Batch={batch_size}, KV_Cache={kv_cache_length})"
    )
    
    # === COMPARISON ===
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")
    
    print("\nPREFILL:")
    print("  - Compute-bound (lots of matrix multiplications)")
    print("  - QKV projection and attention dominate")
    print("  - Batch size helps amortize costs")
    
    print("\nDECODE:")
    print("  - Memory-bound (KV cache reads dominate!)")
    print("  - Each token needs to read all previous KV cache")
    print("  - Longer sequences = slower decode")
    print("  - This is why FlashAttention, PagedAttention help")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
