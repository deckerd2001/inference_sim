"""
Performance modeling for LLM inference.
Calculates compute, memory, and communication costs for Prefill and Decode phases.

Each component uses the roofline model: time = max(compute_time, memory_time)
"""

import math
from typing import Tuple, Optional, Dict, Any
from .config import ModelSpec, ParallelismSpec
from .xpu_spec import DataType, OperationType, xPUSpec
from .communication import (
    TPCommunicationStrategy,
    estimate_collective_time,
    create_megatron_tp_strategy,
)


class PerformanceModel:
    """
    Models the performance of Transformer-based LLM inference.
    
    This includes:
    - Compute costs (FLOPs) for attention and MLP
    - Memory costs (bytes transferred)
    - Communication costs for parallelism
    
    Each component is modeled separately with roofline analysis.
    """
    
    def __init__(self, model_spec: ModelSpec, xpu_spec: xPUSpec, 
                 parallelism_spec: ParallelismSpec,
                 tp_comm_strategy: Optional[TPCommunicationStrategy] = None):
        self.model = model_spec
        self.xpu = xpu_spec
        self.parallel = parallelism_spec
        
        # Communication strategy
        if tp_comm_strategy is None:
            self.tp_comm_strategy = create_megatron_tp_strategy()
        else:
            self.tp_comm_strategy = tp_comm_strategy
        
        # Derived parameters
        self.head_dim = self.model.hidden_size // self.model.n_heads
        self.bytes_per_param = self.model.weight_dtype.bytes_per_element()
        self.bytes_per_activation = self.model.activation_dtype.bytes_per_element()
        
        # FLOPS per operation (2 for multiply-add)
        self.flops_per_mac = 2
    
    # ========== HIGH-LEVEL ESTIMATION ==========
    
    def estimate_prefill_time(self, batch_size: int, seq_length: int) -> float:
        """
        Estimate time for prefill phase.
        
        Args:
            batch_size: Number of sequences in batch
            seq_length: Input sequence length
            
        Returns:
            Estimated time in seconds
        """
        breakdown = self.breakdown_prefill(batch_size, seq_length)
        
        # Sum up all components
        per_layer_time = self._sum_breakdown(breakdown)
        
        # Total for all layers
        total_time = per_layer_time * self.model.n_layers
        
        # Add embedding and output projection overhead
        embedding_time = self._embedding_time(batch_size, seq_length)
        lm_head_time = self._lm_head_time(batch_size, seq_length)
        
        return total_time + embedding_time + lm_head_time
    
    def estimate_decode_time(self, batch_size: int, kv_cache_length: int) -> float:
        """
        Estimate time for a single decode step.
        
        Args:
            batch_size: Number of sequences in batch
            kv_cache_length: Current length of KV cache (context length)
            
        Returns:
            Estimated time in seconds
        """
        breakdown = self.breakdown_decode(batch_size, kv_cache_length)
        
        # Sum up all components
        per_layer_time = self._sum_breakdown(breakdown)
        
        # Total for all layers
        total_time = per_layer_time * self.model.n_layers
        
        # Add output projection overhead
        lm_head_time = self._lm_head_time(batch_size, 1)
        
        return total_time + lm_head_time
    
    # ========== DETAILED BREAKDOWN ==========
    
    def breakdown_prefill(self, batch_size: int, seq_length: int) -> Dict[str, Any]:
        """
        Get detailed breakdown of prefill time by component.
        
        Each component includes roofline analysis where applicable.
        
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            Dictionary with detailed timing for each component
        """
        B, L = batch_size, seq_length
        H = self.model.hidden_size
        
        breakdown = {}
        
        # === ATTENTION BLOCK ===
        breakdown['attention_norm'] = self._norm_time(B, L, H)
        breakdown['qkv_projection'] = self._qkv_projection_time(B, L)
        breakdown['attention_compute'] = self._attention_compute_time(B, L, L)
        breakdown['attention_output_proj'] = self._linear_layer_time(B, L, H, H)
        breakdown['attention_communication'] = self._attention_comm_time(B, L)
        
        # === MLP BLOCK ===
        breakdown['mlp_norm'] = self._norm_time(B, L, H)
        breakdown['mlp_up_proj'] = self._linear_layer_time(B, L, H, self.model.ffn_dim)
        breakdown['mlp_activation'] = self._activation_time(B, L, self.model.ffn_dim)
        breakdown['mlp_down_proj'] = self._linear_layer_time(B, L, self.model.ffn_dim, H)
        breakdown['mlp_communication'] = self._mlp_comm_time(B, L)
        
        # === KV CACHE ===
        breakdown['kv_cache_write'] = self._kv_cache_write_time(B, L)
        
        return breakdown
    
    def breakdown_decode(self, batch_size: int, kv_cache_length: int) -> Dict[str, Any]:
        """
        Get detailed breakdown of decode time by component.
        
        Each component includes roofline analysis where applicable.
        
        Args:
            batch_size: Batch size
            kv_cache_length: Current KV cache length
            
        Returns:
            Dictionary with detailed timing for each component
        """
        B, T = batch_size, kv_cache_length
        H = self.model.hidden_size
        
        breakdown = {}
        
        # === ATTENTION BLOCK ===
        breakdown['attention_norm'] = self._norm_time(B, 1, H)
        breakdown['qkv_projection'] = self._qkv_projection_time(B, 1)
        breakdown['kv_cache_read'] = self._kv_cache_read_time(B, T)  # Major bottleneck!
        breakdown['attention_compute'] = self._attention_compute_time(B, 1, T)
        breakdown['attention_output_proj'] = self._linear_layer_time(B, 1, H, H)
        breakdown['attention_communication'] = self._attention_comm_time(B, 1)
        
        # === MLP BLOCK ===
        breakdown['mlp_norm'] = self._norm_time(B, 1, H)
        breakdown['mlp_up_proj'] = self._linear_layer_time(B, 1, H, self.model.ffn_dim)
        breakdown['mlp_activation'] = self._activation_time(B, 1, self.model.ffn_dim)
        breakdown['mlp_down_proj'] = self._linear_layer_time(B, 1, self.model.ffn_dim, H)
        breakdown['mlp_communication'] = self._mlp_comm_time(B, 1)
        
        # === KV CACHE ===
        breakdown['kv_cache_write'] = self._kv_cache_write_time(B, 1)
        
        return breakdown
    
    # ========== COMPONENT FUNCTIONS (with Roofline) ==========
    
    def _linear_layer_time(self, batch_size: int, seq_length: int,
                          input_dim: int, output_dim: int) -> Dict[str, float]:
        """
        Generic linear layer: Y = X @ W
        
        Applies roofline model: time = max(compute_time, memory_time)
        
        Args:
            batch_size: B
            seq_length: L
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            
        Returns:
            Dict with 'compute', 'memory', 'total', 'bottleneck'
        """
        B, L = batch_size, seq_length
        
        # Compute time: B * L * input_dim * output_dim FLOPs
        flops = B * L * input_dim * output_dim * self.flops_per_mac
        flops /= self.parallel.tensor_parallel_size  # Split across TP
        compute_time = flops / (self.xpu.get_matmul_tflops(self.model.weight_dtype) * 1e12)
        
        # Memory time: Loading weights from HBM
        weight_bytes = input_dim * output_dim * self.bytes_per_param
        weight_bytes /= self.parallel.tensor_parallel_size  # Split across TP
        memory_time = weight_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
        
        # Roofline: Bottleneck is the slower one
        total_time = max(compute_time, memory_time)
        bottleneck = 'compute' if compute_time > memory_time else 'memory'
        
        return {
            'compute': compute_time,
            'memory': memory_time,
            'total': total_time,
            'bottleneck': bottleneck
        }
    
    def _qkv_projection_time(self, batch_size: int, seq_length: int) -> Dict[str, float]:
        """
        QKV projection: 3 linear layers in parallel.
        
        X @ W_q, X @ W_k, X @ W_v (computed together)
        """
        H = self.model.hidden_size
        
        # 3x the size of a single linear layer
        flops = 3 * batch_size * seq_length * H * H * self.flops_per_mac
        flops /= self.parallel.tensor_parallel_size
        compute_time = flops / (self.xpu.get_matmul_tflops(self.model.weight_dtype) * 1e12)
        
        weight_bytes = 3 * H * H * self.bytes_per_param
        weight_bytes /= self.parallel.tensor_parallel_size
        memory_time = weight_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
        
        return {
            'compute': compute_time,
            'memory': memory_time,
            'total': max(compute_time, memory_time),
            'bottleneck': 'compute' if compute_time > memory_time else 'memory'
        }
    
    def _attention_compute_time(self, batch_size: int, query_length: int, 
                               kv_length: int) -> Dict[str, float]:
        """
        Attention computation: QK^T, Softmax, Attention*V
        
        Args:
            query_length: Length of query (L for prefill, 1 for decode)
            kv_length: Length of key/value (L for prefill, T for decode)
            
        Returns:
            Dict with breakdown of QK matmul, softmax, and attention*V
        """
        B = batch_size
        H = self.model.n_heads
        D = self.head_dim
        Q_len = query_length
        KV_len = kv_length
        
        # 1. QK^T: [B, H, Q_len, D] @ [B, H, D, KV_len] -> [B, H, Q_len, KV_len]
        qk_flops = B * H * Q_len * KV_len * D * self.flops_per_mac
        qk_compute = qk_flops / (self.xpu.get_matmul_tflops(self.model.weight_dtype) * 1e12)
        
        # 2. Softmax: Memory-bound operation
        # Read scores, compute exp, sum, divide, write back
        softmax_elements = B * H * Q_len * KV_len
        softmax_bytes = softmax_elements * 4  # FP32 for numerical stability
        softmax_time = softmax_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
        
        # 3. Attention * V: [B, H, Q_len, KV_len] @ [B, H, KV_len, D] -> [B, H, Q_len, D]
        attn_v_flops = B * H * Q_len * KV_len * D * self.flops_per_mac
        attn_v_compute = attn_v_flops / (self.xpu.get_matmul_tflops(self.model.weight_dtype) * 1e12)
        
        total = qk_compute + softmax_time + attn_v_compute
        
        return {
            'qk_matmul': qk_compute,
            'softmax': softmax_time,
            'attn_v_matmul': attn_v_compute,
            'total': total,
            'bottleneck': 'mixed'  # Has both compute and memory components
        }
    
    def _norm_time(self, batch_size: int, seq_length: int, hidden_size: int) -> float:
        """
        LayerNorm / RMSNorm time.
        
        Pure memory-bound operation: read input, compute stats, normalize, write output.
        """
        # Read + Write (2x)
        norm_bytes = 2 * batch_size * seq_length * hidden_size * self.bytes_per_activation
        return norm_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
    
    def _activation_time(self, batch_size: int, seq_length: int, dim: int) -> float:
        """
        Activation function (SwiGLU, GELU, etc).
        
        Pure memory-bound: read input, apply function, write output.
        """
        activation_bytes = 2 * batch_size * seq_length * dim * self.bytes_per_activation
        return activation_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
    
    def _kv_cache_read_time(self, batch_size: int, kv_length: int) -> float:
        """
        KV cache read time (decode phase).
        
        This is often the bottleneck in decode!
        Must read all previous K and V for attention computation.
        
        Pure memory operation.
        """
        # 2 = K and V
        kv_bytes = 2 * batch_size * kv_length * self.model.hidden_size * self.bytes_per_activation
        return kv_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
    
    def _kv_cache_write_time(self, batch_size: int, seq_length: int) -> float:
        """
        KV cache write time.
        
        Write newly computed K and V to cache.
        Pure memory operation.
        """
        kv_bytes = 2 * batch_size * seq_length * self.model.hidden_size * self.bytes_per_activation
        return kv_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
    
    def _attention_comm_time(self, batch_size: int, seq_length: int) -> float:
        """Communication time for attention output (Tensor Parallelism)."""
        if self.parallel.tensor_parallel_size == 1:
            return 0.0
        
        activation_size = batch_size * seq_length * self.model.hidden_size * self.bytes_per_activation
        
        return estimate_collective_time(
            self.tp_comm_strategy.attention_output.collective_op,
            activation_size,
            self.parallel.tensor_parallel_size,
            self.xpu.intra_node_bandwidth_gbs,
            5.0,  # latency_us
            self.tp_comm_strategy.attention_output.algorithm,
        )
    
    def _mlp_comm_time(self, batch_size: int, seq_length: int) -> float:
        """Communication time for MLP output (Tensor Parallelism)."""
        if self.parallel.tensor_parallel_size == 1:
            return 0.0
        
        activation_size = batch_size * seq_length * self.model.hidden_size * self.bytes_per_activation
        
        return estimate_collective_time(
            self.tp_comm_strategy.mlp_down_projection.collective_op,
            activation_size,
            self.parallel.tensor_parallel_size,
            self.xpu.intra_node_bandwidth_gbs,  # NVLink bandwidth for TP communication
            5.0,
            self.tp_comm_strategy.mlp_down_projection.algorithm,
        )
    
    # ========== HELPER METHODS ==========
    
    def _embedding_time(self, batch_size: int, seq_length: int) -> float:
        """Time for embedding lookup (prefill only)."""
        embed_bytes = batch_size * seq_length * self.model.hidden_size * self.bytes_per_activation
        return embed_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
    
    def _lm_head_time(self, batch_size: int, seq_length: int) -> float:
        """Time for final language model head projection."""
        flops = (batch_size * seq_length * self.model.hidden_size * 
                self.model.vocab_size * self.flops_per_mac)
        compute_time = flops / (self.xpu.get_matmul_tflops(self.model.weight_dtype) * 1e12)
        
        weight_bytes = self.model.hidden_size * self.model.vocab_size * self.bytes_per_param
        memory_time = weight_bytes / (self.xpu.memory_bandwidth_gbs * 1e9)
        
        return max(compute_time, memory_time)
    
    def _sum_breakdown(self, breakdown: Dict[str, Any]) -> float:
        """Sum all component times from breakdown."""
        total = 0.0
        for key, val in breakdown.items():
            if isinstance(val, dict):
                total += val['total']
            else:
                total += val
        return total
    
    def estimate_kv_cache_size(self, batch_size: int, max_seq_length: int) -> float:
        """
        Estimate KV cache memory size in GB.
        
        Args:
            batch_size: Number of sequences
            max_seq_length: Maximum sequence length
            
        Returns:
            KV cache size in GB
        """
        kv_cache_elements = (2 * self.model.n_layers * batch_size * 
                            max_seq_length * self.model.hidden_size)
        kv_cache_bytes = kv_cache_elements * self.bytes_per_activation
        kv_cache_gb = kv_cache_bytes / (1024 ** 3)
        
        return kv_cache_gb
