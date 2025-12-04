"""
Performance modeling for LLM inference.
Calculates compute, memory, and communication costs for Prefill and Decode phases.
"""

import math
from typing import Tuple
from .config import ModelSpec, GPUSpec, ParallelismSpec, DataType


class PerformanceModel:
    """
    Models the performance of Transformer-based LLM inference.
    
    This includes:
    - Compute costs (FLOPs) for attention and MLP
    - Memory costs (bytes transferred)
    - Communication costs for parallelism
    """
    
    def __init__(self, model_spec: ModelSpec, gpu_spec: GPUSpec, 
                 parallelism_spec: ParallelismSpec):
        self.model = model_spec
        self.gpu = gpu_spec
        self.parallel = parallelism_spec
        
        # Derived parameters
        self.head_dim = self.model.hidden_size // self.model.n_heads
        self.bytes_per_param = self.model.weight_dtype.bytes_per_element()
        self.bytes_per_activation = self.model.activation_dtype.bytes_per_element()
        
        # FLOPS per operation (2 for multiply-add)
        self.flops_per_mac = 2
    
    def estimate_prefill_time(self, batch_size: int, seq_length: int) -> float:
        """
        Estimate time for prefill phase.
        
        Args:
            batch_size: Number of sequences in batch
            seq_length: Input sequence length
            
        Returns:
            Estimated time in seconds
        """
        # Calculate per-layer costs
        compute_time = self._prefill_compute_time(batch_size, seq_length)
        memory_time = self._prefill_memory_time(batch_size, seq_length)
        comm_time = self._prefill_communication_time(batch_size, seq_length)
        
        # Total time per layer (assuming compute and memory can overlap partially)
        # We use max of compute and memory, plus communication
        per_layer_time = max(compute_time, memory_time) + comm_time
        
        # Total for all layers
        total_time = per_layer_time * self.model.n_layers
        
        # Add embedding and output projection overhead (simplified)
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
        # Calculate per-layer costs for decode
        compute_time = self._decode_compute_time(batch_size, kv_cache_length)
        memory_time = self._decode_memory_time(batch_size, kv_cache_length)
        comm_time = self._decode_communication_time(batch_size, kv_cache_length)
        
        # Total time per layer
        per_layer_time = max(compute_time, memory_time) + comm_time
        
        # Total for all layers
        total_time = per_layer_time * self.model.n_layers
        
        # Add output projection overhead
        lm_head_time = self._lm_head_time(batch_size, 1)
        
        return total_time + lm_head_time
    
    # ========== PREFILL PHASE ==========
    
    def _prefill_compute_time(self, B: int, L: int) -> float:
        """
        Compute time for prefill (one layer).
        
        Args:
            B: batch size
            L: sequence length
        """
        H = self.model.hidden_size
        D_ff = self.model.ffn_dim
        
        # Attention block FLOPs
        # Q, K, V projections: 3 * (B * L * H * H)
        qkv_flops = 3 * B * L * H * H
        
        # QK^T: B * n_heads * L * L * head_dim
        qk_flops = B * self.model.n_heads * L * L * self.head_dim
        
        # Attention * V: B * n_heads * L * L * head_dim
        attn_v_flops = B * self.model.n_heads * L * L * self.head_dim
        
        # Output projection: B * L * H * H
        out_proj_flops = B * L * H * H
        
        attention_flops = qkv_flops + qk_flops + attn_v_flops + out_proj_flops
        
        # MLP block FLOPs
        # Up projection: B * L * H * D_ff
        # Down projection: B * L * D_ff * H
        mlp_flops = B * L * H * D_ff + B * L * D_ff * H
        
        # Total FLOPs per layer
        total_flops = (attention_flops + mlp_flops) * self.flops_per_mac
        
        # Adjust for tensor parallelism (work is divided)
        effective_flops = total_flops / self.parallel.tensor_parallel_size
        
        # Convert to time using GPU compute capability
        # gpu.compute_tflops is in TFLOPS, so we divide by 10^12
        compute_time = effective_flops / (self.gpu.compute_tflops * 1e12)
        
        return compute_time
    
    def _prefill_memory_time(self, B: int, L: int) -> float:
        """
        Memory transfer time for prefill (one layer).
        Memory-bound operations: loading weights and KV cache writes.
        """
        H = self.model.hidden_size
        D_ff = self.model.ffn_dim
        
        # Weight loading (bytes)
        # QKV weights: 3 * H * H
        # Output projection: H * H
        # MLP up: H * D_ff
        # MLP down: D_ff * H
        weight_params = 3 * H * H + H * H + H * D_ff + D_ff * H
        weight_bytes = weight_params * self.bytes_per_param
        
        # Adjust for tensor parallelism (weights are sharded)
        weight_bytes = weight_bytes / self.parallel.tensor_parallel_size
        
        # Activation memory (KV cache writes)
        # K, V cache: 2 * B * L * H
        kv_cache_bytes = 2 * B * L * H * self.bytes_per_activation
        
        # Total memory traffic
        total_bytes = weight_bytes + kv_cache_bytes
        
        # Convert to time using memory bandwidth
        memory_time = total_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
        
        return memory_time
    
    def _prefill_communication_time(self, B: int, L: int) -> float:
        """
        Communication time for prefill (one layer).
        Primarily all-reduce operations for tensor parallelism.
        """
        if self.parallel.tensor_parallel_size == 1:
            return 0.0
        
        H = self.model.hidden_size
        
        # All-reduce after attention and MLP
        # Size: B * L * H for each
        allreduce_size = 2 * B * L * H * self.bytes_per_activation
        
        # Simplified ring all-reduce model
        # Time = (2 * (TP-1) / TP) * (data_size / bandwidth) + latency
        tp_size = self.parallel.tensor_parallel_size
        bandwidth = self.gpu.memory_bandwidth_gbs * 1e9  # Use conservative estimate
        latency = 5e-6  # 5 microseconds
        
        comm_time = (2 * (tp_size - 1) / tp_size) * (allreduce_size / bandwidth) + latency
        
        return comm_time
    
    # ========== DECODE PHASE ==========
    
    def _decode_compute_time(self, B: int, T: int) -> float:
        """
        Compute time for decode (one layer, one token).
        
        Args:
            B: batch size
            T: KV cache length (context length so far)
        """
        H = self.model.hidden_size
        D_ff = self.model.ffn_dim
        
        # Attention block FLOPs (for generating 1 token)
        # Q, K, V projections: 3 * (B * 1 * H * H)
        qkv_flops = 3 * B * H * H
        
        # QK^T: B * n_heads * 1 * T * head_dim
        # (query is 1 token, key is T tokens)
        qk_flops = B * self.model.n_heads * T * self.head_dim
        
        # Attention * V: B * n_heads * 1 * T * head_dim
        attn_v_flops = B * self.model.n_heads * T * self.head_dim
        
        # Output projection: B * 1 * H * H
        out_proj_flops = B * H * H
        
        attention_flops = qkv_flops + qk_flops + attn_v_flops + out_proj_flops
        
        # MLP block FLOPs (for 1 token)
        mlp_flops = B * H * D_ff + B * D_ff * H
        
        # Total FLOPs per layer
        total_flops = (attention_flops + mlp_flops) * self.flops_per_mac
        
        # Adjust for tensor parallelism
        effective_flops = total_flops / self.parallel.tensor_parallel_size
        
        # Convert to time
        compute_time = effective_flops / (self.gpu.compute_tflops * 1e12)
        
        return compute_time
    
    def _decode_memory_time(self, B: int, T: int) -> float:
        """
        Memory transfer time for decode (one layer, one token).
        This is often the bottleneck in decode phase.
        """
        H = self.model.hidden_size
        D_ff = self.model.ffn_dim
        
        # Weight loading (same as prefill)
        weight_params = 3 * H * H + H * H + H * D_ff + D_ff * H
        weight_bytes = weight_params * self.bytes_per_param
        weight_bytes = weight_bytes / self.parallel.tensor_parallel_size
        
        # KV cache read: need to read all T tokens for K and V
        # 2 * B * T * H
        kv_cache_read_bytes = 2 * B * T * H * self.bytes_per_activation
        
        # KV cache write: write new K, V for current token
        # 2 * B * 1 * H
        kv_cache_write_bytes = 2 * B * H * self.bytes_per_activation
        
        # Total memory traffic
        total_bytes = weight_bytes + kv_cache_read_bytes + kv_cache_write_bytes
        
        # Convert to time
        memory_time = total_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
        
        return memory_time
    
    def _decode_communication_time(self, B: int, T: int) -> float:
        """
        Communication time for decode (one layer, one token).
        """
        if self.parallel.tensor_parallel_size == 1:
            return 0.0
        
        H = self.model.hidden_size
        
        # All-reduce after attention and MLP (for 1 token)
        allreduce_size = 2 * B * H * self.bytes_per_activation
        
        # Simplified ring all-reduce model
        tp_size = self.parallel.tensor_parallel_size
        bandwidth = self.gpu.memory_bandwidth_gbs * 1e9
        latency = 5e-6
        
        comm_time = (2 * (tp_size - 1) / tp_size) * (allreduce_size / bandwidth) + latency
        
        return comm_time
    
    # ========== AUXILIARY OPERATIONS ==========
    
    def _embedding_time(self, B: int, L: int) -> float:
        """Time for embedding lookup (prefill)."""
        # Simple lookup: B * L * H
        # This is typically memory-bound
        embed_bytes = B * L * self.model.hidden_size * self.bytes_per_activation
        return embed_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
    
    def _lm_head_time(self, B: int, L: int) -> float:
        """Time for final language model head projection."""
        # Matrix multiply: B * L * H * vocab_size
        flops = B * L * self.model.hidden_size * self.model.vocab_size * self.flops_per_mac
        
        # This can be compute or memory bound
        compute_time = flops / (self.gpu.compute_tflops * 1e12)
        
        # Memory time (loading LM head weights)
        weight_bytes = self.model.hidden_size * self.model.vocab_size * self.bytes_per_param
        memory_time = weight_bytes / (self.gpu.memory_bandwidth_gbs * 1e9)
        
        return max(compute_time, memory_time)
    
    def estimate_kv_cache_size(self, batch_size: int, max_seq_length: int) -> float:
        """
        Estimate KV cache memory size in GB.
        
        Args:
            batch_size: Number of sequences
            max_seq_length: Maximum sequence length
            
        Returns:
            Memory size in GB
        """
        # KV cache: 2 (K and V) * n_layers * batch_size * max_seq_length * hidden_size
        kv_cache_elements = (2 * self.model.n_layers * batch_size * 
                            max_seq_length * self.model.hidden_size)
        kv_cache_bytes = kv_cache_elements * self.bytes_per_activation
        kv_cache_gb = kv_cache_bytes / (1024 ** 3)
        
        return kv_cache_gb
