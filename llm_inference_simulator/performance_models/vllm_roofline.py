"""
vLLM-calibrated Roofline performance model.

This model uses Roofline parameters estimated from vLLM benchmark data
instead of hardware specifications.
"""

from typing import Optional
from .base import BasePerformanceModel
from .roofline import RooflinePerformanceModel
from ..config import ModelSpec, ParallelismSpec
from ..xpu_spec import xPUSpec
from ..communication import TPCommunicationStrategy


class VLLMRooflineModel(BasePerformanceModel):
    """
    Roofline model with parameters calibrated from vLLM benchmarks.
    
    Uses separate Roofline parameters for Prefill and Decode phases,
    estimated from actual vLLM measurements.
    """
    
    def __init__(
        self,
        model_spec: ModelSpec,
        xpu_spec: xPUSpec,
        parallelism_spec: ParallelismSpec,
        roofline_params: 'RooflineParameters',  # From vllm_benchmarks.roofline_estimator
        tp_comm_strategy: Optional[TPCommunicationStrategy] = None
    ):
        """
        Initialize VLLM Roofline model.
        
        Args:
            model_spec: Model specification
            xpu_spec: xPU specification (for fallback/validation)
            parallelism_spec: Parallelism specification
            roofline_params: Roofline parameters estimated from vLLM benchmarks
            tp_comm_strategy: Optional TP communication strategy
        """
        self.model = model_spec
        self.xpu = xpu_spec
        self.parallel = parallelism_spec
        self.roofline_params = roofline_params
        
        # Derived parameters
        self.head_dim = self.model.hidden_size // self.model.n_heads
        self.bytes_per_param = self.model.weight_dtype.bytes_per_element()
        self.bytes_per_activation = self.model.activation_dtype.bytes_per_element()
        self.flops_per_mac = 2
        
        # Use base roofline model for breakdown calculations
        # (we'll override the parameters)
        self._base_roofline = RooflinePerformanceModel(
            model_spec=model_spec,
            xpu_spec=xpu_spec,
            parallelism_spec=parallelism_spec,
            tp_comm_strategy=tp_comm_strategy
        )
    
    def estimate_prefill_time(self, batch_size: int, seq_length: int) -> float:
        """
        Estimate prefill time using vLLM-calibrated Roofline parameters.
        
        Uses Prefill-specific parameters estimated from vLLM benchmarks.
        """
        # Calculate theoretical FLOPs and memory bytes
        theoretical_flops = self._compute_prefill_flops(batch_size, seq_length)
        theoretical_memory_bytes = self._compute_prefill_memory(batch_size, seq_length)
        
        # Roofline model with calibrated parameters
        compute_time = theoretical_flops / (
            self.roofline_params.prefill_effective_tflops * 1e12
        )
        memory_time = theoretical_memory_bytes / (
            self.roofline_params.prefill_effective_bandwidth_gbs * 1e9
        )
        
        # Roofline: max(compute, memory) + communication overhead
        total_time = max(compute_time, memory_time)
        total_time += self.roofline_params.prefill_comm_overhead_s
        
        return total_time
    
    def estimate_decode_time(self, batch_size: int, kv_cache_length: int) -> float:
        """
        Estimate decode time using vLLM-calibrated Roofline parameters.
        
        Uses Decode-specific parameters estimated from vLLM benchmarks.
        """
        # Calculate theoretical FLOPs and memory bytes
        theoretical_flops = self._compute_decode_flops(batch_size, kv_cache_length)
        theoretical_memory_bytes = self._compute_decode_memory(batch_size, kv_cache_length)
        
        # Roofline model with calibrated parameters
        compute_time = theoretical_flops / (
            self.roofline_params.decode_effective_tflops * 1e12
        )
        memory_time = theoretical_memory_bytes / (
            self.roofline_params.decode_effective_bandwidth_gbs * 1e9
        )
        
        # Roofline: max(compute, memory) + communication overhead
        total_time = max(compute_time, memory_time)
        total_time += self.roofline_params.decode_comm_overhead_s
        
        return total_time
    
    # ========== Theoretical computation helpers ==========
    
    def _compute_prefill_flops(self, batch_size: int, seq_length: int) -> float:
        """Compute theoretical FLOPs for prefill."""
        B, L = batch_size, seq_length
        H = self.model.hidden_size
        n_layers = self.model.n_layers
        tp = max(1, self.parallel.tensor_parallel_size)
        
        # Use breakdown from base roofline model for accuracy
        breakdown = self._base_roofline.breakdown_prefill(B, L)
        
        # Sum up FLOPs from breakdown (simplified - use time-based estimate)
        # For more accuracy, we could compute FLOPs directly, but using
        # the breakdown structure maintains consistency
        
        # Simplified FLOPs calculation
        # Attention: QK^T, Attention*V
        attention_flops = 2 * B * L * L * H * self.flops_per_mac
        # MLP
        mlp_flops = 2 * B * L * H * self.model.ffn_dim * self.flops_per_mac
        
        per_layer_flops = (attention_flops + mlp_flops) / tp
        total_flops = per_layer_flops * n_layers
        
        # Embedding and LM head
        embedding_flops = B * L * H * self.flops_per_mac
        lm_head_flops = B * L * H * self.model.vocab_size * self.flops_per_mac
        
        return total_flops + embedding_flops + lm_head_flops
    
    def _compute_prefill_memory(self, batch_size: int, seq_length: int) -> float:
        """Compute theoretical memory bytes for prefill."""
        B, L = batch_size, seq_length
        H = self.model.hidden_size
        tp = max(1, self.parallel.tensor_parallel_size)
        
        # Weight loading (sharded by TP)
        weight_bytes = (self.model.n_params * self.bytes_per_param) / tp
        
        # Activation memory (rough estimate)
        # Attention activations: Q, K, V, scores
        attn_activation_bytes = 4 * B * L * H * self.bytes_per_activation / tp
        # MLP activations
        mlp_activation_bytes = 2 * B * L * self.model.ffn_dim * self.bytes_per_activation / tp
        
        return weight_bytes + attn_activation_bytes + mlp_activation_bytes
    
    def _compute_decode_flops(self, batch_size: int, kv_cache_length: int) -> float:
        """Compute theoretical FLOPs for decode (per token)."""
        B = batch_size
        T = kv_cache_length
        H = self.model.hidden_size
        n_layers = self.model.n_layers
        tp = max(1, self.parallel.tensor_parallel_size)
        
        # Attention: QK^T (1 x T), Attention*V (1 x T)
        attention_flops = 2 * B * 1 * T * H * self.flops_per_mac
        # MLP
        mlp_flops = 2 * B * 1 * H * self.model.ffn_dim * self.flops_per_mac
        
        per_layer_flops = (attention_flops + mlp_flops) / tp
        total_flops = per_layer_flops * n_layers
        
        # LM head
        lm_head_flops = B * 1 * H * self.model.vocab_size * self.flops_per_mac
        
        return total_flops + lm_head_flops
    
    def _compute_decode_memory(self, batch_size: int, kv_cache_length: int) -> float:
        """Compute theoretical memory bytes for decode (KV cache read)."""
        B = batch_size
        T = kv_cache_length
        H = self.model.hidden_size
        n_layers = self.model.n_layers
        tp = max(1, self.parallel.tensor_parallel_size)
        
        # KV cache read (main bottleneck)
        hidden_per_tp = H / tp
        kv_cache_bytes = 2 * n_layers * B * T * hidden_per_tp * self.bytes_per_activation
        
        # Weight loading (sharded by TP)
        weight_bytes = (self.model.n_params * self.bytes_per_param) / tp
        
        return kv_cache_bytes + weight_bytes
