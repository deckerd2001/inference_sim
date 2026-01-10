"""
Roofline parameter estimation from vLLM benchmark data.

This module estimates Roofline model parameters (effective TFLOPS, bandwidth)
from vLLM benchmark measurements by analyzing compute-bound and memory-bound regions.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

from .schema import BenchmarkPoint
from llm_inference_simulator.config import ModelSpec
from llm_inference_simulator.xpu_spec import xPUSpec


@dataclass
class RooflineParameters:
    """
    Roofline model parameters estimated from vLLM benchmarks.
    
    Separate parameters for Prefill and Decode phases.
    """
    # Prefill parameters
    prefill_effective_tflops: float
    prefill_effective_bandwidth_gbs: float
    prefill_comm_overhead_s: float = 0.0
    
    # Decode parameters
    decode_effective_tflops: float
    decode_effective_bandwidth_gbs: float
    decode_comm_overhead_s: float = 0.0
    
    # Metadata
    xpu_name: str = ""
    model_name: str = ""


class RooflineParameterEstimator:
    """
    Estimates Roofline parameters from vLLM benchmark data.
    
    Strategy:
    1. Identify compute-bound points → estimate effective TFLOPS
    2. Identify memory-bound points → estimate effective bandwidth
    3. Separate estimation for Prefill and Decode
    """
    
    def __init__(self, model_spec: ModelSpec, xpu_spec: xPUSpec):
        self.model = model_spec
        self.xpu = xpu_spec
        
        # Derived parameters
        self.head_dim = model_spec.hidden_size // model_spec.n_heads
        self.bytes_per_param = model_spec.weight_dtype.bytes_per_element()
        self.bytes_per_activation = model_spec.activation_dtype.bytes_per_element()
        self.flops_per_mac = 2
    
    def estimate(
        self,
        prefill_benchmarks: List[BenchmarkPoint],
        decode_benchmarks: List[BenchmarkPoint],
        tp_size: int = 1
    ) -> RooflineParameters:
        """
        Estimate Roofline parameters from benchmark data.
        
        Args:
            prefill_benchmarks: List of prefill benchmark points
            decode_benchmarks: List of decode benchmark points
            tp_size: Tensor parallelism size
            
        Returns:
            RooflineParameters with estimated values
        """
        # Estimate Prefill parameters
        prefill_tflops = self._estimate_prefill_tflops(prefill_benchmarks, tp_size)
        prefill_bandwidth = self._estimate_prefill_bandwidth(prefill_benchmarks, tp_size)
        
        # Estimate Decode parameters
        decode_tflops = self._estimate_decode_tflops(decode_benchmarks, tp_size)
        decode_bandwidth = self._estimate_decode_bandwidth(decode_benchmarks, tp_size)
        
        # Use defaults if estimation failed
        if prefill_tflops is None:
            prefill_tflops = self.xpu.get_matmul_tflops(self.model.weight_dtype)
        if prefill_bandwidth is None:
            prefill_bandwidth = self.xpu.memory_bandwidth_gbs
        
        if decode_tflops is None:
            decode_tflops = self.xpu.get_matmul_tflops(self.model.weight_dtype)
        if decode_bandwidth is None:
            decode_bandwidth = self.xpu.memory_bandwidth_gbs
        
        return RooflineParameters(
            prefill_effective_tflops=prefill_tflops,
            prefill_effective_bandwidth_gbs=prefill_bandwidth,
            decode_effective_tflops=decode_tflops,
            decode_effective_bandwidth_gbs=decode_bandwidth,
            xpu_name=prefill_benchmarks[0].xpu_name if prefill_benchmarks else "",
            model_name=prefill_benchmarks[0].model_name if prefill_benchmarks else "",
        )
    
    def _estimate_prefill_tflops(
        self,
        benchmarks: List[BenchmarkPoint],
        tp_size: int
    ) -> Optional[float]:
        """
        Estimate effective TFLOPS from compute-bound prefill points.
        
        Strategy:
        - Identify points where compute intensity is high (FLOPs >> memory bytes)
        - For these points: effective_tflops = theoretical_flops / measured_time
        - Return median of estimates
        """
        compute_bound_points = []
        
        for point in benchmarks:
            if point.prefill_time_ms is None:
                continue
            
            # Calculate theoretical FLOPs and memory bytes
            theoretical_flops = self._compute_prefill_flops(
                point.batch_size, point.seq_length, tp_size
            )
            theoretical_memory = self._compute_prefill_memory(
                point.batch_size, point.seq_length, tp_size
            )
            
            # Compute intensity: FLOPs per byte
            if theoretical_memory > 0:
                intensity = theoretical_flops / theoretical_memory
                
                # Compute-bound: intensity > threshold (e.g., 10 FLOPs/byte)
                if intensity > 10.0:
                    measured_time_s = point.prefill_time_ms / 1000.0
                    effective_tflops = theoretical_flops / (measured_time_s * 1e12)
                    compute_bound_points.append(effective_tflops)
        
        if not compute_bound_points:
            return None
        
        return float(np.median(compute_bound_points))
    
    def _estimate_prefill_bandwidth(
        self,
        benchmarks: List[BenchmarkPoint],
        tp_size: int
    ) -> Optional[float]:
        """
        Estimate effective bandwidth from memory-bound prefill points.
        """
        memory_bound_points = []
        
        for point in benchmarks:
            if point.prefill_time_ms is None:
                continue
            
            theoretical_flops = self._compute_prefill_flops(
                point.batch_size, point.seq_length, tp_size
            )
            theoretical_memory = self._compute_prefill_memory(
                point.batch_size, point.seq_length, tp_size
            )
            
            if theoretical_memory > 0:
                intensity = theoretical_flops / theoretical_memory
                
                # Memory-bound: intensity < threshold (e.g., 1 FLOP/byte)
                if intensity < 1.0:
                    measured_time_s = point.prefill_time_ms / 1000.0
                    effective_bandwidth = theoretical_memory / (measured_time_s * 1e9)  # GB/s
                    memory_bound_points.append(effective_bandwidth)
        
        if not memory_bound_points:
            return None
        
        return float(np.median(memory_bound_points))
    
    def _estimate_decode_tflops(
        self,
        benchmarks: List[BenchmarkPoint],
        tp_size: int
    ) -> Optional[float]:
        """
        Estimate effective TFLOPS from decode points.
        
        Note: Decode is typically memory-bound, so this may be less reliable.
        """
        # Decode compute is usually small, so we might use hardware spec
        # or estimate from points with small KV cache (less memory-bound)
        compute_bound_points = []
        
        for point in benchmarks:
            if point.decode_time_ms is None:
                continue
            
            theoretical_flops = self._compute_decode_flops(
                point.batch_size, point.kv_cache_length, tp_size
            )
            theoretical_memory = self._compute_decode_memory(
                point.batch_size, point.kv_cache_length, tp_size
            )
            
            if theoretical_memory > 0:
                intensity = theoretical_flops / theoretical_memory
                
                # For decode, compute-bound is rare, but check anyway
                if intensity > 5.0:  # Lower threshold for decode
                    measured_time_s = point.decode_time_ms / 1000.0
                    effective_tflops = theoretical_flops / (measured_time_s * 1e12)
                    compute_bound_points.append(effective_tflops)
        
        if compute_bound_points:
            return float(np.median(compute_bound_points))
        
        # Fallback: use hardware spec
        return self.xpu.get_matmul_tflops(self.model.weight_dtype)
    
    def _estimate_decode_bandwidth(
        self,
        benchmarks: List[BenchmarkPoint],
        tp_size: int
    ) -> Optional[float]:
        """
        Estimate effective bandwidth from decode points.
        
        Decode is typically memory-bound (KV cache read), so this is the main parameter.
        """
        bandwidth_points = []
        
        for point in benchmarks:
            if point.decode_time_ms is None:
                continue
            
            # KV cache read is the main memory operation
            kv_cache_bytes = self._compute_kv_cache_read_bytes(
                point.batch_size, point.kv_cache_length, tp_size
            )
            
            measured_time_s = point.decode_time_ms / 1000.0
            if measured_time_s > 0:
                effective_bandwidth = kv_cache_bytes / (measured_time_s * 1e9)  # GB/s
                bandwidth_points.append(effective_bandwidth)
        
        if not bandwidth_points:
            return None
        
        return float(np.median(bandwidth_points))
    
    # ========== Theoretical computation helpers ==========
    
    def _compute_prefill_flops(self, batch_size: int, seq_length: int, tp_size: int) -> float:
        """Compute theoretical FLOPs for prefill."""
        B, L = batch_size, seq_length
        H = self.model.hidden_size
        n_layers = self.model.n_layers
        
        # Per layer FLOPs
        # Attention: QK^T, Attention*V
        attention_flops = 2 * B * L * L * H * self.flops_per_mac  # Simplified
        # MLP
        mlp_flops = 2 * B * L * H * self.model.ffn_dim * self.flops_per_mac
        
        per_layer_flops = attention_flops + mlp_flops
        per_layer_flops /= tp_size  # TP sharding
        
        total_flops = per_layer_flops * n_layers
        
        # Add embedding and LM head (simplified)
        embedding_flops = B * L * H * self.flops_per_mac
        lm_head_flops = B * L * H * self.model.vocab_size * self.flops_per_mac
        
        return total_flops + embedding_flops + lm_head_flops
    
    def _compute_prefill_memory(self, batch_size: int, seq_length: int, tp_size: int) -> float:
        """Compute theoretical memory bytes for prefill."""
        B, L = batch_size, seq_length
        H = self.model.hidden_size
        
        # Weight loading (simplified: assume all weights loaded)
        # This is a rough estimate
        weight_bytes = (self.model.n_params * self.bytes_per_param) / tp_size
        
        # Activation memory (simplified)
        activation_bytes = B * L * H * self.bytes_per_activation * 4  # Rough estimate
        
        return weight_bytes + activation_bytes
    
    def _compute_decode_flops(self, batch_size: int, kv_cache_length: int, tp_size: int) -> float:
        """Compute theoretical FLOPs for decode (per token)."""
        B = batch_size
        T = kv_cache_length
        H = self.model.hidden_size
        n_layers = self.model.n_layers
        
        # Per layer: attention on 1 new token with T cached tokens
        attention_flops = 2 * B * 1 * T * H * self.flops_per_mac  # Simplified
        mlp_flops = 2 * B * 1 * H * self.model.ffn_dim * self.flops_per_mac
        
        per_layer_flops = (attention_flops + mlp_flops) / tp_size
        total_flops = per_layer_flops * n_layers
        
        # LM head
        lm_head_flops = B * 1 * H * self.model.vocab_size * self.flops_per_mac
        
        return total_flops + lm_head_flops
    
    def _compute_decode_memory(self, batch_size: int, kv_cache_length: int, tp_size: int) -> float:
        """Compute theoretical memory bytes for decode (KV cache read)."""
        return self._compute_kv_cache_read_bytes(batch_size, kv_cache_length, tp_size)
    
    def _compute_kv_cache_read_bytes(
        self,
        batch_size: int,
        kv_cache_length: int,
        tp_size: int
    ) -> float:
        """Compute KV cache read bytes."""
        B = batch_size
        T = kv_cache_length
        H = self.model.hidden_size
        n_layers = self.model.n_layers
        
        # KV cache: 2 (K and V) * n_layers * batch * length * hidden_size
        hidden_per_tp = H / tp_size
        kv_bytes = 2 * n_layers * B * T * hidden_per_tp * self.bytes_per_activation
        
        # Also include weight loading (simplified)
        weight_bytes = (self.model.n_params * self.bytes_per_param) / tp_size
        
        return kv_bytes + weight_bytes
