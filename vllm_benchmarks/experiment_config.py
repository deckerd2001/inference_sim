"""
Experiment design for vLLM benchmarks.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ExperimentPoint:
    """Single experiment point configuration."""
    batch_size: int
    seq_length: int = 0  # For prefill
    kv_cache_length: int = 0  # For decode
    is_prefill: bool = True


@dataclass
class ExperimentConfig:
    """
    Configuration for vLLM benchmark experiments.
    
    Defines the parameter space to explore:
    - Batch sizes
    - Sequence lengths (prefill)
    - KV cache lengths (decode)
    """
    model_name: str
    xpu_name: str
    tp_size: int = 1
    
    # Prefill experiment points
    prefill_batch_sizes: List[int] = None
    prefill_seq_lengths: List[int] = None
    
    # Decode experiment points
    decode_batch_sizes: List[int] = None
    decode_kv_cache_lengths: List[int] = None
    
    # Experiment settings
    num_runs: int = 10  # Number of runs per point for averaging
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.prefill_batch_sizes is None:
            self.prefill_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        if self.prefill_seq_lengths is None:
            self.prefill_seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        
        if self.decode_batch_sizes is None:
            self.decode_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        if self.decode_kv_cache_lengths is None:
            self.decode_kv_cache_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    def generate_prefill_points(self) -> List[ExperimentPoint]:
        """
        Generate all prefill experiment points (full grid).
        
        Returns:
            List of ExperimentPoint for prefill
        """
        points = []
        for batch_size in self.prefill_batch_sizes:
            for seq_length in self.prefill_seq_lengths:
                points.append(ExperimentPoint(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    is_prefill=True
                ))
        return points
    
    def generate_decode_points(self) -> List[ExperimentPoint]:
        """
        Generate all decode experiment points (full grid).
        
        Returns:
            List of ExperimentPoint for decode
        """
        points = []
        for batch_size in self.decode_batch_sizes:
            for kv_cache_length in self.decode_kv_cache_lengths:
                points.append(ExperimentPoint(
                    batch_size=batch_size,
                    kv_cache_length=kv_cache_length,
                    is_prefill=False
                ))
        return points
    
    def generate_all_points(self) -> Tuple[List[ExperimentPoint], List[ExperimentPoint]]:
        """
        Generate all experiment points.
        
        Returns:
            (prefill_points, decode_points)
        """
        return self.generate_prefill_points(), self.generate_decode_points()
    
    def generate_sparse_grid(
        self,
        prefill_points: int = 20,
        decode_points: int = 20
    ) -> Tuple[List[ExperimentPoint], List[ExperimentPoint]]:
        """
        Generate sparse grid using Latin Hypercube Sampling (LHS).
        
        Useful for initial exploration with fewer points.
        
        Args:
            prefill_points: Number of prefill points to sample
            decode_points: Number of decode points to sample
            
        Returns:
            (prefill_points, decode_points)
        """
        # Simple implementation: sample from grid
        # TODO: Implement proper LHS if needed
        
        prefill = []
        for batch_size in self.prefill_batch_sizes[::2]:  # Sample every other
            for seq_length in self.prefill_seq_lengths[::2]:
                if len(prefill) < prefill_points:
                    prefill.append(ExperimentPoint(
                        batch_size=batch_size,
                        seq_length=seq_length,
                        is_prefill=True
                    ))
        
        decode = []
        for batch_size in self.decode_batch_sizes[::2]:
            for kv_cache_length in self.decode_kv_cache_lengths[::2]:
                if len(decode) < decode_points:
                    decode.append(ExperimentPoint(
                        batch_size=batch_size,
                        kv_cache_length=kv_cache_length,
                        is_prefill=False
                    ))
        
        return prefill, decode
