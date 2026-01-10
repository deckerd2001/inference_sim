"""
Schema definitions for vLLM benchmark data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class BenchmarkPoint:
    """
    Single benchmark measurement point.
    
    Represents one measurement from vLLM:
    - Prefill: batch_size, seq_length, prefill_time_ms
    - Decode: batch_size, kv_cache_length, decode_time_ms
    """
    # Model and hardware info
    model_name: str
    xpu_name: str
    tp_size: int = 1
    
    # Prefill measurement
    batch_size: int
    seq_length: int = 0  # For prefill
    prefill_time_ms: Optional[float] = None
    
    # Decode measurement
    kv_cache_length: int = 0  # For decode
    decode_time_ms: Optional[float] = None  # Per token
    
    # Statistics
    num_samples: int = 1  # Number of runs averaged
    std_dev: Optional[float] = None
    
    # Metadata
    vllm_version: Optional[str] = None
    measurement_timestamp: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that either prefill or decode measurement exists."""
        if self.prefill_time_ms is None and self.decode_time_ms is None:
            raise ValueError("Either prefill_time_ms or decode_time_ms must be provided")
        
        if self.prefill_time_ms is not None and self.seq_length == 0:
            raise ValueError("seq_length must be provided for prefill measurements")
        
        if self.decode_time_ms is not None and self.kv_cache_length == 0:
            raise ValueError("kv_cache_length must be provided for decode measurements")


@dataclass
class BenchmarkData:
    """
    Collection of benchmark points with metadata.
    """
    # Metadata
    xpu_name: str
    model_name: str
    vllm_version: str
    collection_date: str
    collector: str = "vllm_benchmarks"
    
    # Benchmark points
    benchmarks: List[BenchmarkPoint] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    
    def get_prefill_benchmarks(self) -> List[BenchmarkPoint]:
        """Get all prefill benchmark points."""
        return [b for b in self.benchmarks if b.prefill_time_ms is not None]
    
    def get_decode_benchmarks(self) -> List[BenchmarkPoint]:
        """Get all decode benchmark points."""
        return [b for b in self.benchmarks if b.decode_time_ms is not None]
