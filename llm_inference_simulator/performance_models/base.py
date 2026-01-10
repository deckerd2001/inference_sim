"""
Base performance model interface.
"""

from abc import ABC, abstractmethod


class BasePerformanceModel(ABC):
    """
    Abstract base class for performance models.
    
    All performance models must implement:
    - estimate_prefill_time: Estimate time for prefill phase
    - estimate_decode_time: Estimate time for decode phase (per token)
    """
    
    @abstractmethod
    def estimate_prefill_time(self, batch_size: int, seq_length: int) -> float:
        """
        Estimate time for prefill phase.
        
        Args:
            batch_size: Number of sequences in batch
            seq_length: Input sequence length
            
        Returns:
            Estimated time in seconds
        """
        pass
    
    @abstractmethod
    def estimate_decode_time(self, batch_size: int, kv_cache_length: int) -> float:
        """
        Estimate time for a single decode step.
        
        Args:
            batch_size: Number of sequences in batch
            kv_cache_length: Current length of KV cache (context length)
            
        Returns:
            Estimated time per token in seconds
        """
        pass
