"""
Memory management for GPU memory tracking and scheduling decisions.

UPDATED: Now tracks "resident" requests whose KV cache is in GPU memory.
"""

from dataclasses import dataclass
from typing import List
from .config import ModelSpec, ParallelismSpec
from .xpu_spec import xPUSpec
from .request import Request


@dataclass
class MemoryUsage:
    """Tracks GPU memory usage breakdown."""
    model_weights_gb: float = 0.0
    kv_cache_gb: float = 0.0
    activations_gb: float = 0.0

    @property
    def total_gb(self) -> float:
        """Total memory used in GB."""
        return self.model_weights_gb + self.kv_cache_gb + self.activations_gb

    def __repr__(self):
        return (f"MemoryUsage(weights={self.model_weights_gb:.2f}GB, "
                f"kv_cache={self.kv_cache_gb:.2f}GB, "
                f"activations={self.activations_gb:.2f}GB, "
                f"total={self.total_gb:.2f}GB)")


class MemoryManager:
    """
    Manages GPU memory allocation and tracks usage.

    Responsibilities:
    - Track model weight memory
    - Track KV cache memory per request
    - Check if new batches fit in memory
    - Prevent OOM by rejecting requests

    IMPORTANT: Now tracks "resident" requests - those whose KV cache
    is currently in GPU memory (after prefill, during decode).
    """

    def __init__(self, model_spec: ModelSpec, xpu_spec: xPUSpec,
                 parallelism_spec: ParallelismSpec):
        self.model = model_spec
        self.xpu = xpu_spec
        self.parallel = parallelism_spec

        # Calculate static memory (model weights)
        self.model_memory_gb = self._calculate_model_memory()

        # Track dynamic memory (KV cache + activations)
        self.current_kv_cache_gb = 0.0
        self.current_activation_gb = 0.0

        # Safety margin (reserve some memory)
        self.memory_safety_margin_gb = 0.0  # Reserve 2GB for safety

        # Available memory for dynamic allocation
        self.available_memory_gb = (
            self.xpu.memory_size_gb -
            self.model_memory_gb -
            self.memory_safety_margin_gb
        )

        # NEW: Track requests whose KV cache is resident in GPU memory
        # These are requests that have completed prefill but not yet finished decode
        self.resident_requests = []  # List[Request]

        if self.available_memory_gb < 0:
            raise ValueError(
                f"Model weights ({self.model_memory_gb:.2f}GB) exceed "
                f"GPU memory ({self.xpu.memory_size_gb}GB)!"
            )

    def _calculate_model_memory(self) -> float:
        """
        Calculate memory required for model weights.

        Returns:
            Memory in GB
        """
        # Total parameters
        total_params = self.model.n_params

        # Bytes per parameter
        bytes_per_param = self.model.weight_dtype.bytes_per_element()

        # Total weight memory
        weight_memory_bytes = total_params * bytes_per_param

        # Adjust for tensor parallelism (weights are sharded)
        weight_memory_bytes = weight_memory_bytes / self.parallel.tensor_parallel_size

        # Convert to GB
        weight_memory_gb = weight_memory_bytes / (1024 ** 3)

        return weight_memory_gb

    # ========================================================================
    # NEW: Resident KV Tracking Methods
    # ========================================================================

    def add_resident_requests(self, requests: List[Request]):
        """
        Add requests to resident set (call after prefill completes).

        These requests now have KV cache in GPU memory that must persist
        until decode completes.

        Args:
            requests: Requests whose KV cache is now in GPU memory
        """
        for req in requests:
            if req not in self.resident_requests:
                self.resident_requests.append(req)

    def remove_resident_requests(self, requests: List[Request]):
        """
        Remove requests from resident set (call after decode completes).

        These requests no longer need KV cache in GPU memory.

        Args:
            requests: Requests whose KV cache is no longer needed
        """
        for req in requests:
            if req in self.resident_requests:
                self.resident_requests.remove(req)

    def get_resident_kv_memory(self) -> float:
        """
        Calculate total KV cache memory for all resident requests.

        Uses actual current_kv_cache_length (grows during decode),
        not worst-case max_length.

        Returns:
            Total KV cache memory in GB
        """
        total_kv_gb = 0.0

        for req in self.resident_requests:
            # Use actual current KV length (grows during decode)
            kv_length = req.current_kv_cache_length

            if kv_length == 0:
                # Not yet initialized, use input_length
                kv_length = req.input_length if req.input_length else 0

            if kv_length == 0:
                continue  # Skip if still no length

            # Calculate KV memory for this length
            # KV cache: 2 (K+V) × n_layers × hidden_size × seq_len × bytes_per_elem
            kv_elements = (
                2 *  # K and V
                self.model.n_layers *
                kv_length *
                self.model.hidden_size
            )

            bytes_per_elem = self.model.activation_dtype.bytes_per_element()
            kv_bytes = kv_elements * bytes_per_elem

            # Shard by tensor parallelism
            tp_size = max(1, self.parallel.tensor_parallel_size)
            kv_bytes = kv_bytes / tp_size

            kv_gb = kv_bytes / (1024 ** 3)
            total_kv_gb += kv_gb

        return total_kv_gb

    # ========================================================================
    # MODIFIED: KV Cache Calculation Methods
    # ========================================================================

    def calculate_kv_cache_memory(self, request: Request) -> float:
        """
        Calculate KV cache memory for a single request.

        Args:
            request: Request to calculate memory for

        Returns:
            Memory in GB
        """
        # KV cache: 2 (K and V) * n_layers * seq_length * hidden_size
        # Note: seq_length grows as we generate tokens
        max_length = request.input_length + request.requested_output_tokens

        kv_elements = (
            2 *  # K and V
            self.model.n_layers *
            max_length *
            self.model.hidden_size
        )

        bytes_per_element = self.model.activation_dtype.bytes_per_element()
        kv_bytes = kv_elements * bytes_per_element

        tp = max(1, self.parallel.tensor_parallel_size)
        # Megatron-style TP shards attention heads across TP ranks,
        # so KV cache is sharded as well (per-GPU KV bytes / tp).
        kv_bytes = kv_bytes / tp

        kv_gb = kv_bytes / (1024 ** 3)

        return kv_gb

    def calculate_batch_kv_cache_memory(self, requests: List[Request]) -> float:
        """
        Calculate total KV cache memory for a batch of requests.

        Args:
            requests: List of requests in batch

        Returns:
            Total memory in GB
        """
        total_gb = sum(self.calculate_kv_cache_memory(req) for req in requests)
        return total_gb

    def calculate_activation_memory(self, batch_size: int, seq_length: int) -> float:
        """
        Calculate activation memory for a batch.

        This is temporary memory during computation.

        Args:
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Memory in GB
        """
        # Rough estimate: activations are proportional to batch * seq * hidden
        # We need activations for attention (QKV, scores) and MLP

        hidden_size = self.model.hidden_size

        # Attention activations: Q, K, V, attention scores
        # Simplified: ~4 * batch * seq * hidden
        attn_elements = 4 * batch_size * seq_length * hidden_size

        # MLP activations: intermediate states
        # Simplified: ~2 * batch * seq * ffn_dim
        mlp_elements = 2 * batch_size * seq_length * self.model.ffn_dim

        total_elements = attn_elements + mlp_elements

        bytes_per_element = self.model.activation_dtype.bytes_per_element()
        activation_bytes = total_elements * bytes_per_element

        # Adjust for tensor parallelism
        activation_bytes = activation_bytes / self.parallel.tensor_parallel_size

        activation_gb = activation_bytes / (1024 ** 3)

        return activation_gb

    # ========================================================================
    # MODIFIED: Scheduling Check (considers resident KV)
    # ========================================================================

    def can_schedule_batch(self, requests: List[Request], is_prefill: bool) -> tuple[bool, str]:
        """
        Check if a batch can be scheduled without OOM.

        IMPORTANT: Now considers resident KV cache from previous batches!

        Args:
            requests: List of requests to schedule
            is_prefill: Whether this is a prefill batch

        Returns:
            (can_schedule, reason) tuple
        """
        if not requests:
            return True, "Empty batch"

        # CRITICAL: Calculate resident KV from previous batches
        resident_kv_gb = self.get_resident_kv_memory()

        # Calculate KV cache memory needed for NEW batch (worst-case)
        new_kv_memory = self.calculate_batch_kv_cache_memory(requests)

        # Calculate activation memory needed
        if is_prefill:
            # Prefill: Process all input tokens at once
            max_input_length = max(req.input_length for req in requests)
            activation_memory = self.calculate_activation_memory(
                len(requests), max_input_length
            )
        else:
            # Decode: Generate 1 token per request
            activation_memory = self.calculate_activation_memory(
                len(requests), seq_length=1
            )

        # Total dynamic memory needed = resident + new batch + activations
        total_kv_needed = resident_kv_gb + new_kv_memory
        total_dynamic_memory = total_kv_needed + activation_memory

        # Check if it fits
        if total_dynamic_memory > self.available_memory_gb:
            reason = (
                f"OOM: Need {total_dynamic_memory:.2f}GB "
                f"(Resident KV={resident_kv_gb:.2f}GB + "
                f"New KV={new_kv_memory:.2f}GB + "
                f"Act={activation_memory:.2f}GB), "
                f"but only {self.available_memory_gb:.2f}GB available"
            )
            return False, reason

        return True, "OK"

    def get_max_batch_size(self, avg_input_length: int, avg_output_length: int) -> int:
        """
        Calculate maximum batch size that fits in memory.

        Args:
            avg_input_length: Average input sequence length
            avg_output_length: Average output sequence length

        Returns:
            Maximum batch size
        """
        # Create a dummy request to estimate memory
        from .request import Request, RequestStatus

        max_batch_size = 1

        for batch_size in range(1, 129):  # Try up to 128
            # Create dummy requests
            dummy_requests = [
                Request(
                    request_id=i,
                    arrival_time=0.0,
                    input_text=f"dummy_{i}",
                    requested_output_tokens=avg_output_length,
                    input_length=avg_input_length,
                    status=RequestStatus.QUEUED,
                )
                for i in range(batch_size)
            ]

            can_schedule, _ = self.can_schedule_batch(dummy_requests, is_prefill=True)

            if can_schedule:
                max_batch_size = batch_size
            else:
                break

        return max_batch_size

    def update_memory_usage(self, requests: List[Request], is_prefill: bool):
        """
        Update current memory usage tracking.

        NOTE: This is for monitoring only. Actual memory management
        uses resident_requests tracking.

        Args:
            requests: Currently active requests
            is_prefill: Whether in prefill phase
        """
        self.current_kv_cache_gb = self.calculate_batch_kv_cache_memory(requests)

        if is_prefill and requests:
            max_input_length = max(req.input_length for req in requests)
            self.current_activation_gb = self.calculate_activation_memory(
                len(requests), max_input_length
            )
        elif requests:
            max_kv_length = max(req.current_kv_cache_length for req in requests if req.current_kv_cache_length > 0)
            self.current_activation_gb = self.calculate_activation_memory(
                len(requests), max_kv_length
            )
        else:
            self.current_activation_gb = 0.0

    def get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage breakdown."""
        # Use resident KV for accurate tracking
        resident_kv_gb = self.get_resident_kv_memory()

        return MemoryUsage(
            model_weights_gb=self.model_memory_gb,
            kv_cache_gb=resident_kv_gb,  # Use resident, not current batch
            activations_gb=self.current_activation_gb,
        )

    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        usage = self.get_memory_usage()

        return {
            "total_memory_gb": self.xpu.memory_size_gb,
            "model_weights_gb": self.model_memory_gb,
            "safety_margin_gb": self.memory_safety_margin_gb,
            "available_for_kv_cache_gb": self.available_memory_gb,
            "current_kv_cache_gb": usage.kv_cache_gb,  # Resident KV
            "resident_requests_count": len(self.resident_requests),
            "current_activation_gb": self.current_activation_gb,
            "current_used_gb": usage.total_gb,
            "current_free_gb": self.xpu.memory_size_gb - usage.total_gb,
            "memory_utilization": usage.total_gb / self.xpu.memory_size_gb,
        }