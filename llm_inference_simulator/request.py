"""
Request and Batch classes for the LLM Inference Simulator.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RequestStatus(Enum):
    """Status of an inference request."""
    ARRIVED = "arrived"
    TOKENIZED = "tokenized"
    QUEUED = "queued"
    PREFILLING = "prefilling"
    TRANSFERRING = "transferring"  # KV cache transfer (disaggregation only)
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class Request:
    """Represents a single inference request."""
    request_id: int
    arrival_time: float
    input_text: str
    requested_output_tokens: int

    # Populated after tokenization
    input_length: Optional[int] = None

    # Status tracking
    status: RequestStatus = RequestStatus.ARRIVED

    # Timing information
    tokenization_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    first_token_time: Optional[float] = None  # Time when first token is generated
    completion_time: Optional[float] = None

    # Generation tracking
    tokens_generated: int = 0
    current_kv_cache_length: int = 0  # Length of KV cache for this request

    # Token emission times (for latency distribution)
    token_emission_times: List[float] = field(default_factory=list)

    # Disaggregation: KV cache transfer tracking
    transfer_start_time: Optional[float] = None
    transfer_end_time: Optional[float] = None
    kv_cache_size_gb: Optional[float] = None

    @property
    def is_finished(self) -> bool:
        """Check if request is finished."""
        return (self.status == RequestStatus.FINISHED or
                self.tokens_generated >= self.requested_output_tokens)

    @property
    def first_token_latency(self) -> Optional[float]:
        """Time from arrival to first token (TTFT - Time To First Token)."""
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def end_to_end_latency(self) -> Optional[float]:
        """Time from arrival to completion."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None

    @property
    def prefill_latency(self) -> Optional[float]:
        """Time spent in prefill phase."""
        if self.prefill_start_time and self.prefill_end_time:
            return self.prefill_end_time - self.prefill_start_time
        return None


    @property
    def transfer_latency(self) -> Optional[float]:
        """Time spent transferring KV cache (disaggregation only)."""
        if self.transfer_start_time and self.transfer_end_time:
            return self.transfer_end_time - self.transfer_start_time
        return None
        """Time spent in prefill phase."""
        if self.prefill_start_time and self.prefill_end_time:
            return self.prefill_end_time - self.prefill_start_time
        return None

    def __repr__(self):
        return (f"Request(id={self.request_id}, status={self.status.value}, "
                f"in_len={self.input_length}, out_len={self.tokens_generated}/"
                f"{self.requested_output_tokens})")


@dataclass
class Batch:
    """Represents a batch of requests being processed together."""
    batch_id: int
    requests: List[Request]
    creation_time: float

    # Batch type
    is_prefill: bool  # True for prefill batch, False for decode batch

    # For decode batches
    current_decode_step: int = 0

    # Timing
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None

    @property
    def batch_size(self) -> int:
        """Number of requests in the batch."""
        return len(self.requests)

    @property
    def max_input_length(self) -> int:
        """Maximum input length in the batch (for prefill)."""
        if not self.requests:
            return 0
        return max(req.input_length or 0 for req in self.requests)

    @property
    def max_kv_cache_length(self) -> int:
        """Maximum KV cache length in the batch (for decode)."""
        if not self.requests:
            return 0
        return max(req.current_kv_cache_length for req in self.requests)

    @property
    def active_requests(self) -> List[Request]:
        """Requests that are still generating (not finished)."""
        return [req for req in self.requests if not req.is_finished]

    @property
    def is_batch_finished(self) -> bool:
        """Check if all requests in batch are finished."""
        return len(self.active_requests) == 0

    def remove_finished_requests(self):
        """Remove finished requests from the batch."""
        self.requests = self.active_requests

    def __repr__(self):
        active = len(self.active_requests)
        total = len(self.requests)
        batch_type = "prefill" if self.is_prefill else f"decode(step={self.current_decode_step})"
        return (f"Batch(id={self.batch_id}, {batch_type}, "
                f"requests={active}/{total} active)")


class BatchManager:
    """Manages batch creation and lifecycle."""

    def __init__(self):
        self.next_batch_id = 0
        self.active_batches: List[Batch] = []

    def create_batch(self, requests: List[Request], is_prefill: bool,
                    current_time: float) -> Batch:
        """Create a new batch."""
        batch = Batch(
            batch_id=self.next_batch_id,
            requests=requests,
            creation_time=current_time,
            is_prefill=is_prefill
        )
        self.next_batch_id += 1
        self.active_batches.append(batch)
        return batch

    def remove_batch(self, batch: Batch):
        """Remove a batch from active batches."""
        if batch in self.active_batches:
            self.active_batches.remove(batch)

    def get_batch(self, batch_id: int) -> Optional[Batch]:
        """Get a batch by ID."""
        for batch in self.active_batches:
            if batch.batch_id == batch_id:
                return batch
        return None
