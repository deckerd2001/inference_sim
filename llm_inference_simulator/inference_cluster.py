"""
Abstract cluster layer - BUGFIX VERSION

Fixed issues:
1. Finished requests tracking
2. Metrics collection
3. Event scheduling
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from .request import Request, Batch, RequestStatus
from .scheduler import create_scheduler, Scheduler
from .memory_manager import MemoryManager
from .performance_model import PerformanceModel
from .config import ModelSpec, SimulatorConfig
from .events import (
    PrefillFinishedEvent,
    DecodeStepFinishedEvent,
    KVTransferFinishedEvent,
)


@dataclass
class ClusterResources:
    """Resources for a single cluster (or cluster component)."""
    scheduler: Scheduler
    memory_manager: MemoryManager
    performance_model: PerformanceModel
    is_busy: bool = False
    current_batch: Optional[Batch] = None


@dataclass
class DecodeResult:
    """Result from decode step completion."""
    events: List
    finished_requests: List[Request]
    continuing_requests: List[Request]
    batch_finished: bool


class InferenceCluster(ABC):
    """Abstract base class for inference cluster configurations."""

    def __init__(self, config: SimulatorConfig, tp_comm_strategy=None):
        self.config = config
        self.tp_comm_strategy = tp_comm_strategy

    @abstractmethod
    def add_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def try_schedule_prefill(self, current_time: float) -> Optional[Tuple[float, PrefillFinishedEvent]]:
        pass

    @abstractmethod
    def try_schedule_decode(self, current_time: float) -> Optional[Tuple[float, DecodeStepFinishedEvent]]:
        pass

    @abstractmethod
    def handle_prefill_finished(self, event: PrefillFinishedEvent, current_time: float) -> List:
        pass

    @abstractmethod
    def handle_decode_finished(self, event: DecodeStepFinishedEvent, current_time: float) -> DecodeResult:
        """
        Handle decode step completion.

        Returns:
            DecodeResult with finished_requests for metrics collection
        """
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        pass

    @abstractmethod
    def get_memory_stats(self) -> Dict:
        pass

    @abstractmethod
    def get_scheduler(self) -> Scheduler:
        pass


class AggregatedCluster(InferenceCluster):
    """Aggregated cluster: Single set of resources handles both prefill and decode."""

    def __init__(self, config: SimulatorConfig, tp_comm_strategy=None):
        super().__init__(config, tp_comm_strategy)

        self.resources = ClusterResources(
            scheduler=create_scheduler(config.scheduler_spec),
            memory_manager=MemoryManager(
                config.model_spec,
                config.cluster_spec.xpu_spec,
                config.parallelism_spec
            ),
            performance_model=PerformanceModel(
                config.model_spec,
                config.cluster_spec.xpu_spec,
                config.parallelism_spec,
                tp_comm_strategy=tp_comm_strategy
            )
        )

    def add_request(self, request: Request) -> None:
        self.resources.scheduler.add_request(request)

    def try_schedule_prefill(self, current_time: float) -> Optional[Tuple[float, PrefillFinishedEvent]]:
        if self.resources.is_busy:
            return None

        batch = self.resources.scheduler.schedule_prefill_batch(
            current_time,
            memory_checker=self.resources.memory_manager.can_schedule_batch
        )

        if not batch:
            return None

        self.resources.is_busy = True
        self.resources.current_batch = batch

        prefill_time = self.resources.performance_model.estimate_prefill_time(
            batch_size=batch.batch_size,
            seq_length=batch.max_input_length
        )

        batch.processing_start_time = current_time

        for req in batch.requests:
            req.prefill_start_time = current_time

        self.resources.memory_manager.update_memory_usage(batch.requests, is_prefill=True)

        completion_time = current_time + prefill_time
        event = PrefillFinishedEvent(
            timestamp=completion_time,
            batch_id=batch.batch_id
        )

        return (prefill_time, event)

    def try_schedule_decode(self, current_time: float) -> Optional[Tuple[float, DecodeStepFinishedEvent]]:
        if self.resources.is_busy:
            return None

        batch = self.resources.scheduler.schedule_decode_batch(
            current_time,
            memory_checker=self.resources.memory_manager.can_schedule_batch
        )

        if not batch:
            return None

        self.resources.is_busy = True
        self.resources.current_batch = batch

        decode_time = self.resources.performance_model.estimate_decode_time(
            batch_size=batch.batch_size,
            kv_cache_length=batch.max_kv_cache_length
        )

        batch.processing_start_time = current_time
        self.resources.memory_manager.update_memory_usage(batch.requests, is_prefill=False)

        completion_time = current_time + decode_time
        event = DecodeStepFinishedEvent(
            timestamp=completion_time,
            batch_id=batch.batch_id,
            step=batch.current_decode_step
        )

        return (decode_time, event)

    def handle_prefill_finished(self, event: PrefillFinishedEvent, current_time: float) -> List:
        batch = self.resources.scheduler.batch_manager.get_batch(event.batch_id)
        if not batch:
            return []

        batch.processing_end_time = current_time

        for req in batch.requests:
            req.prefill_end_time = current_time
            req.first_token_time = current_time
            req.current_kv_cache_length = req.input_length

        self.resources.scheduler.move_to_decode_queue(batch.requests)
        self.resources.scheduler.batch_manager.remove_batch(batch)

        self.resources.is_busy = False
        self.resources.current_batch = None

        return []

    def handle_decode_finished(self, event: DecodeStepFinishedEvent, current_time: float) -> DecodeResult:
        """
        BUGFIX: Return finished_requests for metrics collection.
        """
        batch = self.resources.scheduler.batch_manager.get_batch(event.batch_id)
        if not batch:
            return DecodeResult(events=[], finished_requests=[], continuing_requests=[], batch_finished=True)

        batch.processing_end_time = current_time
        batch.current_decode_step += 1

        # Update requests and collect finished ones
        finished_requests = []
        for req in batch.requests:
            req.tokens_generated += 1
            req.current_kv_cache_length += 1

            if req.tokens_generated == 1:
                req.first_token_time = current_time

            if req.is_finished:
                req.completion_time = current_time
                req.status = RequestStatus.FINISHED
                finished_requests.append(req)

        # Remove finished requests
        batch.remove_finished_requests()
        continuing_requests = batch.requests.copy()

        # Determine next action
        new_events = []
        batch_finished = False

        if batch.is_batch_finished:
            # Batch complete
            self.resources.scheduler.batch_manager.remove_batch(batch)
            self.resources.is_busy = False
            self.resources.current_batch = None
            batch_finished = True
        else:
            # Continue decoding - schedule next step
            decode_time = self.resources.performance_model.estimate_decode_time(
                batch_size=batch.batch_size,
                kv_cache_length=batch.max_kv_cache_length
            )

            next_event = DecodeStepFinishedEvent(
                timestamp=current_time + decode_time,
                batch_id=batch.batch_id,
                step=batch.current_decode_step
            )
            new_events.append(next_event)

        return DecodeResult(
            events=new_events,
            finished_requests=finished_requests,
            continuing_requests=continuing_requests,
            batch_finished=batch_finished
        )

    def is_busy(self) -> bool:
        return self.resources.is_busy

    def get_memory_stats(self) -> Dict:
        return self.resources.memory_manager.get_memory_stats()

    def get_scheduler(self) -> Scheduler:
        return self.resources.scheduler


class DisaggregatedCluster(InferenceCluster):
    """Disaggregated cluster: Separate resources for prefill and decode."""

    def __init__(self, config: SimulatorConfig, tp_comm_strategy=None):
        super().__init__(config, tp_comm_strategy)

        disagg_spec = config.disaggregation_spec

        self.prefill_resources = ClusterResources(
            scheduler=create_scheduler(config.scheduler_spec),
            memory_manager=MemoryManager(
                config.model_spec,
                disagg_spec.prefill_cluster.xpu_spec,
                disagg_spec.prefill_parallelism
            ),
            performance_model=PerformanceModel(
                config.model_spec,
                disagg_spec.prefill_cluster.xpu_spec,
                disagg_spec.prefill_parallelism,
                tp_comm_strategy=tp_comm_strategy
            )
        )

        self.decode_resources = ClusterResources(
            scheduler=create_scheduler(config.scheduler_spec),
            memory_manager=MemoryManager(
                config.model_spec,
                disagg_spec.decode_cluster.xpu_spec,
                disagg_spec.decode_parallelism
            ),
            performance_model=PerformanceModel(
                config.model_spec,
                disagg_spec.decode_cluster.xpu_spec,
                disagg_spec.decode_parallelism,
                tp_comm_strategy=tp_comm_strategy
            )
        )

    def add_request(self, request: Request) -> None:
        self.prefill_resources.scheduler.add_request(request)

    def try_schedule_prefill(self, current_time: float) -> Optional[Tuple[float, PrefillFinishedEvent]]:
        if self.prefill_resources.is_busy:
            return None

        batch = self.prefill_resources.scheduler.schedule_prefill_batch(
            current_time,
            memory_checker=self.prefill_resources.memory_manager.can_schedule_batch
        )

        if not batch:
            return None

        self.prefill_resources.is_busy = True
        self.prefill_resources.current_batch = batch

        prefill_time = self.prefill_resources.performance_model.estimate_prefill_time(
            batch_size=batch.batch_size,
            seq_length=batch.max_input_length
        )

        batch.processing_start_time = current_time

        for req in batch.requests:
            req.prefill_start_time = current_time

        self.prefill_resources.memory_manager.update_memory_usage(batch.requests, is_prefill=True)

        completion_time = current_time + prefill_time
        event = PrefillFinishedEvent(
            timestamp=completion_time,
            batch_id=batch.batch_id
        )

        return (prefill_time, event)

    def try_schedule_decode(self, current_time: float) -> Optional[Tuple[float, DecodeStepFinishedEvent]]:
        if self.decode_resources.is_busy:
            return None

        batch = self.decode_resources.scheduler.schedule_decode_batch(
            current_time,
            memory_checker=self.decode_resources.memory_manager.can_schedule_batch
        )

        if not batch:
            return None

        self.decode_resources.is_busy = True
        self.decode_resources.current_batch = batch

        decode_time = self.decode_resources.performance_model.estimate_decode_time(
            batch_size=batch.batch_size,
            kv_cache_length=batch.max_kv_cache_length
        )

        batch.processing_start_time = current_time
        self.decode_resources.memory_manager.update_memory_usage(batch.requests, is_prefill=False)

        completion_time = current_time + decode_time
        event = DecodeStepFinishedEvent(
            timestamp=completion_time,
            batch_id=batch.batch_id,
            step=batch.current_decode_step
        )

        return (decode_time, event)

    def handle_prefill_finished(self, event: PrefillFinishedEvent, current_time: float) -> List:
        batch = self.prefill_resources.scheduler.batch_manager.get_batch(event.batch_id)
        if not batch:
            return []

        batch.processing_end_time = current_time

        for req in batch.requests:
            req.prefill_end_time = current_time
            req.current_kv_cache_length = req.input_length

        # Calculate KV cache transfer
        total_kv_size_gb = self.prefill_resources.memory_manager.calculate_batch_kv_cache_memory(
            batch.requests
        )

        transfer_time = self._calculate_transfer_time(total_kv_size_gb)

        for req in batch.requests:
            req.transfer_start_time = current_time
            req.kv_cache_size_gb = total_kv_size_gb / len(batch.requests)
            req.status = RequestStatus.TRANSFERRING

        # IMPORTANT: Do NOT remove batch yet!
        # Batch will be removed in handle_kv_transfer_finished()
        # Just mark cluster as free
        self.prefill_resources.is_busy = False
        self.prefill_resources.current_batch = None

        transfer_event = KVTransferFinishedEvent(
            timestamp=current_time + transfer_time,
            batch_id=batch.batch_id,
            request_ids=[req.request_id for req in batch.requests]
        )

        return [transfer_event]

    def handle_decode_finished(self, event: DecodeStepFinishedEvent, current_time: float) -> DecodeResult:
        """BUGFIX: Return finished_requests for metrics."""
        batch = self.decode_resources.scheduler.batch_manager.get_batch(event.batch_id)
        if not batch:
            return DecodeResult(events=[], finished_requests=[], continuing_requests=[], batch_finished=True)

        batch.processing_end_time = current_time
        batch.current_decode_step += 1

        finished_requests = []
        for req in batch.requests:
            if req.tokens_generated == 0:
                req.first_token_time = current_time

            req.tokens_generated += 1
            req.current_kv_cache_length += 1

            if req.is_finished:
                req.completion_time = current_time
                req.status = RequestStatus.FINISHED
                finished_requests.append(req)

        batch.remove_finished_requests()
        continuing_requests = batch.requests.copy()

        new_events = []
        batch_finished = False

        if batch.is_batch_finished:
            self.decode_resources.scheduler.batch_manager.remove_batch(batch)
            self.decode_resources.is_busy = False
            self.decode_resources.current_batch = None
            batch_finished = True
        else:
            decode_time = self.decode_resources.performance_model.estimate_decode_time(
                batch_size=batch.batch_size,
                kv_cache_length=batch.max_kv_cache_length
            )

            next_event = DecodeStepFinishedEvent(
                timestamp=current_time + decode_time,
                batch_id=batch.batch_id,
                step=batch.current_decode_step
            )
            new_events.append(next_event)

        return DecodeResult(
            events=new_events,
            finished_requests=finished_requests,
            continuing_requests=continuing_requests,
            batch_finished=batch_finished
        )

    def handle_kv_transfer_finished(self, event: KVTransferFinishedEvent, current_time: float) -> None:
        """Handle KV cache transfer completion."""
        # Get batch from prefill cluster
        batch = self.prefill_resources.scheduler.batch_manager.get_batch(event.batch_id)
        if not batch:
            return

        # Update request status and prepare for decode
        for req in batch.requests:
            req.status = RequestStatus.QUEUED
            req.transfer_end_time = current_time
            req.first_token_time = current_time  # First token ready after transfer
            # Initialize decode parameters
            req.current_kv_cache_length = req.input_length

        # Move to decode scheduler
        self.decode_resources.scheduler.move_to_decode_queue(batch.requests)

        # Remove batch from prefill cluster (it's done with prefill)
        self.prefill_resources.scheduler.batch_manager.remove_batch(batch)

    def is_busy(self) -> bool:
        return self.prefill_resources.is_busy or self.decode_resources.is_busy

    def get_memory_stats(self) -> Dict:
        return {
            'prefill': self.prefill_resources.memory_manager.get_memory_stats(),
            'decode': self.decode_resources.memory_manager.get_memory_stats()
        }

    def get_scheduler(self) -> Scheduler:
        return self.decode_resources.scheduler

    def _calculate_transfer_time(self, kv_size_gb: float) -> float:
        spec = self.config.disaggregation_spec
        effective_size_gb = kv_size_gb / spec.kv_compression_ratio
        bandwidth_time = effective_size_gb / spec.transfer_bandwidth_gbs
        latency_s = spec.transfer_latency_ms / 1000.0
        return bandwidth_time + latency_s


def create_inference_cluster(config: SimulatorConfig, tp_comm_strategy=None) -> InferenceCluster:
    """Factory function to create appropriate cluster type."""
    if config.is_disaggregated:
        return DisaggregatedCluster(config, tp_comm_strategy)
    else:
        return AggregatedCluster(config, tp_comm_strategy)