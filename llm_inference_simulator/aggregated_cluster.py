"""
Aggregated cluster implementation.

Single GPU pool shared between prefill and decode.
Decode has priority over prefill.
"""

from typing import List, Optional
from .cluster import BaseCluster, ScheduleResult
from .request import Request, RequestStatus
from .scheduler import Scheduler
from .memory_manager import MemoryManager
from .performance_model import PerformanceModel
from .events import PrefillFinishedEvent, DecodeStepFinishedEvent


class AggregatedCluster(BaseCluster):
    """
    Aggregated cluster - single GPU pool.

    Scheduling policy:
    - Prefill and decode are mutually exclusive
    - Decode has higher priority (ongoing requests)
    - Only one batch can run at a time
    """

    def __init__(self, model_spec, xpu_spec, n_xpus, parallelism_spec, scheduler_spec):
        # Scheduler
        self.scheduler = Scheduler(scheduler_spec)

        # Memory management
        self.memory_manager = MemoryManager(
            model_spec=model_spec,
            xpu_spec=xpu_spec,
            parallelism_spec=parallelism_spec
        )

        # Performance modeling
        self.performance_model = PerformanceModel(
            model_spec=model_spec,
            xpu_spec=xpu_spec,
            parallelism_spec=parallelism_spec
        )

        # State
        self.is_gpu_busy = False
        self.current_batch = None
        self.last_finished_requests = []


    def try_schedule(self, current_time: float) -> List[ScheduleResult]:
        """
        Try to schedule work in aggregated mode.

        Policy:
        1. If GPU busy → nothing
        2. Try decode first (priority)
        3. If no decode, try prefill
        """
        if self.is_gpu_busy:
            return []

        results = []

        # DECODE FIRST (ongoing requests have priority)
        if self.scheduler.can_schedule_decode():
            result = self._schedule_decode(current_time)
            if result:
                results.append(result)
                return results  # Only one at a time

        # PREFILL SECOND (new requests)
        if self.scheduler.can_schedule_prefill(current_time):
            result = self._schedule_prefill(current_time)
            if result:
                results.append(result)

        return results

    def _schedule_prefill(self, current_time: float) -> Optional[ScheduleResult]:
        """Schedule a prefill batch."""
        # Form batch
        batch = self.scheduler.schedule_prefill_batch(
            current_time=current_time,
            memory_checker=self.memory_manager.can_schedule_batch
        )

        if not batch:
            return None

        # Mark GPU as busy
        self.is_gpu_busy = True
        self.current_batch = batch

        # Update request status and timing
        for req in batch.requests:
            req.status = RequestStatus.PREFILLING
            req.prefill_start_time = current_time

        # Estimate execution time
        prefill_time = self.performance_model.estimate_prefill_time(
            batch_size=batch.batch_size,
            seq_length=batch.max_input_length
        )

        # Update batch timing
        batch.processing_start_time = current_time

        # Update memory tracking
        self.memory_manager.update_memory_usage(batch.requests, is_prefill=True)

        # Create completion event
        event = PrefillFinishedEvent(
            timestamp=current_time + prefill_time,
            batch_id=batch.batch_id
        )

        return ScheduleResult(busy_time=prefill_time, event=event)

    def _schedule_decode(self, current_time: float) -> Optional[ScheduleResult]:
        """Schedule a decode batch."""
        # Form batch
        batch = self.scheduler.schedule_decode_batch(
            current_time=current_time,
            memory_checker=self.memory_manager.can_schedule_batch
        )

        if not batch:
            return None

        # Mark GPU as busy
        self.is_gpu_busy = True
        self.current_batch = batch

        # Update request status
        for req in batch.requests:
            req.status = RequestStatus.DECODING

        # Estimate execution time
        decode_time = self.performance_model.estimate_decode_time(
            batch_size=batch.batch_size,
            kv_cache_length=batch.max_kv_cache_length
        )

        # Update batch timing
        batch.processing_start_time = current_time

        # Update memory tracking
        self.memory_manager.update_memory_usage(batch.requests, is_prefill=False)

        # Create completion event
        event = DecodeStepFinishedEvent(
            timestamp=current_time + decode_time,
            batch_id=batch.batch_id,
            step=batch.current_decode_step
        )

        return ScheduleResult(busy_time=decode_time, event=event)

    def handle_prefill_finished(self, batch_id: int, current_time: float):
        """Handle prefill batch completion."""
        batch = self.scheduler.batch_manager.get_batch(batch_id)
        if not batch:
            return

        batch.processing_end_time = current_time

        # Update request timing and KV cache
        for req in batch.requests:
            req.prefill_end_time = current_time
            req.current_kv_cache_length = req.input_length

        # Move requests to decode queue
        self.scheduler.move_to_decode_queue(batch.requests)

        # CRITICAL: Add to resident KV (prefill complete, KV now in memory)
        self.memory_manager.add_resident_requests(batch.requests)

        # Free GPU
        self.is_gpu_busy = False
        self.current_batch = None

    def handle_decode_step_finished(self, batch_id: int, current_time: float) -> Optional[ScheduleResult]:
        """Handle decode step completion."""
        batch = self.scheduler.batch_manager.get_batch(batch_id)
        if not batch:
            return None

        batch.processing_end_time = current_time
        batch.current_decode_step += 1

        # Update all requests in batch
        finished_requests = []
        for req in batch.requests:
            req.tokens_generated += 1
            req.current_kv_cache_length += 1

            # Track first token
            if req.tokens_generated == 1:
                req.first_token_time = current_time

            # Check if finished
            if req.is_finished:
                req.completion_time = current_time
                req.status = RequestStatus.FINISHED
                finished_requests.append(req)
                self.last_finished_requests = finished_requests.copy()

        # Remove finished requests
        if finished_requests:
            # CRITICAL: Remove from resident KV (decode complete, KV no longer needed)
            self.memory_manager.remove_resident_requests(finished_requests)

            self.scheduler.remove_finished_requests(finished_requests)
            batch.remove_finished_requests()

        # Check if batch is done
        if batch.is_batch_finished:
            # All requests finished
            self.scheduler.batch_manager.remove_batch(batch)
            self.is_gpu_busy = False
            self.current_batch = None
            return None
        else:
            # Continue decoding
            decode_time = self.performance_model.estimate_decode_time(
                batch_size=batch.batch_size,
                kv_cache_length=batch.max_kv_cache_length
            )

            event = DecodeStepFinishedEvent(
                timestamp=current_time + decode_time,
                batch_id=batch.batch_id,
                step=batch.current_decode_step
            )

            return ScheduleResult(busy_time=decode_time, event=event)

    def is_busy(self) -> bool:
        """Check if GPU is busy."""
        return self.is_gpu_busy

    def add_request(self, request: Request):
        """Add request to prefill queue."""
        self.scheduler.add_request(request)

    def get_memory_usage(self):
        """Get memory usage statistics."""
        return self.memory_manager.get_memory_usage()