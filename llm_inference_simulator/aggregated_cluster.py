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
from .performance_models.factory import create_performance_model
from .config import PerformanceModelConfig


class AggregatedCluster(BaseCluster):
    """
    Aggregated cluster - single GPU pool.

    Scheduling policy:
    - Prefill and decode are mutually exclusive
    - Decode has higher priority (ongoing requests)
    - Only one batch can run at a time
    """

    def __init__(self, model_spec, xpu_spec, n_xpus, parallelism_spec, scheduler_spec,
                 performance_model_config: Optional[PerformanceModelConfig] = None):
        # Scheduler
        self.scheduler = Scheduler(scheduler_spec)

        # Memory management
        self.memory_manager = MemoryManager(
            model_spec=model_spec,
            xpu_spec=xpu_spec,
            parallelism_spec=parallelism_spec
        )

        # Performance modeling - use Factory
        if performance_model_config is None:
            # Default: roofline model
            performance_model_config = PerformanceModelConfig(model_type="roofline")
        
        self.performance_model = create_performance_model(
            model_type=performance_model_config.model_type,
            model_spec=model_spec,
            xpu_spec=xpu_spec,
            parallelism_spec=parallelism_spec,
            calibration_data_path=performance_model_config.calibration_data_path
        )

        # State
        self.is_gpu_busy = False
        self.current_batch = None
        self.last_finished_requests = []

        self.n_xpus = n_xpus
        self.xpu_spec = xpu_spec


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

        # Return descriptor (Simulator creates Event)
        return ScheduleResult(
            operation_type="prefill",
            busy_time=prefill_time,
            batch_id=batch.batch_id
        )

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

        # Return descriptor (Simulator creates Event)
        return ScheduleResult(
            operation_type="decode",
            busy_time=decode_time,
            batch_id=batch.batch_id,
            decode_step=batch.current_decode_step
        )

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

        # CRITICAL: Add to resident KV
        self.memory_manager.add_resident_requests(batch.requests)

        # CRITICAL: Remove batch from manager (prevent memory leak)
        self.scheduler.batch_manager.remove_batch(batch)

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
            # Continue decoding - return descriptor (Simulator creates Event)
            decode_time = self.performance_model.estimate_decode_time(
                batch_size=batch.batch_size,
                kv_cache_length=batch.max_kv_cache_length
            )

            return ScheduleResult(
                operation_type="decode",
                busy_time=decode_time,
                batch_id=batch.batch_id,
                decode_step=batch.current_decode_step
            )

    def is_busy(self) -> bool:
        """Check if GPU is busy."""
        return self.is_gpu_busy

    def add_request(self, request: Request):
        """Add request to prefill queue."""
        self.scheduler.add_request(request)

    def get_memory_usage(self):
        """Get memory usage statistics."""
        return self.memory_manager.get_memory_usage()


    def print_info(self):
        """Print configuration for Aggregated Cluster."""
        print(f"Mode: Aggregated (Single Cluster)")
        print(f"-" * 60)

        print(f"[Cluster Resources]")
        print(f"  Hardware: {self.n_xpus}x {self.xpu_spec.name} "
              f"(TP={self.memory_manager.parallel.tensor_parallel_size})")

        self._print_stage_details(self.memory_manager, self.scheduler, self.n_xpus)

    def _print_stage_details(self, memory_manager, scheduler, num_xpus):
        """Helper to print memory and scheduler details."""
        stats = memory_manager.get_memory_stats()
        per_dev_total = stats['total_memory_gb']
        per_dev_model = stats['model_weights_gb']
        per_dev_kv = stats['available_for_kv_cache_gb']
        cluster_total_mem = per_dev_total * num_xpus

        print(f"  Memory Strategy:")
        print(f"    - HBM Capacity: {per_dev_total:.2f} GB (Per xPU) -> {cluster_total_mem:.2f} GB (Cluster Total)")
        print(f"    - Model Weights: {per_dev_model:.2f} GB/xPU ({(per_dev_model/per_dev_total)*100:.1f}%)")
        print(f"    - KV Cache Space: {per_dev_kv:.2f} GB/xPU ({(per_dev_kv/per_dev_total)*100:.1f}%)")

        spec = scheduler.spec
        print(f"  Scheduler:")
        print(f"    - Type:     {spec.batching_type.capitalize()} Batching")
        print(f"    - Strategy: {spec.batching_strategy.upper()}")
        print(f"    - Max Batch: {spec.max_batch_size if spec.max_batch_size else 'Dynamic (Memory-bound)'}")

    def get_prefill_scheduler(self) -> Scheduler:
        return self.scheduler

    def get_decode_scheduler(self) -> Scheduler:
        return self.scheduler