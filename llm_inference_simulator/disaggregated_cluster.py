"""
Disaggregated cluster implementation.

Separate prefill and decode clusters that can run independently.
"""

from typing import List, Optional
from .cluster import BaseCluster, ScheduleResult
from .request import Request, RequestStatus
from .scheduler import Scheduler
from .memory_manager import MemoryManager
from .performance_models.factory import create_performance_model
from .config import PerformanceModelConfig


class DisaggregatedCluster(BaseCluster):
    """
    Disaggregated cluster - separate prefill and decode clusters.

    Scheduling policy:
    - Prefill and decode clusters are independent
    - Both can run simultaneously
    - Each manages its own resources
    - KV cache transfer between clusters
    """

    def __init__(self, model_spec, prefill_xpu_spec, prefill_n_xpus, prefill_parallelism,
                 decode_xpu_spec, decode_n_xpus, decode_parallelism,
                 transfer_bandwidth_gbs, scheduler_spec,
                 performance_model_config: Optional[PerformanceModelConfig] = None):

        # Performance model config (default to roofline)
        if performance_model_config is None:
            performance_model_config = PerformanceModelConfig(model_type="roofline")

        # Prefill cluster
        self.prefill_scheduler = Scheduler(scheduler_spec)
        self.prefill_memory = MemoryManager(
            model_spec=model_spec,
            xpu_spec=prefill_xpu_spec,
            parallelism_spec=prefill_parallelism
        )
        self.prefill_performance = create_performance_model(
            model_type=performance_model_config.model_type,
            model_spec=model_spec,
            xpu_spec=prefill_xpu_spec,
            parallelism_spec=prefill_parallelism,
            calibration_data_path=performance_model_config.calibration_data_path
        )
        self.prefill_gpu_busy = False
        self.prefill_batch = None

        # Decode cluster
        self.decode_scheduler = Scheduler(scheduler_spec)
        self.decode_memory = MemoryManager(
            model_spec=model_spec,
            xpu_spec=decode_xpu_spec,
            parallelism_spec=decode_parallelism
        )
        self.decode_performance = create_performance_model(
            model_type=performance_model_config.model_type,
            model_spec=model_spec,
            xpu_spec=decode_xpu_spec,
            parallelism_spec=decode_parallelism,
            calibration_data_path=performance_model_config.calibration_data_path
        )
        self.decode_gpu_busy = False
        self.decode_batch = None

        # Transfer parameters
        self.transfer_bandwidth_gbs = transfer_bandwidth_gbs
        self.last_finished_requests = []

        self.prefill_n_xpus = prefill_n_xpus
        self.prefill_xpu_spec = prefill_xpu_spec
        self.decode_n_xpus = decode_n_xpus
        self.decode_xpu_spec = decode_xpu_spec

    def try_schedule(self, current_time: float) -> List[ScheduleResult]:
        """
        Try to schedule work in disaggregated mode.

        Policy:
        - Try prefill independently
        - Try decode independently
        - Both can happen simultaneously
        """
        results = []

        # Try prefill (independent)
        if not self.prefill_gpu_busy:
            prefill_result = self._schedule_prefill(current_time)
            if prefill_result:
                results.append(prefill_result)

        # Try decode (independent)
        if not self.decode_gpu_busy:
            decode_result = self._schedule_decode(current_time)
            if decode_result:
                results.append(decode_result)

        return results

    def _schedule_prefill(self, current_time: float) -> Optional[ScheduleResult]:
        """Schedule prefill on prefill cluster."""
        batch = self.prefill_scheduler.schedule_prefill_batch(
            current_time=current_time,
            memory_checker=self.prefill_memory.can_schedule_batch
        )

        if not batch:
            return None

        # Mark prefill cluster as busy
        self.prefill_gpu_busy = True
        self.prefill_batch = batch

        # Update request status
        for req in batch.requests:
            req.status = RequestStatus.PREFILLING
            req.prefill_start_time = current_time

        # Estimate time
        prefill_time = self.prefill_performance.estimate_prefill_time(
            batch_size=batch.batch_size,
            seq_length=batch.max_input_length
        )

        # Update batch timing
        batch.processing_start_time = current_time

        # Update memory
        self.prefill_memory.update_memory_usage(batch.requests, is_prefill=True)

        # Return descriptor (Simulator creates Event)
        return ScheduleResult(
            operation_type="prefill",
            busy_time=prefill_time,
            batch_id=batch.batch_id
        )

    def _schedule_decode(self, current_time: float) -> Optional[ScheduleResult]:
        """Schedule decode on decode cluster."""
        batch = self.decode_scheduler.schedule_decode_batch(
            current_time=current_time,
            memory_checker=self.decode_memory.can_schedule_batch
        )

        if not batch:
            return None

        # Mark decode cluster as busy
        self.decode_gpu_busy = True
        self.decode_batch = batch

        # Update request status
        for req in batch.requests:
            req.status = RequestStatus.DECODING

        # Estimate time
        decode_time = self.decode_performance.estimate_decode_time(
            batch_size=batch.batch_size,
            kv_cache_length=batch.max_kv_cache_length
        )

        # Update batch timing
        batch.processing_start_time = current_time

        # Update memory
        self.decode_memory.update_memory_usage(batch.requests, is_prefill=False)

        # Return descriptor (Simulator creates Event)
        return ScheduleResult(
            operation_type="decode",
            busy_time=decode_time,
            batch_id=batch.batch_id,
            decode_step=batch.current_decode_step
        )

    def handle_prefill_finished(self, batch_id: int, current_time: float):
        """
        Handle prefill completion in disaggregated mode.

        Initiates KV cache transfer to decode cluster.
        """
        batch = self.prefill_scheduler.batch_manager.get_batch(batch_id)
        if not batch:
            return

        batch.processing_end_time = current_time

        # Update request timing and KV cache
        for req in batch.requests:
            req.prefill_end_time = current_time
            req.current_kv_cache_length = req.input_length

        # Note: KV transfer event should be created by simulator
        # We just free the prefill cluster here

        # CRITICAL: Add to prefill resident
        # KV cache is still in prefill memory during transfer
        self.prefill_memory.add_resident_requests(batch.requests)

        # Free prefill cluster
        self.prefill_gpu_busy = False
        self.prefill_batch = None


    def handle_kv_transfer_finished(self, batch_id: int, current_time: float):
        """Handle KV cache transfer completion."""
        # Get batch from prefill cluster
        batch = self.prefill_scheduler.batch_manager.get_batch(batch_id)
        if not batch:
            return

        # Move to decode queue
        for req in batch.requests:
            req.status = RequestStatus.QUEUED
            req.transfer_end_time = current_time

        self.decode_scheduler.move_to_decode_queue(batch.requests)

        # CRITICAL: Remove from prefill resident (transfer complete)
        self.prefill_memory.remove_resident_requests(batch.requests)

        # CRITICAL: Add to decode cluster's resident (KV now in decode memory)
        self.decode_memory.add_resident_requests(batch.requests)

        # CRITICAL: Remove batch from prefill manager (prevent memory leak)
        self.prefill_scheduler.batch_manager.remove_batch(batch)

    def handle_decode_step_finished(self, batch_id: int, current_time: float) -> Optional[ScheduleResult]:
        """Handle decode step completion."""
        batch = self.decode_scheduler.batch_manager.get_batch(batch_id)
        if not batch:
            return None

        batch.processing_end_time = current_time
        batch.current_decode_step += 1

        # Update requests
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
                self.last_finished_requests = finished_requests.copy()

        # Remove finished
        if finished_requests:
            # CRITICAL: Remove from resident KV (decode complete, KV no longer needed)
            self.decode_memory.remove_resident_requests(finished_requests)

            self.decode_scheduler.remove_finished_requests(finished_requests)
            batch.remove_finished_requests()

        # Check if batch done
        if batch.is_batch_finished:
            self.decode_scheduler.batch_manager.remove_batch(batch)
            self.decode_gpu_busy = False
            self.decode_batch = None
            return None
        else:
            # Continue decode - return descriptor (Simulator creates Event)
            decode_time = self.decode_performance.estimate_decode_time(
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
        """Check if any cluster is busy."""
        return self.prefill_gpu_busy or self.decode_gpu_busy

    def add_request(self, request: Request):
        """Add request to prefill queue."""
        self.prefill_scheduler.add_request(request)

    def get_memory_usage(self):
        """Get combined memory usage from both clusters."""
        prefill_usage = self.prefill_memory.get_memory_usage()
        decode_usage = self.decode_memory.get_memory_usage()

        # Return combined - FIX: Use correct attribute names!
        from .memory_manager import MemoryUsage
        return MemoryUsage(
            model_weights_gb=prefill_usage.model_weights_gb + decode_usage.model_weights_gb,
            kv_cache_gb=prefill_usage.kv_cache_gb + decode_usage.kv_cache_gb,
            activations_gb=prefill_usage.activations_gb + decode_usage.activations_gb
        )

    def calculate_transfer_time(self, batch_id: int) -> float:
        """Calculate KV cache transfer time for a batch."""
        batch = self.prefill_scheduler.batch_manager.get_batch(batch_id)
        if not batch:
            return 0.0

        total_kv_size_gb = self.prefill_memory.calculate_batch_kv_cache_memory(batch.requests)
        transfer_time = total_kv_size_gb / self.transfer_bandwidth_gbs

        return transfer_time

    def print_info(self):
        """Print configuration for Disaggregated Cluster."""
        print(f"Mode: Disaggregated Inference")
        print(f"-" * 60)

        # Prefill Info
        print(f"[Prefill Cluster]")
        print(f"  Hardware: {self.prefill_n_xpus}x {self.prefill_xpu_spec.name} "
              f"(TP={self.prefill_memory.parallel.tensor_parallel_size})")
        self._print_stage_details(self.prefill_memory, self.prefill_scheduler, self.prefill_n_xpus)
        print(f"-" * 60)

        # Decode Info
        print(f"[Decode Cluster]")
        print(f"  Hardware: {self.decode_n_xpus}x {self.decode_xpu_spec.name} "
              f"(TP={self.decode_memory.parallel.tensor_parallel_size})")
        self._print_stage_details(self.decode_memory, self.decode_scheduler, self.decode_n_xpus)

        # Interconnect Info
        print(f"-" * 60)
        print(f"[Interconnect]")
        print(f"  Bandwidth: {self.transfer_bandwidth_gbs} GB/s")

    def _print_stage_details(self, memory_manager, scheduler, num_xpus):
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
        return self.prefill_scheduler

    def get_decode_scheduler(self) -> Scheduler:
        return self.decode_scheduler


    def get_kv_transfer_delay(self, batch_id: int) -> Optional[float]:
        """Return KV transfer time."""
        return self.calculate_transfer_time(batch_id)