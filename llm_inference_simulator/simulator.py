"""
Main event-driven simulator for LLM inference.
"""

import heapq
import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .events import (
    Event, EventType,
    RequestArrivedEvent, RequestTokenizedEvent,
    PrefillStartedEvent, PrefillFinishedEvent,
    DecodeStepStartedEvent, DecodeStepFinishedEvent,
    TokenEmittedEvent, RequestFinishedEvent,
    BatchingWakeupEvent,
    KVTransferStartedEvent,
    KVTransferFinishedEvent,
)
from .config import SimulatorConfig
from .request import Request, Batch, RequestStatus
from .scheduler import create_scheduler
from .performance_model import PerformanceModel
from .memory_manager import MemoryManager


@dataclass
class SimulationMetrics:
    """Collected metrics from simulation."""
    # Request-level metrics
    total_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0

    first_token_latencies: List[float] = field(default_factory=list)
    end_to_end_latencies: List[float] = field(default_factory=list)
    prefill_latencies: List[float] = field(default_factory=list)

    # Token-level metrics
    total_tokens_generated: int = 0
    token_generation_times: List[float] = field(default_factory=list)

    # System-level metrics
    total_simulation_time: float = 0.0

    # xPU utilization tracking
    gpu_busy_time: float = 0.0
    gpu_idle_time: float = 0.0

    # Memory tracking
    peak_memory_usage_gb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)

    def compute_statistics(self) -> Dict:
        """Compute summary statistics."""
        stats = {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "rejected_requests": self.rejected_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "simulation_time": self.total_simulation_time,
        }

        # Throughput
        if self.total_simulation_time > 0:
            stats["throughput_requests_per_sec"] = (
                self.completed_requests / self.total_simulation_time
            )
            stats["throughput_tokens_per_sec"] = (
                self.total_tokens_generated / self.total_simulation_time
            )

        # xPU utilization
        total_time = self.gpu_busy_time + self.gpu_idle_time
        if total_time > 0:
            stats["gpu_utilization"] = self.gpu_busy_time / total_time

        # Memory statistics (with percentiles)
        if self.memory_samples:
            stats["memory_peak_gb"] = self.peak_memory_usage_gb
            stats["memory_p95_gb"] = np.percentile(self.memory_samples, 95)
            stats["memory_p50_gb"] = np.percentile(self.memory_samples, 50)

        # Latency statistics
        if self.first_token_latencies:
            stats["first_token_latency"] = {
                "mean": np.mean(self.first_token_latencies),
                "p50": np.percentile(self.first_token_latencies, 50),
                "p90": np.percentile(self.first_token_latencies, 90),
                "p95": np.percentile(self.first_token_latencies, 95),
                "p99": np.percentile(self.first_token_latencies, 99),
            }

        if self.end_to_end_latencies:
            stats["end_to_end_latency"] = {
                "mean": np.mean(self.end_to_end_latencies),
                "p50": np.percentile(self.end_to_end_latencies, 50),
                "p90": np.percentile(self.end_to_end_latencies, 90),
                "p95": np.percentile(self.end_to_end_latencies, 95),
                "p99": np.percentile(self.end_to_end_latencies, 99),
            }

        if self.prefill_latencies:
            stats["prefill_latency"] = {
                "mean": np.mean(self.prefill_latencies),
                "p50": np.percentile(self.prefill_latencies, 50),
                "p90": np.percentile(self.prefill_latencies, 90),
                "p95": np.percentile(self.prefill_latencies, 95),
                "p99": np.percentile(self.prefill_latencies, 99),
            }

        return stats


class LLMInferenceSimulator:
    """
    Event-driven simulator for LLM inference.

    Simulates the end-to-end inference pipeline including:
    - Request arrival
    - Batching and scheduling
    - Prefill phase
    - Decode phase
    - Token generation
    - Memory management (OOM prevention)
    """

    def __init__(self, config: SimulatorConfig, tp_comm_strategy=None):
        self.config = config
        config.validate()

        # Random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)


        # Measurement window (for warm-up support)
        self.measurement_start = self.config.warm_up_duration_s
        self.measurement_end = self.measurement_start + self.config.simulation_duration_s

        # FIX: Add cooldown period to process remaining requests
        # Conservative: Allow enough time to process all remaining requests
        cooldown_duration = 100.0  # Enough time to drain the queue completely
        self.total_duration = self.measurement_end + cooldown_duration


        # Event queue (priority queue)
        self.event_queue = []

        # Simulation state
        self.current_time = 0.0
        self.next_request_id = 0

        # Track completed requests separately
        self.completed_requests_map = {}

        # Metrics
        self.metrics = SimulationMetrics()

        # Event log (optional)
        self.event_log = [] if config.output_event_log else None


        # Store tp_comm_strategy for use in init methods
        self.tp_comm_strategy = tp_comm_strategy
        # Disaggregation mode
        self.is_disaggregated = config.is_disaggregated

        if self.is_disaggregated:
            self._init_disaggregated_resources()
        else:
            self._init_aggregated_resources()


    def _init_aggregated_resources(self):
        """Initialize resources for aggregated mode (current behavior)."""
        # Scheduler
        self.scheduler = create_scheduler(self.config.scheduler_spec)

        # Memory manager
        self.memory_manager = MemoryManager(
            self.config.model_spec,
            self.config.cluster_spec.xpu_spec,
            self.config.parallelism_spec
        )

        # Performance model
        self.performance_model = PerformanceModel(
            self.config.model_spec,
            self.config.cluster_spec.xpu_spec,
            self.config.parallelism_spec,
            tp_comm_strategy=self.tp_comm_strategy
        )

        # GPU state
        self.is_gpu_busy = False
        self.current_batch = None

    def _init_disaggregated_resources(self):
        """Initialize separate resources for prefill and decode clusters."""
        disagg_spec = self.config.disaggregation_spec

        # Prefill cluster resources
        self.prefill_scheduler = create_scheduler(self.config.scheduler_spec)
        self.prefill_memory = MemoryManager(
            self.config.model_spec,
            disagg_spec.prefill_cluster.xpu_spec,
            disagg_spec.prefill_parallelism
        )
        self.prefill_performance = PerformanceModel(
            self.config.model_spec,
            disagg_spec.prefill_cluster.xpu_spec,
            disagg_spec.prefill_parallelism,
            tp_comm_strategy=self.tp_comm_strategy
        )
        self.prefill_gpu_busy = False
        self.prefill_batch = None

        # Decode cluster resources
        self.decode_scheduler = create_scheduler(self.config.scheduler_spec)
        self.decode_memory = MemoryManager(
            self.config.model_spec,
            disagg_spec.decode_cluster.xpu_spec,
            disagg_spec.decode_parallelism
        )
        self.decode_performance = PerformanceModel(
            self.config.model_spec,
            disagg_spec.decode_cluster.xpu_spec,
            disagg_spec.decode_parallelism,
            tp_comm_strategy=self.tp_comm_strategy
        )
        self.decode_gpu_busy = False
        self.decode_batch = None

        # Transfer state
        self.transfer_queue = []
        self.active_transfer = None

        # Use decode cluster's batch manager for completed requests
        self.batch_manager = self.decode_scheduler.batch_manager
        self.scheduler = self.decode_scheduler  # For compatibility

    def schedule_event(self, event: Event):
        """Add an event to the event queue."""
        heapq.heappush(self.event_queue, event)

    def run(self):
        """Run the simulation."""
        print(f"Starting simulation...")
        if self.is_disaggregated:
            print(f"Configuration: {self.config.model_spec.name}, "
                  f"Prefill: {self.config.disaggregation_spec.prefill_cluster.total_xpus} xPUs, "
                  f"Decode: {self.config.disaggregation_spec.decode_cluster.total_xpus} xPUs, "
                  f"arrival_rate={self.config.workload_spec.arrival_rate} req/s")
        else:
            print(f"Configuration: {self.config.model_spec.name}, "
                  f"{self.config.cluster_spec.total_xpus} xPUs, "
              f"arrival_rate={self.config.workload_spec.arrival_rate} req/s")

        # Print memory info
        if self.is_disaggregated:
            prefill_stats = self.prefill_memory.get_memory_stats()
            decode_stats = self.decode_memory.get_memory_stats()
            print(f"Prefill Memory: {prefill_stats['model_weights_gb']:.2f}GB model, "
                f"{prefill_stats['available_for_kv_cache_gb']:.2f}GB available for KV cache")
            print(f"Decode Memory: {decode_stats['model_weights_gb']:.2f}GB model, "
                f"{decode_stats['available_for_kv_cache_gb']:.2f}GB available for KV cache")
        else:
            mem_stats = self.memory_manager.get_memory_stats()
            print(f"Memory: {mem_stats['model_weights_gb']:.2f}GB model, "
                f"{mem_stats['available_for_kv_cache_gb']:.2f}GB available for KV cache")

        # Schedule initial request arrivals
        self._schedule_initial_arrivals()

        # Main event loop
        while self.event_queue:
            # Get next event
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp

            # Log event if enabled
            if self.event_log is not None:
                self.event_log.append(event)

            # Process event
            self._process_event(event)

            # Track memory usage
            self._track_memory_usage()

            # Check stopping conditions
            if self._should_stop():
                break

        # Finalize metrics
        self.metrics.total_simulation_time = self.config.simulation_duration_s

        print(f"\nSimulation completed at t={self.current_time:.2f}s")
        self._print_summary()

        return self.metrics

    def _track_memory_usage(self):
        """Track current memory usage for metrics."""
        if self.is_disaggregated:
            # Track both clusters
            prefill_usage = self.prefill_memory.get_memory_usage()
            decode_usage = self.decode_memory.get_memory_usage()
            total_usage = prefill_usage.total_gb + decode_usage.total_gb
        else:
            usage = self.memory_manager.get_memory_usage()
            total_usage = usage.total_gb

        self.metrics.memory_samples.append(total_usage)
        self.metrics.peak_memory_usage_gb = max(
            self.metrics.peak_memory_usage_gb,
            total_usage
        )
    def _schedule_initial_arrivals(self):
        """
        Schedule initial batch of request arrivals.

        CRITICAL FIX: Requests only arrive during measurement window.
        Simulation continues longer (cooldown period) to process all requests.
        """
        arrival_rate = self.config.workload_spec.arrival_rate
        duration = self.config.simulation_duration_s

        if self.config.workload_spec.arrival_process == "poisson":
            # Generate Poisson arrival times
            current_time = 0.0
            # FIX: Requests arrive until measurement_end (not total_duration)
            while current_time < self.measurement_end:
                # Inter-arrival time is exponentially distributed
                inter_arrival = random.expovariate(arrival_rate)
                current_time += inter_arrival

                if current_time < self.measurement_end:
                    self._create_request_arrival(current_time)

        elif self.config.workload_spec.arrival_process == "deterministic":
            # Fixed inter-arrival time
            inter_arrival = 1.0 / arrival_rate
            current_time = inter_arrival
            # FIX: Requests arrive until measurement_end (not total_duration)
            while current_time < self.measurement_end:
                self._create_request_arrival(current_time)
                current_time += inter_arrival

    def _create_request_arrival(self, arrival_time: float):
        """Create a request arrival event."""
        # Sample input length
        input_length = int(np.random.normal(
            self.config.workload_spec.avg_input_length,
            self.config.workload_spec.input_length_std
        ))
        # Clamp to valid range
        input_length = max(1, min(input_length, self.config.workload_spec.max_input_length))

        # Sample output length
        output_length = int(np.random.normal(
            self.config.workload_spec.avg_output_length,
            self.config.workload_spec.output_length_std
        ))
        # Clamp to valid range (CRITICAL: avoid negative values!)
        output_length = max(1, min(output_length, self.config.workload_spec.max_output_length))

        # Create event
        event = RequestArrivedEvent(
            timestamp=arrival_time,
            request_id=self.next_request_id,
            input_text=f"prompt_{self.next_request_id}",
            requested_output_tokens=output_length
        )
        self.next_request_id += 1

        self.schedule_event(event)

    def _process_event(self, event: Event):
        """Process an event based on its type."""
        if event.event_type == EventType.REQUEST_ARRIVED:
            self._handle_request_arrived(event)
        elif event.event_type == EventType.REQUEST_TOKENIZED:
            self._handle_request_tokenized(event)
        elif event.event_type == EventType.PREFILL_FINISHED:
            self._handle_prefill_finished(event)
        elif event.event_type == EventType.DECODE_STEP_FINISHED:
            self._handle_decode_step_finished(event)
        elif event.event_type == EventType.REQUEST_FINISHED:
            self._handle_request_finished(event)
        elif event.event_type == EventType.BATCHING_WAKEUP:
            # Just trigger scheduling
            if self.is_disaggregated:
                self._try_schedule_prefill()
                self._try_schedule_decode()
            else:
                self._try_schedule()

        elif event.event_type == EventType.KV_TRANSFER_STARTED:
            self._handle_kv_transfer_started(event)

        elif event.event_type == EventType.KV_TRANSFER_FINISHED:
            self._handle_kv_transfer_finished(event)
            pass

        # After processing, try to schedule new work (aggregated mode only)
        if not self.is_disaggregated:
            self._try_schedule_work()


    def _handle_request_arrived(self, event: RequestArrivedEvent):
        """Handle a new request arrival."""
        # Create request object
        request = Request(
            request_id=event.request_id,
            arrival_time=event.timestamp,
            input_text=event.input_text,
            requested_output_tokens=event.requested_output_tokens
        )

        self.metrics.total_requests += 1

        # Simulate tokenization (very fast, ~0.1ms)
        tokenization_delay = 0.0001
        tokenized_event = RequestTokenizedEvent(
            timestamp=self.current_time + tokenization_delay,
            request_id=request.request_id,
            input_length=np.random.randint(
                max(1, self.config.workload_spec.avg_input_length - 100),
                self.config.workload_spec.avg_input_length + 100
            )
        )

        # Store request in appropriate scheduler
        if self.is_disaggregated:
            self.prefill_scheduler.pending_requests[request.request_id] = request
        else:
            self.scheduler.pending_requests[request.request_id] = request

        self.schedule_event(tokenized_event)

    def _handle_request_tokenized(self, event: RequestTokenizedEvent):
        """Handle tokenization completion."""
        # Get request from appropriate scheduler
        if self.is_disaggregated:
            request = self.prefill_scheduler.get_request(event.request_id)
            if request:
                request.input_length = event.input_length
                request.tokenization_time = event.timestamp
                self.prefill_scheduler.add_request(request)
                # Try to schedule prefill immediately
                self._try_schedule_prefill()
        else:
            request = self.scheduler.get_request(event.request_id)
            if request:
                request.input_length = event.input_length
                request.tokenization_time = event.timestamp
                self.scheduler.add_request(request)

    def _schedule_batching_wakeup(self):
        """Schedule a wakeup event for batching window expiration."""
        if self.is_disaggregated:
            # For disaggregated mode, schedule wakeup for prefill scheduler
            wakeup_time = self.prefill_scheduler.get_next_wakeup_time(self.current_time)
        else:
            wakeup_time = self.scheduler.get_next_wakeup_time(self.current_time)

        if wakeup_time is not None:
            from .events import BatchingWakeupEvent
            wakeup_event = BatchingWakeupEvent(timestamp=wakeup_time)
            self.schedule_event(wakeup_event)

    def _try_schedule_work(self):
        """
        Try to schedule prefill or decode work if xPU is idle.

        CRITICAL: Decode has higher priority than Prefill!
        """
        if self.is_gpu_busy:
            # Schedule wakeup if waiting for batching window
            self._schedule_batching_wakeup()
            return

        # DECODE FIRST (ongoing requests)
        if self.scheduler.can_schedule_decode():
            self._schedule_decode()
        # PREFILL SECOND (new requests)
        elif self.scheduler.can_schedule_prefill(self.current_time):
            self._schedule_prefill()
        else:
            # xPU idle, waiting for batching window
            self._schedule_batching_wakeup()

    def _schedule_prefill(self):
        """Schedule a prefill batch with dynamic memory-based sizing."""

        # Memory checker function
        def memory_checker(requests, is_prefill):
            return self.memory_manager.can_schedule_batch(requests, is_prefill)

        # Dynamic batch sizing
        batch = self.scheduler.schedule_prefill_batch(
            self.current_time,
            memory_checker=memory_checker
        )

        if batch is None:
            return

        # Check if batch fits in memory
        can_schedule, reason = self.memory_manager.can_schedule_batch(
            batch.requests, is_prefill=True
        )

        if not can_schedule:
            # OOM! Reject the batch
            print(f"[WARNING] OOM at t={self.current_time:.2f}s: {reason}")
            self.metrics.rejected_requests += len(batch.requests)

            # Remove requests from scheduler
            for req in batch.requests:
                self.scheduler.remove_finished_requests([req])

            self.scheduler.batch_manager.remove_batch(batch)
            return

        self.is_gpu_busy = True
        self.current_batch = batch

        # Calculate prefill time
        prefill_time = self.performance_model.estimate_prefill_time(
            batch_size=batch.batch_size,
            seq_length=batch.max_input_length
        )

        # Update request tracking
        for req in batch.requests:
            req.prefill_start_time = self.current_time

        batch.processing_start_time = self.current_time

        # Update memory usage
        self.memory_manager.update_memory_usage(batch.requests, is_prefill=True)

        # Schedule prefill finish event
        finish_event = PrefillFinishedEvent(
            timestamp=self.current_time + prefill_time,
            batch_id=batch.batch_id
        )
        self.schedule_event(finish_event)

        # Track xPU busy time
        self.metrics.gpu_busy_time += prefill_time

    def _handle_prefill_finished(self, event: PrefillFinishedEvent):
        """Handle prefill completion - either transfer KV or move to decode."""

        if self.is_disaggregated:
            # DISAGGREGATED MODE: Start KV cache transfer
            batch = self.prefill_scheduler.batch_manager.get_batch(event.batch_id)
            if batch is None:
                return

            batch.processing_end_time = self.current_time

            # Update request timing
            for req in batch.requests:
                req.prefill_end_time = self.current_time

            # Calculate total KV cache size to transfer
            total_kv_gb = sum(
                self.prefill_memory.calculate_kv_cache_memory(req)
                for req in batch.requests
            )

            # Calculate transfer time
            transfer_time = self._calculate_transfer_time(total_kv_gb)

            # Schedule transfer started event (immediate)
            transfer_start_event = KVTransferStartedEvent(
                timestamp=self.current_time,
                batch_id=batch.batch_id,
                request_ids=[r.request_id for r in batch.requests],
                total_kv_size_gb=total_kv_gb
            )
            self.schedule_event(transfer_start_event)

            # Schedule transfer finished event (after transfer time)
            transfer_finish_event = KVTransferFinishedEvent(
                timestamp=self.current_time + transfer_time,
                batch_id=batch.batch_id,
                request_ids=[r.request_id for r in batch.requests]
            )
            self.schedule_event(transfer_finish_event)

            # Free prefill cluster
            self.prefill_gpu_busy = False
            self.prefill_batch = None

            # Try to schedule next prefill
            self._try_schedule_prefill()

        else:
            # AGGREGATED MODE: Move directly to decode queue (existing behavior)
            batch = self.scheduler.batch_manager.get_batch(event.batch_id)
            if batch is None:
                return

            batch.processing_end_time = self.current_time

            # Update request timing
            for req in batch.requests:
                req.prefill_end_time = self.current_time
                # Initialize KV cache length for decode (fix: same as disaggregated mode)
                req.current_kv_cache_length = req.input_length

                # Record prefill latency if in measurement window
                if (req.arrival_time >= self.measurement_start and
                    req.prefill_end_time <= self.measurement_end):
                    if req.prefill_latency is not None:
                        self.metrics.prefill_latencies.append(req.prefill_latency)

            # Move to decode queue
            self.scheduler.move_to_decode_queue(batch.requests)

            # GPU is now free
            self.is_gpu_busy = False
            self.current_batch = None

            # Try to schedule next operation (REMOVED - see Issue #1)
            # self._try_schedule()
            # Scheduling will be handled by _try_schedule_work() in _process_event()

    def _schedule_decode(self):
        """Schedule a decode batch with dynamic memory-based sizing."""

        # Memory checker function
        def memory_checker(requests, is_prefill):
            return self.memory_manager.can_schedule_batch(requests, is_prefill)

        # Dynamic batch sizing
        batch = self.scheduler.schedule_decode_batch(
            self.current_time,
            memory_checker=memory_checker
        )

        # Check if batch fits in memory
        can_schedule, reason = self.memory_manager.can_schedule_batch(
            batch.requests, is_prefill=False
        )

        if not can_schedule:
            # OOM during decode
            print(f"[WARNING] OOM during decode at t={self.current_time:.2f}s: {reason}")
            self.scheduler.batch_manager.remove_batch(batch)
            return

        self.is_gpu_busy = True
        self.current_batch = batch

        # Calculate decode time
        decode_time = self.performance_model.estimate_decode_time(
            batch_size=batch.batch_size,
            kv_cache_length=batch.max_kv_cache_length
        )

        batch.processing_start_time = self.current_time

        # Update memory usage
        self.memory_manager.update_memory_usage(batch.requests, is_prefill=False)

        # Schedule decode finish event
        finish_event = DecodeStepFinishedEvent(
            timestamp=self.current_time + decode_time,
            batch_id=batch.batch_id,
            step=batch.current_decode_step
        )
        self.schedule_event(finish_event)

        # Track xPU busy time
        self.metrics.gpu_busy_time += decode_time

    def _handle_decode_step_finished(self, event: DecodeStepFinishedEvent):
        """Handle decode step completion."""

        # Get batch from appropriate scheduler
        if self.is_disaggregated:
            batch = self.decode_scheduler.batch_manager.get_batch(event.batch_id)
            memory_manager = self.decode_memory
        else:
            batch = self.scheduler.batch_manager.get_batch(event.batch_id)
            memory_manager = self.memory_manager

        if batch is None:
            return

        batch.processing_end_time = self.current_time
        batch.current_decode_step += 1

        # Update all requests in batch
        finished_requests = []
        for req in batch.requests:
            req.tokens_generated += 1
            req.current_kv_cache_length += 1
            # Token counting moved to _handle_request_finished (measurement window)

            # Track first token time
            if req.tokens_generated == 1:
                req.first_token_time = self.current_time
                # Note: TTFT recording happens in _handle_request_finished()

            # Check if request is finished
            if req.is_finished:
                req.completion_time = self.current_time
                req.status = RequestStatus.FINISHED
                finished_requests.append(req)
                # Store request before removing from scheduler
                self.completed_requests_map[req.request_id] = req
                # Schedule request finished event
                finish_event = RequestFinishedEvent(
                    timestamp=self.current_time,
                    request_id=req.request_id,
                    total_output_tokens=req.tokens_generated
                )
                self.schedule_event(finish_event)

        # Remove finished requests from batch
        if finished_requests:
            if self.is_disaggregated:
                self.decode_scheduler.remove_finished_requests(finished_requests)
            else:
                self.scheduler.remove_finished_requests(finished_requests)
            batch.remove_finished_requests()

        # Check if batch is finished
        if batch.is_batch_finished:
            # All requests done, remove batch
            if self.is_disaggregated:
                self.decode_scheduler.batch_manager.remove_batch(batch)
                self.decode_gpu_busy = False
                self.decode_batch = None
            else:
                self.scheduler.batch_manager.remove_batch(batch)
                self.is_gpu_busy = False
                self.current_batch = None
        else:
            # Continue decoding
            if self.is_disaggregated:
                decode_time = self.decode_performance.estimate_decode_time(
                    batch_size=batch.batch_size,
                    kv_cache_length=batch.max_kv_cache_length
                )
            else:
                decode_time = self.performance_model.estimate_decode_time(
                    batch_size=batch.batch_size,
                    kv_cache_length=batch.max_kv_cache_length
                )

            next_event = DecodeStepFinishedEvent(
                timestamp=self.current_time + decode_time,
                batch_id=batch.batch_id,
                step=batch.current_decode_step
            )
            self.schedule_event(next_event)
            return

        # GPU is now free, try scheduling
        if self.is_disaggregated:
            self._try_schedule_decode()
        else:
            # REMOVED: self._try_schedule()
            # Scheduling will be handled by _try_schedule_work() in _process_event()
            # This ensures decode priority is maintained
            pass

    def _handle_request_finished(self, event: RequestFinishedEvent):
        """Handle request completion."""
        request = self.completed_requests_map.get(event.request_id)

        if request:
            # FIX: Count requests that ARRIVED in measurement window
            # (not when they completed - this allows cooldown to process all requests)
            if request.arrival_time >= self.measurement_start and \
               request.arrival_time <= self.measurement_end:
                self.metrics.completed_requests += 1
                self.metrics.total_tokens_generated += request.tokens_generated

                if request.first_token_latency:
                    self.metrics.first_token_latencies.append(request.first_token_latency)

                if request.end_to_end_latency:
                    self.metrics.end_to_end_latencies.append(request.end_to_end_latency)


    def _handle_kv_transfer_started(self, event):
        """Handle start of KV cache transfer (disaggregation only)."""
        # Get batch and mark requests as transferring
        batch = self.prefill_scheduler.batch_manager.get_batch(event.batch_id)
        if batch:
            for req in batch.requests:
                req.status = RequestStatus.TRANSFERRING
                req.transfer_start_time = self.current_time
                req.kv_cache_size_gb = event.total_kv_size_gb / len(event.request_ids)

    def _handle_kv_transfer_finished(self, event):
        """Handle completion of KV cache transfer (disaggregation only)."""
        # Get batch from prefill cluster
        batch = self.prefill_scheduler.batch_manager.get_batch(event.batch_id)
        if not batch:
            return

        # Update request status and prepare for decode
        for req in batch.requests:
            req.status = RequestStatus.QUEUED
            req.transfer_end_time = self.current_time
            # Initialize decode parameters
            req.current_kv_cache_length = req.input_length  # KV cache starts with input length

        # Move to decode scheduler
        self.decode_scheduler.move_to_decode_queue(batch.requests)

        # Remove batch from prefill cluster (it's done with prefill)
        self.prefill_scheduler.batch_manager.remove_batch(batch)

        # Try to schedule next operations
        self._try_schedule_prefill()
        self._try_schedule_decode()


    def _try_schedule_prefill(self):
        """Try to schedule a prefill batch."""

        if self.is_disaggregated:
            # DISAGGREGATED MODE: Use prefill cluster
            if self.prefill_gpu_busy:
                return

            batch = self.prefill_scheduler.schedule_prefill_batch(
                self.current_time,
                memory_checker=self.prefill_memory.can_schedule_batch
            )

            if batch:
                # Mark prefill cluster as busy
                self.prefill_gpu_busy = True
                self.prefill_batch = batch

                # Estimate prefill time
                prefill_time = self.prefill_performance.estimate_prefill_time(
                    batch_size=batch.batch_size,
                    seq_length=batch.max_input_length
                )

                # Update batch timing
                batch.processing_start_time = self.current_time

                for req in batch.requests:
                    req.prefill_start_time = self.current_time

                # Update memory tracking
                self.prefill_memory.update_memory_usage(batch.requests, is_prefill=True)

                # Schedule prefill completion
                from .events import PrefillFinishedEvent
                finish_event = PrefillFinishedEvent(
                    timestamp=self.current_time + prefill_time,
                    batch_id=batch.batch_id
                )
                self.schedule_event(finish_event)

                # Track prefill cluster busy time
                self.metrics.gpu_busy_time += prefill_time

        else:
            # AGGREGATED MODE: Use single cluster
            if self.is_gpu_busy:
                return

            batch = self.scheduler.schedule_prefill_batch(
                self.current_time,
                memory_checker=self.memory_manager.can_schedule_batch
            )

            if batch:
                # Mark GPU as busy
                self.is_gpu_busy = True
                self.current_batch = batch

                # Estimate prefill time
                prefill_time = self.performance_model.estimate_prefill_time(
                    batch_size=batch.batch_size,
                    seq_length=batch.max_input_length
                )

                # Update batch timing
                batch.processing_start_time = self.current_time

                for req in batch.requests:
                    req.prefill_start_time = self.current_time

                # Update memory tracking
                self.memory_manager.update_memory_usage(batch.requests, is_prefill=True)

                # Schedule prefill completion
                from .events import PrefillFinishedEvent
                finish_event = PrefillFinishedEvent(
                    timestamp=self.current_time + prefill_time,
                    batch_id=batch.batch_id
                )
                self.schedule_event(finish_event)

    def _try_schedule_decode(self):
        """Try to schedule a decode batch."""

        if self.is_disaggregated:
            # DISAGGREGATED MODE: Use decode cluster
            if self.decode_gpu_busy:
                return

            batch = self.decode_scheduler.schedule_decode_batch(
                self.current_time,
                memory_checker=self.decode_memory.can_schedule_batch
            )

            if batch:
                # Mark decode cluster as busy
                self.decode_gpu_busy = True
                self.decode_batch = batch

                # Estimate decode time (one step)
                decode_time = self.decode_performance.estimate_decode_time(
                    batch_size=batch.batch_size,
                    kv_cache_length=batch.max_kv_cache_length
                )

                # Update batch timing
                batch.processing_start_time = self.current_time

                # Update memory tracking
                self.decode_memory.update_memory_usage(batch.requests, is_prefill=False)

                # Schedule decode step completion
                from .events import DecodeStepFinishedEvent
                finish_event = DecodeStepFinishedEvent(
                    timestamp=self.current_time + decode_time,
                    batch_id=batch.batch_id,
                    step=batch.current_decode_step
                )
                self.schedule_event(finish_event)

                # Track decode cluster busy time
                self.metrics.gpu_busy_time += decode_time

        else:
            # AGGREGATED MODE: Use single cluster
            if self.is_gpu_busy:
                return

            batch = self.scheduler.schedule_decode_batch(
                self.current_time,
                memory_checker=self.memory_manager.can_schedule_batch
            )

            if batch:
                # Mark GPU as busy
                self.is_gpu_busy = True
                self.current_batch = batch

                # Estimate decode time
                decode_time = self.performance_model.estimate_decode_time(
                    batch_size=batch.batch_size,
                    kv_cache_length=batch.max_kv_cache_length
                )

                # Update batch timing
                batch.processing_start_time = self.current_time

                # Update memory tracking
                self.memory_manager.update_memory_usage(batch.requests, is_prefill=False)

                # Schedule decode step completion
                from .events import DecodeStepFinishedEvent
                finish_event = DecodeStepFinishedEvent(
                    timestamp=self.current_time + decode_time,
                    batch_id=batch.batch_id,
                    step=batch.current_decode_step
                )
                self.schedule_event(finish_event)

    def _try_schedule(self):
        """Try to schedule next batch (aggregated mode helper)."""
        if not self.is_disaggregated:
            # Try prefill first
            self._try_schedule_prefill()

            # If GPU is still free, try decode
            if not self.is_gpu_busy:
                self._try_schedule_decode()

    def _calculate_transfer_time(self, kv_size_gb: float) -> float:
        """Calculate KV cache transfer time between clusters."""
        spec = self.config.disaggregation_spec

        # Apply compression if configured
        effective_size_gb = kv_size_gb / spec.kv_compression_ratio

        # Bandwidth time
        bandwidth_time = effective_size_gb / spec.transfer_bandwidth_gbs

        # Latency overhead
        latency_s = spec.transfer_latency_ms / 1000.0

        return bandwidth_time + latency_s

    def _should_stop(self) -> bool:
        """Check if simulation should stop."""
        # Time-based stopping
        if self.current_time >= self.total_duration:
            return True

        # Request count-based stopping
        if (self.config.max_requests is not None and
            self.metrics.completed_requests >= self.config.max_requests):
            return True

        return False

    def _print_summary(self):
        """Print simulation summary with load analysis."""
        import numpy as np

        # Calculate load
        arrival_rate = self.config.workload_spec.arrival_rate
        avg_output = (self.config.workload_spec.avg_output_length +
                     self.config.workload_spec.max_output_length) / 2
        required_throughput = arrival_rate * avg_output
        actual_throughput = (self.metrics.total_tokens_generated /
                            self.metrics.total_simulation_time if self.metrics.total_simulation_time > 0 else 0)
        utilization = required_throughput / actual_throughput if actual_throughput > 0 else float('inf')
        is_overloaded = utilization >= 1.0

        print()
        print("=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print()

        # Load Analysis
        print("Load Analysis:")
        print(f"  Arrival Rate:         {arrival_rate:.1f} req/s")
        print(f"  Avg Output Length:    {avg_output:.0f} tok/req")
        print(f"  Required Throughput:  {required_throughput:.0f} tok/s")
        print(f"  Actual Throughput:    {actual_throughput:.1f} tok/s")
        print(f"  Utilization:          {utilization*100:.0f}%", end="")

        if is_overloaded:
            print(" ⚠️  OVERLOAD")
        elif utilization >= 0.8:
            print(" ⚠️  HIGH LOAD")
        else:
            print(" ✅ NORMAL")
        print()

        if is_overloaded:
            print("-" * 60)
            print("Metrics Reliability:")
            print("  ✅ Throughput:    System capacity (reliable)")
            print("  ✅ Completion:    Capacity/Load ratio (reliable)")
            print("  ❌ TTFT/Latency:  Unreliable (early arrivals only)")
            print("-" * 60)
            print()

        # Requests
        print("Requests:")
        print(f"  Total: {self.metrics.total_requests}")
        print(f"  Completed: {self.metrics.completed_requests}")

        if self.metrics.rejected_requests > 0:
            print(f"  Rejected: {self.metrics.rejected_requests}")

        print()

        # Throughput
        if is_overloaded:
            print("Throughput (= System Capacity):")
        else:
            print("Throughput:")

        if self.metrics.total_simulation_time > 0:
            req_per_sec = self.metrics.completed_requests / self.metrics.total_simulation_time
            tok_per_sec = self.metrics.total_tokens_generated / self.metrics.total_simulation_time
            print(f"  Requests/sec: {req_per_sec:.2f}")
            print(f"  Tokens/sec: {tok_per_sec:.2f}")

        print()
        # Calculate xPU utilization
        # Fix: Calculate idle time from measurement window
        measurement_duration = self.measurement_end - self.measurement_start
        self.metrics.gpu_idle_time = measurement_duration - self.metrics.gpu_busy_time
        total_time = self.metrics.gpu_busy_time + self.metrics.gpu_idle_time
        xpu_util = self.metrics.gpu_busy_time / total_time if total_time > 0 else 0.0
        print(f"xPU Utilization: {xpu_util:.1%}")
        print()

        # Memory
        xpu = self.config.cluster_spec.xpu_spec
        total_mem = self.config.cluster_spec.n_xpus_per_node * self.config.cluster_spec.n_nodes * xpu.memory_size_gb

        print("Memory Usage:")
        print(f"  Peak:        {self.metrics.peak_memory_usage_gb:.2f}GB / {total_mem:.0f}GB ({self.metrics.peak_memory_usage_gb/total_mem*100:.1f}%)")

        if self.metrics.memory_samples:
            mem_array = np.array(self.metrics.memory_samples)
            p95 = np.percentile(mem_array, 95)
            p50 = np.percentile(mem_array, 50)
            print(f"  P95:         {p95:.2f}GB / {total_mem:.0f}GB ({p95/total_mem*100:.1f}%)")
            print(f"  P50 (Med):   {p50:.2f}GB / {total_mem:.0f}GB ({p50/total_mem*100:.1f}%)")

        print()

        # Latency
        if is_overloaded:
            print("First Token Latency (⚠️  Not representative - early arrivals only):")
        else:
            print("First Token Latency (seconds):")

        if self.metrics.first_token_latencies:
            ftl = np.array(self.metrics.first_token_latencies)
            print(f"  Mean: {np.mean(ftl):.4f}")
            print(f"  P50:  {np.percentile(ftl, 50):.4f}")
            print(f"  P90:  {np.percentile(ftl, 90):.4f}")
            print(f"  P95:  {np.percentile(ftl, 95):.4f}")
            print(f"  P99:  {np.percentile(ftl, 99):.4f}")
            print()

        if self.metrics.end_to_end_latencies:
            e2e = np.array(self.metrics.end_to_end_latencies)
            if is_overloaded:
                print("End-to-End Latency (⚠️  Not representative):")
            else:
                print("End-to-End Latency (seconds):")
            print(f"  Mean: {np.mean(e2e):.4f}")
            print(f"  P50:  {np.percentile(e2e, 50):.4f}")
            print(f"  P90:  {np.percentile(e2e, 90):.4f}")
            print(f"  P95:  {np.percentile(e2e, 95):.4f}")
            print(f"  P99:  {np.percentile(e2e, 99):.4f}")
            print()

        print("=" * 60)
