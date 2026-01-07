"""
Refactored simulator using cluster polymorphism.

This version completely removes branching logic from simulator
and delegates all scheduling to cluster implementations.
"""

import heapq
import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .events import (
    Event, EventType,
    RequestArrivedEvent,
    PrefillFinishedEvent,
    DecodeStepFinishedEvent,
    RequestFinishedEvent,
    KVTransferFinishedEvent,
)
from .config import SimulatorConfig
from .request import Request, RequestStatus
from .cluster import ClusterFactory, ScheduleResult


@dataclass
class SimulationMetrics:
    """Collected metrics from simulation."""
    total_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0

    first_token_latencies: List[float] = field(default_factory=list)
    end_to_end_latencies: List[float] = field(default_factory=list)
    prefill_latencies: List[float] = field(default_factory=list)

    total_tokens_generated: int = 0
    token_generation_times: List[float] = field(default_factory=list)

    total_simulation_time: float = 0.0

    gpu_busy_time: float = 0.0
    gpu_idle_time: float = 0.0

    peak_memory_usage_gb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)


class LLMInferenceSimulator:
    """Event-driven simulator using cluster polymorphism."""

    def __init__(self, config: SimulatorConfig, tp_comm_strategy=None):
        self.config = config
        config.validate()

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        self.measurement_start = self.config.warm_up_duration_s
        self.measurement_end = self.measurement_start + self.config.simulation_duration_s

        cooldown_duration = 100.0
        self.total_duration = self.measurement_end + cooldown_duration

        self.event_queue = []

        self.current_time = 0.0
        self.next_request_id = 0

        self.completed_requests_map = {}

        self.metrics = SimulationMetrics()

        self.event_log = [] if config.output_event_log else None

        # Create cluster using factory (polymorphism!)
        self.cluster = ClusterFactory.create(config)

        self._print_configuration()

    def _print_configuration(self):
        """Print the simulation configuration details."""
        print(f"\n{'='*60}")
        print(f"Configuration: {self.config.model_spec.name}")
        print(f"Global Arrival Rate: {self.config.workload_spec.arrival_rate} req/s")
        print(f"-" * 60)

        self.cluster.print_info()


    def schedule_event(self, event: Event):
        """Add an event to the event queue."""
        heapq.heappush(self.event_queue, event)

    def run(self):
        """Run the simulation."""
        print(f"Starting simulation...")

        self._schedule_initial_arrivals()

        # Event loop
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp

            if self.event_log is not None:
                self.event_log.append(event)

            self._process_event(event)
            self._track_memory_usage()

            if self._should_stop():
                break

        self.metrics.total_simulation_time = min(
            self.current_time - self.measurement_start,
            self.config.simulation_duration_s
        ) if self.current_time > self.measurement_start else 0

        print(f"\nSimulation completed at t={self.current_time:.2f}s")
        self._print_summary()

        return self.metrics

    def _track_memory_usage(self):
        """Track current memory usage for metrics."""
        # Get from cluster (polymorphism!)
        usage = self.cluster.get_memory_usage()
        self.metrics.memory_samples.append(usage.total_gb)
        self.metrics.peak_memory_usage_gb = max(
            self.metrics.peak_memory_usage_gb,
            usage.total_gb
        )

    def _schedule_initial_arrivals(self):
        """Schedule initial batch of request arrivals."""
        arrival_rate = self.config.workload_spec.arrival_rate
        num_arrivals = int(arrival_rate * self.total_duration * 1.2)

        arrival_times = []
        current_arrival = 0.0

        for _ in range(num_arrivals):
            inter_arrival = random.expovariate(arrival_rate)
            current_arrival += inter_arrival

            if current_arrival > self.total_duration:
                break

            arrival_times.append(current_arrival)

        for arrival_time in arrival_times:
            event = RequestArrivedEvent(
                timestamp=arrival_time,
                request_id=self.next_request_id,
                input_text="",
                requested_output_tokens=self._generate_output_tokens(),
            )
            self.schedule_event(event)
            self.next_request_id += 1

    def _process_event(self, event: Event):
        """Process an event based on its type."""
        if event.event_type == EventType.REQUEST_ARRIVED:
            self._handle_request_arrived(event)
        elif event.event_type == EventType.PREFILL_FINISHED:
            self._handle_prefill_finished(event)
        elif event.event_type == EventType.DECODE_STEP_FINISHED:
            self._handle_decode_step_finished(event)
        elif event.event_type == EventType.REQUEST_FINISHED:
            self._handle_request_finished(event)
        elif event.event_type == EventType.KV_TRANSFER_FINISHED:
            self._handle_kv_transfer_finished(event)

    # ========== EVENT HANDLERS ==========

    def _handle_request_arrived(self, event: RequestArrivedEvent):
        """Handle new request arrival."""
        request = Request(
            request_id=event.request_id,
            arrival_time=event.timestamp,
            input_text=event.input_text,
            requested_output_tokens=event.requested_output_tokens,
            status=RequestStatus.ARRIVED,
        )

        self.metrics.total_requests += 1

        input_length = self._generate_input_length()
        request.input_length = input_length
        request.tokenization_time = self.current_time
        request.status = RequestStatus.TOKENIZED

        # Add to cluster (polymorphism!)
        self.cluster.add_request(request)

        # Try scheduling
        self._try_schedule()

    def _handle_prefill_finished(self, event: PrefillFinishedEvent):
        """Handle prefill completion."""
        # CRITICAL: Get batch BEFORE cluster handler (cluster may delete it!)
        scheduler = self.cluster.get_prefill_scheduler()
        batch = scheduler.batch_manager.get_batch(event.batch_id)  # ← 먼저 확보!

        # Delegate to cluster (polymorphism!)
        self.cluster.handle_prefill_finished(event.batch_id, self.current_time)  # ← 이제 호출해도 OK

        # Check if KV transfer needed (polymorphism!)
        transfer_delay = self.cluster.get_kv_transfer_delay(event.batch_id)

        if transfer_delay and transfer_delay > 0:
            # Use pre-fetched batch (not re-query!)
            if batch:
                transfer_event = KVTransferFinishedEvent(
                    timestamp=self.current_time + transfer_delay,
                    batch_id=event.batch_id,
                    request_ids=[r.request_id for r in batch.requests]
                )
                self.schedule_event(transfer_event)

        # Collect prefill metrics (use pre-fetched batch!)
        if self._is_in_measurement_window():
            if batch:
                for req in batch.requests:
                    if req.prefill_latency:
                        self.metrics.prefill_latencies.append(req.prefill_latency)

        # Try scheduling next work
        self._try_schedule()


    def _handle_decode_step_finished(self, event: DecodeStepFinishedEvent):
        """Handle decode step completion."""
        # Delegate to cluster (polymorphism!)
        scheduler = self.cluster.get_decode_scheduler()
        batch = scheduler.batch_manager.get_batch(event.batch_id)
        continue_result = self.cluster.handle_decode_step_finished(
            event.batch_id,
            self.current_time
        )

        # Collect finished request metrics BEFORE batch is cleaned up
        if self._is_in_measurement_window():
            # Get scheduler based on cluster type
            if batch:
                # Count all requests that were in batch (for token count)
                self.metrics.total_tokens_generated += len(batch.requests)

                for req in self.cluster.last_finished_requests:
                    self._record_completed_request(req)

        # If batch continues, schedule next iteration
        if continue_result:
            # Simulator creates Event
            event = self._create_event_from_result(continue_result)
            if event:
                self.schedule_event(event)

            if self._is_in_measurement_window():
                self.metrics.gpu_busy_time += continue_result.busy_time


        # Try scheduling next work (if batch finished)
        if not continue_result:
            self._try_schedule()


    def _handle_kv_transfer_finished(self, event: KVTransferFinishedEvent):
        """Handle KV cache transfer completion (disaggregated only)."""
        # All clusters implement this (polymorphism!)
        self.cluster.handle_kv_transfer_finished(event.batch_id, self.current_time)

        # Try scheduling decode after transfer
        self._try_schedule()


    def _handle_request_finished(self, event: RequestFinishedEvent):
        """Handle request completion (legacy - not used in current design)."""
        pass

    def _create_event_from_result(self, result: ScheduleResult) -> Optional[Event]:
        """Convert ScheduleResult descriptor to Event."""
        if result.operation_type == "prefill":
            return PrefillFinishedEvent(
                timestamp=self.current_time + result.busy_time,
                batch_id=result.batch_id
            )
        elif result.operation_type == "decode":
            return DecodeStepFinishedEvent(
                timestamp=self.current_time + result.busy_time,
                batch_id=result.batch_id,
                step=result.decode_step
            )
        return None


    # ========== SCHEDULING ==========

    def _try_schedule(self):
        """
        Try to schedule next batch.

        No branching! Cluster handles scheduling policy via polymorphism!
        """
        results = self.cluster.try_schedule(self.current_time)

        # Schedule all events returned
        for result in results:
            # Simulator creates Event from descriptor
            event = self._create_event_from_result(result)
            if event:
                self.schedule_event(event)

            # Track GPU busy time if in measurement window
            if self._is_in_measurement_window():
                self.metrics.gpu_busy_time += result.busy_time

    # ========== UTILITIES ==========

    def _generate_input_length(self) -> int:
        """Generate input length based on workload spec."""
        spec = self.config.workload_spec
        if spec.max_input_length == spec.avg_input_length:
            return spec.avg_input_length
        else:
            return random.randint(
                max(1, spec.avg_input_length - 100),
                spec.max_input_length
            )

    def _generate_output_tokens(self) -> int:
        """Generate requested output tokens."""
        spec = self.config.workload_spec
        if spec.max_output_length == spec.avg_output_length:
            return spec.avg_output_length
        else:
            return random.randint(
                max(1, spec.avg_output_length - 50),
                spec.max_output_length
            )

    def _is_in_measurement_window(self) -> bool:
        """Check if current time is in measurement window."""
        return self.measurement_start <= self.current_time <= self.measurement_end

    def _should_stop(self) -> bool:
        """Check if simulation should stop."""
        if self.current_time >= self.total_duration:
            return True

        if self.current_time >= self.measurement_end:
            # Check if all requests completed

            prefill_empty = len(self.cluster.get_prefill_scheduler().prefill_queue) == 0
            decode_empty = len(self.cluster.get_decode_scheduler().decode_queue) == 0
            if not self.cluster.is_busy() and prefill_empty and decode_empty:
                return True
        return False

    def _record_completed_request(self, request: Request):
        """Record metrics for a completed request."""
        if request.request_id in self.completed_requests_map:
            return

        self.completed_requests_map[request.request_id] = request
        self.metrics.completed_requests += 1

        if request.first_token_latency:
            self.metrics.first_token_latencies.append(request.first_token_latency)

        if request.end_to_end_latency:
            self.metrics.end_to_end_latencies.append(request.end_to_end_latency)

    def _print_summary(self):
        """Print simulation summary."""
        print(f"\nCompleted requests: {self.metrics.completed_requests} / {self.metrics.total_requests}")

        if self.metrics.first_token_latencies:
            print(f"Avg TTFT: {np.mean(self.metrics.first_token_latencies):.2f}s")

        if self.metrics.end_to_end_latencies:
            print(f"Avg E2E: {np.mean(self.metrics.end_to_end_latencies):.2f}s")

        print(f"Tokens generated: {self.metrics.total_tokens_generated}")
        print(f"Peak memory: {self.metrics.peak_memory_usage_gb:.2f} GB")

        if self.metrics.total_simulation_time > 0:
            xpu_utilization = self.metrics.gpu_busy_time / self.metrics.total_simulation_time
            print(f"xPU Utilization: {xpu_utilization * 100:.1f}%")