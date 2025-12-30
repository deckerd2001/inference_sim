"""
Refactored simulator using abstract cluster layer - BUGFIX VERSION

Fixed issues:
1. Finished requests properly collected
2. Metrics collection works
3. No duplicate "Starting simulation..."
4. Event scheduling fixed
"""

import heapq
import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .events import (
    Event, EventType,
    RequestArrivedEvent, RequestTokenizedEvent,
    PrefillFinishedEvent,
    DecodeStepFinishedEvent,
    RequestFinishedEvent,
    KVTransferFinishedEvent,
)
from .config import SimulatorConfig
from .request import Request, Batch, RequestStatus
from .inference_cluster import create_inference_cluster, InferenceCluster


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
    """Event-driven simulator using abstract cluster layer - BUGFIX VERSION."""

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

        # Create cluster
        self.cluster = create_inference_cluster(config, tp_comm_strategy)

        self.scheduler = self.cluster.get_scheduler()
        self.is_disaggregated = config.is_disaggregated

    def schedule_event(self, event: Event):
        """Add an event to the event queue."""
        heapq.heappush(self.event_queue, event)

    def run(self):
        """Run the simulation."""
        # BUGFIX: Only print once (removed duplicate)
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

        mem_stats = self.cluster.get_memory_stats()
        if self.is_disaggregated:
            print(f"Prefill Memory: {mem_stats['prefill']['model_weights_gb']:.2f}GB model, "
                  f"{mem_stats['prefill']['available_for_kv_cache_gb']:.2f}GB available for KV cache")
            print(f"Decode Memory: {mem_stats['decode']['model_weights_gb']:.2f}GB model, "
                  f"{mem_stats['decode']['available_for_kv_cache_gb']:.2f}GB available for KV cache")
        else:
            print(f"Memory: {mem_stats['model_weights_gb']:.2f}GB model, "
                  f"{mem_stats['available_for_kv_cache_gb']:.2f}GB available for KV cache")

        self._schedule_initial_arrivals()

        # Main event loop
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp

            if self.event_log is not None:
                self.event_log.append(event)

            if self._should_stop():
                break

            self._dispatch_event(event)

            self._try_schedule()

        self.metrics.total_simulation_time = self.measurement_end - self.measurement_start

        self._print_summary()

        # Return metrics for JSON serialization by __main__.py
        return self.metrics

    def _dispatch_event(self, event: Event):
        """Dispatch event to appropriate handler."""
        if event.event_type == EventType.REQUEST_ARRIVED:
            self._handle_request_arrived(event)
        elif event.event_type == EventType.REQUEST_TOKENIZED:
            self._handle_request_tokenized(event)
        elif event.event_type == EventType.PREFILL_FINISHED:
            self._handle_prefill_finished(event)
        elif event.event_type == EventType.DECODE_STEP_FINISHED:
            self._handle_decode_step_finished(event)
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

        self.cluster.add_request(request)

    def _handle_request_tokenized(self, event: RequestTokenizedEvent):
        pass

    def _handle_prefill_finished(self, event: PrefillFinishedEvent):
        """Handle prefill completion."""
        new_events = self.cluster.handle_prefill_finished(event, self.current_time)

        for new_event in new_events:
            self.schedule_event(new_event)

        # Collect prefill metrics (batch still exists at this point)
        if self._is_in_measurement_window():
            batch = self.scheduler.batch_manager.get_batch(event.batch_id)
            if batch:
                for req in batch.requests:
                    if req.prefill_latency:
                        self.metrics.prefill_latencies.append(req.prefill_latency)

    def _handle_decode_step_finished(self, event: DecodeStepFinishedEvent):
        """
        Handle decode step completion - BUGFIX VERSION.

        Now properly collects finished requests for metrics.
        """
        # BUGFIX: Get result with finished_requests
        result = self.cluster.handle_decode_finished(event, self.current_time)

        # Schedule continuation events
        for new_event in result.events:
            self.schedule_event(new_event)

        # BUGFIX: Collect metrics from finished_requests
        if self._is_in_measurement_window():
            # Count tokens generated this step
            total_requests = len(result.finished_requests) + len(result.continuing_requests)
            self.metrics.total_tokens_generated += total_requests

            # Collect finished request metrics
            for req in result.finished_requests:
                self._record_completed_request(req)

    def _handle_kv_transfer_finished(self, event: KVTransferFinishedEvent):
        """Handle KV cache transfer completion (disaggregated only)."""
        if hasattr(self.cluster, 'handle_kv_transfer_finished'):
            self.cluster.handle_kv_transfer_finished(event, self.current_time)

    # ========== SCHEDULING ==========

    def _try_schedule(self):
        """Try to schedule next batch."""
        if self.is_disaggregated:
            # Disaggregated: Try both prefill and decode independently
            prefill_result = self.cluster.try_schedule_prefill(self.current_time)
            if prefill_result:
                busy_time, event = prefill_result
                self.schedule_event(event)
                if self._is_in_measurement_window():
                    self.metrics.gpu_busy_time += busy_time

            decode_result = self.cluster.try_schedule_decode(self.current_time)
            if decode_result:
                busy_time, event = decode_result
                self.schedule_event(event)
                if self._is_in_measurement_window():
                    self.metrics.gpu_busy_time += busy_time
        else:
            # Aggregated: Try prefill first, then decode (mutually exclusive)
            result = self.cluster.try_schedule_prefill(self.current_time)
            if result:
                busy_time, event = result
                self.schedule_event(event)
                if self._is_in_measurement_window():
                    self.metrics.gpu_busy_time += busy_time
                return

            # If cluster is free, try decode
            if not self.cluster.is_busy():
                result = self.cluster.try_schedule_decode(self.current_time)
                if result:
                    busy_time, event = result
                    self.schedule_event(event)
                    if self._is_in_measurement_window():
                        self.metrics.gpu_busy_time += busy_time

    # ========== WORKLOAD GENERATION ==========

    def _schedule_initial_arrivals(self):
        """Schedule initial request arrivals."""
        workload = self.config.workload_spec

        current_time = 0.0
        request_count = 0

        while current_time < self.total_duration:
            inter_arrival_time = random.expovariate(workload.arrival_rate)
            current_time += inter_arrival_time

            if current_time > self.total_duration:
                break

            output_length = max(
                1,
                int(random.gauss(
                    workload.avg_output_length,
                    workload.output_length_std
                ))
            )
            output_length = min(output_length, workload.max_output_length)

            event = RequestArrivedEvent(
                timestamp=current_time,
                request_id=self.next_request_id,
                input_text=f"request_{self.next_request_id}",
                requested_output_tokens=output_length
            )

            self.schedule_event(event)
            self.next_request_id += 1
            request_count += 1

            if (self.config.max_requests is not None and
                request_count >= self.config.max_requests):
                break

    def _generate_input_length(self) -> int:
        """Generate input length from configured distribution."""
        workload = self.config.workload_spec

        length = max(
            1,
            int(random.gauss(
                workload.avg_input_length,
                workload.input_length_std
            ))
        )
        return min(length, workload.max_input_length)

    # ========== METRICS ==========

    def _record_completed_request(self, request: Request):
        """Record metrics for a completed request."""
        self.completed_requests_map[request.request_id] = request
        self.metrics.completed_requests += 1

        if request.first_token_latency:
            self.metrics.first_token_latencies.append(request.first_token_latency)

        if request.end_to_end_latency:
            self.metrics.end_to_end_latencies.append(request.end_to_end_latency)

    def _is_in_measurement_window(self) -> bool:
        """Check if current time is in measurement window."""
        return self.measurement_start <= self.current_time <= self.measurement_end

    def _should_stop(self) -> bool:
        """Check if simulation should stop."""
        if self.current_time >= self.total_duration:
            return True

        if (self.config.max_requests is not None and
            self.metrics.completed_requests >= self.config.max_requests):
            return True

        return False

    # ========== SUMMARY ==========

    def _print_summary(self):
        """Print simulation summary."""
        import numpy as np

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

        print("Requests:")
        print(f"  Total: {self.metrics.total_requests}")
        print(f"  Completed: {self.metrics.completed_requests}")
        if self.metrics.rejected_requests > 0:
            print(f"  Rejected: {self.metrics.rejected_requests}")
        print()

        if self.metrics.total_simulation_time > 0:
            req_per_sec = self.metrics.completed_requests / self.metrics.total_simulation_time
            tok_per_sec = self.metrics.total_tokens_generated / self.metrics.total_simulation_time
            print("Throughput:")
            print(f"  Requests/sec: {req_per_sec:.2f}")
            print(f"  Tokens/sec: {tok_per_sec:.2f}")
            print()

        measurement_duration = self.measurement_end - self.measurement_start
        self.metrics.gpu_idle_time = measurement_duration - self.metrics.gpu_busy_time
        total_time = self.metrics.gpu_busy_time + self.metrics.gpu_idle_time
        xpu_util = self.metrics.gpu_busy_time / total_time if total_time > 0 else 0.0
        print(f"xPU Utilization: {xpu_util:.1%}")
        print()

        mem_stats = self.cluster.get_memory_stats()
        if self.is_disaggregated:
            print("Memory Usage (Decode Cluster):")
            total_mem = (self.config.disaggregation_spec.decode_cluster.total_xpus *
                        self.config.disaggregation_spec.decode_cluster.xpu_spec.memory_size_gb)
            current_used = mem_stats['decode']['current_used_gb']
            print(f"  Current: {current_used:.2f}GB / {total_mem:.0f}GB "
                  f"({current_used/total_mem*100:.1f}%)")
        else:
            print("Memory Usage:")
            total_mem = (self.config.cluster_spec.total_xpus *
                        self.config.cluster_spec.xpu_spec.memory_size_gb)
            current_used = mem_stats['current_used_gb']
            print(f"  Current: {current_used:.2f}GB / {total_mem:.0f}GB "
                  f"({current_used/total_mem*100:.1f}%)")
        print()

        if self.metrics.first_token_latencies:
            ftl = np.array(self.metrics.first_token_latencies)
            print("First Token Latency (seconds):")
            print(f"  Mean: {np.mean(ftl):.4f}")
            print(f"  P50:  {np.percentile(ftl, 50):.4f}")
            print(f"  P90:  {np.percentile(ftl, 90):.4f}")
            print(f"  P95:  {np.percentile(ftl, 95):.4f}")
            print(f"  P99:  {np.percentile(ftl, 99):.4f}")
            print()

        if self.metrics.end_to_end_latencies:
            e2e = np.array(self.metrics.end_to_end_latencies)
            print("End-to-End Latency (seconds):")
            print(f"  Mean: {np.mean(e2e):.4f}")
            print(f"  P50:  {np.percentile(e2e, 50):.4f}")
            print(f"  P90:  {np.percentile(e2e, 90):.4f}")
            print(f"  P95:  {np.percentile(e2e, 95):.4f}")
            print(f"  P99:  {np.percentile(e2e, 99):.4f}")
            print()

        print("=" * 60)