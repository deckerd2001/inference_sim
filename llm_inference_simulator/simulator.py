"""
Main event-driven simulator for LLM inference.
"""

import heapq
import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .events import *
from .config import SimulatorConfig
from .request import Request, Batch, RequestStatus
from .scheduler import create_scheduler
from .performance_model import PerformanceModel


@dataclass
class SimulationMetrics:
    """Collected metrics from simulation."""
    # Request-level metrics
    total_requests: int = 0
    completed_requests: int = 0
    
    first_token_latencies: List[float] = field(default_factory=list)
    end_to_end_latencies: List[float] = field(default_factory=list)
    prefill_latencies: List[float] = field(default_factory=list)
    
    # Token-level metrics
    total_tokens_generated: int = 0
    token_generation_times: List[float] = field(default_factory=list)
    
    # System-level metrics
    total_simulation_time: float = 0.0
    
    # GPU utilization tracking
    gpu_busy_time: float = 0.0
    gpu_idle_time: float = 0.0
    
    def compute_statistics(self) -> Dict:
        """Compute summary statistics."""
        stats = {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
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
        
        # GPU utilization
        total_time = self.gpu_busy_time + self.gpu_idle_time
        if total_time > 0:
            stats["gpu_utilization"] = self.gpu_busy_time / total_time
        
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
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        config.validate()
        
        # Random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize components
        self.scheduler = create_scheduler(config.scheduler_spec)
        self.performance_model = PerformanceModel(
            config.model_spec,
            config.cluster_spec.gpu_spec,
            config.parallelism_spec
        )
        
        # Event queue (priority queue)
        self.event_queue: List[Event] = []
        
        # Simulation state
        self.current_time: float = 0.0
        self.next_request_id: int = 0
        self.is_gpu_busy: bool = False
        self.current_batch: Optional[Batch] = None
        
        # Metrics
        self.metrics = SimulationMetrics()
        
        # Event log (optional)
        self.event_log: List[Event] = [] if config.output_event_log else None
    
    def schedule_event(self, event: Event):
        """Add an event to the event queue."""
        heapq.heappush(self.event_queue, event)
    
    def run(self):
        """Run the simulation."""
        print(f"Starting simulation...")
        print(f"Configuration: {self.config.model_spec.name}, "
              f"{self.config.cluster_spec.total_gpus} GPUs, "
              f"arrival_rate={self.config.workload_spec.arrival_rate} req/s")
        
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
            
            # Check stopping conditions
            if self._should_stop():
                break
        
        # Finalize metrics
        self.metrics.total_simulation_time = self.current_time
        
        print(f"\nSimulation completed at t={self.current_time:.2f}s")
        self._print_summary()
        
        return self.metrics
    
    def _schedule_initial_arrivals(self):
        """Schedule initial batch of request arrivals."""
        arrival_rate = self.config.workload_spec.arrival_rate
        duration = self.config.simulation_duration_s
        
        if self.config.workload_spec.arrival_process == "poisson":
            # Generate Poisson arrival times
            current_time = 0.0
            while current_time < duration:
                # Inter-arrival time is exponentially distributed
                inter_arrival = random.expovariate(arrival_rate)
                current_time += inter_arrival
                
                if current_time < duration:
                    self._create_request_arrival(current_time)
        
        elif self.config.workload_spec.arrival_process == "deterministic":
            # Fixed inter-arrival time
            inter_arrival = 1.0 / arrival_rate
            current_time = inter_arrival
            while current_time < duration:
                self._create_request_arrival(current_time)
                current_time += inter_arrival
    
    def _create_request_arrival(self, arrival_time: float):
        """Create a request arrival event."""
        # Sample input and output lengths
        input_length = int(max(1, np.random.normal(
            self.config.workload_spec.avg_input_length,
            self.config.workload_spec.input_length_std
        )))
        input_length = min(input_length, self.config.workload_spec.max_input_length)
        
        output_length = int(max(1, np.random.normal(
            self.config.workload_spec.avg_output_length,
            self.config.workload_spec.output_length_std
        )))
        output_length = min(output_length, self.config.workload_spec.max_output_length)
        
        # Create event
        event = RequestArrivedEvent(
            timestamp=arrival_time,
            request_id=self.next_request_id,
            input_text=f"prompt_{self.next_request_id}",  # Placeholder
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
        
        # After processing, try to schedule new work
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
        
        # Store request (will be populated with input_length after tokenization)
        self.scheduler.pending_requests[request.request_id] = request
        
        self.schedule_event(tokenized_event)
    
    def _handle_request_tokenized(self, event: RequestTokenizedEvent):
        """Handle tokenization completion."""
        request = self.scheduler.get_request(event.request_id)
        if request:
            request.input_length = event.input_length
            request.tokenization_time = event.timestamp
            self.scheduler.add_request(request)
    
    def _try_schedule_work(self):
        """Try to schedule prefill or decode work if GPU is idle."""
        if self.is_gpu_busy:
            return
        
        # Try prefill first
        if self.scheduler.can_schedule_prefill(self.current_time):
            self._schedule_prefill()
        # Then try decode
        elif self.scheduler.can_schedule_decode():
            self._schedule_decode()
        else:
            # GPU is idle
            pass
    
    def _schedule_prefill(self):
        """Schedule a prefill batch."""
        batch = self.scheduler.schedule_prefill_batch(self.current_time)
        if batch is None:
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
        
        # Schedule prefill finish event
        finish_event = PrefillFinishedEvent(
            timestamp=self.current_time + prefill_time,
            batch_id=batch.batch_id
        )
        self.schedule_event(finish_event)
        
        # Track GPU busy time
        self.metrics.gpu_busy_time += prefill_time
    
    def _handle_prefill_finished(self, event: PrefillFinishedEvent):
        """Handle prefill completion."""
        batch = self.scheduler.batch_manager.get_batch(event.batch_id)
        if batch is None:
            return
        
        batch.processing_end_time = self.current_time
        
        # Update requests
        for req in batch.requests:
            req.prefill_end_time = self.current_time
            req.current_kv_cache_length = req.input_length
            
            # Record prefill latency
            if req.prefill_latency:
                self.metrics.prefill_latencies.append(req.prefill_latency)
        
        # Move requests to decode queue
        self.scheduler.move_to_decode_queue(batch.requests)
        
        # Remove batch
        self.scheduler.batch_manager.remove_batch(batch)
        
        # GPU is now idle
        self.is_gpu_busy = False
        self.current_batch = None
    
    def _schedule_decode(self):
        """Schedule a decode step."""
        batch = self.scheduler.schedule_decode_batch(self.current_time)
        if batch is None:
            return
        
        self.is_gpu_busy = True
        self.current_batch = batch
        
        # Calculate decode time
        decode_time = self.performance_model.estimate_decode_time(
            batch_size=batch.batch_size,
            kv_cache_length=batch.max_kv_cache_length
        )
        
        batch.processing_start_time = self.current_time
        
        # Schedule decode finish event
        finish_event = DecodeStepFinishedEvent(
            timestamp=self.current_time + decode_time,
            batch_id=batch.batch_id,
            step=batch.current_decode_step
        )
        self.schedule_event(finish_event)
        
        # Track GPU busy time
        self.metrics.gpu_busy_time += decode_time
    
    def _handle_decode_step_finished(self, event: DecodeStepFinishedEvent):
        """Handle decode step completion."""
        batch = self.scheduler.batch_manager.get_batch(event.batch_id)
        if batch is None:
            return
        
        batch.processing_end_time = self.current_time
        batch.current_decode_step += 1
        
        # Update all requests in batch
        finished_requests = []
        for req in batch.requests:
            req.tokens_generated += 1
            req.current_kv_cache_length += 1
            self.metrics.total_tokens_generated += 1
            
            # Track first token time
            if req.tokens_generated == 1:
                req.first_token_time = self.current_time
                if req.first_token_latency:
                    self.metrics.first_token_latencies.append(req.first_token_latency)
            
            # Check if request is finished
            if req.is_finished:
                req.completion_time = self.current_time
                finished_requests.append(req)
                
                # Schedule request finished event
                finish_event = RequestFinishedEvent(
                    timestamp=self.current_time,
                    request_id=req.request_id,
                    total_output_tokens=req.tokens_generated
                )
                self.schedule_event(finish_event)
        
        # Remove finished requests
        if finished_requests:
            self.scheduler.remove_finished_requests(finished_requests)
            batch.remove_finished_requests()
        
        # Remove batch if all requests finished
        if batch.is_batch_finished:
            self.scheduler.batch_manager.remove_batch(batch)
        
        # GPU is now idle
        self.is_gpu_busy = False
        self.current_batch = None
    
    def _handle_request_finished(self, event: RequestFinishedEvent):
        """Handle request completion."""
        request = self.scheduler.get_request(event.request_id)
        if request and request.end_to_end_latency:
            self.metrics.completed_requests += 1
            self.metrics.end_to_end_latencies.append(request.end_to_end_latency)
    
    def _should_stop(self) -> bool:
        """Check if simulation should stop."""
        # Time-based stopping
        if self.current_time >= self.config.simulation_duration_s:
            return True
        
        # Request count-based stopping
        if (self.config.max_requests is not None and 
            self.metrics.completed_requests >= self.config.max_requests):
            return True
        
        return False
    
    def _print_summary(self):
        """Print simulation summary."""
        stats = self.metrics.compute_statistics()
        
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        
        print(f"\nRequests:")
        print(f"  Total: {stats['total_requests']}")
        print(f"  Completed: {stats['completed_requests']}")
        
        print(f"\nThroughput:")
        print(f"  Requests/sec: {stats.get('throughput_requests_per_sec', 0):.2f}")
        print(f"  Tokens/sec: {stats.get('throughput_tokens_per_sec', 0):.2f}")
        
        print(f"\nGPU Utilization: {stats.get('gpu_utilization', 0)*100:.1f}%")
        
        if 'first_token_latency' in stats:
            ftl = stats['first_token_latency']
            print(f"\nFirst Token Latency (seconds):")
            print(f"  Mean: {ftl['mean']:.4f}")
            print(f"  P50:  {ftl['p50']:.4f}")
            print(f"  P90:  {ftl['p90']:.4f}")
            print(f"  P95:  {ftl['p95']:.4f}")
            print(f"  P99:  {ftl['p99']:.4f}")
        
        if 'end_to_end_latency' in stats:
            e2e = stats['end_to_end_latency']
            print(f"\nEnd-to-End Latency (seconds):")
            print(f"  Mean: {e2e['mean']:.4f}")
            print(f"  P50:  {e2e['p50']:.4f}")
            print(f"  P90:  {e2e['p90']:.4f}")
            print(f"  P95:  {e2e['p95']:.4f}")
            print(f"  P99:  {e2e['p99']:.4f}")
        
        print("\n" + "="*60)
