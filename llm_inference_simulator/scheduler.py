"""
Scheduler for batching and scheduling inference requests.
"""

from typing import List, Optional, Dict
from collections import deque
import heapq

from .request import Request, Batch, BatchManager, RequestStatus
from .config import SchedulerSpec


class Scheduler:
    """
    Base scheduler class for managing request batching and scheduling.
    """
    
    def __init__(self, spec: SchedulerSpec):
        self.spec = spec
        self.batch_manager = BatchManager()
        
        # Request queues
        self.prefill_queue: deque[Request] = deque()  # Waiting for prefill
        self.decode_queue: List[Request] = []  # Currently decoding
        
        # Track pending requests by ID
        self.pending_requests: Dict[int, Request] = {}
    
    def add_request(self, request: Request):
        """Add a new request to the scheduler."""
        self.pending_requests[request.request_id] = request
        request.status = RequestStatus.QUEUED
        self.prefill_queue.append(request)
    
    def can_schedule_prefill(self, current_time: float) -> bool:
        """Check if we can schedule a prefill batch."""
        if not self.prefill_queue:
            return False
        
        # Check if we have enough requests or waited long enough
        if self.spec.batching_type == "static":
            return len(self.prefill_queue) >= self.spec.max_batch_size
        
        elif self.spec.batching_type == "continuous":
            # For continuous batching, we can schedule as soon as we have requests
            return True
        
        else:  # dynamic
            # Check batching window
            oldest_request = self.prefill_queue[0]
            wait_time = current_time - oldest_request.arrival_time
            
            return (len(self.prefill_queue) >= self.spec.max_batch_size or
                   wait_time >= self.spec.batching_window_ms / 1000.0)
    
    def schedule_prefill_batch(self, current_time: float) -> Optional[Batch]:
        """
        Create a prefill batch from waiting requests.
        
        Returns:
            Batch object or None if no batch can be formed
        """
        if not self.can_schedule_prefill(current_time):
            return None
        
        # Determine batch size
        batch_size = min(len(self.prefill_queue), self.spec.max_batch_size)
        
        # Select requests (FIFO for now)
        selected_requests = []
        for _ in range(batch_size):
            if self.prefill_queue:
                req = self.prefill_queue.popleft()
                req.status = RequestStatus.PREFILLING
                selected_requests.append(req)
        
        if not selected_requests:
            return None
        
        # Create batch
        batch = self.batch_manager.create_batch(
            requests=selected_requests,
            is_prefill=True,
            current_time=current_time
        )
        
        return batch
    
    def move_to_decode_queue(self, requests: List[Request]):
        """Move requests from prefill to decode queue after prefill completes."""
        for req in requests:
            req.status = RequestStatus.DECODING
            self.decode_queue.append(req)
    
    def can_schedule_decode(self) -> bool:
        """Check if we can schedule a decode batch."""
        return len(self.decode_queue) > 0
    
    def schedule_decode_batch(self, current_time: float) -> Optional[Batch]:
        """
        Create a decode batch from decoding requests.
        
        For continuous batching, all active decode requests are batched together.
        """
        if not self.can_schedule_decode():
            return None
        
        # For continuous batching, include all active requests
        if self.spec.token_level_scheduling:
            # All requests in decode queue
            batch_requests = self.decode_queue.copy()
        else:
            # Take up to max_batch_size
            batch_size = min(len(self.decode_queue), self.spec.max_batch_size)
            batch_requests = self.decode_queue[:batch_size]
        
        if not batch_requests:
            return None
        
        # Create decode batch
        batch = self.batch_manager.create_batch(
            requests=batch_requests,
            is_prefill=False,
            current_time=current_time
        )
        
        return batch
    
    def remove_finished_requests(self, requests: List[Request]):
        """Remove finished requests from decode queue."""
        for req in requests:
            if req.is_finished:
                req.status = RequestStatus.FINISHED
                if req in self.decode_queue:
                    self.decode_queue.remove(req)
                if req.request_id in self.pending_requests:
                    del self.pending_requests[req.request_id]
    
    def get_request(self, request_id: int) -> Optional[Request]:
        """Get a request by ID."""
        return self.pending_requests.get(request_id)
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """Get current queue lengths for monitoring."""
        return {
            "prefill_queue": len(self.prefill_queue),
            "decode_queue": len(self.decode_queue),
            "total_pending": len(self.pending_requests)
        }


class PriorityScheduler(Scheduler):
    """
    Scheduler with priority queue support.
    Requests with higher priority are scheduled first.
    """
    
    def __init__(self, spec: SchedulerSpec):
        super().__init__(spec)
        # Override with priority queues
        self.prefill_pqueue: List[tuple] = []  # (priority, arrival_time, request)
    
    def add_request(self, request: Request, priority: int = 0):
        """Add request with priority (lower number = higher priority)."""
        self.pending_requests[request.request_id] = request
        request.status = RequestStatus.QUEUED
        # Use arrival time as tiebreaker
        heapq.heappush(self.prefill_pqueue, 
                      (priority, request.arrival_time, request))
    
    def schedule_prefill_batch(self, current_time: float) -> Optional[Batch]:
        """Schedule prefill batch based on priority."""
        if not self.prefill_pqueue:
            return None
        
        # Select highest priority requests
        batch_size = min(len(self.prefill_pqueue), self.spec.max_batch_size)
        selected_requests = []
        
        for _ in range(batch_size):
            if self.prefill_pqueue:
                _, _, req = heapq.heappop(self.prefill_pqueue)
                req.status = RequestStatus.PREFILLING
                selected_requests.append(req)
        
        if not selected_requests:
            return None
        
        batch = self.batch_manager.create_batch(
            requests=selected_requests,
            is_prefill=True,
            current_time=current_time
        )
        
        return batch


def create_scheduler(spec: SchedulerSpec) -> Scheduler:
    """Factory function to create appropriate scheduler."""
    if spec.priority_policy == "priority_queue":
        return PriorityScheduler(spec)
    else:
        return Scheduler(spec)
