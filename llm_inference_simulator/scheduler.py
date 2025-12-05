"""
Scheduler for batching and managing requests.
"""

from typing import List, Optional, Callable
from collections import deque
from .request import Request, Batch, BatchManager, RequestStatus
from .config import SchedulerSpec


class Scheduler:
    """Base scheduler for managing prefill and decode queues."""
    
    def __init__(self, spec: SchedulerSpec):
        self.spec = spec
        
        # Queues
        self.prefill_queue = deque()
        self.decode_queue = deque()
        
        # Pending requests (waiting for tokenization)
        self.pending_requests = {}
        
        # Batch manager
        self.batch_manager = BatchManager()
    
    def add_request(self, request: Request):
        """Add a tokenized request to prefill queue."""
        request.status = RequestStatus.QUEUED
        self.prefill_queue.append(request)
    
    def can_schedule_prefill(self, current_time: float) -> bool:
        """Check if we can schedule a prefill batch."""
        return len(self.prefill_queue) > 0
    
    def can_schedule_decode(self) -> bool:
        """Check if we can schedule a decode batch."""
        return len(self.decode_queue) > 0
    
    def schedule_prefill_batch(self, current_time: float, 
                              memory_checker: Optional[Callable] = None) -> Optional[Batch]:
        """
        Schedule a prefill batch.
        
        Args:
            current_time: Current simulation time
            memory_checker: Optional callable(requests, is_prefill) -> (bool, str)
                          for dynamic batch sizing
        
        Returns:
            Batch or None
        """
        if not self.can_schedule_prefill(current_time):
            return None
        
        # Collect requests for batch
        batch_requests = []
        
        if memory_checker is None:
            # Fixed batch size mode (legacy)
            max_size = self.spec.max_batch_size if self.spec.max_batch_size else len(self.prefill_queue)
            
            for _ in range(min(max_size, len(self.prefill_queue))):
                req = self.prefill_queue.popleft()
                batch_requests.append(req)
        else:
            # Dynamic batch sizing based on memory
            while len(self.prefill_queue) > 0:
                req = self.prefill_queue[0]  # Peek
                
                # Try adding this request
                test_batch = batch_requests + [req]
                can_fit, reason = memory_checker(test_batch, is_prefill=True)
                
                if can_fit:
                    # Add to batch
                    batch_requests.append(self.prefill_queue.popleft())
                    
                    # Optional: respect max_batch_size if set
                    if (self.spec.max_batch_size is not None and 
                        len(batch_requests) >= self.spec.max_batch_size):
                        break
                else:
                    # Memory full, stop here
                    break
            
            # If we couldn't fit even one request, return None
            if not batch_requests:
                return None
        
        if not batch_requests:
            return None
        
        # Create batch with required arguments
        batch = self.batch_manager.create_batch(
            batch_requests,
            is_prefill=True,
            current_time=current_time
        )
        
        # Update request status (use correct enum value)
        for req in batch_requests:
            req.status = RequestStatus.PREFILLING
        
        return batch
    
    def schedule_decode_batch(self, current_time: float,
                             memory_checker: Optional[Callable] = None) -> Optional[Batch]:
        """
        Schedule a decode batch.
        
        Args:
            current_time: Current simulation time
            memory_checker: Optional callable for dynamic batch sizing
        
        Returns:
            Batch or None
        """
        if not self.can_schedule_decode():
            return None
        
        # Collect requests for batch
        batch_requests = []
        
        if memory_checker is None:
            # Fixed batch size mode
            max_size = self.spec.max_batch_size if self.spec.max_batch_size else len(self.decode_queue)
            
            for _ in range(min(max_size, len(self.decode_queue))):
                req = self.decode_queue.popleft()
                batch_requests.append(req)
        else:
            # Dynamic batch sizing based on memory
            while len(self.decode_queue) > 0:
                req = self.decode_queue[0]  # Peek
                
                # Try adding this request
                test_batch = batch_requests + [req]
                can_fit, reason = memory_checker(test_batch, is_prefill=False)
                
                if can_fit:
                    # Add to batch
                    batch_requests.append(self.decode_queue.popleft())
                    
                    # Optional: respect max_batch_size if set
                    if (self.spec.max_batch_size is not None and 
                        len(batch_requests) >= self.spec.max_batch_size):
                        break
                else:
                    # Memory full, stop here
                    break
            
            if not batch_requests:
                return None
        
        if not batch_requests:
            return None
        
        # Create batch with required arguments
        batch = self.batch_manager.create_batch(
            batch_requests,
            is_prefill=False,
            current_time=current_time
        )
        
        # Update request status (use correct enum value)
        for req in batch_requests:
            req.status = RequestStatus.DECODING
        
        return batch
    
    def move_to_decode_queue(self, requests: List[Request]):
        """Move requests from prefill to decode queue."""
        for req in requests:
            req.status = RequestStatus.QUEUED
            self.decode_queue.append(req)
    
    def remove_finished_requests(self, requests: List[Request]):
        """Remove finished requests from tracking."""
        for req in requests:
            req.status = RequestStatus.FINISHED
            # Remove from pending if still there
            self.pending_requests.pop(req.request_id, None)
    
    def get_request(self, request_id: int) -> Optional[Request]:
        """Get a request by ID."""
        return self.pending_requests.get(request_id)


class ContinuousBatchingScheduler(Scheduler):
    """Scheduler with continuous batching support."""
    
    def __init__(self, spec: SchedulerSpec):
        super().__init__(spec)


def create_scheduler(spec: SchedulerSpec) -> Scheduler:
    """Factory function to create appropriate scheduler."""
    if spec.batching_type == "continuous":
        return ContinuousBatchingScheduler(spec)
    else:
        return Scheduler(spec)
