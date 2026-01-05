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
        """
        Check if we can schedule a prefill batch.

        Two strategies:
        - Greedy: Schedule immediately if queue not empty (default)
        - Windowed: Wait for min_batch_size or timeout
        """
        if len(self.prefill_queue) == 0:
            return False

        # GREEDY: Process immediately (like vLLM, TGI)
        if self.spec.batching_strategy == "greedy":
            return True

        # WINDOWED: Wait for batch to fill
        elif self.spec.batching_strategy == "windowed":
            # Check if minimum batch size reached
            if len(self.prefill_queue) >= self.spec.min_batch_size:
                return True

            # Check if batching window expired for oldest request
            oldest_request = self.prefill_queue[0]
            time_in_queue = current_time - oldest_request.arrival_time
            batching_window_s = self.spec.batching_window_ms / 1000.0

            if time_in_queue >= batching_window_s:
                return True

            return False

        return False

    def can_schedule_decode(self) -> bool:
        """Check if we can schedule a decode batch."""
        return len(self.decode_queue) > 0

    def get_next_wakeup_time(self, current_time: float) -> Optional[float]:
        """
        Get next wakeup time for batching window (only for windowed strategy).

        Returns None if no wakeup needed.
        """
        # Greedy doesn't need wakeup
        if self.spec.batching_strategy == "greedy":
            return None

        if len(self.prefill_queue) == 0:
            return None

        # If already have enough requests, no wakeup needed
        if len(self.prefill_queue) >= self.spec.min_batch_size:
            return None

        # Calculate when oldest request will hit batching window
        oldest_request = self.prefill_queue[0]
        batching_window_s = self.spec.batching_window_ms / 1000.0
        wakeup_time = oldest_request.arrival_time + batching_window_s

        # Only schedule wakeup if it's in the future
        if wakeup_time > current_time:
            return wakeup_time

        return None

    def schedule_prefill_batch(self, current_time: float,
                              memory_checker: Optional[Callable] = None) -> Optional[Batch]:
        """
        Schedule a prefill batch with dynamic memory-based sizing.
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

                # FIX: Append first, then check (avoids O(N) list copy)
                batch_requests.append(req)
                can_fit, reason = memory_checker(batch_requests, is_prefill=True)

                if can_fit:
                    # Fits! Remove from queue
                    self.prefill_queue.popleft()

                    # Optional: respect max_batch_size if set
                    if (self.spec.max_batch_size is not None and
                        len(batch_requests) >= self.spec.max_batch_size):
                        break
                else:
                    # Doesn't fit, remove from batch and stop
                    batch_requests.pop()
                    break


        if not batch_requests:
            return None

        # Create batch
        batch = self.batch_manager.create_batch(
            batch_requests,
            is_prefill=True,
            current_time=current_time
        )

        # Update request status
        for req in batch_requests:
            req.status = RequestStatus.PREFILLING

        return batch

    def schedule_decode_batch(self, current_time: float,
                             memory_checker: Optional[Callable] = None) -> Optional[Batch]:
        """
        Schedule a decode batch with dynamic memory-based sizing.
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

                # FIX: Append first, then check (avoids O(N) list copy)
                batch_requests.append(req)
                can_fit, reason = memory_checker(batch_requests, is_prefill=False)

                if can_fit:
                    # Fits! Remove from queue
                    self.decode_queue.popleft()

                    # Optional: respect max_batch_size if set
                    if (self.spec.max_batch_size is not None and
                        len(batch_requests) >= self.spec.max_batch_size):
                        break
                else:
                    # Doesn't fit, remove from batch and stop
                    batch_requests.pop()
                    break

        if not batch_requests:
            return None

        # Create batch
        batch = self.batch_manager.create_batch(
            batch_requests,
            is_prefill=False,
            current_time=current_time
        )

        # Update request status
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
