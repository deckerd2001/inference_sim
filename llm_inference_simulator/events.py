"""
Event definitions for the simulator.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(Enum):
    """Types of events in the simulation."""
    REQUEST_ARRIVED = "request_arrived"
    REQUEST_TOKENIZED = "request_tokenized"
    PREFILL_STARTED = "prefill_started"
    PREFILL_FINISHED = "prefill_finished"
    DECODE_STEP_STARTED = "decode_step_started"
    DECODE_STEP_FINISHED = "decode_step_finished"
    TOKEN_EMITTED = "token_emitted"
    REQUEST_FINISHED = "request_finished"
    BATCHING_WAKEUP = "batching_wakeup"  # NEW


@dataclass
class Event:
    """Base class for all events."""
    timestamp: float
    event_type: EventType
    
    def __lt__(self, other):
        """Compare events by timestamp for priority queue."""
        return self.timestamp < other.timestamp


@dataclass
class RequestArrivedEvent(Event):
    """A new request has arrived."""
    request_id: int
    input_text: str
    requested_output_tokens: int
    
    def __init__(self, timestamp: float, request_id: int, 
                 input_text: str, requested_output_tokens: int):
        super().__init__(timestamp, EventType.REQUEST_ARRIVED)
        self.request_id = request_id
        self.input_text = input_text
        self.requested_output_tokens = requested_output_tokens


@dataclass
class RequestTokenizedEvent(Event):
    """Request has been tokenized."""
    request_id: int
    input_length: int
    
    def __init__(self, timestamp: float, request_id: int, input_length: int):
        super().__init__(timestamp, EventType.REQUEST_TOKENIZED)
        self.request_id = request_id
        self.input_length = input_length


@dataclass
class PrefillStartedEvent(Event):
    """Prefill phase has started."""
    batch_id: int
    
    def __init__(self, timestamp: float, batch_id: int):
        super().__init__(timestamp, EventType.PREFILL_STARTED)
        self.batch_id = batch_id


@dataclass
class PrefillFinishedEvent(Event):
    """Prefill phase has finished."""
    batch_id: int
    
    def __init__(self, timestamp: float, batch_id: int):
        super().__init__(timestamp, EventType.PREFILL_FINISHED)
        self.batch_id = batch_id


@dataclass
class DecodeStepStartedEvent(Event):
    """A decode step has started."""
    batch_id: int
    step: int
    
    def __init__(self, timestamp: float, batch_id: int, step: int):
        super().__init__(timestamp, EventType.DECODE_STEP_STARTED)
        self.batch_id = batch_id
        self.step = step


@dataclass
class DecodeStepFinishedEvent(Event):
    """A decode step has finished."""
    batch_id: int
    step: int
    
    def __init__(self, timestamp: float, batch_id: int, step: int):
        super().__init__(timestamp, EventType.DECODE_STEP_FINISHED)
        self.batch_id = batch_id
        self.step = step


@dataclass
class TokenEmittedEvent(Event):
    """A token has been generated."""
    request_id: int
    token_id: int
    token_text: str
    
    def __init__(self, timestamp: float, request_id: int, 
                 token_id: int, token_text: str):
        super().__init__(timestamp, EventType.TOKEN_EMITTED)
        self.request_id = request_id
        self.token_id = token_id
        self.token_text = token_text


@dataclass
class RequestFinishedEvent(Event):
    """A request has finished."""
    request_id: int
    total_output_tokens: int
    
    def __init__(self, timestamp: float, request_id: int, 
                 total_output_tokens: int):
        super().__init__(timestamp, EventType.REQUEST_FINISHED)
        self.request_id = request_id
        self.total_output_tokens = total_output_tokens


@dataclass
class BatchingWakeupEvent(Event):
    """Wakeup event to check if batching window has expired."""
    
    def __init__(self, timestamp: float):
        super().__init__(timestamp, EventType.BATCHING_WAKEUP)
