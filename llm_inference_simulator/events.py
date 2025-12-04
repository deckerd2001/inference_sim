"""
Event classes for the LLM Inference Simulator.
All events are timestamped and processed in chronological order.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class EventType(Enum):
    """Types of events in the simulation."""
    REQUEST_ARRIVED = "request_arrived"
    REQUEST_TOKENIZED = "request_tokenized"
    BATCH_FORMED = "batch_formed"
    PREFILL_STARTED = "prefill_started"
    PREFILL_FINISHED = "prefill_finished"
    DECODE_STEP_STARTED = "decode_step_started"
    DECODE_STEP_FINISHED = "decode_step_finished"
    TOKEN_EMITTED = "token_emitted"
    REQUEST_FINISHED = "request_finished"


@dataclass
class Event:
    """Base class for all events."""
    timestamp: float  # Simulation time in seconds
    event_type: EventType
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.timestamp < other.timestamp
    
    def __repr__(self):
        return f"{self.event_type.value}@{self.timestamp:.6f}s"


@dataclass
class RequestArrivedEvent(Event):
    """Event when a new inference request arrives."""
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
    """Event when input text is tokenized."""
    request_id: int
    input_length: int
    
    def __init__(self, timestamp: float, request_id: int, input_length: int):
        super().__init__(timestamp, EventType.REQUEST_TOKENIZED)
        self.request_id = request_id
        self.input_length = input_length


@dataclass
class BatchFormedEvent(Event):
    """Event when a batch is formed for processing."""
    batch_id: int
    request_ids: List[int]
    is_prefill: bool  # True for prefill, False for decode
    
    def __init__(self, timestamp: float, batch_id: int, 
                 request_ids: List[int], is_prefill: bool):
        super().__init__(timestamp, EventType.BATCH_FORMED)
        self.batch_id = batch_id
        self.request_ids = request_ids
        self.is_prefill = is_prefill


@dataclass
class PrefillStartedEvent(Event):
    """Event when prefill computation starts."""
    batch_id: int
    
    def __init__(self, timestamp: float, batch_id: int):
        super().__init__(timestamp, EventType.PREFILL_STARTED)
        self.batch_id = batch_id


@dataclass
class PrefillFinishedEvent(Event):
    """Event when prefill computation finishes."""
    batch_id: int
    
    def __init__(self, timestamp: float, batch_id: int):
        super().__init__(timestamp, EventType.PREFILL_FINISHED)
        self.batch_id = batch_id


@dataclass
class DecodeStepStartedEvent(Event):
    """Event when a decode step starts."""
    batch_id: int
    step: int  # Which token position we're generating
    
    def __init__(self, timestamp: float, batch_id: int, step: int):
        super().__init__(timestamp, EventType.DECODE_STEP_STARTED)
        self.batch_id = batch_id
        self.step = step


@dataclass
class DecodeStepFinishedEvent(Event):
    """Event when a decode step finishes."""
    batch_id: int
    step: int
    
    def __init__(self, timestamp: float, batch_id: int, step: int):
        super().__init__(timestamp, EventType.DECODE_STEP_FINISHED)
        self.batch_id = batch_id
        self.step = step


@dataclass
class TokenEmittedEvent(Event):
    """Event when a token is generated and can be sent to client."""
    request_id: int
    token_position: int
    
    def __init__(self, timestamp: float, request_id: int, token_position: int):
        super().__init__(timestamp, EventType.TOKEN_EMITTED)
        self.request_id = request_id
        self.token_position = token_position


@dataclass
class RequestFinishedEvent(Event):
    """Event when a request completes (EOS or max length reached)."""
    request_id: int
    total_output_tokens: int
    
    def __init__(self, timestamp: float, request_id: int, total_output_tokens: int):
        super().__init__(timestamp, EventType.REQUEST_FINISHED)
        self.request_id = request_id
        self.total_output_tokens = total_output_tokens
