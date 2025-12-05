"""
LLM Inference Simulator Package
"""

from .config import (
    ModelSpec,
    GPUSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    WorkloadSpec,
    SchedulerSpec,
    SimulatorConfig,
    DataType,
)

from .request import Request, RequestStatus, Batch, BatchManager

from .scheduler import Scheduler, create_scheduler

from .performance_model import PerformanceModel

from .memory_manager import MemoryManager, MemoryUsage

from .simulator import LLMInferenceSimulator, SimulationMetrics

from .events import (
    Event,
    EventType,
    RequestArrivedEvent,
    RequestTokenizedEvent,
    PrefillStartedEvent,
    PrefillFinishedEvent,
    DecodeStepStartedEvent,
    DecodeStepFinishedEvent,
    TokenEmittedEvent,
    RequestFinishedEvent,
)

from .communication import (
    CollectiveOp,
    CommunicationAlgorithm,
    CommunicationPattern,
    TPCommunicationStrategy,
    estimate_collective_time,
    create_megatron_tp_strategy,
    create_sequence_parallel_strategy,
)

# Catalog imports
from .gpu_catalog import GPUCatalog, get_gpu
from .model_catalog import ModelCatalog, get_model

__version__ = "0.1.0"

__all__ = [
    # Config
    "ModelSpec",
    "GPUSpec",
    "ClusterSpec",
    "InterconnectSpec",
    "ParallelismSpec",
    "WorkloadSpec",
    "SchedulerSpec",
    "SimulatorConfig",
    "DataType",
    # Request
    "Request",
    "RequestStatus",
    "Batch",
    "BatchManager",
    # Scheduler
    "Scheduler",
    "create_scheduler",
    # Performance
    "PerformanceModel",
    # Memory
    "MemoryManager",
    "MemoryUsage",
    # Simulator
    "LLMInferenceSimulator",
    "SimulationMetrics",
    # Events
    "Event",
    "EventType",
    "RequestArrivedEvent",
    "RequestTokenizedEvent",
    "PrefillStartedEvent",
    "PrefillFinishedEvent",
    "DecodeStepStartedEvent",
    "DecodeStepFinishedEvent",
    "TokenEmittedEvent",
    "RequestFinishedEvent",
    # Communication
    "CollectiveOp",
    "CommunicationAlgorithm",
    "CommunicationPattern",
    "TPCommunicationStrategy",
    "estimate_collective_time",
    "create_megatron_tp_strategy",
    "create_sequence_parallel_strategy",
    # Catalogs
    "GPUCatalog",
    "get_gpu",
    "ModelCatalog",
    "get_model",
]
