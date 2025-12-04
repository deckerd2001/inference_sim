"""
LLM Inference Simulator - Event-driven simulation of LLM inference performance.

This package provides tools to simulate the performance of Large Language Model
inference systems, including prefill and decode phases, with support for various
parallelization strategies and hardware configurations.
"""

from .simulator import LLMInferenceSimulator, SimulationMetrics
from .config import (
    SimulatorConfig,
    ModelSpec,
    WorkloadSpec,
    GPUSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    SchedulerSpec,
    DataType,
    PositionalEncoding,
)
from .request import Request, Batch, RequestStatus
from .scheduler import Scheduler, PriorityScheduler, create_scheduler
from .performance_model import PerformanceModel
from .events import Event, EventType
from .gpu_catalog import GPUCatalog, get_gpu

__version__ = "0.1.0"
__author__ = "LLM Inference Simulator Team"

__all__ = [
    # Main simulator
    "LLMInferenceSimulator",
    "SimulationMetrics",
    
    # Configuration
    "SimulatorConfig",
    "ModelSpec",
    "WorkloadSpec",
    "GPUSpec",
    "ClusterSpec",
    "InterconnectSpec",
    "ParallelismSpec",
    "SchedulerSpec",
    "DataType",
    "PositionalEncoding",
    
    # GPU Catalog
    "GPUCatalog",
    "get_gpu",
    
    # Core components
    "Request",
    "Batch",
    "RequestStatus",
    "Scheduler",
    "PriorityScheduler",
    "create_scheduler",
    "PerformanceModel",
    "Event",
    "EventType",
]
