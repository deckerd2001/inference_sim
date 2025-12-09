"""
LLM Inference Simulator

A discrete-event simulator for modeling Large Language Model inference.
Supports various accelerators (GPU, TPU, NPU) and parallelism strategies.
"""

# Core xPU abstraction
from .xpu_spec import (
    xPUSpec,
    ComputeUnit,
    OperationType,
    DataType,
)

from .xpu_catalog import (
    get_xpu,
    list_xpus,
)

# Configuration
from .config import (
    ModelSpec,
    ClusterSpec,
    InterconnectSpec,
    ParallelismSpec,
    WorkloadSpec,
    SchedulerSpec,
    SimulatorConfig,
    DisaggregationSpec,
)

# Model catalog
from .model_catalog import (
    get_model,
    list_models,
)

# Main simulator
from .simulator import (
    LLMInferenceSimulator,
)

__version__ = "0.2.0"  # xPU refactor!

__all__ = [
    # xPU
    "xPUSpec",
    "ComputeUnit",
    "OperationType",
    "DataType",
    "get_xpu",
    "list_xpus",

    # Config
    "ModelSpec",
    "ClusterSpec",
    "InterconnectSpec",
    "ParallelismSpec",
    "WorkloadSpec",
    "SchedulerSpec",
    "SimulatorConfig",

    # Catalogs
    "get_model",
    "list_models",

    # Simulator
    "LLMInferenceSimulator",
]
