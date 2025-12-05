"""
Configuration classes for the LLM inference simulator.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DataType(Enum):
    """Data types for model parameters and activations."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    
    def bytes_per_element(self) -> int:
        """Return bytes per element."""
        if self == DataType.FP32:
            return 4
        elif self == DataType.FP16:
            return 2
        elif self == DataType.BF16:
            return 2
        elif self == DataType.INT8:
            return 1
        else:
            raise ValueError(f"Unknown data type: {self}")


@dataclass
class ModelSpec:
    """Model architecture specification."""
    name: str
    n_params: int
    n_layers: int
    hidden_size: int
    n_heads: int
    ffn_dim: int
    vocab_size: int
    max_seq_length: int
    weight_dtype: DataType = DataType.BF16
    activation_dtype: DataType = DataType.BF16
    positional_encoding: str = "RoPE"


@dataclass
class GPUSpec:
    """GPU hardware specification."""
    name: str
    memory_size_gb: float
    compute_tflops: float
    memory_bandwidth_gbs: float


@dataclass
class InterconnectSpec:
    """Interconnect specification for multi-GPU communication."""
    intra_node_type: str = "NVLink"
    intra_node_bandwidth_gbs: float = 600.0
    intra_node_latency_us: float = 2.0
    inter_node_type: str = "InfiniBand"
    inter_node_bandwidth_gbs: float = 200.0
    inter_node_latency_us: float = 10.0


@dataclass
class ClusterSpec:
    """Cluster configuration."""
    n_gpus_per_node: int
    n_nodes: int = 1
    gpu_spec: Optional[GPUSpec] = None
    interconnect_spec: Optional[InterconnectSpec] = None
    
    @property
    def total_gpus(self) -> int:
        return self.n_gpus_per_node * self.n_nodes


@dataclass
class ParallelismSpec:
    """Parallelism configuration."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1


@dataclass
class WorkloadSpec:
    """Workload specification."""
    avg_input_length: int = 512
    input_length_std: int = 100
    max_input_length: int = 2048
    
    avg_output_length: int = 128
    output_length_std: int = 50
    max_output_length: int = 512
    
    arrival_rate: float = 1.0
    arrival_process: str = "poisson"
    
    batch_size: int = 8


@dataclass
class SchedulerSpec:
    """Scheduler configuration."""
    batching_type: str = "continuous"
    max_batch_size: Optional[int] = 32
    token_level_scheduling: bool = True
    
    # Batching strategy
    batching_strategy: str = "greedy"  # "greedy" or "windowed"
    min_batch_size: int = 1  # Only used for windowed
    batching_window_ms: float = 10.0  # Only used for windowed
    
    def __post_init__(self):
        assert self.batching_type in ["static", "continuous"]
        assert self.batching_strategy in ["greedy", "windowed"]
        if self.max_batch_size is not None:
            assert self.max_batch_size > 0
        assert self.min_batch_size > 0
        assert self.batching_window_ms >= 0


@dataclass
class SimulatorConfig:
    """Main simulator configuration."""
    model_spec: ModelSpec
    workload_spec: WorkloadSpec
    cluster_spec: ClusterSpec
    parallelism_spec: ParallelismSpec
    scheduler_spec: SchedulerSpec
    
    simulation_duration_s: float = 60.0
    max_requests: Optional[int] = None
    random_seed: int = 42
    output_event_log: bool = False
    
    def validate(self):
        """Validate configuration."""
        assert self.simulation_duration_s > 0
        assert self.parallelism_spec.tensor_parallel_size <= self.cluster_spec.total_gpus
