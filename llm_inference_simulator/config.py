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
    batching_strategy: str = "greedy"
    min_batch_size: int = 1
    batching_window_ms: float = 10.0
    
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
    
    def __post_init__(self):
        """Automatically validate configuration after creation."""
        self.validate()
    
    def validate(self):
        """
        Validate configuration for common errors.
        
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        warnings = []
        
        # 1. Basic validation
        if self.simulation_duration_s <= 0:
            errors.append("simulation_duration_s must be positive")
        
        # 2. Parallelism validation
        total_gpus = self.cluster_spec.total_gpus
        tp_size = self.parallelism_spec.tensor_parallel_size
        pp_size = self.parallelism_spec.pipeline_parallel_size
        dp_size = self.parallelism_spec.data_parallel_size
        
        if tp_size <= 0:
            errors.append("tensor_parallel_size must be positive")
        
        if pp_size <= 0:
            errors.append("pipeline_parallel_size must be positive")
        
        if dp_size <= 0:
            errors.append("data_parallel_size must be positive")
        
        # Check total parallelism
        total_parallel = tp_size * pp_size * dp_size
        if total_parallel > total_gpus:
            errors.append(
                f"Total parallelism ({tp_size}×{pp_size}×{dp_size}={total_parallel}) "
                f"exceeds available GPUs ({total_gpus})"
            )
        
        if total_parallel < total_gpus:
            warnings.append(
                f"Underutilization: Using {total_parallel} GPUs out of {total_gpus} available"
            )
        
        # 3. Memory validation (rough estimate)
        if self.cluster_spec.gpu_spec:
            gpu_memory = self.cluster_spec.gpu_spec.memory_size_gb
            model_params = self.model_spec.n_params
            bytes_per_param = self.model_spec.weight_dtype.bytes_per_element()
            
            # Model memory per GPU with TP
            model_memory_gb = (model_params * bytes_per_param) / (1024**3) / tp_size
            
            if model_memory_gb > gpu_memory:
                errors.append(
                    f"Model weights ({model_memory_gb:.2f}GB per GPU with TP={tp_size}) "
                    f"exceed GPU memory ({gpu_memory:.1f}GB). "
                    f"Increase tensor_parallel_size or use larger GPUs."
                )
            
            # Check if reasonable memory left for KV cache
            available_memory = gpu_memory - model_memory_gb - 2.0  # 2GB safety
            if available_memory < 10.0:
                warnings.append(
                    f"Only {available_memory:.1f}GB available for KV cache. "
                    f"Consider increasing tensor_parallel_size."
                )
        
        # 4. Workload validation
        if self.workload_spec.avg_input_length > self.workload_spec.max_input_length:
            errors.append("avg_input_length cannot exceed max_input_length")
        
        if self.workload_spec.avg_output_length > self.workload_spec.max_output_length:
            errors.append("avg_output_length cannot exceed max_output_length")
        
        if self.workload_spec.arrival_rate <= 0:
            errors.append("arrival_rate must be positive")
        
        # Check if sequence length exceeds model's max
        max_seq = self.workload_spec.max_input_length + self.workload_spec.max_output_length
        if max_seq > self.model_spec.max_seq_length:
            errors.append(
                f"Max sequence length ({max_seq}) exceeds model's max ({self.model_spec.max_seq_length})"
            )
        
        # 5. Raise errors if any
        if errors:
            error_msg = "Configuration validation failed:\n"
            for i, error in enumerate(errors, 1):
                error_msg += f"  {i}. {error}\n"
            raise ValueError(error_msg.strip())
        
        # 6. Print warnings
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"  • {warning}")
            print()
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\nModel: {self.model_spec.name}")
        print(f"  Parameters: {self.model_spec.n_params / 1e9:.1f}B")
        print(f"  Layers: {self.model_spec.n_layers}")
        print(f"  Hidden size: {self.model_spec.hidden_size}")
        
        print(f"\nCluster:")
        print(f"  GPUs: {self.cluster_spec.total_gpus} "
              f"({self.cluster_spec.n_nodes} nodes × {self.cluster_spec.n_gpus_per_node} GPUs)")
        if self.cluster_spec.gpu_spec:
            print(f"  GPU: {self.cluster_spec.gpu_spec.name} "
                  f"({self.cluster_spec.gpu_spec.memory_size_gb:.0f}GB)")
        
        print(f"\nParallelism:")
        print(f"  Tensor Parallel: {self.parallelism_spec.tensor_parallel_size}")
        print(f"  Pipeline Parallel: {self.parallelism_spec.pipeline_parallel_size}")
        print(f"  Data Parallel: {self.parallelism_spec.data_parallel_size}")
        
        print(f"\nWorkload:")
        print(f"  Input length: {self.workload_spec.avg_input_length} "
              f"(max: {self.workload_spec.max_input_length})")
        print(f"  Output length: {self.workload_spec.avg_output_length} "
              f"(max: {self.workload_spec.max_output_length})")
        print(f"  Arrival rate: {self.workload_spec.arrival_rate:.1f} req/s")
        
        print(f"\nScheduler:")
        print(f"  Strategy: {self.scheduler_spec.batching_strategy}")
        print(f"  Max batch size: {self.scheduler_spec.max_batch_size or 'Dynamic'}")
        
        print(f"\nSimulation:")
        print(f"  Duration: {self.simulation_duration_s:.0f}s")
        
        print("="*70 + "\n")
