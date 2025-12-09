"""
Configuration classes for the LLM inference simulator.
"""

from dataclasses import dataclass
from typing import Optional
from .xpu_spec import DataType  # Import from xpu_spec now


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
class InterconnectSpec:
    """Interconnect specification for multi-GPU communication."""
    intra_node_type: str = "NVLink"
    intra_node_bandwidth_gbs: float = 600.0
    intra_node_latency_us: float = 2.0
    inter_node_type: str = "InfiniBand"
    inter_node_bandwidth_gbs: float = 200.0
    inter_node_latency_us: float = 10.0




@dataclass
class DisaggregationSpec:
    """
    Disaggregation configuration for separating prefill and decode clusters.
    
    Benefits:
    - Prefill cluster: Compute-optimized (fewer GPUs, high compute)
    - Decode cluster: Memory-optimized (more GPUs, high memory bandwidth)
    """
    enabled: bool = False
    
    # Prefill cluster (compute-heavy, short bursts)
    prefill_cluster: 'ClusterSpec' = None
    prefill_parallelism: 'ParallelismSpec' = None
    
    # Decode cluster (memory-heavy, continuous)
    decode_cluster: 'ClusterSpec' = None
    decode_parallelism: 'ParallelismSpec' = None
    
    # Inter-cluster network for KV cache transfer
    transfer_bandwidth_gbs: float = 100.0  # GB/s (e.g., 100 Gbps network = 12.5 GB/s)
    transfer_latency_ms: float = 1.0       # Base latency in milliseconds
    
    # Optional: KV cache compression
    kv_compression_ratio: float = 1.0  # 1.0 = no compression


@dataclass
class ClusterSpec:
    """Cluster configuration."""
    n_xpus_per_node: int  # Renamed from n_gpus_per_node
    n_nodes: int = 1
    xpu_spec: Optional['xPUSpec'] = None  # xPU instead of GPU
    interconnect_spec: Optional[InterconnectSpec] = None
    
    @property
    def total_xpus(self) -> int:
        """Total number of xPUs in cluster."""
        return self.n_xpus_per_node * self.n_nodes


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
    
    batch_size: int = 8  # Legacy, used for some examples


@dataclass
class SchedulerSpec:
    """Scheduler configuration."""
    batching_type: str = "continuous"
    max_batch_size: Optional[int] = None  # None = dynamic batching
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
    disaggregation_spec: Optional[DisaggregationSpec] = None  # For disaggregated prefill/decode
    
    
    # Warm-up period (seconds to run before starting measurement)
    warm_up_duration_s: float = 0.0  # 0 = no warm-up
    simulation_duration_s: float = 60.0
    max_requests: Optional[int] = None
    random_seed: int = 42
    output_event_log: bool = False
    
    def __post_init__(self):
        """Automatically validate configuration after creation."""
        self.validate()
    

    @property
    def is_disaggregated(self) -> bool:
        """Check if disaggregation is enabled."""
        return (self.disaggregation_spec is not None and 
                self.disaggregation_spec.enabled)
    
    def _validate_disaggregated(self):
        """Validate disaggregated configuration."""
        errors = []
        spec = self.disaggregation_spec
        
        # Validate prefill cluster
        prefill_mem_per_xpu = (
            self.model_spec.n_params * self.model_spec.weight_dtype.bytes_per_element() / 
            (1024**3) / spec.prefill_parallelism.tensor_parallel_size
        )
        if prefill_mem_per_xpu > spec.prefill_cluster.xpu_spec.memory_size_gb:
            errors.append(
                f"Prefill: Model weights ({prefill_mem_per_xpu:.2f}GB per xPU with TP={spec.prefill_parallelism.tensor_parallel_size}) "
                f"exceed xPU memory ({spec.prefill_cluster.xpu_spec.memory_size_gb}GB). "
                f"Increase tensor_parallel_size or use larger xPUs."
            )
        
        # Validate decode cluster
        decode_mem_per_xpu = (
            self.model_spec.n_params * self.model_spec.weight_dtype.bytes_per_element() /
            (1024**3) / spec.decode_parallelism.tensor_parallel_size
        )
        if decode_mem_per_xpu > spec.decode_cluster.xpu_spec.memory_size_gb:
            errors.append(
                f"Decode: Model weights ({decode_mem_per_xpu:.2f}GB per xPU with TP={spec.decode_parallelism.tensor_parallel_size}) "
                f"exceed xPU memory ({spec.decode_cluster.xpu_spec.memory_size_gb}GB). "
                f"Increase tensor_parallel_size or use larger xPUs."
            )
        
        # Check TP <= num xPUs
        if spec.prefill_parallelism.tensor_parallel_size > spec.prefill_cluster.total_xpus:
            errors.append(
                f"Prefill: tensor_parallel_size ({spec.prefill_parallelism.tensor_parallel_size}) "
                f"exceeds total xPUs ({spec.prefill_cluster.total_xpus})"
            )
        
        if spec.decode_parallelism.tensor_parallel_size > spec.decode_cluster.total_xpus:
            errors.append(
                f"Decode: tensor_parallel_size ({spec.decode_parallelism.tensor_parallel_size}) "
                f"exceeds total xPUs ({spec.decode_cluster.total_xpus})"
            )
        
        if errors:
            error_msg = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))
            raise ValueError(f"Disaggregated configuration validation failed:\n{error_msg}")
    
    def validate(self):
        """
        Validate configuration for common errors.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # For disaggregated mode, validate prefill and decode clusters separately
        if self.is_disaggregated:
            self._validate_disaggregated()
            return  # Skip regular validation
        
        errors = []
        warnings = []
        
        # 1. Basic validation
        if self.simulation_duration_s <= 0:
            errors.append("simulation_duration_s must be positive")
        
        # 2. Parallelism validation
        total_xpus = self.cluster_spec.total_xpus
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
        if total_parallel > total_xpus:
            errors.append(
                f"Total parallelism ({tp_size}×{pp_size}×{dp_size}={total_parallel}) "
                f"exceeds available xPUs ({total_xpus})"
            )
        

        # Check TP doesn't exceed single node
        if self.parallelism_spec.tensor_parallel_size > self.cluster_spec.n_xpus_per_node:
            warnings.append(
                f"TP size ({self.parallelism_spec.tensor_parallel_size}) exceeds xPUs per node "
                f"({self.cluster_spec.n_xpus_per_node}). TP across nodes is unrealistic due to "
                f"high inter-node communication latency. Consider using PP (Pipeline Parallelism) instead."
            )
        
        if total_parallel < total_xpus:
            warnings.append(
                f"Underutilization: Using {total_parallel} xPUs out of {total_xpus} available"
            )
        
        # 3. Memory validation (rough estimate)
        if self.cluster_spec.xpu_spec:
            xpu_memory = self.cluster_spec.xpu_spec.memory_size_gb
            model_params = self.model_spec.n_params
            bytes_per_param = self.model_spec.weight_dtype.bytes_per_element()
            
            # Model memory per xPU with TP
            model_memory_gb = (model_params * bytes_per_param) / (1024**3) / tp_size
            
            if model_memory_gb > xpu_memory:
                errors.append(
                    f"Model weights ({model_memory_gb:.2f}GB per xPU with TP={tp_size}) "
                    f"exceed xPU memory ({xpu_memory:.1f}GB). "
                    f"Increase tensor_parallel_size or use larger xPUs."
                )
            
            # Check if reasonable memory left for KV cache
            available_memory = xpu_memory - model_memory_gb - 2.0  # 2GB safety
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
        print(f"  xPUs: {self.cluster_spec.total_xpus} "
              f"({self.cluster_spec.n_nodes} nodes × {self.cluster_spec.n_xpus_per_node} xPUs)")
        if self.cluster_spec.xpu_spec:
            print(f"  xPU: {self.cluster_spec.xpu_spec.name} "
                  f"({self.cluster_spec.xpu_spec.memory_size_gb:.0f}GB)")
        
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
