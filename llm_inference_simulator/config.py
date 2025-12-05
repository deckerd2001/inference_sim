"""
Configuration classes for the LLM Inference Simulator.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class DataType(Enum):
    """Data types for weights and activations."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    
    def bytes_per_element(self) -> int:
        """Return number of bytes per element."""
        if self == DataType.FP32:
            return 4
        elif self in [DataType.FP16, DataType.BF16]:
            return 2
        elif self == DataType.INT8:
            return 1
        return 4


class PositionalEncoding(Enum):
    """Positional encoding types."""
    ROPE = "rope"
    ALIBI = "alibi"
    ABSOLUTE = "absolute"


@dataclass
class ModelSpec:
    """LLM model specifications."""
    name: str = "llama-7b"
    
    # Architecture parameters
    n_params: int = 7_000_000_000  # 7B parameters
    hidden_size: int = 4096  # H or D_model
    n_layers: int = 32  # L
    n_heads: int = 32  # Number of attention heads
    ffn_dim: int = 11008  # D_ff (typically 4 * hidden_size or similar)
    max_seq_length: int = 2048  # Maximum sequence length
    vocab_size: int = 32000
    
    # Data types
    weight_dtype: DataType = DataType.BF16
    activation_dtype: DataType = DataType.BF16
    
    # Positional encoding
    pos_encoding: PositionalEncoding = PositionalEncoding.ROPE
    
    # MoE settings (optional)
    is_moe: bool = False
    n_experts: Optional[int] = None
    expert_top_k: Optional[int] = None
    moe_layers: Optional[List[int]] = None  # Which layers are MoE
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.n_heads == 0, \
            "hidden_size must be divisible by n_heads"


@dataclass
class WorkloadSpec:
    """Workload and I/O specifications."""
    # Input/Output token distributions
    avg_input_length: int = 512
    max_input_length: int = 2048
    input_length_std: float = 200.0
    
    avg_output_length: int = 128
    max_output_length: int = 512
    output_length_std: float = 50.0
    
    # Arrival process
    arrival_rate: float = 5.0  # requests per second
    arrival_process: str = "poisson"  # "poisson", "deterministic", "trace"
    trace_file: Optional[str] = None
    
    # Batch size
    batch_size: int = 8


@dataclass
class GPUSpec:
    """Single GPU specifications."""
    name: str = "A100-80GB"
    
    # Compute capability
    compute_tflops: float = 312.0  # BF16 TFLOPS for A100
    
    # Memory
    memory_size_gb: float = 80.0  # GB
    memory_bandwidth_gbs: float = 2039.0  # GB/s (HBM bandwidth)
    
    # Cache (optional)
    l2_cache_size_mb: float = 40.0
    
    # Power (optional)
    tdp_watts: float = 400.0


@dataclass
class InterconnectSpec:
    """Interconnect specifications."""
    # Intra-node (between GPUs in same node)
    intra_node_type: str = "NVLink"
    intra_node_bandwidth_gbs: float = 600.0  # GB/s per link
    intra_node_latency_us: float = 2.0  # microseconds
    
    # Inter-node (between nodes)
    inter_node_type: str = "InfiniBand"
    inter_node_bandwidth_gbs: float = 200.0  # GB/s
    inter_node_latency_us: float = 5.0  # microseconds


@dataclass
class ClusterSpec:
    """GPU cluster specifications."""
    # Cluster topology
    n_gpus_per_node: int = 8
    n_nodes: int = 1
    
    # GPU and interconnect specs
    gpu_spec: GPUSpec = field(default_factory=GPUSpec)
    interconnect_spec: InterconnectSpec = field(default_factory=InterconnectSpec)
    
    @property
    def total_gpus(self) -> int:
        """Total number of GPUs in the cluster."""
        return self.n_gpus_per_node * self.n_nodes


@dataclass
class ParallelismSpec:
    """Parallelization strategy specifications."""
    # Parallel dimensions
    tensor_parallel_size: int = 1  # TP
    data_parallel_size: int = 1  # DP
    pipeline_parallel_size: int = 1  # PP
    expert_parallel_size: int = 1  # EP (for MoE)
    
    # Pipeline settings
    pipeline_schedule: str = "1f1b"  # "1f1b", "gpipe"
    microbatch_size: int = 1
    
    # Layer assignment for pipeline stages
    stage_layer_assignment: Optional[List[Tuple[int, int]]] = None  # [(start, end), ...]
    
    def __post_init__(self):
        """Validate parallelism configuration."""
        total_parallel = (self.tensor_parallel_size * 
                         self.data_parallel_size * 
                         self.pipeline_parallel_size * 
                         self.expert_parallel_size)
        # Will be validated against cluster size during initialization


@dataclass
class SchedulerSpec:
    """Scheduler and batching policy specifications."""
    # Batching policy
    batching_type: str = "continuous"  # "static", "dynamic", "continuous"
    max_batch_size: int = 32
    batching_window_ms: float = 10.0  # Maximum wait time for batching
    
    # Priority and scheduling
    priority_policy: str = "fifo"  # "fifo", "priority_queue", "sla_based"
    token_level_scheduling: bool = True  # Allow token-level interleaving in decode
    
    # Multi-tenancy (optional)
    multi_tenancy: bool = False


@dataclass
class SimulatorConfig:
    """Overall simulator configuration."""
    # Component specifications
    model_spec: ModelSpec = field(default_factory=ModelSpec)
    workload_spec: WorkloadSpec = field(default_factory=WorkloadSpec)
    cluster_spec: ClusterSpec = field(default_factory=ClusterSpec)
    parallelism_spec: ParallelismSpec = field(default_factory=ParallelismSpec)
    scheduler_spec: SchedulerSpec = field(default_factory=SchedulerSpec)
    
    # Simulation settings
    simulation_duration_s: float = 3600.0  # 1 hour
    max_requests: Optional[int] = None  # Alternative stopping condition
    random_seed: int = 42
    
    # Output settings
    collect_latency_distribution: bool = True
    latency_percentiles: List[float] = field(
        default_factory=lambda: [0.5, 0.9, 0.95, 0.99]
    )
    output_event_log: bool = False
    event_log_file: Optional[str] = None
    
    def validate(self):
        """Validate the entire configuration."""
        # Check parallelism fits cluster
        total_parallel = (
            self.parallelism_spec.tensor_parallel_size *
            self.parallelism_spec.data_parallel_size *
            self.parallelism_spec.pipeline_parallel_size *
            self.parallelism_spec.expert_parallel_size
        )
        
        if total_parallel > self.cluster_spec.total_gpus:
            raise ValueError(
                f"Parallelism configuration requires {total_parallel} GPUs "
                f"but cluster only has {self.cluster_spec.total_gpus} GPUs"
            )
        
        # Check model fits in GPU memory (basic check)
        model_size_gb = (
            self.model_spec.n_params * 
            self.model_spec.weight_dtype.bytes_per_element() / 
            (1024**3)
        )
        per_gpu_model_size = model_size_gb / self.parallelism_spec.tensor_parallel_size
        
        if per_gpu_model_size > self.cluster_spec.gpu_spec.memory_size_gb * 0.8:
            print(f"Warning: Model size ({per_gpu_model_size:.2f} GB per GPU) "
                  f"may exceed GPU memory ({self.cluster_spec.gpu_spec.memory_size_gb} GB)")


# ============================================================================
# Predefined GPU Specifications
# ============================================================================

class GPUCatalog:
    """Catalog of common GPU specifications."""
    
    @staticmethod
    def get_gpu(name: str) -> GPUSpec:
        """
        Get predefined GPU specification by name.
        
        Args:
            name: GPU model name (case-insensitive)
            
        Returns:
            GPUSpec instance
            
        Available GPUs:
            - A100-40GB, A100-80GB
            - H100-80GB
            - A10-24GB
            - B200-192GB
            - RTX4090
        """
        catalog = {
            "a100-40gb": GPUSpec(
                name="A100-40GB",
                compute_tflops=312.0,  # BF16
                memory_size_gb=40.0,
                memory_bandwidth_gbs=1555.0,
                tdp_watts=400.0,
            ),
            "a100-80gb": GPUSpec(
                name="A100-80GB",
                compute_tflops=312.0,  # BF16
                memory_size_gb=80.0,
                memory_bandwidth_gbs=2039.0,
                tdp_watts=400.0,
            ),
            "h100-80gb": GPUSpec(
                name="H100-80GB",
                compute_tflops=989.0,  # BF16
                memory_size_gb=80.0,
                memory_bandwidth_gbs=3350.0,
                l2_cache_size_mb=50.0,
                tdp_watts=700.0,
            ),
            "a10-24gb": GPUSpec(
                name="A10-24GB",
                compute_tflops=125.0,  # FP16 Tensor Core
                memory_size_gb=24.0,
                memory_bandwidth_gbs=600.0,
                l2_cache_size_mb=6.0,
                tdp_watts=150.0,
            ),
            "b200-192gb": GPUSpec(
                name="B200-192GB",
                compute_tflops=2000.0,  # BF16 (Blackwell generation)
                memory_size_gb=192.0,
                memory_bandwidth_gbs=8000.0,
                l2_cache_size_mb=96.0,
                tdp_watts=1000.0,
            ),
            "rtx4090": GPUSpec(
                name="RTX-4090",
                compute_tflops=165.0,  # FP16
                memory_size_gb=24.0,
                memory_bandwidth_gbs=1008.0,
                l2_cache_size_mb=72.0,
                tdp_watts=450.0,
            ),
        }
        
        key = name.lower().replace(" ", "").replace("_", "")
        if key not in catalog:
            available = ", ".join(sorted(set(gpu.name for gpu in catalog.values())))
            raise ValueError(
                f"GPU '{name}' not found in catalog. "
                f"Available GPUs: {available}"
            )
        
        return catalog[key]
    
    @staticmethod
    def list_available() -> List[str]:
        """List all available GPU models."""
        return [
            "A100-40GB",
            "A100-80GB", 
            "H100-80GB",
            "A10-24GB",
            "B200-192GB",
            "RTX-4090",
        ]
    
    @staticmethod
    def compare(gpu_names: List[str]) -> None:
        """Print comparison table of GPUs."""
        print("\n" + "="*90)
        print("GPU Comparison")
        print("="*90)
        print(f"{'GPU':<20} {'TFLOPS':>10} {'Memory':>10} {'Bandwidth':>12} {'TDP':>8}")
        print("-"*90)
        
        for name in gpu_names:
            try:
                gpu = GPUCatalog.get_gpu(name)
                print(f"{gpu.name:<20} {gpu.compute_tflops:>10.1f} "
                      f"{gpu.memory_size_gb:>9.0f}GB {gpu.memory_bandwidth_gbs:>10.0f} GB/s "
                      f"{gpu.tdp_watts:>7.0f}W")
            except ValueError as e:
                print(f"{name:<20} ERROR: {e}")
        
        print("="*90 + "\n")


# Import communication strategies
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .communication import TPCommunicationStrategy


@dataclass
class ParallelismSpecV2:
    """
    Enhanced parallelization strategy with communication control.
    """
    # Parallel dimensions
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    
    # Pipeline settings
    pipeline_schedule: str = "1f1b"
    microbatch_size: int = 1
    
    # Communication strategy (will be set at runtime)
    tp_communication_strategy: Optional['TPCommunicationStrategy'] = None
    
    # Use sequence parallelism
    use_sequence_parallel: bool = False
    
    # Enable communication-computation overlap
    enable_comm_overlap: bool = False
    
    # Layer assignment for pipeline stages
    stage_layer_assignment: Optional[List[Tuple[int, int]]] = None
