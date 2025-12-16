"""
xPU (Generic Accelerator) Specification.

Supports GPU, TPU, NPU, and other accelerators with diverse compute units.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class DataType(Enum):
    """Data types for computation."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"

    def bytes_per_element(self) -> int:
        """Return bytes per element."""
        if self == DataType.FP32:
            return 4
        elif self == DataType.FP16:
            return 2
        elif self == DataType.BF16:
            return 2
        elif self == DataType.FP8:
            return 1
        elif self == DataType.INT8:
            return 1
        else:
            raise ValueError(f"Unknown data type: {self}")


class OperationType(Enum):
    """Types of operations."""
    MATMUL = "matmul"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    TRANSPOSE = "transpose"


@dataclass
class ComputeUnit:
    """
    Individual compute unit specification.

    Examples:
    - NVIDIA Tensor Core
    - AMD Matrix Core
    - CUDA/SIMD cores
    - TPU Systolic Array
    """
    name: str
    peak_tflops: float  # Peak TFLOPS for primary dtype
    supported_ops: List[OperationType]
    supported_dtypes: List[DataType]

    # Realistic utilization (hardware rarely reaches peak)
    utilization_efficiency: float = 0.8

    def supports_op(self, op_type: OperationType, dtype: DataType) -> bool:
        """Check if this unit supports the operation and dtype."""
        return op_type in self.supported_ops and dtype in self.supported_dtypes

    def get_effective_tflops(self) -> float:
        """Get effective TFLOPS accounting for utilization."""
        return self.peak_tflops * self.utilization_efficiency


@dataclass
class xPUSpec:
    """
    Generic accelerator specification.

    Supports GPUs (NVIDIA, AMD, Intel), TPUs, NPUs, and future accelerators.
    """
    name: str
    device_type: str  # "GPU", "TPU", "NPU", "CPU"
    vendor: str  # "NVIDIA", "AMD", "Intel", "Google", "Apple"

    # Memory hierarchy
    memory_size_gb: float
    memory_bandwidth_gbs: float
    l2_cache_size_mb: Optional[float] = None



    # Interconnect bandwidth (for inter-GPU communication)
    intra_node_bandwidth_gbs: float = 600.0  # NVLink/Infinity Fabric (intra-node)
    inter_node_bandwidth_gbs: float = 50.0   # InfiniBand/RoCE (inter-node)

    # Pricing (cloud on-demand pricing, $/hour)
    price_per_hour: float = 0.0  # $0 = pricing not available

    # Compute units (e.g., {"tensor_core": ..., "cuda_core": ...})
    compute_units: Dict[str, ComputeUnit] = field(default_factory=dict)

    # Default units for operations
    default_matmul_unit: str = "tensor_core"
    default_elementwise_unit: str = "cuda_core"

    def get_compute_unit(self, op_type: OperationType,
                         dtype: DataType) -> Optional[ComputeUnit]:
        """
        Get appropriate compute unit for operation.

        Args:
            op_type: Type of operation
            dtype: Data type

        Returns:
            Best matching ComputeUnit, or None if not supported
        """
        # Try specialized unit first
        if op_type == OperationType.MATMUL:
            preferred_unit = self.default_matmul_unit
        else:
            preferred_unit = self.default_elementwise_unit

        if preferred_unit in self.compute_units:
            unit = self.compute_units[preferred_unit]
            if unit.supports_op(op_type, dtype):
                return unit

        # Fallback: find any unit that supports this op+dtype
        for unit in self.compute_units.values():
            if unit.supports_op(op_type, dtype):
                return unit

        return None

    def get_matmul_tflops(self, dtype: DataType) -> float:
        """Get effective TFLOPS for matrix multiplication."""
        unit = self.get_compute_unit(OperationType.MATMUL, dtype)
        if unit:
            return unit.get_effective_tflops()
        return 0.0

    def print_summary(self):
        """Print summary of xPU capabilities."""
        print(f"\n{'='*70}")
        print(f"{self.name} ({self.vendor} {self.device_type})")
        print(f"{'='*70}")
        print(f"Memory: {self.memory_size_gb:.0f}GB @ {self.memory_bandwidth_gbs:.0f}GB/s")
        if self.l2_cache_size_mb:
            print(f"L2 Cache: {self.l2_cache_size_mb:.0f}MB")
        print(f"\nCompute Units:")
        for name, unit in self.compute_units.items():
            print(f"  • {unit.name}: {unit.peak_tflops:.0f} TFLOPS (peak)")
            print(f"    - Effective: {unit.get_effective_tflops():.0f} TFLOPS")
            print(f"    - Operations: {[op.value for op in unit.supported_ops]}")
            print(f"    - Data types: {[dt.value for dt in unit.supported_dtypes]}")
        print(f"{'='*70}\n")
