"""
xPU catalog with specifications for various accelerators.
"""

from typing import List
from .xpu_spec import xPUSpec, ComputeUnit, OperationType, DataType


# ========== NVIDIA GPUs ==========

A100_80GB = xPUSpec(
    name="A100-80GB",
    device_type="GPU",
    vendor="NVIDIA",
    memory_size_gb=80.0,
    memory_bandwidth_gbs=2039.0,
    l2_cache_size_mb=40.0,
    price_per_hour=3.67,  # AWS p4d.24xlarge / 8 GPUs
    compute_units={
        "tensor_core": ComputeUnit(
            name="Tensor Core (Ampere)",
            peak_tflops=312.0,
            supported_ops=[OperationType.MATMUL],
            supported_dtypes=[DataType.FP16, DataType.BF16],
            utilization_efficiency=0.85
        ),
        "cuda_core": ComputeUnit(
            name="CUDA Core",
            peak_tflops=19.5,
            supported_ops=[OperationType.MATMUL, OperationType.ELEMENTWISE,
                          OperationType.REDUCTION, OperationType.TRANSPOSE],
            supported_dtypes=[DataType.FP32, DataType.FP16],
            utilization_efficiency=0.7
        )
    },
    default_matmul_unit="tensor_core",
    default_elementwise_unit="cuda_core"
)

H100_80GB = xPUSpec(
    name="H100-80GB",
    device_type="GPU",
    vendor="NVIDIA",
    memory_size_gb=80.0,
    memory_bandwidth_gbs=3350.0,
    l2_cache_size_mb=50.0,
    price_per_hour=6.49,  # AWS p5.48xlarge / 8 GPUs
    compute_units={
        "tensor_core": ComputeUnit(
            name="Tensor Core (Hopper)",
            peak_tflops=989.0,
            supported_ops=[OperationType.MATMUL],
            supported_dtypes=[DataType.FP16, DataType.BF16, DataType.FP8],
            utilization_efficiency=0.85
        ),
        "cuda_core": ComputeUnit(
            name="CUDA Core",
            peak_tflops=60.0,
            supported_ops=[OperationType.MATMUL, OperationType.ELEMENTWISE,
                          OperationType.REDUCTION, OperationType.TRANSPOSE],
            supported_dtypes=[DataType.FP32, DataType.FP16],
            utilization_efficiency=0.7
        )
    },
    default_matmul_unit="tensor_core",
    default_elementwise_unit="cuda_core"
)

B200_192GB = xPUSpec(
    name="B200-192GB",
    device_type="GPU",
    vendor="NVIDIA",
    memory_size_gb=192.0,
    memory_bandwidth_gbs=8000.0,
    l2_cache_size_mb=60.0,
    price_per_hour=10.0,  # Estimated
    compute_units={
        "tensor_core": ComputeUnit(
            name="Tensor Core (Blackwell)",
            peak_tflops=2000.0,
            supported_ops=[OperationType.MATMUL],
            supported_dtypes=[DataType.FP16, DataType.BF16, DataType.FP8],
            utilization_efficiency=0.85
        ),
        "cuda_core": ComputeUnit(
            name="CUDA Core",
            peak_tflops=125.0,
            supported_ops=[OperationType.MATMUL, OperationType.ELEMENTWISE,
                          OperationType.REDUCTION, OperationType.TRANSPOSE],
            supported_dtypes=[DataType.FP32],
            utilization_efficiency=0.7
        )
    },
    default_matmul_unit="tensor_core",
    default_elementwise_unit="cuda_core"
)

GTX_1080Ti = xPUSpec(
    name="GTX-1080Ti",
    device_type="GPU",
    vendor="NVIDIA",
    memory_size_gb=11.0,
    memory_bandwidth_gbs=484.0,
    price_per_hour=0.5,  # Estimated (not available in cloud)
    compute_units={
        "cuda_core": ComputeUnit(
            name="CUDA Core (Pascal)",
            peak_tflops=11.3,
            supported_ops=[OperationType.MATMUL, OperationType.ELEMENTWISE,
                          OperationType.REDUCTION, OperationType.TRANSPOSE],
            supported_dtypes=[DataType.FP32],
            utilization_efficiency=0.65
        )
    },
    default_matmul_unit="cuda_core",
    default_elementwise_unit="cuda_core"
)

# ========== AMD GPUs ==========

MI300X = xPUSpec(
    name="MI300X",
    device_type="GPU",
    vendor="AMD",
    memory_size_gb=192.0,
    memory_bandwidth_gbs=5300.0,
    price_per_hour=7.0,  # Estimated (similar to H100)
    compute_units={
        "matrix_core": ComputeUnit(
            name="Matrix Core (CDNA3)",
            peak_tflops=1300.0,
            supported_ops=[OperationType.MATMUL],
            supported_dtypes=[DataType.FP16, DataType.BF16, DataType.FP8],
            utilization_efficiency=0.8
        ),
        "simd_unit": ComputeUnit(
            name="SIMD",
            peak_tflops=82.0,
            supported_ops=[OperationType.MATMUL, OperationType.ELEMENTWISE,
                          OperationType.REDUCTION, OperationType.TRANSPOSE],
            supported_dtypes=[DataType.FP32, DataType.FP16],
            utilization_efficiency=0.7
        )
    },
    default_matmul_unit="matrix_core",
    default_elementwise_unit="simd_unit"
)

# ========== Google TPUs ==========

TPU_V4 = xPUSpec(
    name="TPU-v4",
    device_type="TPU",
    vendor="Google",
    memory_size_gb=32.0,
    memory_bandwidth_gbs=1200.0,
    price_per_hour=3.67,  # GCP TPU v4 pricing
    compute_units={
        "systolic_array": ComputeUnit(
            name="Systolic Array",
            peak_tflops=275.0,
            supported_ops=[OperationType.MATMUL],
            supported_dtypes=[DataType.BF16, DataType.FP32],
            utilization_efficiency=0.9
        ),
        "vector_unit": ComputeUnit(
            name="Vector Unit",
            peak_tflops=20.0,
            supported_ops=[OperationType.ELEMENTWISE, OperationType.REDUCTION],
            supported_dtypes=[DataType.BF16, DataType.FP32],
            utilization_efficiency=0.8
        )
    },
    default_matmul_unit="systolic_array",
    default_elementwise_unit="vector_unit"
)

# ========== Catalog ==========

_CATALOG = {
    "a100-80gb": A100_80GB,
    "h100-80gb": H100_80GB,
    "b200-192gb": B200_192GB,
    "gtx-1080ti": GTX_1080Ti,
    "mi300x": MI300X,
    "tpu-v4": TPU_V4,
}

_ALIASES = {
    "a100": "a100-80gb",
    "h100": "h100-80gb",
    "b200": "b200-192gb",
    "1080ti": "gtx-1080ti",
}


def get_xpu(name: str) -> xPUSpec:
    """Get xPU specification by name."""
    name_lower = name.lower()
    
    if name_lower in _ALIASES:
        name_lower = _ALIASES[name_lower]
    
    if name_lower in _CATALOG:
        return _CATALOG[name_lower]
    
    available = sorted(list(_CATALOG.keys()) + list(_ALIASES.keys()))
    raise ValueError(
        f"xPU '{name}' not found.\nAvailable: {', '.join(available)}"
    )


def list_xpus() -> List[str]:
    """List all available xPUs."""
    return sorted(_CATALOG.keys())
