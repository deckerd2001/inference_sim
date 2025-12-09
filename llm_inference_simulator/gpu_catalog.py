"""
GPU catalog with specifications for common GPUs.
"""

from .config import GPUSpec


class GPUCatalog:
    """Catalog of GPU specifications."""

    # GPU specifications (name -> GPUSpec)
    _gpus = {
        # NVIDIA A100
        "a100-40gb": GPUSpec(
            name="A100-40GB",
            memory_size_gb=40.0,
            compute_tflops=312.0,  # BF16
            memory_bandwidth_gbs=1555.0,
        ),
        "a100-80gb": GPUSpec(
            name="A100-80GB",
            memory_size_gb=80.0,
            compute_tflops=312.0,
            memory_bandwidth_gbs=2039.0,
        ),

        # NVIDIA H100
        "h100-80gb": GPUSpec(
            name="H100-80GB",
            memory_size_gb=80.0,
            compute_tflops=989.0,  # BF16 with sparsity
            memory_bandwidth_gbs=3350.0,
        ),
        "h100-pcie": GPUSpec(
            name="H100-PCIe",
            memory_size_gb=80.0,
            compute_tflops=756.0,
            memory_bandwidth_gbs=2000.0,
        ),

        # NVIDIA A10
        "a10-24gb": GPUSpec(
            name="A10-24GB",
            memory_size_gb=24.0,
            compute_tflops=125.0,  # FP16
            memory_bandwidth_gbs=600.0,
        ),

        # NVIDIA B200 (Blackwell)
        "b200-192gb": GPUSpec(
            name="B200-192GB",
            memory_size_gb=192.0,
            compute_tflops=2000.0,  # Estimated FP16
            memory_bandwidth_gbs=8000.0,
        ),

        # NVIDIA L40S
        "l40s": GPUSpec(
            name="L40S",
            memory_size_gb=48.0,
            compute_tflops=183.0,  # BF16
            memory_bandwidth_gbs=864.0,
        ),

        # NVIDIA RTX 4090
        "rtx-4090": GPUSpec(
            name="RTX-4090",
            memory_size_gb=24.0,
            compute_tflops=165.0,  # FP16
            memory_bandwidth_gbs=1008.0,
        ),

        # NVIDIA RTX 4080
        "rtx-4080": GPUSpec(
            name="RTX-4080",
            memory_size_gb=16.0,
            compute_tflops=97.0,  # FP16
            memory_bandwidth_gbs=716.0,
        ),

        # NVIDIA V100
        "v100-32gb": GPUSpec(
            name="V100-32GB",
            memory_size_gb=32.0,
            compute_tflops=125.0,  # FP16
            memory_bandwidth_gbs=900.0,
        ),
    }

    # Aliases for convenience
    _aliases = {
        "a100": "a100-80gb",
        "h100": "h100-80gb",
        "a10": "a10-24gb",
        "b200": "b200-192gb",
        "v100": "v100-32gb",
        "4090": "rtx-4090",
        "4080": "rtx-4080",
    }

    @classmethod
    def get_gpu(cls, name: str) -> GPUSpec:
        """
        Get GPU specification by name.

        Args:
            name: GPU name (case-insensitive). Can be full name or alias.
                  Examples: "A100-80GB", "a100", "H100", "h100-80gb"

        Returns:
            GPUSpec object

        Raises:
            ValueError: If GPU not found
        """
        name_lower = name.lower()

        # Check aliases first
        if name_lower in cls._aliases:
            name_lower = cls._aliases[name_lower]

        # Get GPU spec
        if name_lower in cls._gpus:
            return cls._gpus[name_lower]

        # Not found
        available = list(cls._gpus.keys()) + list(cls._aliases.keys())
        raise ValueError(
            f"GPU '{name}' not found in catalog. "
            f"Available GPUs: {', '.join(sorted(available))}"
        )

    @classmethod
    def list_available(cls) -> list:
        """List all available GPU names."""
        return sorted(cls._gpus.keys())

    @classmethod
    def compare(cls, gpu_names: list) -> None:
        """Print comparison table of GPUs."""
        print(f"\n{'GPU':<20} {'Memory':<12} {'Compute':<15} {'Bandwidth':<15}")
        print("-" * 65)

        for name in gpu_names:
            gpu = cls.get_gpu(name)
            print(f"{gpu.name:<20} {gpu.memory_size_gb:>6.0f} GB    "
                  f"{gpu.compute_tflops:>8.0f} TFLOPS  "
                  f"{gpu.memory_bandwidth_gbs:>8.0f} GB/s")

    @classmethod
    def print_all(cls) -> None:
        """Print all GPUs in catalog."""
        cls.compare(cls.list_available())

    @classmethod
    def get_by_generation(cls, generation: str) -> list:
        """Get all GPUs from a specific generation."""
        generation = generation.lower()
        result = []

        for name, spec in cls._gpus.items():
            if generation in name:
                result.append(spec)

        return result


# Convenience function
def get_gpu(name: str) -> GPUSpec:
    """
    Get GPU specification by name.

    Convenience wrapper around GPUCatalog.get_gpu().

    Args:
        name: GPU name (case-insensitive)

    Returns:
        GPUSpec object
    """
    return GPUCatalog.get_gpu(name)
