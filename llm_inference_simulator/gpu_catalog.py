"""
GPU Catalog - Predefined GPU specifications for common models.
"""

from typing import List, Dict
from .config import GPUSpec


class GPUCatalog:
    """
    Catalog of common GPU specifications.
    
    Provides easy access to predefined GPU specs for popular models
    including NVIDIA A10, A100, H100, B200, and consumer GPUs.
    """
    
    # GPU specifications database
    _CATALOG: Dict[str, GPUSpec] = {
        "a100-40gb": GPUSpec(
            name="A100-40GB",
            compute_tflops=312.0,  # BF16 Tensor Core
            memory_size_gb=40.0,
            memory_bandwidth_gbs=1555.0,
            l2_cache_size_mb=40.0,
            tdp_watts=400.0,
        ),
        "a100-80gb": GPUSpec(
            name="A100-80GB",
            compute_tflops=312.0,  # BF16 Tensor Core
            memory_size_gb=80.0,
            memory_bandwidth_gbs=2039.0,
            l2_cache_size_mb=40.0,
            tdp_watts=400.0,
        ),
        "a10-24gb": GPUSpec(
            name="A10-24GB",
            compute_tflops=125.0,  # FP16 Tensor Core
            memory_size_gb=24.0,
            memory_bandwidth_gbs=600.0,
            l2_cache_size_mb=6.0,
            tdp_watts=150.0,
        ),
        "h100-80gb": GPUSpec(
            name="H100-80GB",
            compute_tflops=989.0,  # BF16 Tensor Core (Hopper)
            memory_size_gb=80.0,
            memory_bandwidth_gbs=3350.0,  # HBM3
            l2_cache_size_mb=50.0,
            tdp_watts=700.0,
        ),
        "h100-pcie-80gb": GPUSpec(
            name="H100-PCIe-80GB",
            compute_tflops=756.0,  # BF16 (PCIe version, lower power)
            memory_size_gb=80.0,
            memory_bandwidth_gbs=2000.0,
            l2_cache_size_mb=50.0,
            tdp_watts=350.0,
        ),
        "b200-192gb": GPUSpec(
            name="B200-192GB",
            compute_tflops=2000.0,  # BF16 Tensor Core (Blackwell, estimated)
            memory_size_gb=192.0,
            memory_bandwidth_gbs=8000.0,  # HBM3e
            l2_cache_size_mb=96.0,
            tdp_watts=1000.0,
        ),
        "l40s": GPUSpec(
            name="L40S",
            compute_tflops=183.0,  # FP16 Tensor Core
            memory_size_gb=48.0,
            memory_bandwidth_gbs=864.0,
            l2_cache_size_mb=96.0,
            tdp_watts=350.0,
        ),
        "rtx4090": GPUSpec(
            name="RTX-4090",
            compute_tflops=165.0,  # FP16 Tensor Core
            memory_size_gb=24.0,
            memory_bandwidth_gbs=1008.0,
            l2_cache_size_mb=72.0,
            tdp_watts=450.0,
        ),
        "rtx4080": GPUSpec(
            name="RTX-4080",
            compute_tflops=97.0,  # FP16 Tensor Core
            memory_size_gb=16.0,
            memory_bandwidth_gbs=716.0,
            l2_cache_size_mb=64.0,
            tdp_watts=320.0,
        ),
        "v100-32gb": GPUSpec(
            name="V100-32GB",
            compute_tflops=125.0,  # FP16 Tensor Core
            memory_size_gb=32.0,
            memory_bandwidth_gbs=900.0,
            l2_cache_size_mb=6.0,
            tdp_watts=300.0,
        ),
    }
    
    # Aliases for common short names
    _ALIASES = {
        "a10": "a10-24gb",
        "h100": "h100-80gb",
        "h100pcie": "h100-pcie-80gb",
        "b200": "b200-192gb",
        "l40s": "l40s",
        "rtx4090": "rtx4090",
        "rtx4080": "rtx4080",
        "v100": "v100-32gb",
    }
    
    @classmethod
    def get_gpu(cls, name: str) -> GPUSpec:
        """
        Get predefined GPU specification by name.
        
        Args:
            name: GPU model name (case-insensitive, spaces/dashes flexible)
            
        Returns:
            GPUSpec instance
            
        Raises:
            ValueError: If GPU name not found in catalog
            
        Examples:
            >>> gpu = GPUCatalog.get_gpu("A100-80GB")
            >>> gpu = GPUCatalog.get_gpu("a10")  # Uses alias
            >>> gpu = GPUCatalog.get_gpu("H100")  # Uses alias
        """
        # Normalize the input name
        normalized = name.lower().replace(" ", "").replace("_", "").replace("-", "")
        
        # 1. Try exact match in catalog
        if normalized in cls._CATALOG:
            return cls._CATALOG[normalized]
        
        # 2. Try alias lookup
        if normalized in cls._ALIASES:
            catalog_key = cls._ALIASES[normalized]
            return cls._CATALOG[catalog_key]
        
        # 3. Try matching with the display name
        for catalog_key, gpu_spec in cls._CATALOG.items():
            display_normalized = gpu_spec.name.lower().replace(" ", "").replace("_", "").replace("-", "")
            if normalized == display_normalized:
                return gpu_spec
        
        # 4. Try partial match (must be unambiguous)
        matches = []
        for catalog_key, gpu_spec in cls._CATALOG.items():
            catalog_normalized = catalog_key.replace("-", "")
            display_normalized = gpu_spec.name.lower().replace(" ", "").replace("_", "").replace("-", "")
            
            # Check if normalized is in either the key or display name
            if normalized in catalog_normalized or normalized in display_normalized:
                matches.append((catalog_key, gpu_spec))
        
        if len(matches) == 1:
            return matches[0][1]
        elif len(matches) > 1:
            available = ", ".join([gpu.name for _, gpu in matches])
            raise ValueError(
                f"Ambiguous GPU name '{name}'. Multiple matches found: {available}. "
                f"Please be more specific (e.g., 'A100-80GB' instead of 'A100')."
            )
        
        # No match found
        available = ", ".join(sorted(set(gpu.name for gpu in cls._CATALOG.values())))
        raise ValueError(
            f"GPU '{name}' not found in catalog.\n"
            f"Available GPUs: {available}\n"
            f"Use GPUCatalog.list_available() to see all options."
        )
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available GPU models in the catalog.
        
        Returns:
            List of GPU model names
        """
        return sorted(set(gpu.name for gpu in cls._CATALOG.values()))
    
    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """
        List all short name aliases.
        
        Returns:
            Dictionary mapping aliases to full GPU names
        """
        return {
            alias: cls._CATALOG[full_key].name 
            for alias, full_key in cls._ALIASES.items()
        }
    
    @classmethod
    def compare(cls, gpu_names: List[str], show_all: bool = False) -> None:
        """
        Print comparison table of GPUs.
        
        Args:
            gpu_names: List of GPU names to compare
            show_all: If True, show all specs; if False, show key specs only
        """
        print("\n" + "="*100)
        print("GPU Comparison")
        print("="*100)
        
        if show_all:
            print(f"{'GPU':<20} {'TFLOPS':>10} {'Memory':>10} {'Bandwidth':>12} "
                  f"{'L2 Cache':>10} {'TDP':>8}")
            print("-"*100)
        else:
            print(f"{'GPU':<20} {'TFLOPS':>10} {'Memory':>10} {'Bandwidth':>12} {'TDP':>8}")
            print("-"*100)
        
        for name in gpu_names:
            try:
                gpu = cls.get_gpu(name)
                if show_all:
                    print(f"{gpu.name:<20} {gpu.compute_tflops:>10.1f} "
                          f"{gpu.memory_size_gb:>9.0f}GB {gpu.memory_bandwidth_gbs:>10.0f} GB/s "
                          f"{gpu.l2_cache_size_mb:>9.0f}MB {gpu.tdp_watts:>7.0f}W")
                else:
                    print(f"{gpu.name:<20} {gpu.compute_tflops:>10.1f} "
                          f"{gpu.memory_size_gb:>9.0f}GB {gpu.memory_bandwidth_gbs:>10.0f} GB/s "
                          f"{gpu.tdp_watts:>7.0f}W")
            except ValueError as e:
                print(f"{name:<20} ERROR: Not found in catalog")
        
        print("="*100 + "\n")
    
    @classmethod
    def print_all(cls) -> None:
        """Print all GPUs in the catalog."""
        all_gpus = cls.list_available()
        print(f"\nAvailable GPUs in Catalog ({len(all_gpus)} total):\n")
        cls.compare(all_gpus, show_all=True)
    
    @classmethod
    def print_aliases(cls) -> None:
        """Print all short name aliases."""
        print("\n" + "="*60)
        print("GPU Short Name Aliases")
        print("="*60)
        print(f"{'Short Name':<15} → {'Full GPU Name':<30}")
        print("-"*60)
        
        for alias, full_name in sorted(cls.list_aliases().items()):
            print(f"{alias:<15} → {full_name:<30}")
        
        print("="*60 + "\n")
    
    @classmethod
    def get_by_generation(cls, generation: str) -> List[GPUSpec]:
        """
        Get GPUs by architecture generation.
        
        Args:
            generation: "ampere", "hopper", "blackwell", "ada", "volta"
            
        Returns:
            List of GPUSpec for that generation
        """
        generation_map = {
            "ampere": ["a100-40gb", "a100-80gb", "a10-24gb"],
            "hopper": ["h100-80gb", "h100-pcie-80gb"],
            "blackwell": ["b200-192gb"],
            "ada": ["rtx4090", "rtx4080", "l40s"],
            "volta": ["v100-32gb"],
        }
        
        gen_lower = generation.lower()
        if gen_lower not in generation_map:
            raise ValueError(
                f"Generation '{generation}' not recognized. "
                f"Available: {', '.join(generation_map.keys())}"
            )
        
        return [cls._CATALOG[key] for key in generation_map[gen_lower]]


# Convenience function for quick access
def get_gpu(name: str) -> GPUSpec:
    """
    Quick access function to get a GPU spec.
    
    Args:
        name: GPU model name
        
    Returns:
        GPUSpec instance
        
    Example:
        >>> from llm_inference_simulator import get_gpu
        >>> gpu = get_gpu("A10")  # Returns A10-24GB
        >>> gpu = get_gpu("H100")  # Returns H100-80GB
        >>> gpu = get_gpu("A100-80GB")  # Full name also works
    """
    return GPUCatalog.get_gpu(name)
