"""
Model catalog with specifications for common LLM models.
"""

from .config import ModelSpec, DataType


class ModelCatalog:
    """Catalog of LLM model specifications."""
    
    # Model specifications
    _models = {
        # LLaMA family
        "llama-7b": ModelSpec(
            name="LLaMA-7B",
            n_params=7_000_000_000,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            ffn_dim=11008,
            vocab_size=32000,
            max_seq_length=2048,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            positional_encoding="RoPE",
        ),
        "llama-13b": ModelSpec(
            name="LLaMA-13B",
            n_params=13_000_000_000,
            n_layers=40,
            hidden_size=5120,
            n_heads=40,
            ffn_dim=13824,
            vocab_size=32000,
            max_seq_length=2048,
            positional_encoding="RoPE",
        ),
        "llama-30b": ModelSpec(
            name="LLaMA-30B",
            n_params=30_000_000_000,
            n_layers=60,
            hidden_size=6656,
            n_heads=52,
            ffn_dim=17920,
            vocab_size=32000,
            max_seq_length=2048,
            positional_encoding="RoPE",
        ),
        "llama-65b": ModelSpec(
            name="LLaMA-65B",
            n_params=65_000_000_000,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
            ffn_dim=22016,
            vocab_size=32000,
            max_seq_length=2048,
            positional_encoding="RoPE",
        ),
        
        # LLaMA 2 family
        "llama2-7b": ModelSpec(
            name="LLaMA-2-7B",
            n_params=7_000_000_000,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            ffn_dim=11008,
            vocab_size=32000,
            max_seq_length=4096,
            positional_encoding="RoPE",
        ),
        "llama2-13b": ModelSpec(
            name="LLaMA-2-13B",
            n_params=13_000_000_000,
            n_layers=40,
            hidden_size=5120,
            n_heads=40,
            ffn_dim=13824,
            vocab_size=32000,
            max_seq_length=4096,
            positional_encoding="RoPE",
        ),
        "llama2-70b": ModelSpec(
            name="LLaMA-2-70B",
            n_params=70_000_000_000,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
            ffn_dim=28672,
            vocab_size=32000,
            max_seq_length=4096,
            positional_encoding="RoPE",
        ),
        
        # LLaMA 3 family
        "llama3-8b": ModelSpec(
            name="LLaMA-3-8B",
            n_params=8_000_000_000,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            ffn_dim=14336,
            vocab_size=128256,
            max_seq_length=8192,
            positional_encoding="RoPE",
        ),
        "llama3-70b": ModelSpec(
            name="LLaMA-3-70B",
            n_params=70_000_000_000,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
            ffn_dim=28672,
            vocab_size=128256,
            max_seq_length=8192,
            positional_encoding="RoPE",
        ),
        
        # Mistral
        "mistral-7b": ModelSpec(
            name="Mistral-7B",
            n_params=7_000_000_000,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            ffn_dim=14336,
            vocab_size=32000,
            max_seq_length=32768,
            positional_encoding="RoPE",
        ),
        
        # GPT-3 style (for comparison)
        "gpt3-13b": ModelSpec(
            name="GPT-3-13B",
            n_params=13_000_000_000,
            n_layers=40,
            hidden_size=5140,
            n_heads=40,
            ffn_dim=20560,
            vocab_size=50257,
            max_seq_length=2048,
            positional_encoding="Absolute",
        ),
        "gpt3-175b": ModelSpec(
            name="GPT-3-175B",
            n_params=175_000_000_000,
            n_layers=96,
            hidden_size=12288,
            n_heads=96,
            ffn_dim=49152,
            vocab_size=50257,
            max_seq_length=2048,
            positional_encoding="Absolute",
        ),
        
        # Test models
        "tiny-1b": ModelSpec(
            name="Tiny-1B",
            n_params=1_000_000_000,
            n_layers=12,
            hidden_size=2048,
            n_heads=16,
            ffn_dim=8192,
            vocab_size=32000,
            max_seq_length=2048,
            positional_encoding="RoPE",
        ),
        "small-3b": ModelSpec(
            name="Small-3B",
            n_params=3_000_000_000,
            n_layers=24,
            hidden_size=3072,
            n_heads=24,
            ffn_dim=12288,
            vocab_size=32000,
            max_seq_length=2048,
            positional_encoding="RoPE",
        ),
    }
    
    @classmethod
    def get_model(cls, name: str) -> ModelSpec:
        """
        Get model specification by name.
        
        Args:
            name: Model name (case-insensitive)
        
        Returns:
            ModelSpec object
        
        Raises:
            ValueError: If model not found
        """
        name_lower = name.lower()
        
        if name_lower in cls._models:
            return cls._models[name_lower]
        
        available = list(cls._models.keys())
        raise ValueError(
            f"Model '{name}' not found in catalog. "
            f"Available models: {', '.join(sorted(available))}"
        )
    
    @classmethod
    def list_available(cls) -> list:
        """List all available model names."""
        return sorted(cls._models.keys())
    
    @classmethod
    def list_by_size(cls) -> None:
        """List models sorted by parameter count."""
        models = sorted(cls._models.items(), key=lambda x: x[1].n_params)
        
        print(f"\n{'Model':<20} {'Parameters':<15} {'Layers':<8} {'Hidden Size':<12}")
        print("-" * 60)
        
        for name, spec in models:
            params_b = spec.n_params / 1e9
            print(f"{spec.name:<20} {params_b:>7.1f}B       "
                  f"{spec.n_layers:>4}     {spec.hidden_size:>8}")
    
    @classmethod
    def compare(cls, model_names: list) -> None:
        """Print comparison table of models."""
        print(f"\n{'Model':<20} {'Parameters':<15} {'Context':<10} {'Hidden':<10}")
        print("-" * 60)
        
        for name in model_names:
            model = cls.get_model(name)
            params_b = model.n_params / 1e9
            print(f"{model.name:<20} {params_b:>7.1f}B       "
                  f"{model.max_seq_length:>6}     {model.hidden_size:>6}")
    
    @classmethod
    def print_all(cls) -> None:
        """Print all models in catalog."""
        cls.list_by_size()
    
    @classmethod
    def get_by_family(cls, family: str) -> list:
        """Get all models from a specific family."""
        family = family.lower()
        result = []
        
        for name, spec in cls._models.items():
            if family in name:
                result.append(spec)
        
        return result


# Convenience function
def get_model(name: str) -> ModelSpec:
    """
    Get model specification by name.
    
    Convenience wrapper around ModelCatalog.get_model().
    
    Args:
        name: Model name (case-insensitive)
    
    Returns:
        ModelSpec object
    """
    return ModelCatalog.get_model(name)
