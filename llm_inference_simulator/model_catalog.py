"""
Model Catalog - Predefined model specifications for popular LLMs.
"""

from typing import List, Dict
from .config import ModelSpec, DataType, PositionalEncoding


class ModelCatalog:
    """
    Catalog of popular LLM model specifications.
    
    Includes models from LLaMA, GPT, Mistral, and other families.
    """
    
    # Model specifications database
    _CATALOG: Dict[str, ModelSpec] = {
        # LLaMA Family
        "llama-7b": ModelSpec(
            name="LLaMA-7B",
            n_params=7_000_000_000,
            hidden_size=4096,
            n_layers=32,
            n_heads=32,
            ffn_dim=11008,
            max_seq_length=2048,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama-13b": ModelSpec(
            name="LLaMA-13B",
            n_params=13_000_000_000,
            hidden_size=5120,
            n_layers=40,
            n_heads=40,
            ffn_dim=13824,
            max_seq_length=2048,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama-30b": ModelSpec(
            name="LLaMA-30B",
            n_params=30_000_000_000,
            hidden_size=6656,
            n_layers=60,
            n_heads=52,
            ffn_dim=17920,
            max_seq_length=2048,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama-65b": ModelSpec(
            name="LLaMA-65B",
            n_params=65_000_000_000,
            hidden_size=8192,
            n_layers=80,
            n_heads=64,
            ffn_dim=22016,
            max_seq_length=2048,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama2-7b": ModelSpec(
            name="LLaMA-2-7B",
            n_params=7_000_000_000,
            hidden_size=4096,
            n_layers=32,
            n_heads=32,
            ffn_dim=11008,
            max_seq_length=4096,  # Increased context
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama2-13b": ModelSpec(
            name="LLaMA-2-13B",
            n_params=13_000_000_000,
            hidden_size=5120,
            n_layers=40,
            n_heads=40,
            ffn_dim=13824,
            max_seq_length=4096,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama2-70b": ModelSpec(
            name="LLaMA-2-70B",
            n_params=70_000_000_000,
            hidden_size=8192,
            n_layers=80,
            n_heads=64,
            ffn_dim=28672,
            max_seq_length=4096,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama3-8b": ModelSpec(
            name="LLaMA-3-8B",
            n_params=8_000_000_000,
            hidden_size=4096,
            n_layers=32,
            n_heads=32,
            ffn_dim=14336,
            max_seq_length=8192,  # Extended context
            vocab_size=128256,  # Larger vocab
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "llama3-70b": ModelSpec(
            name="LLaMA-3-70B",
            n_params=70_000_000_000,
            hidden_size=8192,
            n_layers=80,
            n_heads=64,
            ffn_dim=28672,
            max_seq_length=8192,
            vocab_size=128256,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        
        # Mistral Family
        "mistral-7b": ModelSpec(
            name="Mistral-7B",
            n_params=7_000_000_000,
            hidden_size=4096,
            n_layers=32,
            n_heads=32,
            ffn_dim=14336,
            max_seq_length=32768,  # Very long context!
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        
        # GPT Family
        "gpt3-175b": ModelSpec(
            name="GPT-3-175B",
            n_params=175_000_000_000,
            hidden_size=12288,
            n_layers=96,
            n_heads=96,
            ffn_dim=49152,
            max_seq_length=2048,
            vocab_size=50257,
            weight_dtype=DataType.FP16,
            activation_dtype=DataType.FP16,
            pos_encoding=PositionalEncoding.ABSOLUTE,
        ),
        "gpt3-13b": ModelSpec(
            name="GPT-3-13B",
            n_params=13_000_000_000,
            hidden_size=5120,
            n_layers=40,
            n_heads=40,
            ffn_dim=20480,
            max_seq_length=2048,
            vocab_size=50257,
            weight_dtype=DataType.FP16,
            activation_dtype=DataType.FP16,
            pos_encoding=PositionalEncoding.ABSOLUTE,
        ),
        
        # Smaller models for testing
        "tiny-1b": ModelSpec(
            name="Tiny-1B",
            n_params=1_000_000_000,
            hidden_size=2048,
            n_layers=12,
            n_heads=16,
            ffn_dim=5120,
            max_seq_length=1024,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
        "small-3b": ModelSpec(
            name="Small-3B",
            n_params=3_000_000_000,
            hidden_size=2560,
            n_layers=24,
            n_heads=20,
            ffn_dim=6912,
            max_seq_length=2048,
            vocab_size=32000,
            weight_dtype=DataType.BF16,
            activation_dtype=DataType.BF16,
            pos_encoding=PositionalEncoding.ROPE,
        ),
    }
    
    # Aliases for common short names
    _ALIASES = {
        "llama7b": "llama-7b",
        "llama13b": "llama-13b",
        "llama70b": "llama2-70b",
        "llama3": "llama3-8b",
        "mistral": "mistral-7b",
        "gpt3": "gpt3-175b",
    }
    
    @classmethod
    def get_model(cls, name: str) -> ModelSpec:
        """
        Get predefined model specification by name.
        
        Args:
            name: Model name (case-insensitive)
            
        Returns:
            ModelSpec instance
            
        Examples:
            >>> model = ModelCatalog.get_model("llama-7b")
            >>> model = ModelCatalog.get_model("llama3")
            >>> model = ModelCatalog.get_model("mistral-7b")
        """
        # Normalize the input name
        normalized = name.lower().replace(" ", "").replace("_", "").replace("-", "")
        
        # Try exact match in catalog
        for key in cls._CATALOG.keys():
            if normalized == key.replace("-", ""):
                return cls._CATALOG[key]
        
        # Try alias lookup
        if normalized in cls._ALIASES:
            catalog_key = cls._ALIASES[normalized]
            return cls._CATALOG[catalog_key]
        
        # Try partial match
        matches = []
        for catalog_key, model_spec in cls._CATALOG.items():
            if normalized in catalog_key.replace("-", ""):
                matches.append((catalog_key, model_spec))
        
        if len(matches) == 1:
            return matches[0][1]
        elif len(matches) > 1:
            available = ", ".join([m.name for _, m in matches])
            raise ValueError(
                f"Ambiguous model name '{name}'. Multiple matches found: {available}. "
                f"Please be more specific."
            )
        
        # No match found
        available = ", ".join(sorted(set(m.name for m in cls._CATALOG.values())))
        raise ValueError(
            f"Model '{name}' not found in catalog.\n"
            f"Available models: {available}\n"
            f"Use ModelCatalog.list_available() to see all options."
        )
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available models in the catalog."""
        return sorted(set(model.name for model in cls._CATALOG.values()))
    
    @classmethod
    def list_by_size(cls) -> Dict[str, List[str]]:
        """Group models by parameter size."""
        size_groups = {
            "< 10B": [],
            "10B - 20B": [],
            "20B - 100B": [],
            "100B+": [],
        }
        
        for model in cls._CATALOG.values():
            if model.n_params < 10_000_000_000:
                size_groups["< 10B"].append(model.name)
            elif model.n_params < 20_000_000_000:
                size_groups["10B - 20B"].append(model.name)
            elif model.n_params < 100_000_000_000:
                size_groups["20B - 100B"].append(model.name)
            else:
                size_groups["100B+"].append(model.name)
        
        return size_groups
    
    @classmethod
    def compare(cls, model_names: List[str]) -> None:
        """Print comparison table of models."""
        print("\n" + "="*100)
        print("Model Comparison")
        print("="*100)
        print(f"{'Model':<20} {'Params':>12} {'Layers':>8} {'Hidden':>8} "
              f"{'Heads':>8} {'Context':>10}")
        print("-"*100)
        
        for name in model_names:
            try:
                model = cls.get_model(name)
                params_str = f"{model.n_params / 1e9:.1f}B"
                print(f"{model.name:<20} {params_str:>12} {model.n_layers:>8} "
                      f"{model.hidden_size:>8} {model.n_heads:>8} "
                      f"{model.max_seq_length:>10}")
            except ValueError:
                print(f"{name:<20} ERROR: Not found")
        
        print("="*100 + "\n")
    
    @classmethod
    def print_all(cls) -> None:
        """Print all models grouped by size."""
        print("\n" + "="*100)
        print("Available Models in Catalog")
        print("="*100)
        
        groups = cls.list_by_size()
        for size_range, models in groups.items():
            if models:
                print(f"\n{size_range}:")
                for model_name in sorted(models):
                    model = next(m for m in cls._CATALOG.values() if m.name == model_name)
                    params_str = f"{model.n_params / 1e9:.1f}B"
                    print(f"  - {model_name:<25} ({params_str:>6}, "
                          f"{model.n_layers} layers, "
                          f"{model.max_seq_length} context)")
        
        print("\n" + "="*100 + "\n")
    
    @classmethod
    def get_by_family(cls, family: str) -> List[ModelSpec]:
        """
        Get models by family.
        
        Args:
            family: "llama", "llama2", "llama3", "mistral", "gpt3"
            
        Returns:
            List of ModelSpec for that family
        """
        family_lower = family.lower()
        matches = []
        
        for key, model in cls._CATALOG.items():
            if key.startswith(family_lower):
                matches.append(model)
        
        if not matches:
            raise ValueError(
                f"No models found for family '{family}'. "
                f"Try: llama, llama2, llama3, mistral, gpt3"
            )
        
        return matches


# Convenience function
def get_model(name: str) -> ModelSpec:
    """
    Quick access function to get a model spec.
    
    Args:
        name: Model name
        
    Returns:
        ModelSpec instance
        
    Example:
        >>> from llm_inference_simulator import get_model
        >>> model = get_model("llama-7b")
        >>> model = get_model("mistral")
    """
    return ModelCatalog.get_model(name)
