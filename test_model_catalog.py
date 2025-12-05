"""Test Model Catalog"""
import sys
sys.path.insert(0, '.')

from llm_inference_simulator import ModelCatalog, get_model

print("="*70)
print("Available Models")
print("="*70)

# List all models
ModelCatalog.print_all()

print("\n" + "="*70)
print("Quick Access Tests")
print("="*70)

test_cases = ["llama-7b", "llama3", "mistral", "gpt3", "tiny-1b"]

for name in test_cases:
    try:
        model = get_model(name)
        print(f"✓ '{name}' → {model.name} "
              f"({model.n_params/1e9:.1f}B params, "
              f"{model.n_layers} layers)")
    except ValueError as e:
        print(f"✗ '{name}' → ERROR")

print("\n" + "="*70)
print("Model Comparison")
print("="*70)

ModelCatalog.compare([
    "llama-7b", 
    "llama2-13b", 
    "llama3-70b", 
    "mistral-7b",
    "gpt3-175b"
])

print("\n" + "="*70)
print("LLaMA Family")
print("="*70)

llama_models = ModelCatalog.get_by_family("llama")
for model in llama_models:
    print(f"  - {model.name}")
