import re

with open('llm_inference_simulator/cli.py', 'r') as f:
    content = f.read()

# null → None (Python syntax)
content = content.replace('"max_batch_size": null', '"max_batch_size": None')

with open('llm_inference_simulator/cli.py', 'w') as f:
    f.write(content)

print("✓ Fixed: null → None")
