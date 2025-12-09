#!/usr/bin/env python3
"""Fix xpu_type → xpu_spec attribute name."""

with open('llm_inference_simulator/simulator.py', 'r') as f:
    content = f.read()

# xpu_type → xpu_spec
content = content.replace(
    "xpu = get_xpu(self.config.cluster_spec.xpu_type)",
    "xpu = get_xpu(self.config.cluster_spec.xpu_spec)"
)

with open('llm_inference_simulator/simulator.py', 'w') as f:
    f.write(content)

print("✓ Fixed attribute name: xpu_type → xpu_spec")
