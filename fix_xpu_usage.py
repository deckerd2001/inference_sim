#!/usr/bin/env python3
"""Fix xpu usage - xpu_spec is already an xPUSpec object."""

with open('llm_inference_simulator/simulator.py', 'r') as f:
    content = f.read()

# xpu_spec은 이미 객체이므로 get_xpu() 불필요
content = content.replace(
    "from llm_inference_simulator import get_xpu\n        xpu = get_xpu(self.config.cluster_spec.xpu_spec)",
    "xpu = self.config.cluster_spec.xpu_spec"
)

# 혹시 다른 형태로 있다면
content = content.replace(
    "xpu = get_xpu(self.config.cluster_spec.xpu_spec)",
    "xpu = self.config.cluster_spec.xpu_spec"
)

with open('llm_inference_simulator/simulator.py', 'w') as f:
    f.write(content)

print("✓ Fixed: xpu_spec is already an xPUSpec object, no need to call get_xpu()")
