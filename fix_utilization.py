#!/usr/bin/env python3
"""Fix xPU utilization calculation."""

with open('llm_inference_simulator/simulator.py', 'r') as f:
    content = f.read()

# get_xpu_utilization() 대신 직접 계산
content = content.replace(
    "print(f\"xPU Utilization: {self.metrics.get_xpu_utilization():.1%}\")",
    """# Calculate xPU utilization
        total_time = self.metrics.gpu_busy_time + self.metrics.gpu_idle_time
        xpu_util = self.metrics.gpu_busy_time / total_time if total_time > 0 else 0.0
        print(f"xPU Utilization: {xpu_util:.1%}")"""
)

with open('llm_inference_simulator/simulator.py', 'w') as f:
    f.write(content)

print("✓ Fixed xPU utilization calculation")
