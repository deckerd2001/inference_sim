#!/bin/bash

echo "Updating PerformanceModel to use xPU compute units..."

# PerformanceModel에서 실제로 compute unit 선택하도록 수정
# self.xpu.compute_tflops → self.xpu.get_matmul_tflops(dtype)

sed -i 's/self\.xpu\.compute_tflops/self.xpu.get_matmul_tflops(self.model.weight_dtype)/g' llm_inference_simulator/performance_model.py

echo "✓ PerformanceModel updated!"
