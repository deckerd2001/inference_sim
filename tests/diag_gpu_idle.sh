#!/bin/bash

echo "=========================================================================="
echo "DIAGNOSIS: Why GPU Utilization Drops with Long Duration"
echo "=========================================================================="

echo ""
echo "Checking simulation logs for clues..."
echo ""

# Check for OOM messages
echo "1. OOM Messages:"
grep -i "oom\|out of memory\|reject" test_mi300x_50rps_500s_without_params.log 2>/dev/null | head -5 || echo "  None found"

echo ""
echo "2. Queue Size Over Time:"
echo "  (Check if queue grows unbounded)"
grep -i "queue\|requests.*waiting" test_mi300x_50rps_500s_without_params.log 2>/dev/null | tail -10 || echo "  Not available in logs"

echo ""
echo "3. Batch Formation:"
echo "  (Check if batches are getting smaller)"
grep -i "batch.*size\|scheduling.*requests" test_mi300x_50rps_500s_without_params.log 2>/dev/null | tail -10 || echo "  Not available in logs"

echo ""
echo "=========================================================================="
echo "THEORY: GPU Idle in Long Simulations"
echo "=========================================================================="

echo ""
echo "Evidence:"
echo "  Duration=20s:  xPU Util 73%, Throughput 47 req/s"
echo "  Duration=500s: xPU Util 30%, Throughput 13 req/s"
echo ""
echo "Hypothesis 1: Queue Processing Bottleneck"
echo "  - Long queue (18,500 requests)"
echo "  - Scheduler can't efficiently form batches from large queue"
echo "  - O(N) search time slows down batch formation"
echo "  - GPU sits idle waiting for scheduler"
echo ""
echo "Hypothesis 2: Memory Fragmentation"
echo "  - Long simulation causes memory fragmentation"
echo "  - Can't allocate large contiguous blocks"
echo "  - Forced to use smaller batches"
echo "  - Lower GPU utilization"
echo ""
echo "Hypothesis 3: Prefill Starvation"
echo "  - Decode queue grows large"
echo "  - Decode monopolizes GPU time"
echo "  - New prefill requests starve"
echo "  - TTFT explodes (298s confirms this!)"
echo ""

echo "=========================================================================="
echo "QUICK TEST: Short Cooldown"
echo "=========================================================================="

echo ""
echo "Theory: If we reduce cooldown, does GPU util stay high?"
echo ""
echo "Test A: Duration=100s (manageable queue)"
echo ""

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 50.0 \
    --duration 100 \
    --warm-up 10 \
    --output test_100s.json \
    2>&1 | grep -E "xPU Utilization|Throughput|Completed"

echo ""
echo "Test B: Duration=200s (moderate queue)"
echo ""

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 50.0 \
    --duration 200 \
    --warm-up 10 \
    --output test_200s.json \
    2>&1 | grep -E "xPU Utilization|Throughput|Completed"

echo ""
echo "=========================================================================="
echo "ANALYSIS"
echo "=========================================================================="

python3 << 'EOF'
import json

tests = [
    ("20s", "verify_rate_50.0.json", 20),  # From earlier
    ("100s", "test_100s.json", 100),
    ("200s", "test_200s.json", 200),
    ("500s", "test_mi300x_50rps_500s_without_params.json", 500)
]

print("\nGPU Utilization vs Duration:")
print(f"{'Duration':<12} {'xPU Util':<12} {'Throughput':<15} {'Queue':<12}")
print("-" * 55)

for name, file, dur in tests:
    try:
        with open(file) as f:
            d = json.load(f)
        
        util = d.get('xpu_utilization', 0) * 100
        tput = d.get('throughput', {}).get('requests_per_sec', 0)
        total = d.get('total_requests', 0)
        completed = d.get('completed_requests', 0)
        queue = total - completed
        
        print(f"{name:<12} {util:<12.1f} {tput:<15.2f} {queue:<12}")
    except:
        print(f"{name:<12} {'ERROR':<12} {'ERROR':<15} {'ERROR':<12}")

print("\nConclusion:")
print("  If xPU Util decreases as duration increases:")
print("  → Scheduler/Queue management degradation")
print("  If xPU Util stays constant:")
print("  → Normal overload behavior")
EOF

echo ""
echo "=========================================================================="
