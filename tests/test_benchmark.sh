#!/bin/bash

################################################################################
# Test Benchmark - Exactly matches cluster_benchmark.sh settings
################################################################################

MODEL="llama2-70b"
ARRIVAL_RATE=50.0
DURATION=500.0
WARM_UP=60.0

# Workload settings
AVG_INPUT=512
MAX_INPUT=1024
AVG_OUTPUT=192
MAX_OUTPUT=256

# Test MI300X configuration
XPU="mi300x"
N_XPUS=8
TP=8

echo "=========================================================================="
echo "TEST BENCHMARK - MI300X"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  xPU: $XPU × $N_XPUS (TP=$TP)"
echo "  Arrival Rate: $ARRIVAL_RATE req/s"
echo "  Duration: ${DURATION}s (+ ${WARM_UP}s warm-up)"
echo "  Input: $AVG_INPUT/$MAX_INPUT tokens (avg/max)"
echo "  Output: $AVG_OUTPUT/$MAX_OUTPUT tokens (avg/max)"
echo ""
echo "This will take ~$(python3 -c "print(int($DURATION + $WARM_UP))")s to complete..."
echo ""

RESULT_FILE="test_mi300x_50rps_500s.json"
LOG_FILE="test_mi300x_50rps_500s.log"

# Run WITH output length parameters (correct way)
echo "Test 1: WITH output length parameters (CORRECT)"
echo "----------------------------------------------------------------"

python3 -m llm_inference_simulator \
    --model $MODEL \
    --xpu $XPU \
    --n-xpus-per-node $N_XPUS \
    --tp $TP \
    --arrival-rate $ARRIVAL_RATE \
    --duration $DURATION \
    --warm-up $WARM_UP \
    --avg-input-length $AVG_INPUT \
    --max-input-length $MAX_INPUT \
    --avg-output-length $AVG_OUTPUT \
    --max-output-length $MAX_OUTPUT \
    --output "${RESULT_FILE%.json}_with_params.json" \
    2>&1 | tee "${LOG_FILE%.log}_with_params.log"

echo ""
echo "Test 2: WITHOUT output length parameters (like cluster_benchmark.sh)"
echo "----------------------------------------------------------------"

# Run WITHOUT output length parameters (cluster_benchmark bug)
python3 -m llm_inference_simulator \
    --model $MODEL \
    --xpu $XPU \
    --n-xpus-per-node $N_XPUS \
    --tp $TP \
    --arrival-rate $ARRIVAL_RATE \
    --duration $DURATION \
    --warm-up $WARM_UP \
    --output "${RESULT_FILE%.json}_without_params.json" \
    2>&1 | tee "${LOG_FILE%.log}_without_params.log"

echo ""
echo "=========================================================================="
echo "COMPARISON"
echo "=========================================================================="

echo ""
echo "Test 1 Results (WITH parameters):"
python3 << 'EOF'
import json
try:
    with open('test_mi300x_50rps_500s_with_params.json') as f:
        d = json.load(f)
    print(f"  Total requests: {d.get('total_requests', 0)}")
    print(f"  Completed: {d.get('completed_requests', 0)}")
    print(f"  Throughput: {d.get('throughput', {}).get('requests_per_sec', 0):.2f} req/s")
    print(f"  Token rate: {d.get('throughput', {}).get('tokens_per_sec', 0):.1f} tok/s")
    print(f"  TTFT mean: {d.get('first_token_latency', {}).get('mean', 0):.2f}s")
    print(f"  Memory peak: {d.get('memory', {}).get('peak_memory_gb', 0):.1f}GB")
    print(f"  xPU Util: {d.get('xpu_utilization', 0)*100:.1f}%")
except Exception as e:
    print(f"  Error: {e}")
EOF

echo ""
echo "Test 2 Results (WITHOUT parameters - cluster_benchmark bug):"
python3 << 'EOF'
import json
try:
    with open('test_mi300x_50rps_500s_without_params.json') as f:
        d = json.load(f)
    print(f"  Total requests: {d.get('total_requests', 0)}")
    print(f"  Completed: {d.get('completed_requests', 0)}")
    print(f"  Throughput: {d.get('throughput', {}).get('requests_per_sec', 0):.2f} req/s")
    print(f"  Token rate: {d.get('throughput', {}).get('tokens_per_sec', 0):.1f} tok/s")
    print(f"  TTFT mean: {d.get('first_token_latency', {}).get('mean', 0):.2f}s")
    print(f"  Memory peak: {d.get('memory', {}).get('peak_memory_gb', 0):.1f}GB")
    print(f"  xPU Util: {d.get('xpu_utilization', 0)*100:.1f}%")
except Exception as e:
    print(f"  Error: {e}")
EOF

echo ""
echo "=========================================================================="
echo "DIAGNOSIS"
echo "=========================================================================="

python3 << 'EOF'
import json

try:
    with open('test_mi300x_50rps_500s_with_params.json') as f:
        d1 = json.load(f)
    with open('test_mi300x_50rps_500s_without_params.json') as f:
        d2 = json.load(f)
    
    tput1 = d1.get('throughput', {}).get('requests_per_sec', 0)
    tput2 = d2.get('throughput', {}).get('requests_per_sec', 0)
    
    print("")
    if abs(tput1 - tput2) < 5:
        print("✓ Both tests have similar throughput")
        print("  → cluster_benchmark.sh is NOT the issue")
        print("  → Problem might be elsewhere")
    else:
        print("🚨 SIGNIFICANT DIFFERENCE FOUND!")
        print(f"  With params: {tput1:.2f} req/s")
        print(f"  Without params: {tput2:.2f} req/s")
        print(f"  Difference: {abs(tput1 - tput2):.2f} req/s ({abs(tput1-tput2)/max(tput1,tput2)*100:.1f}%)")
        print("")
        print("  → cluster_benchmark.sh is missing --avg-output-length and --max-output-length!")
        print("  → This causes default values to be used")
        print("  → Fix: Add these parameters to cluster_benchmark.sh")
    print("")
    
except Exception as e:
    print(f"Error comparing: {e}")
EOF

echo "=========================================================================="
