#!/bin/bash

################################################################################
# Disaggregation Comparison - Core Metrics Only
################################################################################

MODEL="llama2-70b"
ARRIVAL_RATE=1.5
DURATION=20.0
WARM_UP=5.0

RESULTS_DIR="results/disagg_compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "======================================================================="
echo "Disaggregation Comparison: Aggregated vs Disaggregated"
echo "Model: $MODEL | Arrival Rate: $ARRIVAL_RATE req/s"
echo "======================================================================="
echo ""

# Test 1: Aggregated MI300X
echo "[1/2] Testing Aggregated MI300X (8 GPUs, TP=8)..."
python3 -m llm_inference_simulator \
    --model $MODEL \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate $ARRIVAL_RATE \
    --duration $DURATION \
    --warm-up $WARM_UP \
    --output $RESULTS_DIR/aggregated.json \
    2>&1 | grep -E "(Throughput|TTFT|E2E|Cost)" || echo "Failed"

echo ""

# Test 2: Disaggregated A100 + MI300X
echo "[2/2] Testing Disaggregated A100→MI300X (100GB/s)..."
python3 -m llm_inference_simulator \
    --model $MODEL \
    --disaggregated \
    --prefill-xpu a100-80gb \
    --prefill-n-xpus 4 \
    --prefill-tp 4 \
    --decode-xpu mi300x \
    --decode-n-xpus 8 \
    --decode-tp 8 \
    --transfer-bandwidth 100 \
    --arrival-rate $ARRIVAL_RATE \
    --duration $DURATION \
    --warm-up $WARM_UP \
    --output $RESULTS_DIR/disaggregated.json \
    2>&1 | grep -E "(Throughput|TTFT|E2E|Prefill|Decode|Cost)" || echo "Failed"

echo ""
echo "======================================================================="
echo "COMPARISON SUMMARY"
echo "======================================================================="

# Extract and compare
if [ -f $RESULTS_DIR/aggregated.json ] && [ -f $RESULTS_DIR/disaggregated.json ]; then
    python3 -c "
import json

with open('$RESULTS_DIR/aggregated.json') as f:
    agg = json.load(f)
with open('$RESULTS_DIR/disaggregated.json') as f:
    dis = json.load(f)

agg_tput = agg.get('throughput',{}).get('tokens_per_sec',0)
dis_tput = dis.get('throughput',{}).get('tokens_per_sec',0)
agg_ttft = agg.get('first_token_latency',{}).get('mean',0)
dis_ttft = dis.get('first_token_latency',{}).get('mean',0)
agg_e2e = agg.get('end_to_end_latency',{}).get('mean',0)
dis_e2e = dis.get('end_to_end_latency',{}).get('mean',0)

print(f'Metric           Aggregated    Disaggregated   Difference')
print(f'---------------------------------------------------------------')
print(f'Throughput       {agg_tput:>8.1f}        {dis_tput:>8.1f}        {((dis_tput/agg_tput-1)*100):>+6.1f}%')
print(f'TTFT             {agg_ttft:>8.4f}        {dis_ttft:>8.4f}        {((dis_ttft/agg_ttft-1)*100):>+6.1f}%')
print(f'E2E Latency      {agg_e2e:>8.4f}        {dis_e2e:>8.4f}        {((dis_e2e/agg_e2e-1)*100):>+6.1f}%')
"
fi

echo "======================================================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
