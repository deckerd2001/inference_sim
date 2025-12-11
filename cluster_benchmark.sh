#!/bin/bash

################################################################################
# LLM Inference Simulator - Simplified Benchmark
# Shows only core metrics: Throughput, TTFT, E2E Latency
################################################################################

MODEL="llama2-70b"

# Test configurations
XPUS=(
    "a100-80gb"
    "h100-80gb"
    "mi300x"
)

TP_CONFIGS=(
    "1:1"
    "2:2"
    "4:4"
    "8:8"
)

# Workload settings
ARRIVAL_RATE=2.0
DURATION=20.0
WARM_UP=5.0
RANDOM_SEED=$RANDOM

# Create results directory
RESULTS_DIR="results/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

echo "=======================================================================" | tee $SUMMARY_FILE
echo "LLM Inference Simulator - Cluster Benchmark"
echo "Model: $MODEL | Arrival Rate: $ARRIVAL_RATE req/s | Duration: ${DURATION}s"
echo "=======================================================================" | tee -a $SUMMARY_FILE
echo ""

# Results array
declare -a RESULTS

total_tests=$((${#XPUS[@]} * ${#TP_CONFIGS[@]}))
current_test=0

for xpu in "${XPUS[@]}"; do
    for config in "${TP_CONFIGS[@]}"; do
        current_test=$((current_test + 1))

        IFS=':' read -r n_xpus tp <<< "$config"

        echo "[$current_test/$total_tests] Testing: $xpu (${n_xpus} xPUs, TP=$tp)..."

        result_file="$RESULTS_DIR/${xpu}_tp${tp}.json"

        # Run simulation
        python3 -m llm_inference_simulator \
            --model $MODEL \
            --xpu $xpu \
            --n-xpus-per-node $n_xpus \
            --tp $tp \
            --arrival-rate $ARRIVAL_RATE \
            --duration $DURATION \
            --warm-up $WARM_UP \
            --seed $RANDOM_SEED \
            --output $result_file \
            2>&1 | grep -E "(Simulation completed|Throughput|TTFT|E2E|Cost)" || echo "  ✗ Failed"

        if [ -f $result_file ]; then
            # Extract metrics using jq (if available) or python
            if command -v jq &> /dev/null; then
                throughput=$(jq -r '.throughput.tokens_per_sec // 0' $result_file)
                ttft=$(jq -r '.first_token_latency.mean // 0' $result_file)
                e2e=$(jq -r '.end_to_end_latency.mean // 0' $result_file)
            else
                # Fallback: use python
                metrics=$(python3 -c "
import json, sys
with open('$result_file') as f:
    d = json.load(f)
    print(f\"{d.get('throughput',{}).get('tokens_per_sec',0):.1f}\")
    print(f\"{d.get('first_token_latency',{}).get('mean',0):.4f}\")
    print(f\"{d.get('end_to_end_latency',{}).get('mean',0):.4f}\")
" 2>/dev/null)
                read throughput ttft e2e <<< "$metrics"
            fi

            # Store result
            RESULTS+=("$xpu|TP=$tp|$throughput|$ttft|$e2e")
            echo "  ✓ Throughput: ${throughput} tok/s | TTFT: ${ttft}s | E2E: ${e2e}s"
        fi

        echo ""
    done
done

echo "=======================================================================" | tee -a $SUMMARY_FILE
echo "BENCHMARK SUMMARY" | tee -a $SUMMARY_FILE
echo "=======================================================================" | tee -a $SUMMARY_FILE
printf "%-15s %-8s %12s %10s %10s\n" "xPU" "Config" "Throughput" "TTFT" "E2E" | tee -a $SUMMARY_FILE
echo "-----------------------------------------------------------------------" | tee -a $SUMMARY_FILE

for result in "${RESULTS[@]}"; do
    IFS='|' read -r xpu config throughput ttft e2e <<< "$result"
    printf "%-15s %-8s %12s %10s %10s\n" "$xpu" "$config" "${throughput} tok/s" "${ttft}s" "${e2e}s" | tee -a $SUMMARY_FILE
done

echo "=======================================================================" | tee -a $SUMMARY_FILE
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"
