#!/bin/bash

################################################################################
# LLM Inference Simulator - Comprehensive Benchmark
# Tests: Aggregated + Disaggregated configurations
# Shows: Throughput, TTFT, E2E, CapEx
################################################################################

MODEL="llama2-70b"
ARRIVAL_RATE=2.0
DURATION=20.0
WARM_UP=5.0
RANDOM_SEED=$RANDOM

# Workload settings (for display)
AVG_INPUT=512
MAX_INPUT=1024
AVG_OUTPUT=192
MAX_OUTPUT=256

# GPU Prices (CapEx per GPU in USD)
declare -A GPU_PRICES
GPU_PRICES["a100-80gb"]=10000
GPU_PRICES["h100-80gb"]=30000
GPU_PRICES["mi300x"]=10000

# Results directory
RESULTS_DIR="results/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

echo "=======================================================================" | tee $SUMMARY_FILE
echo "LLM Inference Simulator - Comprehensive Benchmark"
echo "Model: $MODEL | Arrival Rate: $ARRIVAL_RATE req/s | Duration: ${DURATION}s"
echo "=======================================================================" | tee -a $SUMMARY_FILE
echo ""

# Results arrays
declare -a RESULTS

#==============================================================================
# AGGREGATED CONFIGURATIONS
#==============================================================================

echo "Testing Aggregated Configurations..." | tee -a $SUMMARY_FILE
echo ""

AGGREGATED_CONFIGS=(
    "mi300x:8:8:8× MI300X (TP=8)"
    "a100-80gb:8:8:8× A100-80GB (TP=8)"
    "h100-80gb:8:8:8× H100-80GB (TP=8)"
)

test_num=0
total_tests=$((${#AGGREGATED_CONFIGS[@]} + 3))  # 3 disaggregated configs

for config in "${AGGREGATED_CONFIGS[@]}"; do
    test_num=$((test_num + 1))

    IFS=':' read -r xpu n_xpus tp description <<< "$config"

    echo "[$test_num/$total_tests] $description..."

    result_file="$RESULTS_DIR/agg_${xpu}_tp${tp}.json"
    output_file="$RESULTS_DIR/agg_${xpu}_tp${tp}.log"

    # Run simulation (capture output)
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
        2>&1 | tee $output_file > /dev/null

    if [ -f $result_file ]; then
        # Extract metrics from JSON
        metrics=$(python3 -c "
import json
try:
    with open('$result_file') as f:
        d = json.load(f)
    tput = d.get('throughput',{}).get('tokens_per_sec', 0)
    ttft = d.get('first_token_latency',{}).get('mean', 0)
    e2e = d.get('end_to_end_latency',{}).get('mean', 0)
    print(f'{tput}|{ttft}|{e2e}')
except:
    print('0|0|0')
" 2>/dev/null)

        IFS='|' read -r throughput ttft e2e <<< "$metrics"

        # Calculate CapEx
        capex=$((n_xpus * ${GPU_PRICES[$xpu]}))

        # Store result (keep as numbers for proper formatting)
        RESULTS+=("$description|$throughput|$ttft|$e2e|$capex")
        echo "  ✓ Throughput: ${throughput} tok/s | TTFT: ${ttft}s | E2E: ${e2e}s | CapEx: \$$capex"
    else
        echo "  ✗ Failed"
        RESULTS+=("$description|0|0|0|0")
    fi

    echo ""
done

#==============================================================================
# DISAGGREGATED CONFIGURATIONS
#==============================================================================

echo "Testing Disaggregated Configurations..." | tee -a $SUMMARY_FILE
echo ""

DISAGGREGATED_CONFIGS=(
    "a100-80gb:4:4:mi300x:8:8:100:Disagg: 4×A100 + 8×MI300X (100GB/s)"
    "a100-80gb:4:4:mi300x:8:8:400:Disagg: 4×A100 + 8×MI300X (400GB/s)"
    "h100-80gb:4:4:mi300x:8:8:400:Disagg: 4×H100 + 8×MI300X (400GB/s)"
)

for config in "${DISAGGREGATED_CONFIGS[@]}"; do
    test_num=$((test_num + 1))

    IFS=':' read -r prefill_xpu prefill_n prefill_tp decode_xpu decode_n decode_tp bandwidth description <<< "$config"

    echo "[$test_num/$total_tests] $description..."

    result_file="$RESULTS_DIR/disagg_${prefill_xpu}_${decode_xpu}_${bandwidth}gbps.json"
    output_file="$RESULTS_DIR/disagg_${prefill_xpu}_${decode_xpu}_${bandwidth}gbps.log"

    # Run simulation (capture output)
    python3 -m llm_inference_simulator \
        --model $MODEL \
        --disaggregated \
        --prefill-xpu $prefill_xpu \
        --prefill-n-xpus $prefill_n \
        --prefill-tp $prefill_tp \
        --decode-xpu $decode_xpu \
        --decode-n-xpus $decode_n \
        --decode-tp $decode_tp \
        --transfer-bandwidth $bandwidth \
        --arrival-rate $ARRIVAL_RATE \
        --duration $DURATION \
        --warm-up $WARM_UP \
        --seed $RANDOM_SEED \
        --output $result_file \
        2>&1 | tee $output_file > /dev/null

    if [ -f $result_file ]; then
        # Extract metrics from JSON
        metrics=$(python3 -c "
import json
try:
    with open('$result_file') as f:
        d = json.load(f)
    tput = d.get('throughput',{}).get('tokens_per_sec', 0)
    ttft = d.get('first_token_latency',{}).get('mean', 0)
    e2e = d.get('end_to_end_latency',{}).get('mean', 0)
    print(f'{tput}|{ttft}|{e2e}')
except:
    print('0|0|0')
" 2>/dev/null)

        IFS='|' read -r throughput ttft e2e <<< "$metrics"

        # Calculate CapEx (prefill + decode)
        capex=$(( (prefill_n * ${GPU_PRICES[$prefill_xpu]}) + (decode_n * ${GPU_PRICES[$decode_xpu]}) ))

        # Store result
        RESULTS+=("$description|$throughput|$ttft|$e2e|$capex")
        echo "  ✓ Throughput: ${throughput} tok/s | TTFT: ${ttft}s | E2E: ${e2e}s | CapEx: \$$capex"
    else
        echo "  ✗ Failed"
        RESULTS+=("$description|0|0|0|0")
    fi

    echo ""
done

#==============================================================================
# SUMMARY TABLE
#==============================================================================

echo "=======================================================================================" | tee -a $SUMMARY_FILE
echo "BENCHMARK SUMMARY" | tee -a $SUMMARY_FILE
echo "=======================================================================================" | tee -a $SUMMARY_FILE
echo "Model: ${MODEL}" | tee -a $SUMMARY_FILE
echo "Workload: ${ARRIVAL_RATE} req/s | Duration: ${DURATION}s (+ ${WARM_UP}s warm-up)" | tee -a $SUMMARY_FILE
echo "Tokens: Input ${AVG_INPUT}/${MAX_INPUT} (avg/max) | Output ${AVG_OUTPUT}/${MAX_OUTPUT} (avg/max)" | tee -a $SUMMARY_FILE
echo "---------------------------------------------------------------------------------------" | tee -a $SUMMARY_FILE
printf "%-47s %15s %12s %12s %15s\n" "Configuration" "Throughput" "TTFT" "E2E" "CapEx" | tee -a $SUMMARY_FILE
printf "%-47s %15s %12s %12s %15s\n" "" "(tok/s)" "(sec)" "(sec)" "(USD)" | tee -a $SUMMARY_FILE
echo "---------------------------------------------------------------------------------------" | tee -a $SUMMARY_FILE

for result in "${RESULTS[@]}"; do
    IFS='|' read -r description throughput ttft e2e capex <<< "$result"
    # Use printf with proper numeric formatting for alignment
    printf "%-47s %15.1f %12.4f %12.4f %15s\n" \
        "$description" \
        "$throughput" \
        "$ttft" \
        "$e2e" \
        "\$$capex" | tee -a $SUMMARY_FILE
done

echo "=======================================================================================" | tee -a $SUMMARY_FILE
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"