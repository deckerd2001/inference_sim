#!/bin/bash

################################################################################
# LLM Inference Simulator - Comprehensive Benchmark
# Tests: Aggregated + Disaggregated configurations
# Shows: Throughput, Output Rate, TTFT, E2E, CapEx
################################################################################

MODEL="llama2-70b"
ARRIVAL_RATE=5.0
DURATION=500.0
WARM_UP=0.0

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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/benchmark_${TIMESTAMP}"
mkdir -p $RESULT_DIR
SUMMARY_FILE="$RESULT_DIR/summary.txt"

# Arrays to store results
declare -A RESULTS

#==============================================================================
# BENCHMARK
#==============================================================================

echo "======================================================================="
echo "LLM Inference Simulator - Comprehensive Benchmark"
echo "Model: $MODEL | Arrival Rate: $ARRIVAL_RATE req/s | Duration: ${DURATION}s"
echo "======================================================================="
echo ""
echo "Testing Aggregated Configurations..."
echo ""

AGGREGATED_CONFIGS=(
    "mi300x:8:8:8× MI300X (TP=8)"
    "a100-80gb:8:8:8× A100-80GB (TP=8)"
    "h100-80gb:8:8:8× H100-80GB (TP=8)"
)

test_num=0
total_tests=$((${#AGGREGATED_CONFIGS[@]} + 3))

for config in "${AGGREGATED_CONFIGS[@]}"; do
    test_num=$((test_num + 1))

    IFS=':' read -r xpu n_xpus tp description <<< "$config"

    echo "[$test_num/$total_tests] $description..."

    result_file="$RESULT_DIR/agg_${xpu}_tp${tp}.json"
    output_file="$RESULT_DIR/agg_${xpu}_tp${tp}.log"

    RANDOM_SEED=$(date +%s)

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
        metrics=$(python3 -c "
import json
try:
    with open('$result_file') as f:
        d = json.load(f)
    tput = d.get('throughput',{}).get('tokens_per_sec', 0)
    req_rate = d.get('throughput',{}).get('requests_per_sec', 0)
    ttft = d.get('first_token_latency',{}).get('mean', 0)
    e2e = d.get('end_to_end_latency',{}).get('mean', 0)
    print(f'{tput}|{req_rate}|{ttft}|{e2e}')
except:
    print('0|0|0|0')
" 2>/dev/null)

        IFS='|' read -r throughput req_rate ttft e2e <<< "$metrics"

        capex=$((n_xpus * ${GPU_PRICES[$xpu]}))

        RESULTS["agg_${xpu}_${tp}"]="$description|$throughput|$req_rate|$ttft|$e2e|$capex"
        printf "  ✓ Throughput: %.1f tok/s | Output: %.2f req/s | TTFT: %.4fs | E2E: %.4fs | CapEx: \$%d\n" \
            "$throughput" "$req_rate" "$ttft" "$e2e" "$capex"
    else
        echo "  ✗ Failed"
        RESULTS["agg_${xpu}_${tp}"]="$description|0|0|0|0|0"
    fi
done

echo ""
echo "Testing Disaggregated Configurations..."
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

    config_name="disagg_${prefill_xpu}_${decode_xpu}_${bandwidth}"
    result_file="$RESULT_DIR/${config_name}.json"
    output_file="$RESULT_DIR/${config_name}.log"

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
        metrics=$(python3 -c "
import json
try:
    with open('$result_file') as f:
        d = json.load(f)
    tput = d.get('throughput',{}).get('tokens_per_sec', 0)
    req_rate = d.get('throughput',{}).get('requests_per_sec', 0)
    ttft = d.get('first_token_latency',{}).get('mean', 0)
    e2e = d.get('end_to_end_latency',{}).get('mean', 0)
    print(f'{tput}|{req_rate}|{ttft}|{e2e}')
except:
    print('0|0|0|0')
" 2>/dev/null)

        IFS='|' read -r throughput req_rate ttft e2e <<< "$metrics"

        prefill_capex=$((prefill_n * ${GPU_PRICES[$prefill_xpu]}))
        decode_capex=$((decode_n * ${GPU_PRICES[$decode_xpu]}))
        capex=$((prefill_capex + decode_capex))

        RESULTS[$config_name]="$description|$throughput|$req_rate|$ttft|$e2e|$capex"
        printf "  ✓ Throughput: %.1f tok/s | Output: %.2f req/s | TTFT: %.4fs | E2E: %.4fs | CapEx: \$%d\n" \
            "$throughput" "$req_rate" "$ttft" "$e2e" "$capex"
    else
        echo "  ✗ Failed"
        RESULTS[$config_name]="$description|0|0|0|0|0"
    fi
done

echo ""

#==============================================================================
# SUMMARY
#==============================================================================

echo "====================================================================================================" | tee $SUMMARY_FILE
echo "BENCHMARK SUMMARY" | tee -a $SUMMARY_FILE
echo "====================================================================================================" | tee -a $SUMMARY_FILE
echo "Model: $MODEL" | tee -a $SUMMARY_FILE
echo "Workload: $ARRIVAL_RATE req/s (arrival rate) | Duration: ${DURATION}s (+ ${WARM_UP}s warm-up)" | tee -a $SUMMARY_FILE
echo "Tokens: Input $AVG_INPUT/$MAX_INPUT (avg/max) | Output $AVG_OUTPUT/$MAX_OUTPUT (avg/max)" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE
echo "NOTE: All metrics measured during hot window (after ${WARM_UP}s warm-up)" | tee -a $SUMMARY_FILE
echo "----------------------------------------------------------------------------------------------------" | tee -a $SUMMARY_FILE
printf "%-45s %12s %10s %10s %10s %12s\n" \
    "Configuration" "Throughput" "Output" "TTFT" "E2E" "CapEx" | tee -a $SUMMARY_FILE
printf "%-45s %12s %10s %10s %10s %12s\n" \
    "" "(tok/s)" "(req/s)" "(sec)" "(sec)" "(USD)" | tee -a $SUMMARY_FILE
echo "----------------------------------------------------------------------------------------------------" | tee -a $SUMMARY_FILE

# Print results in order
for key in "agg_mi300x_8" "agg_a100-80gb_8" "agg_h100-80gb_8" \
           "disagg_a100-80gb_mi300x_100" "disagg_a100-80gb_mi300x_400" "disagg_h100-80gb_mi300x_400"; do
    if [ -n "${RESULTS[$key]}" ]; then
        IFS='|' read -r desc tput req_rate ttft e2e capex <<< "${RESULTS[$key]}"
        printf "%-45s %12.1f %10.2f %10.4f %10.4f %12s\n" \
            "$desc" "$tput" "$req_rate" "$ttft" "$e2e" "\$$capex" | tee -a $SUMMARY_FILE
    fi
done

echo "====================================================================================================" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE
echo "Measurement Window:" | tee -a $SUMMARY_FILE
echo "  Warm-up:  ${WARM_UP}s (system stabilization, metrics not counted)" | tee -a $SUMMARY_FILE
echo "  Hot:      ${DURATION}s (actual measurement period)" | tee -a $SUMMARY_FILE
TOTAL_TIME=$(python3 -c "print($DURATION + $WARM_UP)" 2>/dev/null || echo "N/A")
echo "  Total:    ${TOTAL_TIME}s" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE
echo "Results saved to: $RESULT_DIR" | tee -a $SUMMARY_FILE
echo "Summary: $SUMMARY_FILE" | tee -a $SUMMARY_FILE