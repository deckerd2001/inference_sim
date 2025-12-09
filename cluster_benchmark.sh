#!/bin/bash

################################################################################
# LLM Inference Simulator - Cluster Configuration Benchmark
################################################################################

MODEL="llama2-70b"

XPUS=(
    "a100-80gb"
    "h100-80gb"
    "mi300x"
)

WORKLOAD_AVG_INPUT=512
WORKLOAD_MAX_INPUT=1024
WORKLOAD_AVG_OUTPUT=128
WORKLOAD_MAX_OUTPUT=256
WORKLOAD_ARRIVAL_RATE=10.0
SIMULATION_DURATION=60.0
RANDOM_SEED=$RANDOM

TP_CONFIGS=(
    "1:1:TP=1"
    "2:2:TP=2"
    "4:4:TP=4"
    "8:8:TP=8"
)

RESULTS_DIR="results/cluster_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
LOG_FILE="$RESULTS_DIR/benchmark.log"

echo "=======================================================================" | tee $LOG_FILE
echo "LLM Inference Simulator - TP Scaling Benchmark" | tee -a $LOG_FILE
echo "=======================================================================" | tee -a $LOG_FILE
echo "Start Time: $(date)" | tee -a $LOG_FILE
echo "Random Seed: $RANDOM_SEED" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "⚠️  Note: Only TP (Tensor Parallelism) is implemented" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Configuration:" | tee -a $LOG_FILE
echo "  Model:           $MODEL" | tee -a $LOG_FILE
echo "  xPUs:            ${XPUS[@]}" | tee -a $LOG_FILE
echo "  Arrival Rate:    $WORKLOAD_ARRIVAL_RATE req/s" | tee -a $LOG_FILE
echo "  Duration:        ${SIMULATION_DURATION}s" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Testing ${#XPUS[@]} xPUs × ${#TP_CONFIGS[@]} TP configs = $((${#XPUS[@]} * ${#TP_CONFIGS[@]})) total tests" | tee -a $LOG_FILE
echo "=======================================================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

total_tests=$((${#XPUS[@]} * ${#TP_CONFIGS[@]}))
current_test=0

for xpu in "${XPUS[@]}"; do
    for config in "${TP_CONFIGS[@]}"; do
        current_test=$((current_test + 1))

        IFS=':' read -r n_xpus tp description <<< "$config"

        result_file="$RESULTS_DIR/${xpu}_${n_xpus}xpu_tp${tp}.json"
        error_log_file="$RESULTS_DIR/${xpu}_${n_xpus}xpu_tp${tp}_error.log"

        echo "-----------------------------------------------------------------------" | tee -a $LOG_FILE
        echo "[$current_test/$total_tests] Testing: $xpu - $description" | tee -a $LOG_FILE
        echo "  Cluster: ${n_xpus} xPUs (TP=$tp)" | tee -a $LOG_FILE
        echo "-----------------------------------------------------------------------" | tee -a $LOG_FILE

        output=$(python3 -m llm_inference_simulator \
            --model $MODEL \
            --xpu $xpu \
            --n-xpus-per-node $n_xpus \
            --tp $tp \
            --avg-input-length $WORKLOAD_AVG_INPUT \
            --max-input-length $WORKLOAD_MAX_INPUT \
            --avg-output-length $WORKLOAD_AVG_OUTPUT \
            --max-output-length $WORKLOAD_MAX_OUTPUT \
            --arrival-rate $WORKLOAD_ARRIVAL_RATE \
            --duration $SIMULATION_DURATION \
            --seed $RANDOM_SEED \
            --output $result_file \
            2>&1)

        exit_code=$?
        echo "$output" | tee -a $LOG_FILE

        if [ $exit_code -eq 0 ] && [ -f $result_file ]; then
            echo "✓ Completed successfully" | tee -a $LOG_FILE

            metrics=$(python3 << PYTHON
import json
try:
    data = json.load(open('$result_file'))
    throughput = data.get('throughput', {}).get('tokens_per_sec', 0)
    completed = data.get('completed_requests', 0)
    ttft_p95 = data.get('first_token_latency', {}).get('p95', 0)
    print(f"Throughput: {throughput:.1f} tok/s, Completed: {completed}, P95 TTFT: {ttft_p95:.2f}s")
except:
    print("Failed to parse results")
PYTHON
)
            echo "  $metrics" | tee -a $LOG_FILE
        else
            echo "✗ Failed" | tee -a $LOG_FILE
            echo "$output" > $error_log_file

            error_summary=$(python3 << ERRORPARSE
import re
output = """$output"""
error_msg = "Unknown error"
if "Configuration validation failed:" in output:
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if "Configuration validation failed:" in line:
            error_lines = []
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and not lines[j].startswith('Traceback'):
                    error_lines.append(lines[j].strip())
            if error_lines:
                error_msg = ' '.join(error_lines)
                break
elif "ValueError:" in output:
    match = re.search(r'ValueError: (.+)', output)
    if match:
        error_msg = match.group(1).strip()
elif "Error:" in output:
    match = re.search(r'(\w+Error): (.+)', output)
    if match:
        error_msg = f"{match.group(1)}: {match.group(2).strip()}"
if len(error_msg) > 200:
    error_msg = error_msg[:200] + "..."
print(error_msg)
ERRORPARSE
)

            python3 << ERRORJSON
import json
error_data = {
    "status": "failed",
    "error_summary": """${error_summary}""",
    "error_log": "${error_log_file}",
    "xpu": "$xpu",
    "n_xpus": $n_xpus,
    "tp": $tp
}
with open('$result_file', 'w') as f:
    json.dump(error_data, f, indent=2)
ERRORJSON

            echo "  Error: $error_summary" | tee -a $LOG_FILE
        fi

        echo "" | tee -a $LOG_FILE
    done
done

echo "=======================================================================" | tee -a $LOG_FILE
echo "Benchmark Complete!" | tee -a $LOG_FILE
echo "=======================================================================" | tee -a $LOG_FILE
echo "End Time: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Generate comprehensive summary
export MODEL XPUS WORKLOAD_ARRIVAL_RATE SIMULATION_DURATION RESULTS_DIR
python3 scripts/analyze_benchmark_results.py "$RESULTS_DIR" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "Results saved to: $RESULTS_DIR" | tee -a $LOG_FILE
echo "=======================================================================" | tee -a $LOG_FILE
