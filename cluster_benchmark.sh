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
python3 << 'PYTHON' | tee -a $LOG_FILE
import json
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_inference_simulator import get_xpu

RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results/cluster_benchmark_*')
results_dirs = sorted(glob.glob(RESULTS_DIR))
if not results_dirs:
    results_dir = RESULTS_DIR
else:
    results_dir = results_dirs[-1] if '*' in RESULTS_DIR else RESULTS_DIR

print("="*140)
print("TP SCALING BENCHMARK REPORT")
print("="*140)
print()

print("Configuration:")
print("-"*140)
print(f"  Model:           {os.environ.get('MODEL', 'N/A')}")
print(f"  xPUs Tested:     {os.environ.get('XPUS', 'N/A')}")
print(f"  Workload:        {os.environ.get('WORKLOAD_ARRIVAL_RATE', 'N/A')} req/s")
print(f"  Duration:        {os.environ.get('SIMULATION_DURATION', 'N/A')}s")
print()

# =============================================================================
# Unified Performance & Cost Table with Out(req/s)
# =============================================================================
print("Performance & Cost Analysis:")
print("="*140)
# Header with Out(req/s) column
print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Out':>7} {'Throughput':>11} {'P95 TTFT':>9} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'(req/s)':>7} {'(tok/s)':>11} {'(sec)':>9} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")
print("-"*140)

results = []
failed_results = []

for result_file in sorted(glob.glob(f"{results_dir}/*xpu_tp*.json")):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        basename = os.path.basename(result_file).replace('.json', '')
        match = re.match(r'(.+?)_(\d+)xpu_tp(\d+)', basename)
        if not match:
            continue
        
        xpu_name = match.group(1)
        n_xpus = int(match.group(2))
        tp = int(match.group(3))
        
        if data.get('status') == 'failed':
            failed_results.append({
                'xpu': xpu_name,
                'n_xpus': n_xpus,
                'tp': tp,
                'error': data.get('error_summary', 'Unknown error'),
            })
            # All N/A with proper spacing
            print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} {'N/A':>7} {'N/A':>11} {'N/A':>9} {'N/A':>8} {'N/A':>11} {'N/A':>11}")
        else:
            throughput = data.get('throughput', {}).get('tokens_per_sec', 0)
            completed = data.get('completed_requests', 0)
            total = data.get('total_requests', 0)
            ttft_p95 = data.get('first_token_latency', {}).get('p95', 0)
            sim_time = data.get('simulation_time', 60)
            
            # Calculate output rate (req/s)
            output_rate = completed / sim_time if sim_time > 0 else 0
            
            # Get cost info
            try:
                xpu = get_xpu(xpu_name)
                total_cost = xpu.price_per_hour * n_xpus
                efficiency = throughput / total_cost if total_cost > 0 else 0
                cost_per_1m = (total_cost / throughput * 1_000_000) if throughput > 0 else 0
            except:
                total_cost = 0
                efficiency = 0
                cost_per_1m = 0
            
            results.append({
                'xpu': xpu_name,
                'n_xpus': n_xpus,
                'tp': tp,
                'output_rate': output_rate,
                'throughput': throughput,
                'completed': completed,
                'total': total,
                'ttft_p95': ttft_p95,
                'total_cost': total_cost,
                'efficiency': efficiency,
                'cost_per_1m': cost_per_1m
            })
            
            print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'✅ OK':>8} "
                  f"{output_rate:>7.1f} {throughput:>11.1f} {ttft_p95:>9.2f} "
                  f"{total_cost:>8.2f} {efficiency:>11.1f} {cost_per_1m:>11.2f}")
    except Exception as e:
        print(f"Error reading {result_file}: {e}")

print("-"*140)

# Rest of the summary sections...
# (keeping existing recommendations, failed tests, TP scaling sections)

if results:
    print()
    print("🏆 Recommended Configurations:")
    print("-"*140)
    
    best_value = max(results, key=lambda x: x['efficiency'])
    print(f"  💰 Best Value (tok/$/hour):")
    print(f"     {best_value['xpu'].upper()}: {best_value['n_xpus']} GPUs, TP={best_value['tp']}")
    print(f"     {best_value['efficiency']:.1f} tok/$/hour | ${best_value['cost_per_1m']:.2f}/1M tokens | {best_value['throughput']:.1f} tok/s")
    print()
    
    best_perf = max(results, key=lambda x: x['throughput'])
    print(f"  🚀 Best Performance (throughput):")
    print(f"     {best_perf['xpu'].upper()}: {best_perf['n_xpus']} GPUs, TP={best_perf['tp']}")
    print(f"     {best_perf['throughput']:.1f} tok/s | ${best_perf['total_cost']:.2f}/hour | P95 TTFT: {best_perf['ttft_p95']:.2f}s")
    print()
    
    best_latency = min(results, key=lambda x: x['ttft_p95'])
    print(f"  ⚡ Best Latency (TTFT):")
    print(f"     {best_latency['xpu'].upper()}: {best_latency['n_xpus']} GPUs, TP={best_latency['tp']}")
    print(f"     P95 TTFT: {best_latency['ttft_p95']:.2f}s | {best_latency['throughput']:.1f} tok/s | ${best_latency['total_cost']:.2f}/hour")
    print()

if failed_results:
    print()
    print("❌ Failed Tests:")
    print("-"*140)
    for f in failed_results:
        print(f"  {f['xpu'].upper()}: {f['n_xpus']} GPUs, TP={f['tp']}")
        print(f"    → {f['error']}")
    print()

if results:
    print()
    print("📈 TP Scaling Efficiency:")
    print("-"*140)
    
    xpu_names = sorted(set(r['xpu'] for r in results))
    for xpu_name in xpu_names:
        xpu_results = sorted([r for r in results if r['xpu'] == xpu_name], key=lambda x: x['n_xpus'])
        if len(xpu_results) > 1:
            baseline = xpu_results[0]
            print(f"  {xpu_name.upper()} (baseline: {baseline['n_xpus']} GPU, TP={baseline['tp']}, {baseline['throughput']:.1f} tok/s):")
            for r in xpu_results[1:]:
                speedup = r['throughput'] / baseline['throughput'] if baseline['throughput'] > 0 else 0
                ideal = r['n_xpus'] / baseline['n_xpus']
                efficiency_pct = (speedup / ideal) * 100 if ideal > 0 else 0
                print(f"    {r['n_xpus']} GPUs, TP={r['tp']}: {speedup:.2f}x speedup ({efficiency_pct:.1f}% efficiency, ideal: {ideal:.1f}x)")
            print()

print("="*140)
print(f"Summary: {len(results)} succeeded, {len(failed_results)} failed")
print("="*140)
PYTHON

echo "" | tee -a $LOG_FILE
echo "Results saved to: $RESULTS_DIR" | tee -a $LOG_FILE
echo "=======================================================================" | tee -a $LOG_FILE
