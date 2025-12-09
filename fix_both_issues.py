#!/usr/bin/env python3
"""Fix both Out(req/s) column and N/A alignment."""

with open('cluster_benchmark.sh', 'r') as f:
    lines = f.readlines()

# Find the Python section for table output
in_python = False
python_start = None
python_end = None

for i, line in enumerate(lines):
    if 'python3 << \'PYTHON\'' in line:
        in_python = True
        python_start = i
    elif in_python and line.strip() == 'PYTHON':
        python_end = i
        break

if python_start is None or python_end is None:
    print("✗ Could not find Python section")
    exit(1)

print(f"Found Python section: lines {python_start+1} to {python_end+1}")

# Replace the entire table printing section
new_python_section = '''python3 << 'PYTHON' | tee -a $LOG_FILE
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
'''

# Replace the Python section
new_lines = lines[:python_start] + [new_python_section] + lines[python_end+1:]

with open('cluster_benchmark.sh', 'w') as f:
    f.writelines(new_lines)

print("✓ Fixed both Out(req/s) column and N/A alignment")
print("\nRun: ./cluster_benchmark.sh")
