#!/usr/bin/env python3
"""
Analyze and generate comprehensive benchmark report.

Usage:
    python3 scripts/analyze_benchmark_results.py <results_dir>
"""

import json
import glob
import os
import re
import sys


def parse_result_file(filepath):
    """Parse a single result JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        basename = os.path.basename(filepath).replace('.json', '')
        match = re.match(r'(.+?)_(\d+)xpu_tp(\d+)', basename)
        if not match:
            return None
        
        return {
            'xpu_name': match.group(1),
            'n_xpus': int(match.group(2)),
            'tp': int(match.group(3)),
            'data': data,
            'filepath': filepath
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return None


def get_cost_info(xpu_name, n_xpus, throughput):
    """Calculate cost metrics for a configuration."""
    try:
        # Add parent directory to path to import llm_inference_simulator
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from llm_inference_simulator import get_xpu
        
        xpu = get_xpu(xpu_name)
        total_cost = xpu.price_per_hour * n_xpus
        efficiency = throughput / total_cost if total_cost > 0 else 0
        cost_per_1m = (total_cost / throughput * 1_000_000) if throughput > 0 else 0
        
        return {
            'total_cost': total_cost,
            'efficiency': efficiency,
            'cost_per_1m': cost_per_1m
        }
    except Exception as e:
        print(f"Error getting cost info for {xpu_name}: {e}", file=sys.stderr)
        return {
            'total_cost': 0,
            'efficiency': 0,
            'cost_per_1m': 0
        }


def process_results(results_dir):
    """Process all result files and categorize."""
    results = []
    failed = []
    
    for filepath in sorted(glob.glob(f"{results_dir}/*xpu_tp*.json")):
        parsed = parse_result_file(filepath)
        if not parsed:
            continue
        
        xpu_name = parsed['xpu_name']
        n_xpus = parsed['n_xpus']
        tp = parsed['tp']
        data = parsed['data']
        
        if data.get('status') == 'failed':
            failed.append({
                'xpu': xpu_name,
                'n_xpus': n_xpus,
                'tp': tp,
                'error': data.get('error_summary', 'Unknown error')
            })
        else:
            throughput = data.get('throughput', {}).get('tokens_per_sec', 0)
            completed = data.get('completed_requests', 0)
            total = data.get('total_requests', 0)
            ttft_p95 = data.get('first_token_latency', {}).get('p95', 0)
            sim_time = data.get('simulation_time', 60)
            output_rate = completed / sim_time if sim_time > 0 else 0
            
            cost_info = get_cost_info(xpu_name, n_xpus, throughput)
            
            results.append({
                'xpu': xpu_name,
                'n_xpus': n_xpus,
                'tp': tp,
                'output_rate': output_rate,
                'throughput': throughput,
                'completed': completed,
                'total': total,
                'ttft_p95': ttft_p95,
                **cost_info
            })
    
    return results, failed


def print_workload_summary(arrival_rate, duration):
    """Print workload summary."""
    avg_output = 192  # (128 + 256) / 2
    required = arrival_rate * avg_output
    
    print("Workload:")
    print("-" * 140)
    print(f"  Arrival Rate:         {arrival_rate:.1f} req/s")
    print(f"  Avg Output Length:    {avg_output} tok/req")
    print(f"  Required Throughput:  {required:.0f} tok/s")
    print(f"  Duration:             {duration}s (+ 20s warm-up)")
    print()


def print_performance_table(results, failed):
    """Print unified performance & cost table."""
    print("Performance & Cost Analysis:")
    print("=" * 140)
    print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Out':>7} "
          f"{'Throughput':>11} {'P95 TTFT':>9} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
    print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'(req/s)':>7} "
          f"{'(tok/s)':>11} {'(sec)':>9} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")
    print("-" * 140)
    
    # Combine and sort all configs
    all_configs = []
    for r in results:
        all_configs.append(('success', r))
    for f in failed:
        all_configs.append(('failed', f))
    
    all_configs.sort(key=lambda x: (x[1]['xpu'], x[1]['n_xpus']))
    
    for status, item in all_configs:
        xpu = item['xpu'].upper()
        n_xpus = item['n_xpus']
        tp = item['tp']
        
        if status == 'failed':
            print(f"{xpu:>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} "
                  f"{'N/A':>7} {'N/A':>11} {'N/A':>9} {'N/A':>8} {'N/A':>11} {'N/A':>11}")
        else:
            print(f"{xpu:>12} {n_xpus:>5} {tp:>3} {'✅ OK':>8} "
                  f"{item['output_rate']:>7.1f} {item['throughput']:>11.1f} {item['ttft_p95']:>9.2f} "
                  f"{item['total_cost']:>8.2f} {item['efficiency']:>11.1f} {item['cost_per_1m']:>11.2f}")
    
    print("-" * 140)


def print_recommendations(results):
    """Print recommended configurations."""
    if not results:
        return
    
    print()
    print("🏆 Recommended Configurations:")
    print("-" * 140)
    
    best_value = max(results, key=lambda x: x['efficiency'])
    print(f"  💰 Best Value (tok/$/hour):")
    print(f"     {best_value['xpu'].upper()}: {best_value['n_xpus']} GPUs, TP={best_value['tp']}")
    print(f"     {best_value['efficiency']:.1f} tok/$/hour | "
          f"${best_value['cost_per_1m']:.2f}/1M tokens | {best_value['throughput']:.1f} tok/s")
    print()
    
    best_perf = max(results, key=lambda x: x['throughput'])
    print(f"  🚀 Best Performance (throughput):")
    print(f"     {best_perf['xpu'].upper()}: {best_perf['n_xpus']} GPUs, TP={best_perf['tp']}")
    print(f"     {best_perf['throughput']:.1f} tok/s | "
          f"${best_perf['total_cost']:.2f}/hour | P95 TTFT: {best_perf['ttft_p95']:.2f}s")
    print()
    
    best_latency = min(results, key=lambda x: x['ttft_p95'])
    print(f"  ⚡ Best Latency (TTFT):")
    print(f"     {best_latency['xpu'].upper()}: {best_latency['n_xpus']} GPUs, TP={best_latency['tp']}")
    print(f"     P95 TTFT: {best_latency['ttft_p95']:.2f}s | "
          f"{best_latency['throughput']:.1f} tok/s | ${best_latency['total_cost']:.2f}/hour")
    print()


def print_failed_tests(failed):
    """Print failed test summary."""
    if not failed:
        return
    
    print()
    print("❌ Failed Tests:")
    print("-" * 140)
    for f in failed:
        print(f"  {f['xpu'].upper()}: {f['n_xpus']} GPUs, TP={f['tp']}")
        print(f"    → {f['error']}")
    print()


def print_scaling_efficiency(results):
    """Print TP scaling efficiency analysis."""
    if not results:
        return
    
    print()
    print("📈 TP Scaling Efficiency:")
    print("-" * 140)
    
    xpu_names = sorted(set(r['xpu'] for r in results))
    for xpu_name in xpu_names:
        xpu_results = sorted([r for r in results if r['xpu'] == xpu_name], 
                           key=lambda x: x['n_xpus'])
        if len(xpu_results) > 1:
            baseline = xpu_results[0]
            print(f"  {xpu_name.upper()} (baseline: {baseline['n_xpus']} GPU, "
                  f"TP={baseline['tp']}, {baseline['throughput']:.1f} tok/s):")
            
            for r in xpu_results[1:]:
                speedup = r['throughput'] / baseline['throughput'] if baseline['throughput'] > 0 else 0
                ideal = r['n_xpus'] / baseline['n_xpus']
                efficiency_pct = (speedup / ideal) * 100 if ideal > 0 else 0
                print(f"    {r['n_xpus']} GPUs, TP={r['tp']}: {speedup:.2f}x speedup "
                      f"({efficiency_pct:.1f}% efficiency, ideal: {ideal:.1f}x)")
            print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze_benchmark_results.py <results_dir>", file=sys.stderr)
        sys.exit(1)
    
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found", file=sys.stderr)
        sys.exit(1)
    
    # Get config from environment
    model = os.environ.get('MODEL', 'N/A')
    xpus = os.environ.get('XPUS', 'N/A')
    arrival_rate = float(os.environ.get('WORKLOAD_ARRIVAL_RATE', '10'))
    duration = os.environ.get('SIMULATION_DURATION', '60')
    
    # Process results
    results, failed = process_results(results_dir)
    
    # Print report
    print("=" * 140)
    print("TP SCALING BENCHMARK REPORT")
    print("=" * 140)
    print()
    
    print("Configuration:")
    print("-" * 140)
    print(f"  Model:           {model}")
    print(f"  xPUs Tested:     {xpus}")
    print(f"  Workload:        {arrival_rate:.1f} req/s")
    print(f"  Duration:        {duration}s")
    print()
    
    print_workload_summary(arrival_rate, duration)
    print_performance_table(results, failed)
    print_recommendations(results)
    print_failed_tests(failed)
    print_scaling_efficiency(results)
    
    print("=" * 140)
    print(f"Summary: {len(results)} succeeded, {len(failed)} failed")
    print("=" * 140)


if __name__ == '__main__':
    main()
