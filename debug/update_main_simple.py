#!/usr/bin/env python3
"""Update __main__.py with simple load analysis."""

# Add load analysis function
load_analysis_func = '''
def analyze_load(metrics, config):
    """Simple load analysis: compare required vs actual throughput."""
    from llm_inference_simulator import get_xpu
    
    # Required throughput
    arrival_rate = config.workload_spec.arrival_rate
    avg_output = (config.workload_spec.avg_output_length + 
                 config.workload_spec.max_output_length) / 2
    required_throughput = arrival_rate * avg_output
    
    # Actual throughput
    actual_throughput = (metrics.total_tokens_generated / 
                        metrics.total_simulation_time if metrics.total_simulation_time > 0 else 0)
    
    # Utilization
    utilization = required_throughput / actual_throughput if actual_throughput > 0 else float('inf')
    
    # Status
    if utilization >= 1.0:
        status = "OVERLOAD"
    elif utilization >= 0.8:
        status = "HIGH_LOAD"
    else:
        status = "NORMAL"
    
    return {
        'status': status,
        'utilization': utilization,
        'utilization_pct': utilization * 100,
        'required_throughput': required_throughput,
        'actual_throughput': actual_throughput,
        'is_overloaded': utilization >= 1.0,
        'arrival_rate': arrival_rate,
        'avg_output': avg_output,
    }
'''

# Updated print_summary
print_summary_func = '''
def print_summary(metrics, config):
    """Print simulation summary with load analysis."""
    import numpy as np
    from llm_inference_simulator import get_xpu
    
    # Analyze load
    load = analyze_load(metrics, config)
    
    print()
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print()
    
    # Load Analysis
    print("Load Analysis:")
    print(f"  Arrival Rate:         {load['arrival_rate']:.1f} req/s")
    print(f"  Required Throughput:  {load['required_throughput']:.0f} tok/s")
    print(f"  Actual Throughput:    {load['actual_throughput']:.1f} tok/s")
    print(f"  Utilization:          {load['utilization_pct']:.0f}%", end="")
    
    if load['status'] == "OVERLOAD":
        print(" ⚠️  OVERLOAD")
    elif load['status'] == "HIGH_LOAD":
        print(" ⚠️  HIGH LOAD")
    else:
        print(" ✅ NORMAL")
    
    print()
    
    # Metrics reliability
    if load['is_overloaded']:
        print("-" * 60)
        print("Metrics Reliability:")
        print("  ✅ Throughput:    System capacity (reliable)")
        print("  ✅ Completion:    Capacity/Load ratio (reliable)")
        print("  ❌ TTFT/Latency:  Unreliable (early arrivals only)")
        print("-" * 60)
        print()
    
    # Requests
    print("Requests:")
    print(f"  Total: {metrics.total_requests}")
    print(f"  Completed: {metrics.completed_requests}", end="")
    if metrics.total_requests > 0:
        pct = metrics.completed_requests / metrics.total_requests * 100
        print(f" ({pct:.1f}%)")
    else:
        print()
    
    if metrics.rejected_requests > 0:
        print(f"  Rejected: {metrics.rejected_requests}")
    
    print()
    
    # Throughput
    if load['is_overloaded']:
        print("Throughput (= System Capacity):")
    else:
        print("Throughput:")
    
    if metrics.total_simulation_time > 0:
        req_per_sec = metrics.completed_requests / metrics.total_simulation_time
        tok_per_sec = metrics.total_tokens_generated / metrics.total_simulation_time
        print(f"  Requests/sec: {req_per_sec:.2f}")
        print(f"  Tokens/sec: {tok_per_sec:.2f}")
    
    print()
    print(f"xPU Utilization: {metrics.get_xpu_utilization():.1%}")
    print()
    
    # Memory
    xpu = get_xpu(config.cluster_spec.xpu_type)
    total_mem = config.cluster_spec.n_xpus_per_node * config.cluster_spec.n_nodes * xpu.memory_size_gb
    
    print("Memory Usage:")
    print(f"  Peak:        {metrics.peak_memory_usage_gb:.2f}GB / {total_mem:.0f}GB ({metrics.peak_memory_usage_gb/total_mem*100:.1f}%)")
    
    if metrics.memory_samples:
        mem_array = np.array(metrics.memory_samples)
        p95 = np.percentile(mem_array, 95)
        p50 = np.percentile(mem_array, 50)
        print(f"  P95:         {p95:.2f}GB / {total_mem:.0f}GB ({p95/total_mem*100:.1f}%)")
        print(f"  P50 (Med):   {p50:.2f}GB / {total_mem:.0f}GB ({p50/total_mem*100:.1f}%)")
    
    print()
    
    # Latency with warning if overloaded
    if load['is_overloaded']:
        print("First Token Latency (⚠️  Not representative - early arrivals only):")
    else:
        print("First Token Latency (seconds):")
    
    if metrics.first_token_latencies:
        ftl = np.array(metrics.first_token_latencies)
        print(f"  Mean: {np.mean(ftl):.4f}")
        print(f"  P50:  {np.percentile(ftl, 50):.4f}")
        print(f"  P90:  {np.percentile(ftl, 90):.4f}")
        print(f"  P95:  {np.percentile(ftl, 95):.4f}")
        print(f"  P99:  {np.percentile(ftl, 99):.4f}")
        print()
    
    if metrics.end_to_end_latencies:
        e2e = np.array(metrics.end_to_end_latencies)
        if load['is_overloaded']:
            print("End-to-End Latency (⚠️  Not representative):")
        else:
            print("End-to-End Latency (seconds):")
        print(f"  Mean: {np.mean(e2e):.4f}")
        print(f"  P50:  {np.percentile(e2e, 50):.4f}")
        print(f"  P90:  {np.percentile(e2e, 90):.4f}")
        print(f"  P95:  {np.percentile(e2e, 95):.4f}")
        print(f"  P99:  {np.percentile(e2e, 99):.4f}")
        print()
    
    # Recommendations
    if load['is_overloaded']:
        shortfall = load['utilization']
        stable_arrival = load['actual_throughput'] * 0.8 / load['avg_output']
        
        print("Recommendation:")
        print(f"  Current config cannot handle workload ({load['utilization']:.1f}x overloaded)")
        print(f"  For stable operation: Reduce arrival to {stable_arrival:.1f} req/s")
        print()
    elif load['status'] == "HIGH_LOAD":
        print("Recommendation:")
        print(f"  System near capacity ({load['utilization_pct']:.0f}% util)")
        print(f"  Consider adding capacity for headroom")
        print()
    
    print("=" * 60)
'''

# Write to file
with open('llm_inference_simulator/__main__.py', 'r') as f:
    content = f.read()

# Add analyze_load before print_summary
if 'def analyze_load' not in content:
    # Find print_summary and insert before it
    marker = 'def print_summary'
    pos = content.find(marker)
    if pos > 0:
        content = content[:pos] + load_analysis_func + '\n\n' + content[pos:]

# Replace print_summary
import re
pattern = r'def print_summary\([^)]+\):.*?(?=\ndef [a-z_]+\(|\Z)'
content = re.sub(pattern, print_summary_func, content, flags=re.DOTALL)

with open('llm_inference_simulator/__main__.py', 'w') as f:
    f.write(content)

print("✓ Updated __main__.py with simple load analysis")
