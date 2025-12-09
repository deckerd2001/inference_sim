#!/usr/bin/env python3
"""Directly replace print_summary in __main__.py"""

with open('llm_inference_simulator/__main__.py', 'r') as f:
    lines = f.readlines()

# Find where print_summary is defined
summary_start = None
for i, line in enumerate(lines):
    if 'def print_summary' in line:
        summary_start = i
        break

if summary_start is None:
    print("✗ Could not find print_summary")
    exit(1)

# Find the end of print_summary (next function or end of file)
summary_end = len(lines)
for i in range(summary_start + 1, len(lines)):
    if lines[i].startswith('def ') and not lines[i].startswith('    '):
        summary_end = i
        break

print(f"Found print_summary at lines {summary_start+1} to {summary_end}")

# New print_summary with Load Analysis
new_summary = '''def print_summary(metrics, config):
    """Print simulation summary with load analysis."""
    import numpy as np
    from llm_inference_simulator import get_xpu
    
    # Calculate load
    arrival_rate = config.workload_spec.arrival_rate
    avg_output = (config.workload_spec.avg_output_length + 
                 config.workload_spec.max_output_length) / 2
    required_throughput = arrival_rate * avg_output
    actual_throughput = (metrics.total_tokens_generated / 
                        metrics.total_simulation_time if metrics.total_simulation_time > 0 else 0)
    utilization = required_throughput / actual_throughput if actual_throughput > 0 else float('inf')
    is_overloaded = utilization >= 1.0
    
    print()
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print()
    
    # Load Analysis
    print("Load Analysis:")
    print(f"  Arrival Rate:         {arrival_rate:.1f} req/s")
    print(f"  Avg Output Length:    {avg_output:.0f} tok/req")
    print(f"  Required Throughput:  {required_throughput:.0f} tok/s")
    print(f"  Actual Throughput:    {actual_throughput:.1f} tok/s")
    print(f"  Utilization:          {utilization*100:.0f}%", end="")
    
    if is_overloaded:
        print(" ⚠️  OVERLOAD")
    elif utilization >= 0.8:
        print(" ⚠️  HIGH LOAD")
    else:
        print(" ✅ NORMAL")
    print()
    
    if is_overloaded:
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
    print(f"  Completed: {metrics.completed_requests}")
    if metrics.rejected_requests > 0:
        print(f"  Rejected: {metrics.rejected_requests}")
    print()
    
    # Throughput
    if is_overloaded:
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
    
    # Latency
    if is_overloaded:
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
        if is_overloaded:
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
    if is_overloaded:
        stable_arrival = actual_throughput * 0.8 / avg_output
        print("Recommendation:")
        print(f"  System cannot handle workload ({utilization:.1f}x overloaded)")
        print(f"  For stable operation: Reduce arrival to {stable_arrival:.1f} req/s (80% util)")
        print()
    elif utilization >= 0.8:
        print("Recommendation:")
        print(f"  System near capacity ({utilization*100:.0f}% util)")
        print(f"  Consider adding capacity for headroom")
        print()
    
    print("=" * 60)

'''

# Replace
new_lines = lines[:summary_start] + [new_summary + '\n'] + lines[summary_end:]

with open('llm_inference_simulator/__main__.py', 'w') as f:
    f.writelines(new_lines)

print(f"✓ Replaced print_summary ({summary_end - summary_start} lines → 1 function)")
