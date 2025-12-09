#!/usr/bin/env python3
"""Fix simulator.py _print_summary method to show Load Analysis."""

with open('llm_inference_simulator/simulator.py', 'r') as f:
    lines = f.readlines()

# Find _print_summary method
start = None
for i, line in enumerate(lines):
    if 'def _print_summary(self):' in line:
        start = i
        break

if start is None:
    print("✗ Could not find _print_summary")
    exit(1)

# Find end of method (next def at same indentation level)
end = len(lines)
for i in range(start + 1, len(lines)):
    if lines[i].startswith('    def ') and not lines[i].startswith('        '):
        end = i
        break

print(f"Found _print_summary at lines {start+1} to {end}")

# New _print_summary with Load Analysis
new_method = '''    def _print_summary(self):
        """Print simulation summary with load analysis."""
        import numpy as np
        
        # Calculate load
        arrival_rate = self.config.workload_spec.arrival_rate
        avg_output = (self.config.workload_spec.avg_output_length + 
                     self.config.workload_spec.max_output_length) / 2
        required_throughput = arrival_rate * avg_output
        actual_throughput = (self.metrics.total_tokens_generated / 
                            self.metrics.total_simulation_time if self.metrics.total_simulation_time > 0 else 0)
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
        print(f"  Total: {self.metrics.total_requests}")
        print(f"  Completed: {self.metrics.completed_requests}")
        
        if self.metrics.rejected_requests > 0:
            print(f"  Rejected: {self.metrics.rejected_requests}")
        
        print()
        
        # Throughput
        if is_overloaded:
            print("Throughput (= System Capacity):")
        else:
            print("Throughput:")
        
        if self.metrics.total_simulation_time > 0:
            req_per_sec = self.metrics.completed_requests / self.metrics.total_simulation_time
            tok_per_sec = self.metrics.total_tokens_generated / self.metrics.total_simulation_time
            print(f"  Requests/sec: {req_per_sec:.2f}")
            print(f"  Tokens/sec: {tok_per_sec:.2f}")
        
        print()
        print(f"xPU Utilization: {self.metrics.get_xpu_utilization():.1%}")
        print()
        
        # Memory
        from llm_inference_simulator import get_xpu
        xpu = get_xpu(self.config.cluster_spec.xpu_type)
        total_mem = self.config.cluster_spec.n_xpus_per_node * self.config.cluster_spec.n_nodes * xpu.memory_size_gb
        
        print("Memory Usage:")
        print(f"  Peak:        {self.metrics.peak_memory_usage_gb:.2f}GB / {total_mem:.0f}GB ({self.metrics.peak_memory_usage_gb/total_mem*100:.1f}%)")
        
        if self.metrics.memory_samples:
            mem_array = np.array(self.metrics.memory_samples)
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
        
        if self.metrics.first_token_latencies:
            ftl = np.array(self.metrics.first_token_latencies)
            print(f"  Mean: {np.mean(ftl):.4f}")
            print(f"  P50:  {np.percentile(ftl, 50):.4f}")
            print(f"  P90:  {np.percentile(ftl, 90):.4f}")
            print(f"  P95:  {np.percentile(ftl, 95):.4f}")
            print(f"  P99:  {np.percentile(ftl, 99):.4f}")
            print()
        
        if self.metrics.end_to_end_latencies:
            e2e = np.array(self.metrics.end_to_end_latencies)
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
new_lines = lines[:start] + [new_method + '\n'] + lines[end:]

with open('llm_inference_simulator/simulator.py', 'w') as f:
    f.writelines(new_lines)

print(f"✓ Replaced _print_summary in simulator.py ({end - start} lines)")
print()
print("Now test:")
print("python3 -m llm_inference_simulator --model llama2-70b --xpu a100-80gb \\")
print("  --n-xpus-per-node 8 --tp 8 --arrival-rate 50 --duration 60")
