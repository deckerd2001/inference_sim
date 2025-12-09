#!/usr/bin/env python3
"""Add Utilization column to benchmark table."""

with open('cluster_benchmark.sh', 'r') as f:
    content = f.read()

# 1. Update table header
old_header = '''print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Throughput':>11} {'P95 TTFT':>9} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
    print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'(tok/s)':>11} {'(sec)':>9} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")'''

new_header = '''print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Throughput':>11} {'P95 TTFT':>9} {'Util':>6} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
    print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'(tok/s)':>11} {'(sec)':>9} {'(%)':>6} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")'''

if old_header in content:
    content = content.replace(old_header, new_header)
    print("✓ Updated table header with Util column")
else:
    print("⚠️  Could not find exact header match")

# 2. Update FAIL line
old_fail = '''print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} {'N/A':>11} {'N/A':>9} {'N/A':>8} {'N/A':>11} {'N/A':>11}")'''

new_fail = '''print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} {'N/A':>11} {'N/A':>9} {'N/A':>6} {'N/A':>8} {'N/A':>11} {'N/A':>11}")'''

if old_fail in content:
    content = content.replace(old_fail, new_fail)
    print("✓ Updated FAIL line with N/A for Util")
else:
    print("⚠️  Could not find FAIL line")

# 3. Add utilization calculation before results.append
old_results_append = '''            results.append({
                'xpu': xpu_name,
                'n_xpus': n_xpus,
                'tp': tp,
                'throughput': throughput,
                'completed': completed,
                'total': total,
                'ttft_p95': ttft_p95,
                'total_cost': total_cost,
                'efficiency': efficiency,
                'cost_per_1m': cost_per_1m
            })
            
            print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'✅ OK':>8} "
                  f"{throughput:>11.1f} {ttft_p95:>9.2f} {total_cost:>8.2f} {efficiency:>11.1f} {cost_per_1m:>11.2f}")'''

new_results_append = '''            # Calculate utilization
            avg_output = 192  # (128 + 256) / 2
            arrival_rate = float(os.environ.get('WORKLOAD_ARRIVAL_RATE', '10'))
            required_throughput = arrival_rate * avg_output
            utilization = (required_throughput / throughput * 100) if throughput > 0 else 0
            
            results.append({
                'xpu': xpu_name,
                'n_xpus': n_xpus,
                'tp': tp,
                'throughput': throughput,
                'completed': completed,
                'total': total,
                'ttft_p95': ttft_p95,
                'total_cost': total_cost,
                'efficiency': efficiency,
                'cost_per_1m': cost_per_1m,
                'utilization': utilization
            })
            
            print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'✅ OK':>8} "
                  f"{throughput:>11.1f} {ttft_p95:>9.2f} {utilization:>6.0f} {total_cost:>8.2f} {efficiency:>11.1f} {cost_per_1m:>11.2f}")'''

if old_results_append in content:
    content = content.replace(old_results_append, new_results_append)
    print("✓ Added utilization calculation and display")
else:
    print("⚠️  Could not find results.append section")

with open('cluster_benchmark.sh', 'w') as f:
    f.write(content)

print("\n✓ All updates complete!")
print("\nRun: ./cluster_benchmark.sh")
