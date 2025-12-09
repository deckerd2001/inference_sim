#!/usr/bin/env python3
"""Add Output Rate (req/s) column instead of Util%."""

with open('cluster_benchmark.sh', 'r') as f:
    content = f.read()

# 1. Update table header - add Out(req/s) column
old_header = '''print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Throughput':>11} {'P95 TTFT':>9} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
    print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'(tok/s)':>11} {'(sec)':>9} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")'''

new_header = '''print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Out(req/s)':>10} {'Throughput':>11} {'P95 TTFT':>9} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
    print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'':>10} {'(tok/s)':>11} {'(sec)':>9} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")'''

content = content.replace(old_header, new_header)
print("✓ Updated table header with Out(req/s) column")

# 2. Update FAIL line
old_fail = '''print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} {'N/A':>11} {'N/A':>9} {'N/A':>8} {'N/A':>11} {'N/A':>11}")'''

new_fail = '''print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} {'N/A':>10} {'N/A':>11} {'N/A':>9} {'N/A':>8} {'N/A':>11} {'N/A':>11}")'''

content = content.replace(old_fail, new_fail)
print("✓ Updated FAIL line")

# 3. Add output_rate calculation and display
old_results = '''            results.append({
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

new_results = '''            # Calculate output rate (req/s)
            sim_time = data.get('simulation_time', 60)
            output_rate = completed / sim_time if sim_time > 0 else 0
            
            results.append({
                'xpu': xpu_name,
                'n_xpus': n_xpus,
                'tp': tp,
                'throughput': throughput,
                'output_rate': output_rate,
                'completed': completed,
                'total': total,
                'ttft_p95': ttft_p95,
                'total_cost': total_cost,
                'efficiency': efficiency,
                'cost_per_1m': cost_per_1m
            })
            
            print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'✅ OK':>8} "
                  f"{output_rate:>10.1f} {throughput:>11.1f} {ttft_p95:>9.2f} {total_cost:>8.2f} {efficiency:>11.1f} {cost_per_1m:>11.2f}")'''

content = content.replace(old_results, new_results)
print("✓ Added output_rate calculation and display")

with open('cluster_benchmark.sh', 'w') as f:
    f.write(content)

print("\n✓ All updates complete!")
print("\nNow the table shows:")
print("  Arrival Rate: 10 req/s (input)")
print("  Out(req/s):   Actual output rate (directly comparable!)")
