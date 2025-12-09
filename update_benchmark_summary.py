#!/usr/bin/env python3
"""Add Load Analysis to cluster_benchmark.sh summary."""

with open('cluster_benchmark.sh', 'r') as f:
    content = f.read()

# Find the Performance & Cost Analysis table section
# Add Utilization column

old_header = """print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Throughput':>11} {'P95 TTFT':>9} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
    print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'(tok/s)':>11} {'(sec)':>9} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")"""

new_header = """print(f"{'xPU':>12} {'GPUs':>5} {'TP':>3} {'Status':>8} {'Throughput':>11} {'P95 TTFT':>9} {'Util':>6} {'Cost':>8} {'Efficiency':>11} {'Cost':>11}")
    print(f"{'':>12} {'':>5} {'':>3} {'':>8} {'(tok/s)':>11} {'(sec)':>9} {'(%)':>6} {'($/hr)':>8} {'(tok/$/h)':>11} {'($/1Mtok)':>11}")"""

content = content.replace(old_header, new_header)

# Add utilization calculation and display
old_fail_line = """print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} {'N/A':>11} {'N/A':>9} {'N/A':>8} {'N/A':>11} {'N/A':>11}")"""

new_fail_line = """print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'❌ FAIL':>8} {'N/A':>11} {'N/A':>9} {'N/A':>6} {'N/A':>8} {'N/A':>11} {'N/A':>11}")"""

content = content.replace(old_fail_line, new_fail_line)

# Add utilization for success case
old_success = """            results.append({
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
            
            # Print with ONLY numbers (no units)
            print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'✅ OK':>8} "
                  f"{throughput:>11.1f} {ttft_p95:>9.2f} {total_cost:>8.2f} {efficiency:>11.1f} {cost_per_1m:>11.2f}")"""

new_success = """            # Calculate utilization
            avg_output = 192  # Average of 128 and 256
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
            
            # Print with ONLY numbers (no units)
            print(f"{xpu_name.upper():>12} {n_xpus:>5} {tp:>3} {'✅ OK':>8} "
                  f"{throughput:>11.1f} {ttft_p95:>9.2f} {utilization:>6.0f} {total_cost:>8.2f} {efficiency:>11.1f} {cost_per_1m:>11.2f}")"""

content = content.replace(old_success, new_success)

# Add Load Analysis summary section after recommendations
load_summary = """
# =============================================================================
# Load Analysis Summary
# =============================================================================
if results:
    print()
    print("📊 Load Analysis:")
    print("-"*130)
    print(f"  Workload: {os.environ.get('WORKLOAD_ARRIVAL_RATE', '10')} req/s × 192 tok/req = {float(os.environ.get('WORKLOAD_ARRIVAL_RATE', '10')) * 192:.0f} tok/s required")
    print()
    
    # Sort by utilization
    sorted_by_util = sorted(results, key=lambda x: x.get('utilization', 0))
    
    print("  Configurations by Load Level:")
    for r in sorted_by_util:
        util = r.get('utilization', 0)
        config_name = f"{r['xpu'].upper()} {r['n_xpus']} GPUs TP={r['tp']}"
        
        if util >= 100:
            status = "⚠️  OVERLOAD"
        elif util >= 80:
            status = "⚠️  HIGH"
        else:
            status = "✅ NORMAL"
        
        print(f"    {config_name:<30} {r['throughput']:>8.1f} tok/s  ({util:>5.0f}% util)  {status}")
    print()

"""

# Insert before the final summary line
import re
pattern = r"(print\(\"=\"\*130\)\nprint\(f\"Summary: )"
content = re.sub(pattern, load_summary + r"\1", content)

with open('cluster_benchmark.sh', 'w') as f:
    f.write(content)

print("✓ Updated cluster_benchmark.sh with Load Analysis in summary")
