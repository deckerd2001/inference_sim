#!/usr/bin/env python3
"""Add workload summary above Performance & Cost Analysis table."""

with open('cluster_benchmark.sh', 'r') as f:
    content = f.read()

# Find the Performance & Cost Analysis section
old_header = '''print("Performance & Cost Analysis:")
print("="*140)'''

new_header = '''print("Workload:")
print("-"*140)
arrival_rate = float(os.environ.get('WORKLOAD_ARRIVAL_RATE', '10'))
avg_output = 192  # (128 + 256) / 2
required = arrival_rate * avg_output
print(f"  Arrival Rate:         {arrival_rate:.1f} req/s")
print(f"  Avg Output Length:    {avg_output} tok/req")
print(f"  Required Throughput:  {required:.0f} tok/s")
print(f"  Duration:             {os.environ.get('SIMULATION_DURATION', '60')}s (+ 20s warm-up)")
print()

print("Performance & Cost Analysis:")
print("="*140)'''

content = content.replace(old_header, new_header)

with open('cluster_benchmark.sh', 'w') as f:
    f.write(content)

print("✓ Added workload summary above table")
