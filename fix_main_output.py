#!/usr/bin/env python3
"""Add Load Analysis to __main__.py output."""

with open('llm_inference_simulator/__main__.py', 'r') as f:
    content = f.read()

# 1. analyze_load 함수 추가 (없으면)
if 'def analyze_load' not in content:
    analyze_load_code = '''
def analyze_load(metrics, config):
    """Calculate load analysis: arrival rate vs output rate."""
    arrival_rate = config.workload_spec.arrival_rate
    avg_output = (config.workload_spec.avg_output_length + 
                 config.workload_spec.max_output_length) / 2
    required_throughput = arrival_rate * avg_output
    
    actual_throughput = (metrics.total_tokens_generated / 
                        metrics.total_simulation_time if metrics.total_simulation_time > 0 else 0)
    
    utilization = required_throughput / actual_throughput if actual_throughput > 0 else float('inf')
    
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
    # print_summary 함수 앞에 추가
    import re
    match = re.search(r'(def print_summary)', content)
    if match:
        pos = match.start()
        content = content[:pos] + analyze_load_code + content[pos:]
        print("✓ Added analyze_load function")

# 2. print_summary에 Load Analysis 출력 추가
# SIMULATION SUMMARY 바로 아래에 Load Analysis 추가
old_pattern = r'(print\("=" \* 60\)\s+print\(\)\s+)(print\("Requests:"\))'

new_section = r'''\1# Load Analysis
    load = analyze_load(metrics, config)
    
    print("Load Analysis:")
    print(f"  Arrival Rate:         {load['arrival_rate']:.1f} req/s")
    print(f"  Avg Output Length:    {load['avg_output']:.0f} tok/req")
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
    
    \2'''

import re
content = re.sub(old_pattern, new_section, content)

# 3. Latency 출력에 경고 추가
# First Token Latency 제목 수정
content = re.sub(
    r'print\("First Token Latency \(seconds\):"\)',
    r'''if load.get('is_overloaded', False):
        print("First Token Latency (⚠️  Not representative - early arrivals only):")
    else:
        print("First Token Latency (seconds):")''',
    content
)

# End-to-End Latency 제목 수정
content = re.sub(
    r'(if metrics\.end_to_end_latencies:.*?)(print\("End-to-End Latency \(seconds\):"\))',
    r'''\1if load.get('is_overloaded', False):
        print("End-to-End Latency (⚠️  Not representative):")
    else:
        print("End-to-End Latency (seconds):")''',
    content,
    flags=re.DOTALL
)

# 4. Throughput 제목 수정
content = re.sub(
    r'print\("Throughput:"\)',
    r'''if load.get('is_overloaded', False):
        print("Throughput (= System Capacity):")
    else:
        print("Throughput:")''',
    content
)

# 5. Recommendation 추가 (마지막 구분선 앞에)
recommendation_code = '''
    # Recommendations
    if load.get('is_overloaded', False):
        stable_arrival = load['actual_throughput'] * 0.8 / load['avg_output']
        print("Recommendation:")
        print(f"  System cannot handle workload ({load['utilization']:.1f}x overloaded)")
        print(f"  For stable operation: Reduce arrival to {stable_arrival:.1f} req/s (80% util)")
        print()
    elif load.get('status') == "HIGH_LOAD":
        print("Recommendation:")
        print(f"  System near capacity ({load['utilization_pct']:.0f}% util)")
        print(f"  Consider adding capacity for headroom")
        print()
    
    '''

# 마지막 구분선 바로 앞에 추가
content = re.sub(
    r'(\n    print\("=" \* 60\)\n\s*$)',
    recommendation_code + r'\1',
    content,
    flags=re.MULTILINE
)

with open('llm_inference_simulator/__main__.py', 'w') as f:
    f.write(content)

print("✓ Successfully updated __main__.py with Load Analysis output")
print()
print("Test it with:")
print("  python3 -m llm_inference_simulator --model llama2-70b --xpu a100-80gb \\")
print("    --n-xpus-per-node 8 --tp 8 --arrival-rate 50 --duration 60")
