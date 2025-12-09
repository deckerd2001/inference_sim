#!/bin/bash

echo "======================================================================="
echo "QUEUEING DELAY VERIFICATION"
echo "======================================================================="
echo ""

# Check simulator logic
echo "1. Checking how TTFT is calculated..."
grep -A 10 "first_token_time\|ttft" llm_inference_simulator/simulator.py | head -20

echo ""
echo "2. Checking request arrival and queueing..."
grep -A 10 "def.*arrive\|queue" llm_inference_simulator/simulator.py | head -30

echo ""
echo "3. Checking how requests are completed..."
grep -A 5 "completed_requests\|complete_request" llm_inference_simulator/simulator.py | head -20

echo ""
echo "4. Detailed analysis of a slow GPU result..."
python3 << 'PYTHON'
import json
import glob

results_dirs = sorted(glob.glob("results/cluster_benchmark_*"))
if results_dirs:
    results_dir = results_dirs[-1]
    
    # Find A100 8 GPU result
    for f in glob.glob(f"{results_dir}/a100-80gb_8xpu_tp8.json"):
        with open(f) as file:
            data = json.load(file)
        
        if data.get('status') == 'failed':
            continue
        
        sim_time = data.get('simulation_time', 60)
        total = data.get('total_requests', 0)
        completed = data.get('completed_requests', 0)
        throughput = data['throughput']['tokens_per_sec']
        ttft_mean = data['first_token_latency']['mean']
        ttft_p95 = data['first_token_latency']['p95']
        
        arrival_rate = total / sim_time
        required_throughput = arrival_rate * 128  # avg output length
        
        print("A100 8 GPUs Analysis:")
        print("="*60)
        print(f"Arrival Rate:        {arrival_rate:.1f} req/s")
        print(f"Required Throughput: {required_throughput:.1f} tok/s")
        print(f"Actual Throughput:   {throughput:.1f} tok/s")
        print(f"Capacity:            {throughput/required_throughput*100:.1f}%")
        print()
        print(f"Total Requests:      {total}")
        print(f"Completed:           {completed} ({completed/total*100:.1f}%)")
        print(f"Incomplete:          {total-completed} ({(total-completed)/total*100:.1f}%)")
        print()
        print(f"TTFT Mean:           {ttft_mean:.2f}s")
        print(f"TTFT P95:            {ttft_p95:.2f}s")
        print()
        print("🤔 Questions:")
        print(f"  - Why {completed/total*100:.1f}% completed if only {throughput/required_throughput*100:.1f}% capacity?")
        print(f"  - Is queueing delay included in TTFT?")
        print(f"  - Are requests rejected or queued?")
        print()
PYTHON

echo ""
echo "======================================================================="
