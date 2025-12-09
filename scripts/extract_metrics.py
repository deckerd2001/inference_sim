#!/usr/bin/env python3
"""Extract metrics from result JSON file."""
import sys
import json

if len(sys.argv) < 2:
    print("Failed to parse results")
    sys.exit(1)

try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    
    throughput = data.get('throughput', {}).get('tokens_per_sec', 0)
    completed = data.get('completed_requests', 0)
    ttft_p95 = data.get('first_token_latency', {}).get('p95', 0)
    
    print(f"Throughput: {throughput:.1f} tok/s, Completed: {completed}, P95 TTFT: {ttft_p95:.2f}s")
except Exception as e:
    print(f"Failed to parse results: {e}")
    sys.exit(1)
