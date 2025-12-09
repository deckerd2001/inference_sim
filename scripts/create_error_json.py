#!/usr/bin/env python3
"""Create error JSON file."""
import sys
import json

if len(sys.argv) < 6:
    print("Usage: create_error_json.py <output_file> <error_summary> <error_log> <xpu> <n_xpus> <tp>")
    sys.exit(1)

output_file = sys.argv[1]
error_summary = sys.argv[2]
error_log = sys.argv[3]
xpu = sys.argv[4]
n_xpus = int(sys.argv[5])
tp = int(sys.argv[6])

error_data = {
    "status": "failed",
    "error_summary": error_summary,
    "error_log": error_log,
    "xpu": xpu,
    "n_xpus": n_xpus,
    "tp": tp
}

with open(output_file, 'w') as f:
    json.dump(error_data, f, indent=2)

print(f"Error JSON created: {output_file}")
