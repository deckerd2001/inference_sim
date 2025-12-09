#!/usr/bin/env python3
"""Parse error message from simulator output."""
import sys
import re

if len(sys.argv) < 2:
    print("Unknown error")
    sys.exit(1)

output = sys.argv[1]
error_msg = "Unknown error"

if "Configuration validation failed:" in output:
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if "Configuration validation failed:" in line:
            error_lines = []
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and not lines[j].startswith('Traceback'):
                    error_lines.append(lines[j].strip())
            if error_lines:
                error_msg = ' '.join(error_lines)
                break
elif "ValueError:" in output:
    match = re.search(r'ValueError: (.+)', output)
    if match:
        error_msg = match.group(1).strip()
elif "Error:" in output:
    match = re.search(r'(\w+Error): (.+)', output)
    if match:
        error_msg = f"{match.group(1)}: {match.group(2).strip()}"

if len(error_msg) > 200:
    error_msg = error_msg[:200] + "..."

print(error_msg)
