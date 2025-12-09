#!/usr/bin/env python3
"""Fix indentation error in __main__.py"""

with open('llm_inference_simulator/__main__.py', 'r') as f:
    content = f.read()

# Remove any duplicate or malformed warm-up argument
import re

# Remove any existing warm-up additions
content = re.sub(
    r"    parser\.add_argument\('--warm-up'.*?\n.*?help=.*?\)\s*\n",
    "",
    content,
    flags=re.DOTALL
)

# Find the duration argument and add warm-up properly after it
duration_pattern = r"(    parser\.add_argument\('--duration'[^)]+\))"

warmup_arg = """    parser.add_argument('--warm-up', type=float, default=0.0,
                        help='Warm-up duration in seconds before measurement (default: 0)')"""

content = re.sub(duration_pattern, r"\1\n" + warmup_arg, content)

# Update config creation to include warm_up_duration_s
if 'warm_up_duration_s' not in content:
    # Find the simulation config creation
    config_pattern = r'(simulation_duration_s=args\.duration,)'
    replacement = r'\1\n        warm_up_duration_s=args.warm_up,'
    content = re.sub(config_pattern, replacement, content)

with open('llm_inference_simulator/__main__.py', 'w') as f:
    f.write(content)

print("✓ Fixed indentation and added warm-up argument properly")
