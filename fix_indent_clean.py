#!/usr/bin/env python3
"""Clean fix for indentation."""

with open('llm_inference_simulator/__main__.py', 'r') as f:
    lines = f.readlines()

# Find and fix the problematic lines
fixed_lines = []
skip_next = False

for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    
    # Fix duration line (remove extra indent)
    if "parser.add_argument('--duration'" in line and line.startswith('        '):
        # Change 8 spaces to 4
        fixed_lines.append('    ' + line.lstrip())
    # Remove malformed warm-up if exists
    elif "parser.add_argument('--warm-up'" in line:
        # Skip this and next line
        if i + 1 < len(lines) and 'help=' in lines[i+1]:
            skip_next = True
        continue
    else:
        fixed_lines.append(line)

# Now add warm-up properly after duration
final_lines = []
for i, line in enumerate(fixed_lines):
    final_lines.append(line)
    # Add warm-up right after duration
    if "parser.add_argument('--duration'" in line:
        final_lines.append("    parser.add_argument('--warm-up', type=float, default=0.0,\n")
        final_lines.append("                        help='Warm-up duration in seconds (default: 0)')\n")

with open('llm_inference_simulator/__main__.py', 'w') as f:
    f.writelines(final_lines)

print("✓ Fixed indentation issues")
print("\nChecking result:")

# Verify
with open('llm_inference_simulator/__main__.py', 'r') as f:
    for i, line in enumerate(f, 1):
        if 'duration' in line and 'add_argument' in line:
            print(f"Line {i}: {repr(line[:50])}")
