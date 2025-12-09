#!/usr/bin/env python3
"""Fix indentation in simulator.py"""

with open('llm_inference_simulator/simulator.py', 'r') as f:
    lines = f.readlines()

# Find and fix the problematic area around line 207
fixed_lines = []
for i, line in enumerate(lines):
    # Skip empty comment lines
    if line.strip() == '# Warm-up notification removed to avoid issues':
        continue
    # Fix any line that starts with unexpected indent after removal
    if i > 0 and line.strip().startswith('print(f"\\nSimulation completed'):
        # Check previous line
        prev_line = lines[i-1] if i > 0 else ''
        # If previous was removed, this needs proper indent
        if prev_line.strip() == '' or prev_line.strip().startswith('#'):
            # Should be at method level (8 spaces)
            fixed_lines.append('        ' + line.lstrip())
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

with open('llm_inference_simulator/simulator.py', 'w') as f:
    f.writelines(fixed_lines)

print("✓ Fixed indentation")

# Verify the area
print("\nChecking lines around 'Simulation completed':")
with open('llm_inference_simulator/simulator.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if 'Simulation completed' in line:
            print(f"Line {i-1}: {repr(lines[i-2][:60]) if i > 1 else 'N/A'}")
            print(f"Line {i}: {repr(line[:60])}")
            print(f"Line {i+1}: {repr(lines[i][:60]) if i < len(lines) else 'N/A'}")
            break
