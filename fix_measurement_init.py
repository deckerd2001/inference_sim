#!/usr/bin/env python3
"""Add measurement_start to Simulator.__init__"""

with open('llm_inference_simulator/simulator.py', 'r') as f:
    content = f.read()

# Find __init__ method and add measurement window attributes
init_pattern = r'(self\.event_log = \[\] if config\.enable_event_log else None)'

measurement_attrs = r'''\1
        
        # Measurement window (for warm-up support)
        self.measurement_start = self.config.warm_up_duration_s
        self.measurement_end = self.measurement_start + self.config.simulation_duration_s
        self.total_duration = self.measurement_end'''

import re
content = re.sub(init_pattern, measurement_attrs, content)

# Update arrival generation to use total_duration
content = content.replace(
    'while current_time < duration:',
    'while current_time < self.total_duration:'
)

# Fix the condition in run() - simplify it
content = content.replace(
    'if self.config.warm_up_duration_s > 0 and self.current_time >= self.measurement_start and self.current_time < self.measurement_start + 0.1:',
    '# Warm-up notification removed to avoid issues'
)

# Remove the measurement window start print
content = content.replace(
    'print(f"\\n📊 Measurement window started at t={self.current_time:.2f}s")\n        \n        ',
    ''
)

with open('llm_inference_simulator/simulator.py', 'w') as f:
    f.write(content)

print("✓ Added measurement_start, measurement_end, total_duration to __init__")
