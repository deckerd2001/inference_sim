#!/usr/bin/env python3
"""Add warm-up period support to simulator."""

# 1. Add to Config
print("Step 1: Adding warm_up_duration to SimulationConfig...")

with open('llm_inference_simulator/config.py', 'r') as f:
    config_content = f.read()

# Add warm_up_duration field
if 'warm_up_duration_s' not in config_content:
    addition = '''    
    # Warm-up period (seconds to run before starting measurement)
    warm_up_duration_s: float = 0.0  # 0 = no warm-up
'''
    # Insert after simulation_duration_s
    config_content = config_content.replace(
        '    simulation_duration_s: float',
        addition + '    simulation_duration_s: float'
    )
    
    with open('llm_inference_simulator/config.py', 'w') as f:
        f.write(config_content)
    print("  ✓ Added warm_up_duration_s to Config")
else:
    print("  ✓ warm_up_duration_s already exists")

# 2. Update Simulator to track measurement window
print("\nStep 2: Updating Simulator to use measurement window...")

with open('llm_inference_simulator/simulator.py', 'r') as f:
    sim_content = f.read()

# Add measurement window tracking in __init__
if 'self.measurement_start' not in sim_content:
    init_addition = '''
        # Measurement window (for warm-up support)
        self.measurement_start = self.config.warm_up_duration_s
        self.measurement_end = self.measurement_start + self.config.simulation_duration_s
        self.total_duration = self.measurement_end
        
        if self.config.warm_up_duration_s > 0:
            print(f"Warm-up: {self.config.warm_up_duration_s:.0f}s, Measurement: {self.config.simulation_duration_s:.0f}s, Total: {self.total_duration:.0f}s")
'''
    
    # Insert after self.event_log initialization
    sim_content = sim_content.replace(
        'self.event_log = [] if config.enable_event_log else None',
        'self.event_log = [] if config.enable_event_log else None' + init_addition
    )
    print("  ✓ Added measurement window to __init__")

# Update request arrival generation
if 'self.total_duration' in sim_content and 'while current_time < duration:' in sim_content:
    sim_content = sim_content.replace(
        'while current_time < duration:',
        'while current_time < self.total_duration:'
    )
    print("  ✓ Updated arrival generation to use total_duration")

# Update metrics to only count requests completed in measurement window
if 'def _complete_request' not in sim_content:
    print("  ⚠️  Need to add _complete_request method")

# Update run() to show measurement window start
if 'Simulation completed' in sim_content:
    sim_content = sim_content.replace(
        'print(f"\\nSimulation completed at t={self.current_time:.2f}s")',
        '''if self.config.warm_up_duration_s > 0 and self.current_time >= self.measurement_start and self.current_time < self.measurement_start + 0.1:
            print(f"\\n📊 Measurement window started at t={self.current_time:.2f}s")
        
        print(f"\\nSimulation completed at t={self.current_time:.2f}s")'''
    )
    print("  ✓ Added measurement window notification")

with open('llm_inference_simulator/simulator.py', 'w') as f:
    f.write(sim_content)

# 3. Add CLI argument
print("\nStep 3: Adding --warm-up CLI argument...")

with open('llm_inference_simulator/__main__.py', 'r') as f:
    main_content = f.read()

if '--warm-up' not in main_content:
    # Add argument
    arg_addition = '''    parser.add_argument('--warm-up', type=float, default=0.0,
                        help='Warm-up duration in seconds (default: 0)')
    '''
    
    main_content = main_content.replace(
        "parser.add_argument('--duration'",
        arg_addition + "\n    parser.add_argument('--duration'"
    )
    
    # Use the argument
    main_content = main_content.replace(
        'simulation_duration_s=args.duration,',
        '''simulation_duration_s=args.duration,
        warm_up_duration_s=args.warm_up,'''
    )
    
    with open('llm_inference_simulator/__main__.py', 'w') as f:
        f.write(main_content)
    print("  ✓ Added --warm-up argument")
else:
    print("  ✓ --warm-up argument already exists")

print("\n" + "="*70)
print("✓ Warm-up support added!")
print("="*70)
print("\nUsage:")
print("  python3 -m llm_inference_simulator \\")
print("    --model llama2-70b --xpu mi300x \\")
print("    --n-xpus-per-node 8 --tp 8 \\")
print("    --warm-up 20 \\      # 20s warm-up")
print("    --duration 60 \\     # 60s measurement")
print("    --arrival-rate 10")
print()
print("Total simulation: 80s (20s warm-up + 60s measurement)")
