#!/usr/bin/env python3
"""
Debug script to trace simulator behavior
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm_inference_simulator.config import SimulationConfig, WorkloadSpec, ClusterSpec
from llm_inference_simulator.simulator import Simulator
import numpy as np

def debug_simulation():
    """Run a debug simulation with detailed logging."""
    
    # Simple config
    config = SimulationConfig(
        cluster_spec=ClusterSpec(
            model_name="llama2-70b",
            xpu_type="mi300x",
            n_xpus_per_node=8,
            tensor_parallelism=8,
            is_disaggregated=False
        ),
        workload_spec=WorkloadSpec(
            arrival_rate=1.0,
            avg_input_length=512,
            max_input_length=1024,
            input_length_std=50,
            avg_output_length=192,
            max_output_length=256,
            output_length_std=20,
            arrival_process="poisson"
        ),
        simulation_duration_s=100.0,  # Short test
        warm_up_duration_s=10.0,
        random_seed=42
    )
    
    print("=" * 80)
    print("DEBUG SIMULATION")
    print("=" * 80)
    print(f"Arrival rate: {config.workload_spec.arrival_rate} req/s")
    print(f"Duration: {config.simulation_duration_s}s")
    print(f"Warm-up: {config.warm_up_duration_s}s")
    print()
    
    # Create simulator
    sim = Simulator(config)
    
    # Add debug hooks
    original_handle_arrived = sim._handle_request_arrived
    original_handle_finished = sim._handle_request_finished
    
    arrived_count = 0
    finished_count = 0
    arrived_times = []
    finished_times = []
    
    def debug_arrived(event):
        nonlocal arrived_count
        arrived_count += 1
        arrived_times.append(event.timestamp)
        result = original_handle_arrived(event)
        if arrived_count <= 5 or arrived_count % 20 == 0:
            print(f"[ARRIVED] #{arrived_count} at t={event.timestamp:.2f}s")
        return result
    
    def debug_finished(event):
        nonlocal finished_count
        finished_count += 1
        finished_times.append(sim.current_time)
        result = original_handle_finished(event)
        if finished_count <= 5 or finished_count % 20 == 0:
            print(f"[FINISHED] #{finished_count} at t={sim.current_time:.2f}s")
        return result
    
    sim._handle_request_arrived = debug_arrived
    sim._handle_request_finished = debug_finished
    
    # Run simulation
    print("\n--- Starting simulation ---\n")
    sim.run()
    print("\n--- Simulation ended ---\n")
    
    # Analysis
    print("=" * 80)
    print("DEBUG ANALYSIS")
    print("=" * 80)
    
    print("\n1. TIMING:")
    print(f"   Measurement start: {sim.measurement_start:.2f}s")
    print(f"   Measurement end: {sim.measurement_end:.2f}s")
    print(f"   Total duration: {sim.total_duration:.2f}s")
    print(f"   Current time (end): {sim.current_time:.2f}s")
    
    print("\n2. ARRIVALS:")
    print(f"   Total arrived: {arrived_count}")
    print(f"   First arrival: {min(arrived_times):.2f}s" if arrived_times else "None")
    print(f"   Last arrival: {max(arrived_times):.2f}s" if arrived_times else "None")
    print(f"   In measurement window: {sum(1 for t in arrived_times if sim.measurement_start <= t <= sim.measurement_end)}")
    
    print("\n3. COMPLETIONS:")
    print(f"   Total finished: {finished_count}")
    print(f"   First finish: {min(finished_times):.2f}s" if finished_times else "None")
    print(f"   Last finish: {max(finished_times):.2f}s" if finished_times else "None")
    print(f"   After measurement_end: {sum(1 for t in finished_times if t > sim.measurement_end)}")
    
    print("\n4. METRICS:")
    print(f"   metrics.total_requests: {sim.metrics.total_requests}")
    print(f"   metrics.completed_requests: {sim.metrics.completed_requests}")
    print(f"   metrics.simulation_time: {sim.metrics.total_simulation_time:.2f}s")
    
    print("\n5. MISSING:")
    in_window = sum(1 for t in arrived_times if sim.measurement_start <= t <= sim.measurement_end)
    missing = in_window - sim.metrics.completed_requests
    print(f"   Arrived in window: {in_window}")
    print(f"   Counted completed: {sim.metrics.completed_requests}")
    print(f"   Missing: {missing} ({missing/in_window*100:.1f}%)" if in_window > 0 else "N/A")
    
    print("\n6. QUEUE STATUS:")
    if hasattr(sim.scheduler, 'prefill_queue'):
        print(f"   Prefill queue: {len(sim.scheduler.prefill_queue)}")
    if hasattr(sim.scheduler, 'decode_queue'):
        print(f"   Decode queue: {len(sim.scheduler.decode_queue)}")
    
    print("\n7. GPU STATUS:")
    print(f"   Is busy: {sim.is_gpu_busy}")
    print(f"   Current batch: {sim.current_batch is not None}")
    
    print("\n8. EVENT QUEUE:")
    print(f"   Events remaining: {len(sim.event_queue)}")
    
    print("\n9. ARRIVAL TIME CHECK:")
    # Check if counting uses arrival_time or completion_time
    print("   Checking _handle_request_finished code...")
    import inspect
    source = inspect.getsource(sim._handle_request_finished)
    if "arrival_time >= self.measurement_start" in source:
        print("   ✓ Uses arrival_time (CORRECT)")
    elif "completion_time >= self.measurement_start" in source:
        print("   ✗ Uses completion_time (WRONG)")
    else:
        print("   ? Cannot determine")
    
    print("\n" + "=" * 80)
    
    # Return detailed info
    return {
        'arrived_count': arrived_count,
        'finished_count': finished_count,
        'arrived_times': arrived_times,
        'finished_times': finished_times,
        'measurement_start': sim.measurement_start,
        'measurement_end': sim.measurement_end,
        'total_duration': sim.total_duration,
        'current_time': sim.current_time,
        'metrics': sim.metrics
    }

if __name__ == "__main__":
    result = debug_simulation()
    
    print("\nDEBUG COMPLETE!")
    print("\nKey findings:")
    print(f"1. Simulation ended at t={result['current_time']:.2f}s (expected {result['total_duration']:.2f}s)")
    print(f"2. Total arrived: {result['arrived_count']}, finished: {result['finished_count']}")
    print(f"3. Metrics counted: {result['metrics'].completed_requests}")
    
    if result['finished_count'] != result['metrics'].completed_requests:
        print(f"\n⚠️  DISCREPANCY: {result['finished_count']} finished but only {result['metrics'].completed_requests} counted!")
        print("   → Check measurement window logic")
    
    if result['current_time'] < result['total_duration'] - 1:
        print(f"\n⚠️  EARLY TERMINATION: Ended at {result['current_time']:.2f}s, expected {result['total_duration']:.2f}s")
        print("   → Check run() loop exit condition")
