#!/usr/bin/env python3
"""Update metrics to only count requests completed in measurement window."""

with open('llm_inference_simulator/simulator.py', 'r') as f:
    content = f.read()

# Find where requests are marked as completed
# Update to check if completion time is in measurement window

update_code = '''
    def _finalize_request(self, request):
        """Finalize a completed request (only count if in measurement window)."""
        # Only count requests completed during measurement window
        if self.current_time >= self.measurement_start and self.current_time <= self.measurement_end:
            self.metrics.completed_requests += 1
            self.metrics.total_tokens_generated += request.tokens_generated
            
            if request.first_token_latency:
                self.metrics.first_token_latencies.append(request.first_token_latency)
            if request.end_to_end_latency:
                self.metrics.end_to_end_latencies.append(request.end_to_end_latency)
        
        # Always update total simulation time
        if self.current_time > self.measurement_start:
            self.metrics.total_simulation_time = min(
                self.current_time - self.measurement_start,
                self.config.simulation_duration_s
            )
'''

print("Manual update required:")
print("  - Modify request completion to check measurement window")
print("  - Update metrics.total_simulation_time calculation")
print("\nThis is complex - let's test warm-up first, then refine metrics")
