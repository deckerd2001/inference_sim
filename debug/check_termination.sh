#!/bin/bash

echo "======================================================================="
echo "SIMULATION TERMINATION LOGIC CHECK"
echo "======================================================================="
echo ""

echo "1. Main simulation loop..."
grep -B 3 -A 25 "def run" llm_inference_simulator/simulator.py | grep -A 20 "while self.event_queue"

echo ""
echo "2. Event processing - duration check..."
grep -B 5 -A 10 "simulation_duration\|current_time.*duration" llm_inference_simulator/simulator.py

echo ""
echo "3. How are in-flight requests handled at termination?"
grep -B 5 -A 10 "Simulation completed\|finalize\|cleanup" llm_inference_simulator/simulator.py

