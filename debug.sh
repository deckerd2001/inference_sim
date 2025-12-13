#!/bin/bash
# Quick verification script

echo "======================================"
echo "VERIFICATION: Are fixes applied?"
echo "======================================"
echo

echo "1. Check arrival_time fix in simulator.py:"
echo "---"
if grep -q "arrival_time >= self.measurement_start" llm_inference_simulator/simulator.py; then
    echo "✓ FOUND: arrival_time >= self.measurement_start"
    echo "  Fix is APPLIED"
else
    echo "✗ NOT FOUND"
    echo "  Fix is NOT applied"
    echo "  Looking for completion_time instead..."
    grep -n "completion_time >= self.measurement_start" llm_inference_simulator/simulator.py | head -2
fi
echo

echo "2. Check cooldown in simulator.py:"
echo "---"
grep -A 2 "cooldown_duration" llm_inference_simulator/simulator.py | head -5
echo

echo "3. Check measurement_end in _schedule_initial_arrivals:"
echo "---"
if grep -q "current_time < self.measurement_end" llm_inference_simulator/simulator.py; then
    echo "✓ FOUND: while current_time < self.measurement_end"
    echo "  Fix is APPLIED"
else
    echo "✗ NOT FOUND"
    echo "  May still use total_duration"
fi
echo

echo "4. Check decode activation memory in memory_manager.py:"
echo "---"
if grep -q "seq_length=1" llm_inference_simulator/memory_manager.py; then
    echo "✓ FOUND: seq_length=1"
    echo "  Gemini fix is APPLIED"
else
    echo "✗ NOT FOUND"
    echo "  May still use max_kv_length"
fi
echo

echo "5. Check KV cache initialization in simulator.py:"
echo "---"
if grep -q "current_kv_cache_length = req.input_length" llm_inference_simulator/simulator.py; then
    echo "✓ FOUND: current_kv_cache_length = req.input_length"
    count=$(grep -c "current_kv_cache_length = req.input_length" llm_inference_simulator/simulator.py)
    echo "  Found $count occurrence(s)"
    if [ $count -ge 2 ]; then
        echo "  ChatGPT fix is APPLIED (aggregated + disaggregated)"
    else
        echo "  Partial - check if aggregated mode is fixed"
    fi
else
    echo "✗ NOT FOUND"
    echo "  ChatGPT fix is NOT applied"
fi
echo

echo "======================================"
echo "SUMMARY"
echo "======================================"
echo
echo "Run this to apply all fixes:"
echo "  cp simulator_final.py llm_inference_simulator/simulator.py"
echo "  cp memory_manager_fixed.py llm_inference_simulator/memory_manager.py"
echo "  cp request_fixed.py llm_inference_simulator/request.py"
echo
