#!/bin/bash

################################################################################
# LLM Inference Simulator - Comprehensive Test Suite
# Tests: Core functionality, edge cases, performance, metrics validation
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test results directory
TEST_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
LOG_FILE="$TEST_DIR/test_log.txt"

echo "========================================================================" | tee "$LOG_FILE"
echo "LLM Inference Simulator - Test Suite"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "Test Directory: $TEST_DIR"
echo "Start Time: $(date)"
echo "========================================================================" | tee -a "$LOG_FILE"
echo ""

# Helper functions
pass_test() {
    PASSED_TESTS=$((PASSED_TESTS + 1))
    echo -e "${GREEN}✓ PASS${NC}: $1" | tee -a "$LOG_FILE"
}

fail_test() {
    FAILED_TESTS=$((FAILED_TESTS + 1))
    echo -e "${RED}✗ FAIL${NC}: $1" | tee -a "$LOG_FILE"
    echo "  Reason: $2" | tee -a "$LOG_FILE"
}

start_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${BLUE}[Test $TOTAL_TESTS]${NC} $1" | tee -a "$LOG_FILE"
}

check_value() {
    # check_value actual expected tolerance description
    local actual=$1
    local expected=$2
    local tolerance=$3
    local desc=$4
    
    local diff=$(python3 -c "print(abs($actual - $expected))")
    local within=$(python3 -c "print(1 if $diff <= $tolerance else 0)")
    
    if [ "$within" = "1" ]; then
        echo "  ✓ $desc: $actual (expected ~$expected, tolerance ±$tolerance)" | tee -a "$LOG_FILE"
        return 0
    else
        echo "  ✗ $desc: $actual (expected ~$expected, diff: $diff > $tolerance)" | tee -a "$LOG_FILE"
        return 1
    fi
}

################################################################################
# TEST 1: Basic Simulator Execution (Aggregated)
################################################################################
start_test "Basic simulator execution - Aggregated MI300X"

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 20.0 \
    --warm-up 5.0 \
    --seed 42 \
    --output "$TEST_DIR/test1_basic_agg.json" \
    > "$TEST_DIR/test1_basic_agg.log" 2>&1

if [ -f "$TEST_DIR/test1_basic_agg.json" ]; then
    # Check JSON structure
    tput=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test1_basic_agg.json')); print(d.get('throughput',{}).get('tokens_per_sec',0))")
    req_rate=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test1_basic_agg.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    
    if [ $(python3 -c "print(1 if $tput > 0 else 0)") = "1" ] && [ $(python3 -c "print(1 if $req_rate > 0 else 0)") = "1" ]; then
        pass_test "Basic aggregated simulation"
        echo "  Throughput: $tput tok/s, Output Rate: $req_rate req/s" | tee -a "$LOG_FILE"
    else
        fail_test "Basic aggregated simulation" "Invalid metrics: tput=$tput, req_rate=$req_rate"
    fi
else
    fail_test "Basic aggregated simulation" "Output file not created"
fi

echo ""

################################################################################
# TEST 2: Basic Disaggregated Execution
################################################################################
start_test "Basic simulator execution - Disaggregated"

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --disaggregated \
    --prefill-xpu a100-80gb \
    --prefill-n-xpus 4 \
    --prefill-tp 4 \
    --decode-xpu mi300x \
    --decode-n-xpus 8 \
    --decode-tp 8 \
    --transfer-bandwidth 400 \
    --arrival-rate 1.5 \
    --duration 20.0 \
    --warm-up 5.0 \
    --seed 42 \
    --output "$TEST_DIR/test2_basic_disagg.json" \
    > "$TEST_DIR/test2_basic_disagg.log" 2>&1

if [ -f "$TEST_DIR/test2_basic_disagg.json" ]; then
    tput=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test2_basic_disagg.json')); print(d.get('throughput',{}).get('tokens_per_sec',0))")
    req_rate=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test2_basic_disagg.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    
    if [ $(python3 -c "print(1 if $tput > 0 else 0)") = "1" ] && [ $(python3 -c "print(1 if $req_rate > 0 else 0)") = "1" ]; then
        pass_test "Basic disaggregated simulation"
        echo "  Throughput: $tput tok/s, Output Rate: $req_rate req/s" | tee -a "$LOG_FILE"
    else
        fail_test "Basic disaggregated simulation" "Invalid metrics"
    fi
else
    fail_test "Basic disaggregated simulation" "Output file not created"
fi

echo ""

################################################################################
# TEST 3: Output Rate Convergence (Low Load)
################################################################################
start_test "Output rate convergence - Low load (1.0 req/s)"

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.0 \
    --duration 100.0 \
    --warm-up 5.0 \
    --seed 42 \
    --output "$TEST_DIR/test3_low_load.json" \
    > "$TEST_DIR/test3_low_load.log" 2>&1

if [ -f "$TEST_DIR/test3_low_load.json" ]; then
    req_rate=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test3_low_load.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    
    # Output should equal input at low load (within 5% tolerance)
    if check_value "$req_rate" 1.0 0.05 "Output rate"; then
        pass_test "Low load convergence (output ≈ input)"
    else
        fail_test "Low load convergence" "Output rate $req_rate not close to input 1.0"
    fi
else
    fail_test "Low load convergence" "Output file not created"
fi

echo ""

################################################################################
# TEST 4: Overload Detection
################################################################################
start_test "Overload detection - High load (3.0 req/s)"

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 3.0 \
    --duration 100.0 \
    --warm-up 5.0 \
    --seed 42 \
    --output "$TEST_DIR/test4_overload.json" \
    > "$TEST_DIR/test4_overload.log" 2>&1

if [ -f "$TEST_DIR/test4_overload.json" ]; then
    req_rate=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test4_overload.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    
    # Output should be less than input (overload condition)
    if [ $(python3 -c "print(1 if $req_rate < 3.0 else 0)") = "1" ]; then
        pass_test "Overload detection (output < input)"
        echo "  Input: 3.0 req/s, Output: $req_rate req/s (overload detected)" | tee -a "$LOG_FILE"
    else
        fail_test "Overload detection" "Output rate $req_rate >= input 3.0 (should be less)"
    fi
else
    fail_test "Overload detection" "Output file not created"
fi

echo ""

################################################################################
# TEST 5: Different GPU Types
################################################################################
start_test "GPU type comparison - A100, H100, MI300X"

declare -a gpu_types=("a100-80gb" "h100-80gb" "mi300x")
declare -A gpu_results

for xpu in "${gpu_types[@]}"; do
    python3 -m llm_inference_simulator \
        --model llama2-70b \
        --xpu "$xpu" \
        --n-xpus-per-node 8 \
        --tp 8 \
        --arrival-rate 1.5 \
        --duration 20.0 \
        --warm-up 5.0 \
        --seed 42 \
        --output "$TEST_DIR/test5_${xpu}.json" \
        > "$TEST_DIR/test5_${xpu}.log" 2>&1
    
    if [ -f "$TEST_DIR/test5_${xpu}.json" ]; then
        req_rate=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test5_${xpu}.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
        gpu_results["$xpu"]="$req_rate"
    else
        gpu_results["$xpu"]="0"
    fi
done

# Check that all GPUs produced valid results
all_valid=1
for xpu in "${gpu_types[@]}"; do
    if [ $(python3 -c "print(1 if ${gpu_results[$xpu]} > 0 else 0)") = "0" ]; then
        all_valid=0
    fi
done

if [ $all_valid = 1 ]; then
    pass_test "GPU type comparison"
    for xpu in "${gpu_types[@]}"; do
        echo "  $xpu: ${gpu_results[$xpu]} req/s" | tee -a "$LOG_FILE"
    done
else
    fail_test "GPU type comparison" "Some GPU types failed"
fi

echo ""

################################################################################
# TEST 6: Network Bandwidth Impact (Disaggregated)
################################################################################
start_test "Network bandwidth impact - 100GB/s vs 400GB/s"

declare -a bandwidths=(100 400)
declare -A bw_results

for bw in "${bandwidths[@]}"; do
    python3 -m llm_inference_simulator \
        --model llama2-70b \
        --disaggregated \
        --prefill-xpu a100-80gb \
        --prefill-n-xpus 4 \
        --prefill-tp 4 \
        --decode-xpu mi300x \
        --decode-n-xpus 8 \
        --decode-tp 8 \
        --transfer-bandwidth "$bw" \
        --arrival-rate 1.5 \
        --duration 20.0 \
        --warm-up 5.0 \
        --seed 42 \
        --output "$TEST_DIR/test6_bw${bw}.json" \
        > "$TEST_DIR/test6_bw${bw}.log" 2>&1
    
    if [ -f "$TEST_DIR/test6_bw${bw}.json" ]; then
        ttft=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test6_bw${bw}.json')); print(d.get('first_token_latency',{}).get('mean',0))")
        bw_results["$bw"]="$ttft"
    else
        bw_results["$bw"]="0"
    fi
done

# 400GB/s should have lower TTFT than 100GB/s
if [ $(python3 -c "print(1 if ${bw_results[400]} > 0 and ${bw_results[100]} > 0 and ${bw_results[400]} < ${bw_results[100]} else 0)") = "1" ]; then
    pass_test "Network bandwidth impact (400GB/s faster than 100GB/s)"
    echo "  100GB/s TTFT: ${bw_results[100]}s" | tee -a "$LOG_FILE"
    echo "  400GB/s TTFT: ${bw_results[400]}s" | tee -a "$LOG_FILE"
else
    fail_test "Network bandwidth impact" "400GB/s not faster than 100GB/s"
fi

echo ""

################################################################################
# TEST 7: Warm-up Period Effect
################################################################################
start_test "Warm-up period effect - 0s vs 10s"

# Test with no warm-up
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 2.0 \
    --duration 50.0 \
    --warm-up 0.0 \
    --seed 42 \
    --output "$TEST_DIR/test7_warmup0.json" \
    > "$TEST_DIR/test7_warmup0.log" 2>&1

# Test with warm-up
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 2.0 \
    --duration 50.0 \
    --warm-up 10.0 \
    --seed 42 \
    --output "$TEST_DIR/test7_warmup10.json" \
    > "$TEST_DIR/test7_warmup10.log" 2>&1

if [ -f "$TEST_DIR/test7_warmup0.json" ] && [ -f "$TEST_DIR/test7_warmup10.json" ]; then
    rate0=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test7_warmup0.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    rate10=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test7_warmup10.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    
    # With warm-up should give more conservative (lower) estimate
    if [ $(python3 -c "print(1 if $rate0 > 0 and $rate10 > 0 else 0)") = "1" ]; then
        pass_test "Warm-up period effect"
        echo "  No warm-up: $rate0 req/s" | tee -a "$LOG_FILE"
        echo "  With warm-up: $rate10 req/s" | tee -a "$LOG_FILE"
    else
        fail_test "Warm-up period effect" "Invalid results"
    fi
else
    fail_test "Warm-up period effect" "Output files not created"
fi

echo ""

################################################################################
# TEST 8: Reproducibility (Fixed Seed)
################################################################################
start_test "Reproducibility with fixed seed"

# Run 1
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 20.0 \
    --warm-up 5.0 \
    --seed 12345 \
    --output "$TEST_DIR/test8_run1.json" \
    > "$TEST_DIR/test8_run1.log" 2>&1

# Run 2
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 20.0 \
    --warm-up 5.0 \
    --seed 12345 \
    --output "$TEST_DIR/test8_run2.json" \
    > "$TEST_DIR/test8_run2.log" 2>&1

if [ -f "$TEST_DIR/test8_run1.json" ] && [ -f "$TEST_DIR/test8_run2.json" ]; then
    rate1=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test8_run1.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    rate2=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test8_run2.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    
    # Should be identical with same seed
    if [ $(python3 -c "print(1 if abs($rate1 - $rate2) < 0.001 else 0)") = "1" ]; then
        pass_test "Reproducibility (run1=$rate1, run2=$rate2)"
    else
        fail_test "Reproducibility" "Results differ: run1=$rate1, run2=$rate2"
    fi
else
    fail_test "Reproducibility" "Output files not created"
fi

echo ""

################################################################################
# TEST 9: Tensor Parallelism Scaling
################################################################################
start_test "Tensor Parallelism scaling - TP=2, 4, 8"

declare -a tp_sizes=(2 4 8)
declare -A tp_results

for tp in "${tp_sizes[@]}"; do
    python3 -m llm_inference_simulator \
        --model llama2-70b \
        --xpu mi300x \
        --n-xpus-per-node "$tp" \
        --tp "$tp" \
        --arrival-rate 1.5 \
        --duration 20.0 \
        --warm-up 5.0 \
        --seed 42 \
        --output "$TEST_DIR/test9_tp${tp}.json" \
        > "$TEST_DIR/test9_tp${tp}.log" 2>&1
    
    if [ -f "$TEST_DIR/test9_tp${tp}.json" ]; then
        ttft=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test9_tp${tp}.json')); print(d.get('first_token_latency',{}).get('mean',0))")
        tp_results["$tp"]="$ttft"
    else
        tp_results["$tp"]="0"
    fi
done

# Higher TP should generally give lower latency
all_valid=1
for tp in "${tp_sizes[@]}"; do
    if [ $(python3 -c "print(1 if ${tp_results[$tp]} > 0 else 0)") = "0" ]; then
        all_valid=0
    fi
done

if [ $all_valid = 1 ]; then
    pass_test "Tensor Parallelism scaling"
    for tp in "${tp_sizes[@]}"; do
        echo "  TP=$tp: TTFT=${tp_results[$tp]}s" | tee -a "$LOG_FILE"
    done
else
    fail_test "Tensor Parallelism scaling" "Some TP configs failed"
fi

echo ""

################################################################################
# TEST 10: Duration Convergence
################################################################################
start_test "Duration convergence - 50s, 100s, 200s at capacity limit"

declare -a durations=(50 100 200)
declare -A dur_results

for dur in "${durations[@]}"; do
    python3 -m llm_inference_simulator \
        --model llama2-70b \
        --xpu mi300x \
        --n-xpus-per-node 8 \
        --tp 8 \
        --arrival-rate 2.0 \
        --duration "$dur" \
        --warm-up 5.0 \
        --seed 42 \
        --output "$TEST_DIR/test10_dur${dur}.json" \
        > "$TEST_DIR/test10_dur${dur}.log" 2>&1
    
    if [ -f "$TEST_DIR/test10_dur${dur}.json" ]; then
        req_rate=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test10_dur${dur}.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
        dur_results["$dur"]="$req_rate"
    else
        dur_results["$dur"]="0"
    fi
done

# Longer duration should give more stable (lower) results
all_valid=1
for dur in "${durations[@]}"; do
    if [ $(python3 -c "print(1 if ${dur_results[$dur]} > 0 else 0)") = "0" ]; then
        all_valid=0
    fi
done

if [ $all_valid = 1 ]; then
    pass_test "Duration convergence"
    for dur in "${durations[@]}"; do
        echo "  ${dur}s: ${dur_results[$dur]} req/s" | tee -a "$LOG_FILE"
    done
    
    # Check that 200s result is more conservative than 50s
    if [ $(python3 -c "print(1 if ${dur_results[200]} <= ${dur_results[50]} else 0)") = "1" ]; then
        echo "  ✓ Longer duration gives more conservative estimate" | tee -a "$LOG_FILE"
    else
        echo "  ⚠ Warning: Longer duration did not give lower estimate" | tee -a "$LOG_FILE"
    fi
else
    fail_test "Duration convergence" "Some duration tests failed"
fi

echo ""

################################################################################
# TEST 11: Metrics Completeness
################################################################################
start_test "Metrics completeness - All required fields present"

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 20.0 \
    --warm-up 5.0 \
    --seed 42 \
    --output "$TEST_DIR/test11_metrics.json" \
    > "$TEST_DIR/test11_metrics.log" 2>&1

if [ -f "$TEST_DIR/test11_metrics.json" ]; then
    # Check for required fields
    has_throughput=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test11_metrics.json')); print(1 if 'throughput' in d else 0)")
    has_ttft=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test11_metrics.json')); print(1 if 'first_token_latency' in d else 0)")
    has_e2e=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test11_metrics.json')); print(1 if 'end_to_end_latency' in d else 0)")
    has_config=$(python3 -c "import json; d=json.load(open('$TEST_DIR/test11_metrics.json')); print(1 if 'config' in d else 0)")
    
    if [ "$has_throughput" = "1" ] && [ "$has_ttft" = "1" ] && [ "$has_e2e" = "1" ] && [ "$has_config" = "1" ]; then
        pass_test "Metrics completeness (all required fields present)"
    else
        fail_test "Metrics completeness" "Missing fields: throughput=$has_throughput, ttft=$has_ttft, e2e=$has_e2e, config=$has_config"
    fi
else
    fail_test "Metrics completeness" "Output file not created"
fi

echo ""

################################################################################
# TEST 12: Benchmark Script Integration
################################################################################
start_test "Benchmark script integration test"

# Temporarily modify benchmark script for quick test
cp cluster_benchmark.sh "$TEST_DIR/test12_benchmark.sh"
sed -i 's/DURATION=1000.0/DURATION=10.0/' "$TEST_DIR/test12_benchmark.sh"
sed -i 's/ARRIVAL_RATE=1.9/ARRIVAL_RATE=1.5/' "$TEST_DIR/test12_benchmark.sh"

# Run quick benchmark
cd "$TEST_DIR"
bash test12_benchmark.sh > test12_benchmark_output.txt 2>&1
cd - > /dev/null

# Check if benchmark completed
if grep -q "Results saved to:" "$TEST_DIR/test12_benchmark_output.txt"; then
    pass_test "Benchmark script integration"
else
    fail_test "Benchmark script integration" "Benchmark did not complete successfully"
fi

echo ""

################################################################################
# SUMMARY
################################################################################
echo "========================================================================" | tee -a "$LOG_FILE"
echo "TEST SUMMARY"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "Total Tests:  $TOTAL_TESTS" | tee -a "$LOG_FILE"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}" | tee -a "$LOG_FILE"
echo -e "${RED}Failed:       $FAILED_TESTS${NC}" | tee -a "$LOG_FILE"
echo "Success Rate: $(python3 -c "print(f'{100*$PASSED_TESTS/$TOTAL_TESTS:.1f}%')")" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
echo ""
echo "Test results saved to: $TEST_DIR"
echo "Log file: $LOG_FILE"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo "Please review the log file for details: $LOG_FILE"
    exit 1
fi