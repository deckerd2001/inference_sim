#!/bin/bash

################################################################################
# Quick Test Suite - 5분 안에 핵심 기능 검증
# 전체 test.sh의 짧은 버전 (빠른 검증용)
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

TEST_DIR="quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "========================================================================"
echo "Quick Test Suite (5 minutes)"
echo "========================================================================"
echo ""

pass_test() {
    PASSED_TESTS=$((PASSED_TESTS + 1))
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

fail_test() {
    FAILED_TESTS=$((FAILED_TESTS + 1))
    echo -e "${RED}✗ FAIL${NC}: $1"
    echo "  Reason: $2"
}

start_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${BLUE}[Test $TOTAL_TESTS]${NC} $1"
}

# Test 1: Basic Execution
start_test "Basic simulation (10s)"
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 10.0 \
    --warm-up 2.0 \
    --seed 42 \
    --output "$TEST_DIR/quick1.json" \
    > "$TEST_DIR/quick1.log" 2>&1

if [ -f "$TEST_DIR/quick1.json" ]; then
    pass_test "Basic simulation"
else
    fail_test "Basic simulation" "No output"
fi

# Test 2: Disaggregated
start_test "Disaggregated (10s)"
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
    --duration 10.0 \
    --warm-up 2.0 \
    --seed 42 \
    --output "$TEST_DIR/quick2.json" \
    > "$TEST_DIR/quick2.log" 2>&1

if [ -f "$TEST_DIR/quick2.json" ]; then
    pass_test "Disaggregated"
else
    fail_test "Disaggregated" "No output"
fi

# Test 3: Overload Detection
start_test "Overload detection (10s)"
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 3.0 \
    --duration 10.0 \
    --warm-up 2.0 \
    --seed 42 \
    --output "$TEST_DIR/quick3.json" \
    > "$TEST_DIR/quick3.log" 2>&1

if [ -f "$TEST_DIR/quick3.json" ]; then
    req_rate=$(python3 -c "import json; d=json.load(open('$TEST_DIR/quick3.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    if [ $(python3 -c "print(1 if $req_rate < 3.0 else 0)") = "1" ]; then
        pass_test "Overload detection (output=$req_rate < input=3.0)"
    else
        fail_test "Overload detection" "Output $req_rate >= 3.0"
    fi
else
    fail_test "Overload detection" "No output"
fi

# Test 4: Reproducibility
start_test "Reproducibility (10s × 2)"
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 10.0 \
    --warm-up 2.0 \
    --seed 99999 \
    --output "$TEST_DIR/quick4a.json" \
    > "$TEST_DIR/quick4a.log" 2>&1

python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 10.0 \
    --warm-up 2.0 \
    --seed 99999 \
    --output "$TEST_DIR/quick4b.json" \
    > "$TEST_DIR/quick4b.log" 2>&1

if [ -f "$TEST_DIR/quick4a.json" ] && [ -f "$TEST_DIR/quick4b.json" ]; then
    rate_a=$(python3 -c "import json; d=json.load(open('$TEST_DIR/quick4a.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    rate_b=$(python3 -c "import json; d=json.load(open('$TEST_DIR/quick4b.json')); print(d.get('throughput',{}).get('requests_per_sec',0))")
    
    if [ $(python3 -c "print(1 if abs($rate_a - $rate_b) < 0.001 else 0)") = "1" ]; then
        pass_test "Reproducibility (a=$rate_a, b=$rate_b)"
    else
        fail_test "Reproducibility" "Differ: a=$rate_a, b=$rate_b"
    fi
else
    fail_test "Reproducibility" "No output"
fi

# Test 5: Metrics Completeness
start_test "Metrics completeness (10s)"
python3 -m llm_inference_simulator \
    --model llama2-70b \
    --xpu mi300x \
    --n-xpus-per-node 8 \
    --tp 8 \
    --arrival-rate 1.5 \
    --duration 10.0 \
    --warm-up 2.0 \
    --seed 42 \
    --output "$TEST_DIR/quick5.json" \
    > "$TEST_DIR/quick5.log" 2>&1

if [ -f "$TEST_DIR/quick5.json" ]; then
    has_all=$(python3 -c "
import json
d = json.load(open('$TEST_DIR/quick5.json'))
required = ['throughput', 'first_token_latency', 'end_to_end_latency', 'config']
print(1 if all(k in d for k in required) else 0)
")
    if [ "$has_all" = "1" ]; then
        pass_test "Metrics completeness"
    else
        fail_test "Metrics completeness" "Missing fields"
    fi
else
    fail_test "Metrics completeness" "No output"
fi

echo ""
echo "========================================================================"
echo "QUICK TEST SUMMARY"
echo "========================================================================"
echo "Total Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
echo "Success Rate: $(python3 -c "print(f'{100*$PASSED_TESTS/$TOTAL_TESTS:.1f}%')")"
echo "========================================================================"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL QUICK TESTS PASSED!${NC}"
    echo "Run './test.sh' for comprehensive testing"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi