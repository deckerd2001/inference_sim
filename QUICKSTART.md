# LLM Inference Simulator - 퀵스타트 가이드

## 설치

```bash
# 의존성 설치
pip install numpy

# 패키지 디렉토리로 이동
cd llm_inference_simulator
```

## 5분 튜토리얼

### 1. 기본 시뮬레이션

가장 간단한 예제부터 시작합니다:

```python
from llm_inference_simulator import *

# 설정 생성
config = SimulatorConfig(
    model_spec=ModelSpec(
        name="llama-7b",
        n_params=7_000_000_000,
        hidden_size=4096,
        n_layers=32,
        n_heads=32,
        ffn_dim=11008,
    ),
    workload_spec=WorkloadSpec(
        avg_input_length=512,
        avg_output_length=128,
        arrival_rate=2.0,  # 2 requests/sec
    ),
    cluster_spec=ClusterSpec(
        n_gpus_per_node=1,
        n_nodes=1,
        gpu_spec=GPUSpec(name="A100-80GB"),
    ),
    simulation_duration_s=60.0,
)

# 실행
simulator = LLMInferenceSimulator(config)
metrics = simulator.run()

# 결과 확인
stats = metrics.compute_statistics()
print(f"Throughput: {stats['throughput_tokens_per_sec']:.2f} tokens/sec")
```

### 2. 성능 비교

두 가지 설정을 비교하려면:

```python
# 설정 1: Single GPU
config1 = SimulatorConfig(...)
simulator1 = LLMInferenceSimulator(config1)
metrics1 = simulator1.run()

# 설정 2: TP=4
config2 = config1
config2.parallelism_spec = ParallelismSpec(tensor_parallel_size=4)
config2.cluster_spec.n_gpus_per_node = 4
simulator2 = LLMInferenceSimulator(config2)
metrics2 = simulator2.run()

# 비교
stats1 = metrics1.compute_statistics()
stats2 = metrics2.compute_statistics()
print(f"Speedup: {stats2['throughput_tokens_per_sec'] / stats1['throughput_tokens_per_sec']:.2f}x")
```

### 3. 다양한 워크로드 테스트

부하가 다른 시나리오를 테스트:

```python
for arrival_rate in [1, 5, 10, 20]:
    config.workload_spec.arrival_rate = arrival_rate
    
    simulator = LLMInferenceSimulator(config)
    metrics = simulator.run()
    stats = metrics.compute_statistics()
    
    print(f"Load: {arrival_rate} req/s")
    print(f"  Throughput: {stats['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  P95 Latency: {stats['first_token_latency']['p95']:.4f}s")
    print()
```

## 주요 파라미터 설명

### ModelSpec
- `n_params`: 모델 파라미터 수 (7B, 13B, 70B 등)
- `hidden_size`: Hidden dimension (보통 4096, 8192 등)
- `n_layers`: Transformer layer 수
- `n_heads`: Attention head 수
- `ffn_dim`: FFN dimension (보통 4 × hidden_size)

### WorkloadSpec
- `avg_input_length`: 평균 입력 토큰 수
- `avg_output_length`: 평균 출력 토큰 수
- `arrival_rate`: 초당 요청 수
- `arrival_process`: "poisson" 또는 "deterministic"

### GPUSpec
- `name`: GPU 이름 (표시용)
- `compute_tflops`: BF16/FP16 TFLOPS
- `memory_size_gb`: GPU 메모리 크기
- `memory_bandwidth_gbs`: HBM 대역폭

### ParallelismSpec
- `tensor_parallel_size`: Tensor parallelism degree
- `data_parallel_size`: Data parallelism degree
- `pipeline_parallel_size`: Pipeline parallelism degree

### SchedulerSpec
- `batching_type`: "static", "dynamic", "continuous"
- `max_batch_size`: 최대 배치 크기
- `token_level_scheduling`: 토큰 단위 스케줄링 여부

## 일반적인 사용 사례

### 사례 1: GPU 선택

어떤 GPU를 사용할지 결정:

```python
gpus = [
    ("A100-80GB", 312.0, 80.0, 2039.0),
    ("H100-80GB", 989.0, 80.0, 3350.0),
    ("A100-40GB", 312.0, 40.0, 1555.0),
]

for name, tflops, mem, bw in gpus:
    config.cluster_spec.gpu_spec = GPUSpec(
        name=name,
        compute_tflops=tflops,
        memory_size_gb=mem,
        memory_bandwidth_gbs=bw,
    )
    
    metrics = LLMInferenceSimulator(config).run()
    stats = metrics.compute_statistics()
    
    print(f"{name}: {stats['throughput_tokens_per_sec']:.2f} tokens/sec")
```

### 사례 2: 배치 크기 최적화

최적의 배치 크기 찾기:

```python
for batch_size in [1, 2, 4, 8, 16, 32]:
    config.scheduler_spec.max_batch_size = batch_size
    config.workload_spec.batch_size = batch_size
    
    metrics = LLMInferenceSimulator(config).run()
    stats = metrics.compute_statistics()
    
    print(f"Batch {batch_size}: "
          f"{stats['throughput_tokens_per_sec']:.2f} tok/s, "
          f"P95: {stats['first_token_latency']['p95']:.4f}s")
```

### 사례 3: 텐서 병렬화 스케일링

TP 스케일링 효율성 측정:

```python
for tp_size in [1, 2, 4, 8]:
    config.parallelism_spec.tensor_parallel_size = tp_size
    config.cluster_spec.n_gpus_per_node = tp_size
    
    metrics = LLMInferenceSimulator(config).run()
    stats = metrics.compute_statistics()
    
    print(f"TP={tp_size}: {stats['throughput_tokens_per_sec']:.2f} tokens/sec")
```

### 사례 4: 워크로드 패턴 분석

다양한 입출력 길이 조합:

```python
for input_len, output_len in [(128, 32), (512, 128), (2048, 512)]:
    config.workload_spec.avg_input_length = input_len
    config.workload_spec.avg_output_length = output_len
    
    metrics = LLMInferenceSimulator(config).run()
    stats = metrics.compute_statistics()
    
    print(f"I/O: {input_len}/{output_len}")
    print(f"  Throughput: {stats['throughput_tokens_per_sec']:.2f}")
    print(f"  P95 TTFT: {stats['first_token_latency']['p95']:.4f}s")
```

## 메트릭 해석

### First Token Latency (TTFT)
- 사용자가 첫 응답을 받기까지의 시간
- Prefill 성능에 의존
- 낮을수록 좋음 (보통 수십 ms ~ 수백 ms)

### End-to-End Latency
- 전체 응답 완료까지의 시간
- TTFT + 디코딩 시간
- 긴 응답일수록 증가

### Throughput
- 시스템이 처리할 수 있는 초당 토큰 수
- 높을수록 좋음
- Batch size와 GPU 성능에 의존

### GPU Utilization
- GPU가 실제로 작업한 시간 비율
- 100%에 가까울수록 효율적
- 낮으면 배치 크기 증가 고려

## 문제 해결

### Q: Completed requests가 0입니다
A: 시뮬레이션 시간이 짧아서 요청이 완료되지 못했습니다.
   `simulation_duration_s`를 늘리거나 `avg_output_length`를 줄이세요.

### Q: GPU Utilization이 낮습니다
A: 배치 크기를 늘리거나 arrival rate를 높이세요.

### Q: Latency가 너무 높습니다
A: 배치 크기를 줄이거나, 더 빠른 GPU를 사용하거나,
   TP를 늘려보세요.

### Q: 메모리 부족 경고가 나옵니다
A: 모델 크기와 배치 크기를 줄이거나,
   메모리가 더 큰 GPU를 사용하세요.

## 다음 단계

1. **예제 실행**: `python example.py`로 다양한 예제 확인
2. **문서 읽기**: `README.md`와 `ARCHITECTURE.md` 참고
3. **커스터마이징**: 자신의 모델과 워크로드로 테스트
4. **확장**: 새로운 스케줄링 정책이나 성능 모델 추가

## 추가 리소스

- **예제 코드**: `example.py`
- **간단한 테스트**: `simple_test.py`
- **아키텍처 문서**: `ARCHITECTURE.md`
- **전체 문서**: `README.md`

즐거운 시뮬레이션 되세요! 🚀
