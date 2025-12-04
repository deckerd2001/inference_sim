# LLM Inference Simulator

이벤트 기반(Event-Driven) LLM 추론 성능 시뮬레이터입니다. Transformer 기반 대규모 언어 모델의 Prefill 및 Decode 단계 성능을 시뮬레이션하여 throughput, latency 분포 등을 예측합니다.

## 주요 기능

### 1. **이벤트 드리븐 시뮬레이션**
- 우선순위 큐 기반으로 시간순 이벤트 처리
- Request arrival, tokenization, prefill, decode, token emission 등 모든 단계를 이벤트로 모델링

### 2. **상세한 성능 모델링**
- **Compute 비용**: Attention 및 MLP 블록의 FLOPs 계산
- **Memory 비용**: 가중치 로딩, KV cache 읽기/쓰기
- **Communication 비용**: Tensor Parallelism을 위한 All-Reduce 등

### 3. **다양한 모델 및 하드웨어 지원**
- 모델: LLaMA-7B, 13B, 70B 등 임의의 Transformer 모델
- GPU: A100, H100, RTX 4090 등 다양한 GPU 사양
- 병렬화: Tensor Parallel (TP), Data Parallel (DP), Pipeline Parallel (PP)

### 4. **유연한 워크로드 설정**
- Poisson/Deterministic request arrival
- 입출력 토큰 길이 분포 설정
- 다양한 배칭 및 스케줄링 정책

### 5. **상세한 메트릭 수집**
- First Token Latency (TTFT)
- End-to-End Latency
- Throughput (requests/s, tokens/s)
- GPU Utilization
- P50, P90, P95, P99 latency 분포

## 설치

```bash
pip install numpy
```

## 빠른 시작

### 기본 예제

```python
from llm_inference_simulator import (
    LLMInferenceSimulator,
    SimulatorConfig,
    ModelSpec,
    WorkloadSpec,
    GPUSpec,
    ClusterSpec,
    ParallelismSpec,
    SchedulerSpec,
    DataType,
)

# 설정 생성
config = SimulatorConfig(
    model_spec=ModelSpec(
        name="llama-7b",
        n_params=7_000_000_000,
        hidden_size=4096,
        n_layers=32,
        n_heads=32,
        ffn_dim=11008,
        max_seq_length=2048,
        weight_dtype=DataType.BF16,
    ),
    workload_spec=WorkloadSpec(
        avg_input_length=512,
        avg_output_length=128,
        arrival_rate=2.0,  # 2 requests/sec
    ),
    cluster_spec=ClusterSpec(
        n_gpus_per_node=1,
        n_nodes=1,
        gpu_spec=GPUSpec(
            name="A100-80GB",
            compute_tflops=312.0,
            memory_size_gb=80.0,
            memory_bandwidth_gbs=2039.0,
        ),
    ),
    parallelism_spec=ParallelismSpec(
        tensor_parallel_size=1,
    ),
    scheduler_spec=SchedulerSpec(
        batching_type="continuous",
        max_batch_size=8,
    ),
    simulation_duration_s=60.0,
)

# 시뮬레이션 실행
simulator = LLMInferenceSimulator(config)
metrics = simulator.run()

# 결과 확인
stats = metrics.compute_statistics()
print(f"Throughput: {stats['throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"P95 Latency: {stats['first_token_latency']['p95']:.4f} sec")
```

### 다양한 예제 실행

```bash
python example.py
```

예제 스크립트는 다음을 포함합니다:
1. **LLaMA-7B on Single GPU**: 단일 A100 GPU에서의 성능
2. **LLaMA-70B with TP=8**: 8-way Tensor Parallelism
3. **High Load Test**: 높은 request rate에서의 시스템 동작
4. **H100 vs A100 Comparison**: GPU 성능 비교

## 아키텍처

```
llm_inference_simulator/
├── __init__.py           # 패키지 초기화
├── config.py             # 설정 데이터 클래스
├── events.py             # 이벤트 정의
├── request.py            # Request 및 Batch 클래스
├── scheduler.py          # 스케줄링 정책
├── performance_model.py  # 성능 모델링 (핵심)
└── simulator.py          # 메인 시뮬레이터
```

### 핵심 컴포넌트

#### 1. Event System
모든 동작은 이벤트로 표현됩니다:
- `RequestArrivedEvent`: 요청 도착
- `PrefillFinishedEvent`: Prefill 완료
- `DecodeStepFinishedEvent`: Decode 스텝 완료
- `TokenEmittedEvent`: 토큰 생성
- `RequestFinishedEvent`: 요청 완료

#### 2. Performance Model
Transformer의 각 블록에 대해 다음을 계산합니다:

**Prefill Phase:**
- Attention: QKV projection, QK^T, softmax, attention*V, output projection
- MLP: Up/down projection
- 메모리: 가중치 로딩, KV cache 쓰기
- 통신: All-reduce (TP)

**Decode Phase:**
- Attention: 1개 토큰에 대한 projection, KV cache 읽기/쓰기
- MLP: 1개 토큰에 대한 projection
- 메모리 bound 특성 반영 (KV cache 읽기가 병목)

#### 3. Scheduler
요청을 배칭하고 실행 순서를 결정합니다:
- **Static batching**: 고정 크기 배치
- **Dynamic batching**: 시간 윈도우 내 동적 배칭
- **Continuous batching**: 토큰 단위 스케줄링

## 설정 상세

### ModelSpec
```python
ModelSpec(
    n_params=7_000_000_000,    # 파라미터 수
    hidden_size=4096,           # Hidden dimension
    n_layers=32,                # Layer 수
    n_heads=32,                 # Attention head 수
    ffn_dim=11008,              # FFN dimension
    max_seq_length=2048,        # 최대 시퀀스 길이
    weight_dtype=DataType.BF16, # 가중치 데이터 타입
)
```

### WorkloadSpec
```python
WorkloadSpec(
    avg_input_length=512,       # 평균 입력 토큰 수
    avg_output_length=128,      # 평균 출력 토큰 수
    arrival_rate=5.0,           # 초당 요청 수
    arrival_process="poisson",  # Arrival 패턴
)
```

### ParallelismSpec
```python
ParallelismSpec(
    tensor_parallel_size=4,     # TP degree
    data_parallel_size=2,       # DP degree
    pipeline_parallel_size=2,   # PP degree
)
```

## 출력 메트릭

시뮬레이션은 다음 메트릭을 수집합니다:

### Request-Level
- **First Token Latency**: 요청 도착부터 첫 토큰까지의 시간
- **End-to-End Latency**: 전체 응답 완료까지의 시간
- **Prefill Latency**: Prefill 단계 소요 시간

### System-Level
- **Throughput**: requests/sec, tokens/sec
- **GPU Utilization**: GPU 사용률
- **Queue Lengths**: Prefill/Decode 큐 크기

### Latency Distribution
- Mean, P50, P90, P95, P99

## 확장 가능성

이 시뮬레이터는 다음과 같은 확장이 가능하도록 설계되었습니다:

1. **새로운 모델 구조**
   - MoE (Mixture of Experts)
   - Encoder-Decoder 모델
   - Custom attention mechanisms

2. **고급 스케줄링**
   - SLA-based scheduling
   - Priority queues
   - Multi-tenancy

3. **하드웨어 모델링**
   - 더 정교한 메모리 계층 모델
   - Inter-node communication
   - PCIe vs NVLink 차이

4. **캘리브레이션**
   - 실제 측정 데이터로 성능 모델 조정
   - GPU 특화 최적화 반영

## 제한사항

1. **단순화된 가정**
   - 완벽한 파이프라이닝 가정
   - Kernel fusion 무시
   - Attention 최적화 (FlashAttention 등) 미반영

2. **통신 모델**
   - Ring All-Reduce 기반 단순 모델
   - 실제 네트워크 congestion 무시

3. **메모리**
   - OOM 체크 미구현
   - 페이징/스와핑 무시

## 라이선스

MIT License

## 기여

이슈 및 PR을 환영합니다!

## 참고 문헌

- Attention is All You Need (Vaswani et al., 2017)
- FlashAttention (Dao et al., 2022)
- Efficient Large-Scale Language Model Training (Narayanan et al., 2021)
- Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., 2022)
