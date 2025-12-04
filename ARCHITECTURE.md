# LLM Inference Simulator - 아키텍처 설계 문서

## 1. 전체 아키텍처 개요

### 1.1 설계 철학

이 시뮬레이터는 다음 원칙을 따라 설계되었습니다:

1. **이벤트 드리븐**: 모든 동작을 이벤트로 모델링하여 실제 시간 진행을 시뮬레이션
2. **모듈화**: 각 컴포넌트를 독립적으로 테스트하고 확장 가능하도록 설계
3. **확장성**: 새로운 모델, 하드웨어, 스케줄링 정책을 쉽게 추가
4. **정확성**: Transformer의 실제 연산 특성을 반영한 성능 모델링

### 1.2 시스템 구조

```
┌─────────────────────────────────────────────────────┐
│              LLMInferenceSimulator                  │
│  ┌───────────────────────────────────────────────┐  │
│  │           Event Queue (Priority Queue)        │  │
│  └───────────────────────────────────────────────┘  │
│                        ↓                            │
│  ┌───────────────────────────────────────────────┐  │
│  │         Event Processing Loop                 │  │
│  └───────────────────────────────────────────────┘  │
│           ↓                    ↓                    │
│  ┌──────────────┐    ┌──────────────────────┐      │
│  │  Scheduler   │←→  │  Performance Model   │      │
│  └──────────────┘    └──────────────────────┘      │
│           ↓                    ↓                    │
│  ┌──────────────┐    ┌──────────────────────┐      │
│  │  Request     │    │  GPU / Cluster       │      │
│  │  Management  │    │  State               │      │
│  └──────────────┘    └──────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

## 2. 핵심 컴포넌트 상세

### 2.1 Event System (events.py)

모든 시스템 동작은 이벤트로 표현됩니다.

**주요 이벤트 타입:**
- `RequestArrivedEvent`: 새 요청 도착
- `RequestTokenizedEvent`: 입력 토크나이즈 완료
- `BatchFormedEvent`: 배치 구성 완료
- `PrefillStartedEvent/FinishedEvent`: Prefill 시작/완료
- `DecodeStepStartedEvent/FinishedEvent`: Decode 스텝 시작/완료
- `TokenEmittedEvent`: 토큰 생성
- `RequestFinishedEvent`: 요청 완료

**설계 특징:**
- 모든 이벤트는 `timestamp` 필드를 가짐
- 우선순위 큐를 통해 시간순 처리
- `__lt__` 메서드로 자동 정렬

### 2.2 Performance Model (performance_model.py)

Transformer의 연산 비용을 계산하는 핵심 모듈입니다.

**계산 모델:**

#### Prefill Phase
```
시간 = max(Compute Time, Memory Time) + Communication Time

Compute Time:
  - QKV Projection: 3 × B × L × H × H
  - Attention: B × n_heads × L² × head_dim
  - Output Projection: B × L × H × H
  - MLP: 2 × B × L × H × D_ff

Memory Time:
  - Weight Loading: (3H² + H² + 2H×D_ff) × bytes
  - KV Cache Write: 2 × B × L × H × bytes

Communication Time (TP):
  - All-Reduce: 2 × (TP-1)/TP × data_size/bandwidth
```

#### Decode Phase
```
시간 = max(Compute Time, Memory Time) + Communication Time

Compute Time:
  - QKV Projection: 3 × B × H × H (1개 토큰)
  - Attention: B × n_heads × T × head_dim
  - Output Projection: B × H × H
  - MLP: 2 × B × H × D_ff

Memory Time (보통 병목):
  - Weight Loading: 동일
  - KV Cache Read: 2 × B × T × H × bytes
  - KV Cache Write: 2 × B × H × bytes
```

**주요 가정:**
1. Compute-bound vs Memory-bound는 max로 결정
2. Perfect overlap 가정 (실제로는 overlap 불완전)
3. FlashAttention 등의 최적화는 미반영

### 2.3 Scheduler (scheduler.py)

요청을 배칭하고 실행 순서를 결정합니다.

**배칭 전략:**

1. **Static Batching**
   - 고정 크기의 배치를 구성
   - 배치가 채워질 때까지 대기
   - 간단하지만 latency가 높을 수 있음

2. **Dynamic Batching**
   - 배칭 윈도우(시간 제한) 적용
   - 최대 배치 크기 또는 시간 제한 중 먼저 도달
   - Latency와 throughput 균형

3. **Continuous Batching**
   - 토큰 단위로 요청 처리
   - Decode에서 요청을 동적으로 추가/제거
   - 가장 효율적이지만 구현 복잡

**큐 관리:**
- `prefill_queue`: Prefill 대기 중인 요청
- `decode_queue`: Decode 진행 중인 요청
- `pending_requests`: 모든 활성 요청 추적

### 2.4 Request & Batch (request.py)

**Request 클래스:**
```python
class Request:
    - request_id: 고유 식별자
    - arrival_time: 도착 시간
    - input_length: 입력 토큰 수
    - requested_output_tokens: 요청된 출력 토큰 수
    - tokens_generated: 현재까지 생성된 토큰 수
    - current_kv_cache_length: 현재 KV cache 길이
    - status: 현재 상태 (ARRIVED, QUEUED, PREFILLING, DECODING, FINISHED)
```

**Batch 클래스:**
```python
class Batch:
    - batch_id: 배치 식별자
    - requests: 배치에 포함된 요청 리스트
    - is_prefill: Prefill 배치인지 여부
    - current_decode_step: Decode 단계 (decode 배치만)
    - max_input_length: 배치 내 최대 입력 길이
    - max_kv_cache_length: 배치 내 최대 KV cache 길이
```

**설계 특징:**
- Continuous batching을 위해 배치에서 완료된 요청 동적 제거
- KV cache 길이 추적으로 메모리 사용량 예측

### 2.5 Simulator (simulator.py)

메인 시뮬레이터 엔진입니다.

**이벤트 처리 루프:**
```python
while event_queue:
    event = heappop(event_queue)
    current_time = event.timestamp
    
    process_event(event)
    
    # 새로운 작업 스케줄링 시도
    try_schedule_work()
    
    if should_stop():
        break
```

**상태 관리:**
- `current_time`: 현재 시뮬레이션 시간
- `is_gpu_busy`: GPU 사용 중 여부
- `current_batch`: 현재 처리 중인 배치
- `metrics`: 수집된 메트릭

**메트릭 수집:**
- Request-level: First token latency, end-to-end latency
- Token-level: 토큰 생성 시간 분포
- System-level: Throughput, GPU utilization

## 3. 데이터 흐름

### 3.1 Request Lifecycle

```
1. Request Arrival
   ↓
2. Tokenization (빠름, ~0.1ms)
   ↓
3. Queue in prefill_queue
   ↓
4. Batch Formation
   ↓
5. Prefill Phase
   - 전체 입력 시퀀스 처리
   - KV cache 초기화
   - First token 생성
   ↓
6. Move to decode_queue
   ↓
7. Decode Phase (반복)
   - 1개 토큰씩 생성
   - KV cache 업데이트
   - 각 단계마다 배치 재구성 가능
   ↓
8. Request Completion
   - EOS 토큰 또는 max length 도달
```

### 3.2 시간 진행 메커니즘

시뮬레이터는 **Discrete Event Simulation**을 사용합니다:

1. 이벤트 큐에서 가장 이른 이벤트 추출
2. 시뮬레이션 시간을 해당 이벤트 시간으로 점프
3. 이벤트 처리 및 새 이벤트 스케줄링
4. 반복

**장점:**
- 효율적 (불필요한 시간 단계 스킵)
- 정확한 이벤트 순서 보장
- 확장성 (이벤트 수에만 비례)

## 4. 병렬화 전략 지원

### 4.1 Tensor Parallelism (TP)

**구현 방식:**
- 연산량을 TP로 나눔: `effective_flops = total_flops / TP`
- 가중치를 TP로 나눔: `weight_bytes = total_bytes / TP`
- All-Reduce 통신 비용 추가

**통신 모델:**
```python
# Ring All-Reduce
comm_time = 2 × (TP-1)/TP × data_size/bandwidth + latency
```

### 4.2 Data Parallelism (DP)

**구현 계획:** (현재 부분적 지원)
- 복제된 모델로 서로 다른 배치 처리
- 가중치 동기화 비용 무시 (추론 시)

### 4.3 Pipeline Parallelism (PP)

**구현 계획:** (미래 확장)
- 레이어를 스테이지로 분할
- Microbatch 기반 파이프라이닝
- Bubble 시간 계산

## 5. 성능 최적화 고려사항

### 5.1 현재 구현의 단순화

1. **Perfect Overlap 가정**
   - Compute와 Memory를 max로만 처리
   - 실제로는 부분 overlap

2. **단순화된 통신 모델**
   - Ring All-Reduce만 지원
   - Tree, Hierarchical 미지원

3. **KV Cache 관리**
   - OOM 체크 없음
   - Paging/Eviction 미지원

### 5.2 향후 개선 방향

1. **더 정교한 성능 모델**
   - Roofline 모델 적용
   - Kernel fusion 반영
   - FlashAttention 등 최적화 지원

2. **메모리 관리**
   - PagedAttention 지원
   - KV cache 압축
   - Dynamic memory allocation

3. **고급 스케줄링**
   - SLA 기반 스케줄링
   - 멀티 테넌시
   - Preemption

4. **캘리브레이션**
   - 실제 측정 데이터로 모델 튜닝
   - GPU별 특성 반영

## 6. 확장 가이드

### 6.1 새로운 모델 추가

1. `ModelSpec`에 파라미터 추가
2. `PerformanceModel`에 계산 로직 추가
3. 필요시 새 이벤트 타입 정의

### 6.2 새로운 스케줄링 정책 추가

```python
class MyScheduler(Scheduler):
    def schedule_prefill_batch(self, current_time):
        # 커스텀 로직 구현
        pass
```

### 6.3 새로운 메트릭 추가

1. `SimulationMetrics`에 필드 추가
2. 해당 이벤트 핸들러에서 수집
3. `compute_statistics()`에서 계산

## 7. 테스트 및 검증

### 7.1 단위 테스트 권장사항

- `PerformanceModel`: 알려진 설정에 대해 예상 시간 검증
- `Scheduler`: 배치 구성 로직 검증
- `Event`: 우선순위 큐 순서 검증

### 7.2 통합 테스트

- End-to-end 시뮬레이션
- 실제 시스템과 비교 (가능한 경우)
- Edge case 처리

## 8. 알려진 제한사항

1. **모델 정확도**
   - 단순화된 가정들로 인해 ±20% 오차 가능
   - 캘리브레이션 필요

2. **확장성**
   - 매우 큰 규모 (수천 GPU)에서는 미검증

3. **특수 케이스**
   - MoE 모델: 부분 지원
   - Speculative decoding: 미지원
   - Continuous batching의 일부 최적화: 미지원

## 9. 참고 자료

### 논문
- Transformer 아키텍처: "Attention is All You Need"
- FlashAttention: "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Megatron-LM: "Megatron-LM: Training Multi-Billion Parameter Language Models"
- Orca: "Orca: A Distributed Serving System for Transformer-Based Generative Models"

### 구현 참고
- vLLM: Continuous batching 구현
- FasterTransformer: 최적화된 Transformer 추론
- DeepSpeed-Inference: 병렬화 전략

---

이 문서는 시뮬레이터의 설계 결정과 구현 세부사항을 설명합니다.
추가 질문이나 개선 제안은 언제든 환영합니다!
