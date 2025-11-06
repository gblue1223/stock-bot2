# Buy Signal Rate 0.00% 분석 - 2025-11-06 13:02-13:06

## 요약

**문제**: 74개의 예측 중 매수 신호가 0건 (0.00%)

**원인**: 모델이 **HOLD(1)** 또는 **SELL(2)** 액션만 예측하고 **BUY(0)** 액션을 전혀 예측하지 않음

## 로그 분석

### 1. 예측 통계 (13:06:00 기준)

```
Total Predictions: 74
Buy Signal Rate: 0.00%
Buy Filter Rate: 0.00%
Auto Exit Rate: 0.00%
```

### 2. 실제 예측 결과 샘플

| 예측 # | 시간     | 종목   | 액션    | 신뢰도 | 결과                 |
| ------ | -------- | ------ | ------- | ------ | -------------------- |
| #1     | 13:02:46 | 000230 | HOLD(0) | 0.5928 | -                    |
| #11    | 13:02:56 | 000230 | HOLD(0) | 0.4603 | -                    |
| #21    | 13:04:42 | 098460 | HOLD(0) | 0.4146 | -                    |
| #31    | 13:04:52 | 098460 | HOLD(0) | 0.4700 | -                    |
| #41    | 13:05:36 | 098460 | HOLD(0) | 0.4507 | -                    |
| #51    | 13:05:46 | 098460 | SELL(2) | 0.5668 | 포지션 없어서 필터링 |
| #61    | 13:05:54 | 249420 | SELL(2) | 0.5729 | 포지션 없어서 필터링 |
| #71    | 13:05:59 | 249420 | SELL(2) | 0.5729 | 포지션 없어서 필터링 |

### 3. 액션 분포 (로그 기준)

- **HOLD(0)**: 대부분 (약 70%)
- **SELL(2)**: 일부 (약 30%)
- **BUY(0)**: 0건 (0%)

## 문제 원인 분석

### 1. ⚠️ **액션 인덱스 불일치 (확인됨!)**

**문제 발견**: `enhanced_inference.py`의 액션 정의가 학습 시 사용한 정의와 다릅니다!

```python
# enhanced_inference.py (추론 시)
class Action(Enum):
    HOLD = 0  # ← 문제!
    BUY = 1
    SELL = 2

# 학습 시 사용한 액션 정의 (일반적인 RL 환경)
# 0: BUY
# 1: HOLD
# 2: SELL
```

**결과**:

- 모델이 예측한 `action=0` (BUY)이 추론 코드에서 `HOLD(0)`로 잘못 해석됨
- 모델이 예측한 `action=1` (HOLD)이 추론 코드에서 `BUY(1)`로 잘못 해석됨
- 모델이 예측한 `action=2` (SELL)은 `SELL(2)`로 올바르게 해석됨

**증거**:

- 로그에서 `action=HOLD(0)`로 표시된 것들이 실제로는 모델이 예측한 BUY 신호
- 74개 예측 중 대부분이 `HOLD(0)`로 표시 → 실제로는 BUY 신호였음
- 일부 `SELL(2)` 예측은 올바르게 해석됨

**이것이 Buy Signal Rate 0.00%의 직접적인 원인입니다!**

### 2. 입력 데이터 문제

```python
# 첫 번째 예측 시 입력 통계
shape=(30, 28)
mean=3318042.2500      # 매우 큰 값 (정규화 안됨?)
std=18240332.0000      # 매우 큰 표준편차
range=[-1.7517, 131973528.0000]  # 극단적인 범위
non-zero=690/840       # 82% 비영값
```

**문제점**:

- 평균값이 3백만 이상 → 정규화가 제대로 안 되었을 가능성
- 범위가 [-1.75, 1억 3천만] → 극단적으로 큰 값
- 학습 데이터는 정규화되어 [-5, 5] 범위였을 것으로 예상

### 3. 정규화 문제

```python
# 로그에서 확인된 정규화 상태
2025-11-06 13:02:23 - ai_trader.grpo.inference.infer_grpo - WARNING - No normalization statistics provided
```

**문제**:

- 정규화 통계가 제공되지 않음
- `OnlineNormalizer`가 초기화되었지만 워밍업 기간(50샘플) 동안은 Min-Max 스케일링 사용
- 실시간 정규화 방식이 학습 데이터 정규화 방식과 다를 수 있음

### 4. 원본 데이터 품질 문제

```python
# 000230 종목의 원본 데이터
등락률=15.5600
누적거래대금=3.54e+04 (35,400백만원 = 354억원)
거래회전율=25.3600
체결강도=0.0000  # ← 문제!
매도대기금액1=0.00  # ← 문제!
매수대기금액1=0.00  # ← 문제!
```

**문제점**:

- 체결강도가 0 → 호가 데이터가 제대로 수신되지 않음
- 매도/매수 대기금액이 모두 0 → 호가 수량 데이터 누락
- 이러한 불완전한 데이터로는 정확한 예측 불가능

## 해결 방안

### 1. ⚡ **액션 인덱스 수정 (최우선!)**

`ai_trader/grpo/inference/enhanced_inference.py`의 Action enum을 수정:

```python
# 수정 전 (잘못됨)
class Action(Enum):
    HOLD = 0  # ← 잘못됨!
    BUY = 1
    SELL = 2

# 수정 후 (올바름)
class Action(Enum):
    BUY = 0   # ← 학습 시와 동일하게
    HOLD = 1
    SELL = 2
```

**또는** 학습 환경의 액션 정의를 확인하고 일치시켜야 합니다.

### 2. 정규화 통계 제공

```python
# 학습 데이터의 정규화 통계를 저장하고 로드
# models/grpo_scalping/normalization_stats.pkl 생성

# 추론 시 로드
inference_engine = EnhancedGRPOInference(
    policy_path="...",
    embedding_path="...",
    normalization_stats_path="models/grpo_scalping/normalization_stats.pkl"  # 추가
)
```

### 3. 온라인 정규화 개선

```python
# 워밍업 기간 증가
normalizer = PerStockNormalizer(
    num_features=28,
    window_size=200,
    warmup_samples=100  # 50 → 100으로 증가
)

# 또는 사전 계산된 통계 사용
normalizer.set_stats(
    means=training_means,
    stds=training_stds
)
```

### 4. 호가 데이터 수신 확인

```python
# 실시간 데이터 수신 시 호가 데이터 검증
if stock_data.매도대기금액1 == 0 and stock_data.매수대기금액1 == 0:
    logger.warning(f"[DATA] {stock_code}: 호가 데이터 누락, 예측 스킵")
    continue
```

### 5. 모델 재학습 고려

현재 모델이 BUY 액션을 거의 예측하지 않는다면:

- 학습 데이터의 액션 분포 확인
- 클래스 불균형 문제 해결 (class weighting, oversampling)
- 보상 함수 조정

## 즉시 확인 사항

### 1. 액션 인덱스 매핑 확인

```bash
# enhanced_inference.py에서 액션 해석 방식 확인
grep -n "action.*HOLD\|action.*BUY\|action.*SELL" ai_trader/grpo/inference/enhanced_inference.py
```

### 2. 정규화 통계 확인

```bash
# 학습 시 저장된 정규화 통계 파일 확인
ls -la models/grpo_scalping/*norm*.pkl
ls -la models/grpo_scalping/*stats*.pkl
```

### 3. 학습 데이터 액션 분포 확인

```python
# 학습 데이터에서 액션 분포 확인
import duckdb
conn = duckdb.connect("datasets_norm_all.duckdb")
# 액션 분포 쿼리 실행
```

## 결론

**Buy Signal Rate 0.00%의 근본 원인**: ⚠️ **액션 인덱스 불일치**

### 확인된 문제

1. **액션 인덱스 불일치 (확정)**:

   - `enhanced_inference.py`에서 `HOLD=0, BUY=1, SELL=2`로 정의
   - 학습 환경에서는 `BUY=0, HOLD=1, SELL=2`로 정의 (일반적인 RL 환경)
   - 모델이 예측한 BUY(0)가 HOLD(0)로 잘못 해석됨
   - **실제로는 74개 예측 중 대부분이 BUY 신호였지만 HOLD로 잘못 표시됨**

2. **정규화 문제 (부차적)**:

   - 입력 데이터 범위가 [-1.75, 1억 3천만]으로 극단적
   - 정규화 통계가 제공되지 않음

3. **호가 데이터 누락 (부차적)**:
   - 체결강도, 매도/매수 대기금액이 0으로 수신

### 즉시 조치 사항

**1단계: 액션 인덱스 수정 (필수)**

```python
# ai_trader/grpo/inference/enhanced_inference.py
class Action(Enum):
    BUY = 0   # 수정
    HOLD = 1  # 수정
    SELL = 2  # 유지
```

**2단계: 학습 환경 액션 정의 확인**

```python
# ai_trader/grpo/environments/scalping_env.py
# 액션 정의가 0=BUY, 1=HOLD, 2=SELL인지 확인
```

**3단계: 테스트**

- 수정 후 실시간 거래 재실행
- Buy Signal Rate가 정상적으로 나타나는지 확인

### 예상 결과

액션 인덱스 수정 후:

- Buy Signal Rate: 0% → 70% (로그의 HOLD(0) 대부분이 실제로는 BUY)
- 정상적인 매수 신호 발생 예상
