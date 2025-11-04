# 정규화 통합 가이드

## 개요

학습 데이터 정규화와 실시간 추론 정규화가 **동일한 방식**을 사용하도록 통합되었습니다.

---

## 정규화 전략

### 1. Log + Z-Score (23개 특징)

**대상 특징:**
- 누적거래대금
- 거래회전율
- 체결강도
- 매도대기금액 1~10 (10개)
- 매수대기금액 1~10 (10개)

**변환 과정:**
```python
# 1단계: Signed Log1p 변환
log_value = sign(x) * log1p(|x|)

# 2단계: Z-Score 정규화
normalized = (log_value - mean) / std
```

**이유:**
- 이 특징들은 분포가 매우 치우쳐 있음 (heavy-tailed)
- 큰 값들이 지배적 (예: 거래대금 1조 vs 1억)
- 로그 변환으로 분포를 정규화

### 2. Z-Score Only (1개 특징)

**대상 특징:**
- 등락률

**변환 과정:**
```python
normalized = (value - mean) / std
```

**이유:**
- 등락률은 이미 비교적 정규 분포
- 음수/양수 모두 중요한 의미

### 3. No Normalization (4개 특징)

**대상 특징:**
- 종목명_scalar (이미 [0, 1] 범위)
- 시간_sin (이미 [-1, 1] 범위)
- 시간_cos (이미 [-1, 1] 범위)
- 시간_scalar (이미 Z-Score 정규화됨)

**이유:**
- 파생 피처 생성 시 이미 정규화됨
- 추가 정규화 불필요

---

## 구현

### 공통 모듈

**`lib/normalization.py`**
- 정규화 로직의 단일 소스
- 학습과 실시간에서 공유

```python
from lib.normalization import (
    signed_log1p,
    standard_scale,
    get_normalization_strategy,
    FEATURE_NAMES
)
```

### 학습 데이터 정규화

**`scripts/data/normalize_datasets.py`**
- 배치 정규화
- 전체 데이터셋의 통계 사용

```python
def apply_feature_normalization(df: pd.DataFrame) -> pd.DataFrame:
    # Log + Standard
    for col in logstd_cols:
        df[col] = _standard_scale(_signed_log1p(df[col]))
    
    # Standard only
    for col in stdonly_cols:
        df[col] = _standard_scale(df[col])
    
    # Derived features: 이미 정규화됨
    return df
```

### 실시간 정규화

**`scripts/live/training_compatible_normalizer.py`**
- 온라인 정규화 (Rolling window)
- 학습과 동일한 전략 사용
- 종목별 독립 통계

```python
normalizer = PerStockTrainingCompatibleNormalizer(
    num_features=28,
    window_size=200,
    warmup_samples=50
)

normalized = normalizer.normalize(stock_code, features, update=True)
```

---

## 검증

### 정규화 일치 확인

**학습 데이터:**
```python
# Log + Z-Score
누적거래대금: log1p(1조) → 13.8 → Z-score → 정규화
```

**실시간 데이터:**
```python
# 동일: Log + Z-Score
누적거래대금: log1p(1조) → 13.8 → Z-score → 정규화
```

### 테스트 방법

```python
# 1. 동일한 원본 값
value = 1_000_000.0  # 100만

# 2. 학습 방식 (normalize_datasets.py)
log_value = signed_log1p(value)  # ≈ 13.8
normalized_train = (log_value - mean_train) / std_train

# 3. 실시간 방식 (training_compatible_normalizer.py)
log_value = signed_log1p(value)  # ≈ 13.8
normalized_live = (log_value - mean_live) / std_live

# 통계가 유사하면 결과도 유사
assert abs(normalized_train - normalized_live) < 0.1
```

---

## 특징 비교표

| 특징 | 인덱스 | 전략 | 변환 |
|------|--------|------|------|
| 종목명_scalar | 0 | derived | 그대로 |
| 시간_sin | 1 | derived | 그대로 |
| 시간_cos | 2 | derived | 그대로 |
| 시간_scalar | 3 | derived | 그대로 |
| 등락률 | 4 | std_only | Z-score |
| 누적거래대금 | 5 | log_std | log1p + Z-score |
| 거래회전율 | 6 | log_std | log1p + Z-score |
| 체결강도 | 7 | log_std | log1p + Z-score |
| 매도대기금액1~10 | 8-17 | log_std | log1p + Z-score |
| 매수대기금액1~10 | 18-27 | log_std | log1p + Z-score |

---

## 마이그레이션

### 기존 코드 업데이트

**Before (PerStockNormalizer):**
```python
from scripts.live.online_normalizer import PerStockNormalizer

normalizer = PerStockNormalizer(...)
```

**After (TrainingCompatible):**
```python
from scripts.live.training_compatible_normalizer import PerStockTrainingCompatibleNormalizer

normalizer = PerStockTrainingCompatibleNormalizer(...)
```

### API 호환성

동일한 인터페이스:
```python
# 정규화
normalized = normalizer.normalize(stock_code, features, update=True)

# 상태 확인
is_ready = normalizer.is_ready(stock_code)
stats = normalizer.get_stats(stock_code)

# 초기화
normalizer.reset(stock_code)
```

---

## 문제 해결

### Q: 여전히 Buy Signal이 0%입니다
**A:** 다음을 확인하세요:
1. Rolling window가 충분히 채워졌는지 (50 샘플 이상)
2. 로그에서 정규화 통계 확인
3. 모델이 올바른 경로에서 로드되었는지

### Q: 정규화 통계가 이상합니다
**A:** 로그 확인:
```
[NORMALIZER] 486990 warming up... (1/50)
[NORMALIZER] 486990 ready for inference
[FEATURES] 486990 특징 추출 샘플: 평균=0.0, 표준편차=1.0
```

### Q: 학습 데이터를 재정규화해야 하나요?
**A:** 아니요, 실시간 정규화만 수정하면 됩니다. 학습 데이터는 이미 올바르게 정규화되어 있습니다.

---

## 참고

- **학습 정규화:** `scripts/data/normalize_datasets.py`
- **실시간 정규화:** `scripts/live/training_compatible_normalizer.py`
- **공통 모듈:** `lib/normalization.py`
- **사용 예시:** `scripts/live/live_trading.py`
