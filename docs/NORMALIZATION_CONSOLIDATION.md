# 정규화 코드 통합 (2025-11-04)

## 🎯 목적
정규화 로직을 단일 소스로 통합하여 학습과 실시간에서 일관성 보장

---

## 📦 변경 사항

### 1. 공통 모듈 사용
**모든 정규화 코드가 `lib/normalization.py`를 사용**

#### Before (분산)
```
normalize_datasets.py: 자체 구현 (_signed_log1p, _standard_scale)
online_normalizer.py: 자체 구현 (단순 Z-score)
training_compatible_normalizer.py: 별도 구현
```

#### After (통합)
```
lib/normalization.py: 단일 소스
  ├─ normalize_datasets.py: lib/normalization 사용
  └─ online_normalizer.py: lib/normalization 사용
```

### 2. 파일 통합
**PerStockNormalizer에 학습 호환 로직 통합**

#### Before
```
online_normalizer.py:
  - OnlineNormalizer (단순 Z-score)
  - PerStockNormalizer (래퍼)

training_compatible_normalizer.py:
  - TrainingCompatibleNormalizer (학습 호환)
  - PerStockTrainingCompatibleNormalizer (래퍼)
```

#### After
```
online_normalizer.py:
  - OnlineNormalizer (학습 호환 로직 포함)
  - PerStockNormalizer (래퍼)

training_compatible_normalizer.py: 삭제됨
```

---

## 🔧 구현 상세

### lib/normalization.py
공통 정규화 함수 및 상수 정의

```python
# 함수
- signed_log1p(x): 부호 보존 로그 변환
- standard_scale(x, mean, std): Z-Score 정규화
- get_normalization_strategy(feature_name): 특징별 전략 반환

# 상수
- LOGSTD_FEATURES: Log + Z-Score 대상 특징
- STDONLY_FEATURES: Z-Score만 적용할 특징
- DERIVED_FEATURES: 이미 정규화된 파생 특징
- FEATURE_NAMES: 28개 특징 이름
```

### scripts/data/normalize_datasets.py
학습 데이터 정규화 (배치)

```python
from lib.normalization import (
    signed_log1p as _signed_log1p_np,
    LOGSTD_FEATURES,
    STDONLY_FEATURES,
    DERIVED_FEATURES
)

def _signed_log1p(arr: pd.Series) -> pd.Series:
    """lib.normalization.signed_log1p의 래퍼"""
    x = pd.to_numeric(arr, errors="coerce").fillna(0)
    result = _signed_log1p_np(x.values)
    return pd.Series(result, index=arr.index)

def apply_feature_normalization(df: pd.DataFrame) -> pd.DataFrame:
    # lib/normalization의 상수 사용
    logstd_cols = LOGSTD_FEATURES
    stdonly_cols = STDONLY_FEATURES
    ...
```

### scripts/live/online_normalizer.py
실시간 정규화 (온라인)

```python
from lib.normalization import (
    FEATURE_NAMES,
    signed_log1p,
    standard_scale,
    get_normalization_strategy,
    DERIVED_FEATURES
)

class OnlineNormalizer:
    """학습 호환 실시간 정규화"""
    
    def __init__(self, ...):
        # 특징별 전략 설정
        self.strategies = [
            get_normalization_strategy(name) 
            for name in FEATURE_NAMES
        ]
    
    def _transform_for_storage(self, features):
        """log_std 전략은 로그 변환 후 저장"""
        for i, strategy in enumerate(self.strategies):
            if strategy == "log_std":
                transformed[i] = signed_log1p(features[i])
            else:
                transformed[i] = features[i]
        return transformed
    
    def normalize(self, features, update=True):
        """전략별 정규화"""
        for i, strategy in enumerate(self.strategies):
            if strategy == "derived":
                normalized[i] = features[i]  # 그대로
            elif strategy == "log_std":
                log_value = signed_log1p(features[i])
                normalized[i] = standard_scale(log_value, mean, std)
            elif strategy == "std_only":
                normalized[i] = standard_scale(features[i], mean, std)
        return normalized
```

---

## ✅ 검증

### 테스트 통과
```bash
.venv64/Scripts/python tests/test_normalization_compatibility.py
```

```
[PASS] signed_log1p
[PASS] standard_scale
[PASS] normalization strategies
[PASS] TrainingCompatibleNormalizer
[PASS] Feature consistency
[PASS] Comparison

[SUCCESS] All tests passed!
```

### 정규화 일치 확인
```python
# 학습 데이터
누적거래대금: 1,000,000 → log1p → 13.8 → Z-score

# 실시간 데이터
누적거래대금: 1,000,000 → log1p → 13.8 → Z-score

✅ 동일한 함수 사용 → 동일한 결과
```

---

## 📊 정규화 전략

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

## 🚀 사용 방법

### 변경 없음!
```bash
# 학습 데이터 정규화
python scripts/data/normalize_datasets.py --input datasets.duckdb

# 실시간 거래
python scripts/live/live_trading.py --config config/trading_config.json
```

### 내부적으로
```python
# 학습 시
from lib.normalization import LOGSTD_FEATURES, signed_log1p
for col in LOGSTD_FEATURES:
    df[col] = signed_log1p(df[col])  # ← 동일 함수

# 실시간 시
from lib.normalization import signed_log1p
log_value = signed_log1p(features[i])  # ← 동일 함수
```

---

## 📂 파일 구조

```
lib/
  └─ normalization.py         # ✨ 공통 모듈 (새로 생성)

scripts/
  ├─ data/
  │   └─ normalize_datasets.py  # 🔄 lib/normalization 사용
  └─ live/
      ├─ online_normalizer.py   # 🔄 lib/normalization 사용 + 학습 호환
      └─ training_compatible_normalizer.py  # ❌ 삭제됨

tests/
  └─ test_normalization_compatibility.py  # 🔄 업데이트

docs/
  └─ NORMALIZATION.md           # 📄 상세 문서
```

---

## 🎯 장점

### 1. 단일 소스
- 모든 정규화 로직이 `lib/normalization.py`에 집중
- 버그 수정 시 한 곳만 수정
- 일관성 자동 보장

### 2. 코드 간소화
- 중복 코드 제거
- `training_compatible_normalizer.py` 삭제
- `PerStockNormalizer`에 기능 통합

### 3. 유지보수 용이
- 새로운 정규화 전략 추가 시 `lib/normalization.py`만 수정
- 테스트 코드 단순화

### 4. 타입 안전성
- 공통 함수 사용으로 타입 에러 감소
- 상수 사용으로 오타 방지

---

## 📝 마이그레이션 가이드

### 기존 코드가 있다면
**아무 것도 변경할 필요 없음!**

```python
# live_trading.py
from scripts.live.online_normalizer import PerStockNormalizer

normalizer = PerStockNormalizer(...)  # 동일한 API
```

### 새로운 정규화 로직 추가
```python
# lib/normalization.py에 추가
NEW_STRATEGY_FEATURES = {"새특징1", "새특징2"}

def get_normalization_strategy(feature_name: str) -> str:
    if feature_name in NEW_STRATEGY_FEATURES:
        return "new_strategy"
    ...

# online_normalizer.py에서 자동 적용됨
```

---

## 🔄 롤백

문제 발생 시 이전 버전으로 복구:
```bash
git checkout HEAD~1 scripts/live/online_normalizer.py
git checkout HEAD~1 scripts/data/normalize_datasets.py
```

---

**변경 일자:** 2025-11-04  
**목적:** 코드 통합 및 정규화 일관성 보장  
**이슈:** Buy Signal Rate 0% 해결
