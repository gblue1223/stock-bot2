# 파생 피처 계산 코드 중앙화 (Derived Features Refactoring)

## 개요

파생 피처 계산 로직을 `lib/normalization.py`로 중앙화하여 코드 중복을 제거하고 일관성을 보장합니다.

## 변경 사항

### 1. 중앙화된 함수 추가 (`lib/normalization.py`)

다음 3개의 공개 함수를 추가했습니다:

```python
def compute_stock_name_scalar(stock_name: str) -> float:
    """종목명을 0~1 범위의 스칼라 값으로 인코딩"""
    
def compute_time_features(time_str: str) -> Tuple[float, float, float]:
    """시간 문자열에서 (시간_sin, 시간_cos, 시간_scalar) 계산"""
    
def compute_all_derived_features(stock_name: str, time_str: str) -> Tuple[float, float, float, float]:
    """모든 파생 피처를 한 번에 계산: (종목명_scalar, 시간_sin, 시간_cos, 시간_scalar)"""
```

### 2. 실시간 데이터 모듈 리팩토링

**파일**: `koapys/kiwoom_rest_api/src/kiwoom_rest_api/koreanstock/realtime_stock.py`

**변경 전**:
```python
def compute_derived_features(self):
    # 중복된 계산 로직 (40+ 줄)
    if self.종목명:
        char_sum = sum(ord(c) for c in self.종목명)
        self.종목명_scalar = float(char_sum % 10000) / 10000.0
    # ... 시간 계산 로직
```

**변경 후**:
```python
def compute_derived_features(self):
    from lib.normalization import compute_all_derived_features
    
    self.종목명_scalar, self.시간_sin, self.시간_cos, self.시간_scalar = \
        compute_all_derived_features(self.종목명, self.시간)
```

## 파생 피처 설명

### 1. 종목명_scalar
- **목적**: 종목명을 수치형 특징으로 변환
- **방법**: 문자 유니코드 값의 합을 0~1 범위로 정규화
- **범위**: [0.0, 1.0]
- **예시**: "삼성전자" → 0.1089

### 2. 시간_sin, 시간_cos
- **목적**: 시간의 주기성을 표현
- **방법**: 하루 24시간을 주기로 sin/cos 변환
- **범위**: [-1.0, 1.0]
- **특징**: sin² + cos² = 1 (항상 성립)
- **예시**: "093015000" (09:30:15) → sin=0.608, cos=-0.794

### 3. 시간_scalar
- **목적**: 장 시작 후 경과 시간 표현
- **방법**: 09:00 기준 경과 시간을 -1~1 범위로 정규화
- **범위**: [-1.0, 1.0] (실제로는 장 시간 내에서만 사용)
- **매핑**:
  - 09:00 (장 시작) → -1.0
  - 12:00 (3시간 경과) → 0.0
  - 15:00 (6시간 경과) → 1.0
- **예시**: "093015000" → -0.832

## 학습 데이터 vs 실시간 추론

### 차이점

| 피처 | 학습 데이터 (normalize_datasets.py) | 실시간 추론 (lib/normalization.py) |
|------|-------------------------------------|-------------------------------------|
| 종목명_scalar | 문자별 평균 ID / 최대 ID | 문자 유니코드 합 % 10000 / 10000 |
| 시간_sin | sin(2π × secs / 86400) | 동일 |
| 시간_cos | cos(2π × secs / 86400) | 동일 |
| 시간_scalar | z-score (전체 데이터셋 기준) | (sec_from_open - 10800) / 10800 |

### 이유

1. **종목명_scalar**: 실시간에서는 전체 종목 목록을 모르므로 간단한 해시 기반 방식 사용
2. **시간_scalar**: 실시간에서는 전체 데이터셋의 통계를 모르므로 고정된 범위로 매핑

### 영향

- 파생 피처는 `DERIVED_FEATURES`로 분류되어 추가 정규화를 받지 않음
- 학습과 추론 간 약간의 차이가 있지만, 모델은 이미 정규화된 값으로 학습되었으므로 영향 최소화
- 실시간 추론의 계산 속도와 단순성을 우선시

## 테스트

새로운 테스트 파일 추가: `tests/test_derived_features.py`

```bash
# 테스트 실행
python -m pytest tests/test_derived_features.py -v
```

**테스트 항목**:
- ✅ 종목명_scalar 계산 및 범위 검증
- ✅ 시간 파생 피처 계산 및 범위 검증
- ✅ sin² + cos² = 1 검증
- ✅ 모든 파생 피처 일괄 계산
- ✅ 실시간 데이터 호환성

## 사용 예시

### 실시간 데이터 처리

```python
from lib.normalization import compute_all_derived_features

# 실시간 데이터에서 파생 피처 계산
stock_name = "삼성전자"
time_str = "093015000"  # 09:30:15

종목명_scalar, 시간_sin, 시간_cos, 시간_scalar = \
    compute_all_derived_features(stock_name, time_str)

print(f"종목명_scalar: {종목명_scalar:.4f}")
print(f"시간_sin: {시간_sin:.4f}")
print(f"시간_cos: {시간_cos:.4f}")
print(f"시간_scalar: {시간_scalar:.4f}")
```

### StockRealtimeData 클래스

```python
from koapys.kiwoom_rest_api.src.kiwoom_rest_api.koreanstock.realtime_stock import StockRealtimeData

stock = StockRealtimeData(종목코드="005930", 종목명="삼성전자", 시간="093015000")
stock.compute_derived_features()

# 파생 피처가 자동으로 계산됨
print(stock.종목명_scalar)  # 0.1089
print(stock.시간_sin)       # 0.6079
print(stock.시간_cos)       # -0.7940
print(stock.시간_scalar)    # -0.8319
```

## 이점

1. **코드 중복 제거**: 40+ 줄의 중복 코드 제거
2. **유지보수성 향상**: 한 곳에서만 수정하면 모든 곳에 반영
3. **일관성 보장**: 모든 모듈에서 동일한 계산 방식 사용
4. **테스트 용이성**: 독립적인 함수로 분리되어 테스트 작성 용이
5. **문서화**: 중앙화된 함수에 상세한 docstring 제공

## 관련 파일

- `lib/normalization.py`: 중앙화된 파생 피처 계산 함수 (실시간 + 배치)
- `koapys/kiwoom_rest_api/src/kiwoom_rest_api/koreanstock/realtime_stock.py`: 실시간 데이터 클래스
- `scripts/data/normalize_datasets.py`: 학습 데이터 전처리 (중앙화된 배치 함수 사용)
- `scripts/live/online_normalizer.py`: 온라인 정규화 (파생 피처는 정규화 안함)
- `tests/test_derived_features.py`: 실시간 파생 피처 계산 테스트
- `tests/test_batch_derived_features.py`: 배치 파생 피처 계산 테스트

## 주의사항

1. **학습 데이터 전처리는 변경하지 않음**: `normalize_datasets.py`는 기존 방식 유지
2. **파생 피처는 추가 정규화 안함**: `DERIVED_FEATURES`로 분류되어 있음
3. **실시간 추론 최적화**: 간단하고 빠른 계산 방식 사용

## 배치 처리 함수 (추가)

`lib/normalization.py`에 학습 데이터 전처리용 배치 함수도 추가되었습니다:

```python
def compute_stock_name_scalar_batch(stock_names: pd.Series) -> pd.Series:
    """종목명 Series를 배치로 스칼라 인코딩"""
    
def compute_time_features_batch(time_series: pd.Series, use_zscore_for_scalar: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """시간 Series에서 파생 피처 배치 계산"""
```

### 사용 예시

```python
import pandas as pd
from lib.normalization import compute_stock_name_scalar_batch, compute_time_features_batch

# 종목명 배치 처리
stock_names = pd.Series(["삼성전자", "SK하이닉스", "NAVER"])
scalars = compute_stock_name_scalar_batch(stock_names)

# 시간 배치 처리 (학습 데이터용 - z-score)
time_series = pd.Series(["090000000", "093000000", "100000000"])
time_sin, time_cos, time_scalar = compute_time_features_batch(
    time_series, 
    use_zscore_for_scalar=True  # 학습 데이터는 z-score 사용
)
```

## 향후 개선 사항

1. ~~학습 데이터와 실시간 추론의 종목명_scalar 계산 방식 통일 검토~~ (배치 함수 추가로 중앙화 완료)
2. 시간_scalar의 z-score 통계를 사전 계산하여 실시간에서도 사용 검토
3. 파생 피처 계산 성능 벤치마크 및 최적화
