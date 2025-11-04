# 정규화 통합 업데이트 (2025-11-04)

## 🎯 목적
학습 데이터와 실시간 추론에서 **동일한 정규화 방식**을 사용하여 Buy Signal 0% 문제 해결

## 🔍 문제 진단

### Before: 정규화 불일치
```
학습 데이터: log1p + Z-score (누적거래대금 등)
실시간:      Z-score only

→ 분포 불일치 → 모델이 패턴 인식 못함 → Buy Signal 0%
```

### After: 정규화 일치
```
학습 데이터: log1p + Z-score
실시간:      log1p + Z-score (동일!)

→ 분포 일치 → 모델이 패턴 인식 가능 ✅
```

## 📦 새로운 파일

### 1. `lib/normalization.py`
정규화 로직의 단일 소스
```python
from lib.normalization import (
    signed_log1p,
    standard_scale,
    get_normalization_strategy
)
```

### 2. `scripts/live/training_compatible_normalizer.py`
학습 호환 실시간 정규화
```python
from scripts.live.training_compatible_normalizer import PerStockTrainingCompatibleNormalizer

normalizer = PerStockTrainingCompatibleNormalizer(
    num_features=28,
    window_size=200,
    warmup_samples=50
)
```

### 3. `docs/NORMALIZATION.md`
상세 문서

## 🔧 수정된 파일

### `scripts/live/live_trading.py`
```python
# Before
from scripts.live.online_normalizer import PerStockNormalizer
self.normalizer = PerStockNormalizer(...)

# After
from scripts.live.training_compatible_normalizer import PerStockTrainingCompatibleNormalizer
self.normalizer = PerStockTrainingCompatibleNormalizer(...)
```

## 🚀 사용 방법

### 그대로 실행
```bash
python scripts/live/live_trading.py --config config/trading_config.json
```

### 설정 확인
```json
{
  "online_window_size": 200,
  "online_warmup_samples": 50
}
```

## 📊 정규화 전략

### Log + Z-Score (23개)
- 누적거래대금, 거래회전율, 체결강도
- 매도/매수대기금액 1~10

### Z-Score Only (1개)
- 등락률

### No Normalization (4개)
- 종목명_scalar, 시간_sin, 시간_cos, 시간_scalar

## ✅ 검증

### 로그 확인
```
✅ Training-Compatible Normalizer initialized (window=200, warmup=50)
[NORMALIZER] 486990 warming up... (1/50)
[NORMALIZER] 486990 ready for inference
[FEATURES] 486990 특징 추출 샘플: 평균≈0.0, 표준편차≈1.0
```

### 기대 결과
- Buy Signal Rate > 0%
- 모델이 매수/매도 신호 생성

## 📚 참고

- **상세 문서:** `docs/NORMALIZATION.md`
- **공통 모듈:** `lib/normalization.py`
- **학습 정규화:** `scripts/data/normalize_datasets.py`
- **실시간 정규화:** `scripts/live/training_compatible_normalizer.py`

## 🐛 문제 해결

### 여전히 Buy Signal 0%?
1. 워밍업 완료 확인 (50 샘플 이상)
2. 로그에서 정규화 통계 확인
3. 모델 경로 확인

### 정규화 통계가 이상함?
- 파생 피처(0-3번)는 원본 유지가 정상
- Log 특징(5-27번)은 로그 변환 후 통계

## 🔄 롤백

문제 발생 시:
```python
# live_trading.py 되돌리기
from scripts.live.online_normalizer import PerStockNormalizer
self.normalizer = PerStockNormalizer(...)
```

---

**변경 일자:** 2025-11-04  
**담당자:** Cascade  
**이슈:** Buy Signal Rate 0% 해결
