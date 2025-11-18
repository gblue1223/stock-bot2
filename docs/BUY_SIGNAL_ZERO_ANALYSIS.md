# Buy Signal Rate 0% 문제 분석 및 해결 방안

## 문제 요약

실시간 거래 시스템에서 Buy Signal Rate가 0%로 나타나는 문제가 발생했습니다.
- 예측은 정상적으로 실행됨 (1,225회)
- 하지만 모든 예측이 HOLD 또는 SELL로 나옴
- BUY 신호가 전혀 발생하지 않음

## 근본 원인 분석

### 1. 정규화 문제 (해결됨 ✅)

**문제**: 이중 정규화
- 정규화된 데이터의 통계를 원본 데이터에 적용
- 결과: 정규화 후 평균이 2.7~2.8 (정상은 ~0)

**해결**: 원본 데이터의 정규화 통계 사용
```python
# models/grpo_scalping@2025120/raw_normalization_stats.json
# 원본 데이터에서 직접 계산한 통계 사용
```

### 2. 체결강도=0 필터링 (해결됨 ✅)

**문제**: 거래 비활발 종목에 대한 예측 실행
- 체결강도=0인 종목은 거래가 없는 상태
- 이런 종목에 대한 예측은 의미 없음

**해결**: 실시간 데이터 필터링
```python
# scripts/live/live_trading.py
if 체결강도 == 0:
    continue  # 예측 건너뛰기
```

### 3. 분포 이동 (Distribution Shift) - 핵심 문제 ⚠️

**문제**: 학습 데이터와 실시간 데이터의 시장 환경 차이

| 항목 | 학습 데이터 | 실시간 데이터 | 차이 |
|------|------------|--------------|------|
| 기간 | 2024년 10월 | 2024년 11월 | 1개월 차이 |
| 시장 상황 | 특정 시장 환경 | 다른 시장 환경 | 분포 이동 |
| 정규화 통계 | 10월 데이터 기준 | 10월 통계 사용 | 불일치 |

**결과**:
- 모델이 학습한 패턴과 실시간 패턴이 다름
- 정규화 후에도 특징 분포가 학습 시와 다름
- 모델이 안전하게 HOLD/SELL만 선택

## 해결 방안

### 방안 1: Rolling Window 정규화 (권장 ⭐)

**개념**: 학습 통계 대신 최근 N개 데이터의 통계 사용

**장점**:
- 시장 환경 변화에 실시간 적응
- 분포 이동 문제 자동 해결
- 추가 학습 불필요

**구현**:
```python
# lib/rolling_normalization.py (이미 구현됨)
from lib.rolling_normalization import RollingNormalizer

normalizer = RollingNormalizer(
    window_size=1000,  # 최근 1000개 데이터
    min_samples=100    # 최소 100개 수집 후 정규화
)

# 실시간 사용
normalized = normalizer.normalize(features, update=True)
```

**적용 방법**:
```python
# scripts/live/live_trading.py 수정
# 기존: PerStockNormalizer with training stats
# 변경: RollingNormalizer

from lib.rolling_normalization import RollingNormalizer

self.normalizer = RollingNormalizer(
    window_size=1000,
    min_samples=100,
    feature_names=FEATURE_NAMES
)
```

### 방안 2: 시간대별 정규화

**개념**: 시간대마다 다른 정규화 통계 사용

**장점**:
- 장 초반/중반의 특성 차이 반영
- 시간대별 거래 패턴 고려

**구현**:
```python
from lib.rolling_normalization import TimeBasedNormalizer

normalizer = TimeBasedNormalizer(
    time_windows=[
        (90000, 93000),   # 09:00-09:30
        (93000, 100000),  # 09:30-10:00
        (100000, 103000), # 10:00-10:30
        (103000, 110000), # 10:30-11:00
    ],
    window_size=500,
    min_samples=50
)

# 사용
normalized = normalizer.normalize(features, time_value=90530, update=True)
```

### 방안 3: 모델 재학습

**개념**: 최신 데이터(11월)로 모델 재학습

**장점**:
- 최신 시장 환경에 최적화
- 근본적인 해결

**단점**:
- 시간 소요 (2-3시간)
- 정기적 재학습 필요

**실행**:
```bash
# 최신 데이터로 재학습
python scripts/grpo/quick_train_grpo.py \
    --db datasets_202411.duckdb \
    --out models/grpo_scalping_202411
```

### 방안 4: 온라인 학습 (장기 해결책)

**개념**: 실시간으로 모델 업데이트

**장점**:
- 지속적인 적응
- 재학습 불필요

**단점**:
- 구현 복잡도 높음
- 안정성 검증 필요

## 권장 실행 순서

### 1단계: Rolling Window 정규화 적용 (즉시 실행 가능)

```bash
# 1. live_trading.py 수정
# PerStockNormalizer → RollingNormalizer

# 2. 테스트 실행
python scripts/live/live_trading.py --config config/trading_config.json

# 3. 결과 확인
# - Buy Signal Rate > 0% 확인
# - 정규화 통계 모니터링
```

### 2단계: 결과 모니터링 (1-2일)

- Buy Signal Rate 추이 관찰
- 거래 성과 분석
- 정규화 통계 안정성 확인

### 3단계: 필요시 추가 조치

**Case A: Buy Signal Rate 여전히 낮음 (< 5%)**
→ 모델 재학습 고려

**Case B: Buy Signal Rate 정상 (5-20%)**
→ 현재 설정 유지

**Case C: Buy Signal Rate 너무 높음 (> 30%)**
→ 신뢰도 임계값 조정

## 구현 예제

### Rolling Window 정규화 적용

```python
# scripts/live/live_trading.py 수정

# 기존 코드 (line 260-300)
from scripts.live.online_normalizer import PerStockNormalizer

self.normalizer = PerStockNormalizer(
    num_features=config.num_features,
    window_size=config.online_window_size,
    warmup_samples=config.online_warmup_samples,
    use_training_stats=use_training_stats,
    training_mean=training_mean,
    training_std=training_std
)

# 새로운 코드
from lib.rolling_normalization import RollingNormalizer

# 종목별 normalizer 딕셔너리
self.normalizers = {}

def get_normalizer(self, stock_code):
    """종목별 normalizer 반환"""
    if stock_code not in self.normalizers:
        self.normalizers[stock_code] = RollingNormalizer(
            window_size=1000,
            min_samples=100,
            feature_names=FEATURE_NAMES
        )
    return self.normalizers[stock_code]

# 사용 (line 687)
# 기존
features_array = self.normalizer.normalize(code, features_array, update=True)

# 변경
normalizer = self.get_normalizer(code)
features_array = normalizer.normalize(features_array, update=True)
```

## 예상 결과

### Rolling Window 정규화 적용 후

| 지표 | 현재 | 예상 |
|------|------|------|
| Buy Signal Rate | 0% | 5-15% |
| 예측 실행 | 1,225회 | 유지 |
| 정규화 품질 | 정상 | 유지 |
| 거래 발생 | 0건 | 5-20건/일 |

### 성공 기준

1. **Buy Signal Rate**: 5% 이상
2. **정규화 통계**: 평균 ~0, 표준편차 ~1
3. **거래 발생**: 하루 5건 이상
4. **Win Rate**: 30% 이상 (백테스트 기준)

## 모니터링 체크리스트

### 실시간 모니터링

- [ ] Buy Signal Rate (1분마다)
- [ ] 정규화 통계 (5분마다)
- [ ] 예측 실행 횟수
- [ ] 체결강도=0 필터링 비율

### 일일 분석

- [ ] 총 거래 건수
- [ ] Win Rate
- [ ] 평균 수익률
- [ ] 최대 손실

### 주간 분석

- [ ] 누적 수익률
- [ ] Sharpe Ratio
- [ ] Maximum Drawdown
- [ ] 모델 재학습 필요성 검토

## 참고 자료

- `lib/rolling_normalization.py`: Rolling window 정규화 구현
- `scripts/data/analyze_distribution_shift.py`: 분포 이동 분석 도구
- `docs/DATA_QUALITY_VALIDATION.md`: 데이터 품질 검증 가이드
- `docs/NORMALIZATION_VALIDATION_REPORT.md`: 정규화 검증 보고서

## 결론

Buy Signal Rate 0% 문제는 **분포 이동(Distribution Shift)**이 근본 원인입니다.

**즉시 실행 가능한 해결책**:
1. Rolling Window 정규화 적용
2. 실시간 모니터링
3. 필요시 모델 재학습

**장기 해결책**:
1. 정기적 모델 재학습 (월 1회)
2. 온라인 학습 시스템 구축
3. 앙상블 모델 (여러 시기 학습 모델 조합)

---

**작성일**: 2024-11-17
**작성자**: AI Trading System
**버전**: 1.0
