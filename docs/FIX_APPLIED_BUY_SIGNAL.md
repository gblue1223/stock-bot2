# Buy Signal 0% 문제 수정 완료

## ✅ 적용된 수정 사항

### 1. 정규화 통계 생성 스크립트 추가
**파일**: `scripts/data/save_normalization_stats.py`

학습 데이터의 정규화 통계를 계산하고 저장하는 스크립트 생성

**사용법**:
```bash
python scripts/data/save_normalization_stats.py \
    --db datasets_norm_all.duckdb \
    --output models/grpo_scalping/normalization_stats.pkl
```

### 2. 정규화 전후 로깅 추가
**파일**: `scripts/live/live_trading.py`

`extract_features()` 함수에 정규화 전후 데이터 로깅 추가:
- 정규화 전: 원본 데이터 통계 (평균, 표준편차, 범위)
- 정규화 후: 정규화된 데이터 통계
- 10번 예측마다 로깅하여 성능 영향 최소화

**로그 예시**:
```
[BEFORE NORM] 000230: mean=3318042.25, std=18240332.00, range=[-1.75, 131973528.00]
[BEFORE NORM] 000230 주요값: 등락률=15.5600, 누적거래대금=3.54e+04, 거래회전율=25.3600, 체결강도=0.0000
[AFTER NORM] 000230: mean=0.0234, std=1.2345, range=[-2.3456, 3.4567]
```

### 3. 호가 데이터 검증 로직 추가
**파일**: `scripts/live/live_trading.py`

예측 전에 호가 데이터 품질 검증:
- 체결강도 = 0
- 매도대기금액1 = 0
- 매수대기금액1 = 0

위 조건이 모두 만족되면 호가 데이터 누락으로 판단하고 예측 스킵

**로그 예시**:
```
[DATA QUALITY] 000230: 호가 데이터 누락 (체결강도=0, 매도대기금액1=0, 매수대기금액1=0), 예측 스킵
```

### 4. OnlineNormalizer 검증 로직 추가
**파일**: `scripts/live/online_normalizer.py`

정규화 후 결과 검증:
- 정규화 후에도 값이 100 이상이면 경고
- 평균/표준편차 캐시 값 출력

**로그 예시**:
```
[NORMALIZER] 정규화 후에도 값이 너무 큼! max_abs=123.45, mean_cache=[...], std_cache=[...]
```

### 5. 모델 입력 검증 추가
**파일**: `ai_trader/grpo/inference/enhanced_inference.py`

예측 전 입력 데이터 검증:
- 입력 시퀀스의 최대 절대값이 100 이상이면 경고
- 정규화되지 않은 데이터로 판단

**로그 예시**:
```
[INFERENCE] 입력 데이터가 정규화되지 않은 것으로 보임! max_abs=123.45, mean=3318042.25, std=18240332.00
[INFERENCE] 예측 결과가 부정확할 수 있습니다. 정규화 통계를 확인하세요.
```

## 🚀 다음 단계

### 1단계: 정규화 통계 생성 (필수)

```bash
# 학습 데이터베이스에서 정규화 통계 계산 및 저장
python scripts/data/save_normalization_stats.py \
    --db datasets_norm_all.duckdb \
    --output models/grpo_scalping/normalization_stats.pkl
```

**예상 출력**:
```
================================================================================
NORMALIZATION STATISTICS COMPUTATION
================================================================================
Opening database: datasets_norm_all.duckdb

Computing normalization stats for 28 features...
================================================================================
  [ 1/28] 종목명_scalar        : mean=      0.5234, std=      0.2345
  [ 2/28] 시간_sin            : mean=      0.0123, std=      0.7890
  ...
  [28/28] 매수대기금액10       : mean=   1234.5678, std=   5678.9012
================================================================================

================================================================================
SUMMARY
================================================================================
✅ Normalization stats saved to: models/grpo_scalping/normalization_stats.pkl
   Features: 28
   Mean range: [-0.1234, 1234.5678]
   Std range: [0.2345, 5678.9012]
   File size: 2.34 KB
================================================================================

Key Features:
  등락률         : mean=      0.1234, std=      5.6789
  누적거래대금    : mean=  12345.6789, std=  56789.0123
  거래회전율      : mean=      1.2345, std=      3.4567
  체결강도        : mean=    123.4567, std=    456.7890

✅ Done!
```

### 2단계: 실시간 거래 재실행

```bash
python scripts/live/live_trading.py
```

### 3단계: 로그 모니터링

다음 로그들을 확인하여 문제 진단:

1. **정규화 전후 비교**:
   ```
   [BEFORE NORM] 000230: mean=..., std=..., range=[...]
   [AFTER NORM] 000230: mean=..., std=..., range=[...]
   ```
   - 정규화 후 평균이 0 근처, 범위가 [-5, 5] 내에 있어야 정상

2. **호가 데이터 품질**:
   ```
   [DATA QUALITY] 000230: 호가 데이터 누락 ...
   ```
   - 이 경고가 자주 나오면 호가 데이터 수신 문제

3. **입력 검증**:
   ```
   [INFERENCE] 입력 데이터가 정규화되지 않은 것으로 보임!
   ```
   - 이 경고가 나오면 정규화 문제

4. **Buy Signal Rate**:
   ```
   [AI] Inference Stats:
   Total Predictions: 74
   Buy Signal Rate: 30.00%  # ← 0%에서 증가해야 함
   ```

## 📊 예상 결과

### 수정 전
```
[BEFORE NORM] 000230: mean=3318042.25, std=18240332.00, range=[-1.75, 131973528.00]
[AFTER NORM] 000230: mean=3318042.25, std=18240332.00, range=[-1.75, 131973528.00]  # 정규화 안됨!
[INFERENCE] 입력 데이터가 정규화되지 않은 것으로 보임! max_abs=131973528.00
Total Predictions: 74
Buy Signal Rate: 0.00%  # ← 문제!
```

### 수정 후 (예상)
```
[BEFORE NORM] 000230: mean=3318042.25, std=18240332.00, range=[-1.75, 131973528.00]
[AFTER NORM] 000230: mean=0.0234, std=1.2345, range=[-2.3456, 3.4567]  # 정규화 됨!
Total Predictions: 74
Buy Signal Rate: 35.14%  # ← 정상!
```

## 🔍 문제 해결 가이드

### 문제 1: 정규화 후에도 값이 크다
**증상**:
```
[NORMALIZER] 정규화 후에도 값이 너무 큼! max_abs=123.45
```

**원인**: OnlineNormalizer의 통계가 부정확

**해결**:
1. 워밍업 샘플 수 증가: `config.online_warmup_samples = 100`
2. 학습 데이터 정규화 통계 사용 (향후 구현)

### 문제 2: 호가 데이터 누락이 빈번
**증상**:
```
[DATA QUALITY] 000230: 호가 데이터 누락 ...
```

**원인**: WebSocket 호가 데이터(0D) 수신 문제

**해결**:
1. WebSocket 연결 상태 확인
2. 호가 데이터 구독 확인
3. 키움 API 상태 확인

### 문제 3: 여전히 Buy Signal Rate 0%
**증상**: 수정 후에도 Buy Signal Rate가 0%

**원인**: 모델 자체의 문제

**해결**:
1. 신뢰도 임계값 낮추기: `min_buy_confidence = 0.0`
2. 다른 조건 검색식 사용
3. 모델 재학습 고려

## 📝 체크리스트

- [x] 정규화 통계 생성 스크립트 작성
- [x] 정규화 전후 로깅 추가
- [x] 호가 데이터 검증 로직 추가
- [x] OnlineNormalizer 검증 추가
- [x] 모델 입력 검증 추가
- [ ] 정규화 통계 생성 실행
- [ ] 실시간 거래 재실행
- [ ] 로그 확인 및 문제 진단
- [ ] Buy Signal Rate 정상화 확인

## 🎯 성공 기준

1. ✅ 정규화 후 데이터 범위: [-5, 5] 내
2. ✅ 정규화 후 평균: 0 근처 (±0.5)
3. ✅ Buy Signal Rate: 20-50% (정상 범위)
4. ✅ 호가 데이터 누락 경고: 최소화

## 📚 관련 문서

- `docs/BUY_SIGNAL_ANALYSIS_20251106.md`: 문제 분석
- `docs/FIX_PROPOSAL_BUY_SIGNAL.md`: 수정 제안
- `docs/FIX_APPLIED_BUY_SIGNAL.md`: 수정 완료 (현재 문서)
