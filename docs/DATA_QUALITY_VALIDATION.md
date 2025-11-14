# 데이터 품질 검증 가이드

**날짜**: 2025-11-14  
**목적**: 실시간 트레이딩에서 데이터 품질 문제 감지 및 처리

---

## 📋 문제 정의

### 발견된 문제
실시간 트레이딩 중 **체결강도=0**인 데이터가 수신되어 모델이 잘못된 예측을 생성:

```
[BEFORE NORM] 체결강도=0.0000
[AFTER NORM] 체결강도=-5.0000  ← 클리핑됨
→ 모델이 모든 예측을 SELL로 출력 (Buy Signal Rate: 0.00%)
```

### 원인
1. **호가 데이터 누락**: WebSocket에서 호가 정보가 제대로 수신되지 않음
2. **체결강도 계산 실패**: 호가 데이터 없이는 체결강도를 계산할 수 없음
3. **정규화 왜곡**: 체결강도=0 → 로그 변환 → -∞ → 클리핑(-5.0)

---

## 🛡️ 구현된 검증 메커니즘

### 1. 거래 활발도 필터링 (on_data_update)

**위치**: `scripts/live/live_trading.py` - `on_data_update()` 콜백

```python
# ✅ 거래 활발도 검증 (체결강도 > 0)
if stock_data.체결강도 == 0:
    # 체결강도=0인 종목은 거래가 활발하지 않음
    logger.debug(
        f"[INACTIVE] {code}: 체결강도=0 (거래 비활발), "
        f"등락률={stock_data.등락률:.2f}%, 거래회전율={stock_data.거래회전율:.2f}%"
    )
    return
```

**검증 항목**:
- ✅ 체결강도 = 0 → 데이터 수신 스킵 (거래 비활발)
- ⏱️ 1분마다 디버그 로그 (중복 방지)
- 🎯 **핵심**: 거래가 활발한 종목만 처리

### 2. 실시간 데이터 검증 (process_stock)

**위치**: `scripts/live/live_trading.py` - `process_stock()` 메서드

```python
# ✅ 거래 활발도 검증
if self.realtime_client:
    stock_data = self.realtime_client.get_stock_data(code)
    if stock_data:
        # 체결강도가 0이면 거래 비활발
        if stock_data.체결강도 == 0:
            return
        
        # 매도/매수 대기금액이 모두 0인 경우
        if (stock_data.매도대기금액1 == 0 and stock_data.매수대기금액1 == 0):
            logger.debug(f"[DATA QUALITY] {code}: 매도/매수대기금액=0")
            return
```

**검증 항목**:
- ✅ 체결강도 = 0 → 예측 스킵
- ✅ 매도대기금액1 = 0 AND 매수대기금액1 = 0 → 예측 스킵
- ⏱️ 30초마다 디버그 로그

### 3. 특징 추출 모니터링 (extract_features)

**위치**: `scripts/live/live_trading.py` - `extract_features()` 메서드

```python
# ✅ 체결강도 모니터링 (정규화 전)
if len(features_array) > 7 and features_array[7] == 0.0:
    logger.debug(
        f"[FEATURE INFO] {code}: 체결강도=0 in features "
        f"(이미 on_data_update에서 필터링됨)"
    )
```

**검증 항목**:
- ℹ️ features[7] (체결강도) = 0 → 정보성 로그만
- ⏱️ 1분마다 디버그 로그
- 📝 **변경**: 제로 벡터 반환하지 않음 (이미 필터링됨)

---

## 📊 검증 흐름도

```
실시간 데이터 수신 (on_data_update)
    ↓
[1단계] 거래 활발도 필터링 ⭐ 핵심
    ├─ 체결강도 = 0? → 데이터 수신 스킵 (거래 비활발)
    └─ 체결강도 > 0 → 다음 단계 진행
    ↓
[2단계] 특징 추출
    ├─ extract_features() 호출
    └─ 정규화 적용
    ↓
[3단계] 버퍼 업데이트
    └─ 정상 데이터 → 버퍼 추가
    ↓
[4단계] 시퀀스 준비 확인 (process_stock)
    ├─ 체결강도 재검증
    ├─ 버퍼 길이 < 60? → 대기
    └─ 버퍼 길이 = 60 → 예측 실행
    ↓
모델 예측
```

**핵심 변경사항**:
- ✅ **1단계에서 거래 비활발 종목 필터링** (체결강도=0)
- ✅ 활발한 종목만 버퍼에 추가
- ✅ 제로 벡터 검증 제거 (불필요)

---

## 🔍 모니터링 방법

### 로그 확인

#### 1. 거래 비활발 종목 필터링
```bash
grep "INACTIVE" logs/live_trading_*.log
```

**예시 출력**:
```
[INACTIVE] 015760: 체결강도=0 (거래 비활발), 등락률=2.54%, 거래회전율=0.63%
[INACTIVE] 054540: 체결강도=0 (거래 비활발), 등락률=17.09%, 거래회전율=73.97%
```

#### 2. 데이터 품질 경고
```bash
grep "DATA QUALITY" logs/live_trading_*.log
```

**예시 출력**:
```
[DATA QUALITY] 015760: 매도/매수대기금액=0
```

#### 3. 활발한 종목 처리
```bash
grep "Buffer.*60/60" logs/live_trading_*.log
```

**예시 출력**:
```
[REALTIME] 005930: Buffer 59 -> 60/60  ← 삼성전자 (활발)
[REALTIME] 000660: Buffer 59 -> 60/60  ← SK하이닉스 (활발)
```

### 통계 확인

#### AI 통계 (1분마다)
```
[AI] Inference Stats:
Total Predictions: 91
Buy Signal Rate: 0.00%  ← 문제 지표
```

**정상 범위**:
- Buy Signal Rate: 5-20% (학습 데이터 기준)
- 0%가 지속되면 데이터 품질 문제 의심

---

## 🚨 문제 해결 가이드

### 문제 1: Buy Signal Rate = 0%

**증상**:
```
Total Predictions: 100+
Buy Signal Rate: 0.00%
```

**원인**:
- 체결강도=0 데이터만 수신
- 호가 데이터 누락

**해결**:
1. 로그 확인: `grep "DATA QUALITY" logs/live_trading_*.log`
2. WebSocket 연결 상태 확인
3. 조건검색 종목 확인: 거래량이 있는 종목인가?

### 문제 2: 모든 예측이 SELL

**증상**:
```
[PREDICT] 015760: SELL signal (반복)
```

**원인**:
- 정규화 후 값이 음수로 편향
- 체결강도=-5.0 (클리핑)

**해결**:
1. 정규화 후 값 확인: `grep "AFTER NORM" logs/live_trading_*.log`
2. 체결강도 검증 로그 확인
3. 호가 데이터 수신 확인

### 문제 3: 버퍼가 채워지지 않음

**증상**:
```
[BUFFER] 015760: Collecting data (10/60)  ← 계속 증가하지 않음
```

**원인**:
- 제로 벡터만 수신되어 버퍼 업데이트 스킵
- 실시간 데이터 수신 중단

**해결**:
1. 제로 벡터 로그 확인: `grep "Zero vector" logs/live_trading_*.log`
2. 실시간 데이터 수신 확인: `grep "REALTIME.*Received" logs/live_trading_*.log`
3. WebSocket 재연결

---

## ✅ 검증 체크리스트

실시간 트레이딩 시작 전:

- [ ] 정규화 통계 로딩 확인
  ```
  ✅ Loaded RAW normalization stats
  ```

- [ ] 실시간 데이터 수신 확인
  ```
  [REALTIME] Received X data updates in last 5 seconds
  ```

- [ ] 조건검색 종목 확인
  ```
  [CONDITION] Current tracked stocks: X
  ```

실시간 트레이딩 중 (5분마다):

- [ ] Buy Signal Rate 확인 (> 0%)
- [ ] 데이터 품질 경고 빈도 확인 (< 10%)
- [ ] 버퍼 상태 확인 (60/60 도달)

---

## 📈 성능 지표

### 정상 동작 기준

| 지표 | 정상 범위 | 경고 | 위험 |
|------|----------|------|------|
| Buy Signal Rate | 5-20% | 0-5% | 0% |
| 데이터 품질 경고 | < 5% | 5-10% | > 10% |
| 제로 벡터 비율 | < 1% | 1-5% | > 5% |
| 버퍼 채움 시간 | < 60초 | 60-120초 | > 120초 |

### 예시: 정상 동작

```
[12:40:00] Total Predictions: 50
[12:40:00] Buy Signal Rate: 12.00%  ✅
[12:40:00] 데이터 품질 경고: 2회 (4%)  ✅
[12:40:00] 제로 벡터: 1회 (2%)  ✅
```

### 예시: 문제 상황

```
[12:40:00] Total Predictions: 91
[12:40:00] Buy Signal Rate: 0.00%  ❌
[12:40:00] 데이터 품질 경고: 45회 (49%)  ❌
[12:40:00] 제로 벡터: 30회 (33%)  ❌
```

---

## 🔧 추가 개선 사항

### 향후 구현 예정

1. **자동 복구 메커니즘**
   - 데이터 품질 문제 지속 시 WebSocket 재연결
   - 종목 자동 교체 (품질 좋은 종목으로)

2. **실시간 대시보드**
   - 데이터 품질 지표 시각화
   - 종목별 품질 점수

3. **알림 시스템**
   - 데이터 품질 저하 시 알림
   - Buy Signal Rate 0% 지속 시 알림

---

## 📚 관련 문서

- [정규화 검증 보고서](NORMALIZATION_VALIDATION_REPORT.md)
- [실시간 트레이딩 가이드](../scripts/live/results/DEPLOYMENT_READY.md)
- [구현 완료 보고서](../scripts/live/results/IMPLEMENTATION_COMPLETE.md)

---

**작성**: 2025-11-14  
**버전**: 1.0  
**상태**: ✅ 구현 완료
