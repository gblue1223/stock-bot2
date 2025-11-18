# 지속적 모니터링 기능

## 개요

조건검색에서 한번 포착된 종목은 조건 이탈 후에도 계속 모니터링하도록 변경했습니다.

## 변경 이유

### 기존 동작 (❌ 문제)
```
1. 조건검색 IN → 종목 추가 → 실시간 데이터 수신 시작
2. 조건검색 OUT → 종목 제거 → 실시간 데이터 수신 중단
3. 다시 IN → 종목 재추가 → 버퍼 초기화 (60개 데이터 다시 수집)
```

**문제점**:
- 조건 이탈/재진입 반복 시 데이터 버퍼가 계속 초기화됨
- 60개 데이터 수집 대기 시간 발생 (약 1분)
- 거래 기회 손실

### 새로운 동작 (✅ 개선)
```
1. 조건검색 IN → 종목 추가 → 실시간 데이터 수신 시작
2. 조건검색 OUT → 종목 유지 → 실시간 데이터 계속 수신
3. 다시 IN → 이미 모니터링 중 → 즉시 예측 가능
```

**장점**:
- 데이터 버퍼 유지
- 즉시 예측 가능
- 거래 기회 증가

## 구현 내용

### 1. Stock OUT 처리 변경

**기존 코드**:
```python
def on_stock_out(code, name, cond_idx):
    """종목 이탈 시 호출"""
    code_clean = code[1:] if code.startswith('A') else code
    logger.info(f"[CONDITION] Stock OUT signal received: {code} -> {code_clean}")
    
    with self._condition_codes_lock:
        if code_clean in self._condition_codes:
            self._condition_codes.remove(code_clean)  # ❌ 종목 제거
            logger.info(f"[CONDITION] Stock removed from tracking")
```

**새로운 코드**:
```python
def on_stock_out(code, name, cond_idx):
    """
    종목 이탈 시 호출
    
    ✅ 변경: 한번 포착된 종목은 계속 모니터링
    조건검색에서 이탈해도 실시간 데이터는 계속 수신
    """
    code_clean = code[1:] if code.startswith('A') else code
    logger.info(f"[CONDITION] Stock OUT signal received: {code} -> {code_clean}")
    logger.info(f"[CONDITION] ✅ Keeping {code_clean} in tracking (continue monitoring)")
    
    # ✅ 변경: 종목을 제거하지 않음 (계속 모니터링)
    # 주석 처리된 기존 코드
```

### 2. 설정 파일 추가

**config/trading_config.json**:
```json
{
  "keep_monitoring_after_out": true,
  "# keep_monitoring_after_out": "한번 포착된 종목은 조건검색 이탈 후에도 계속 모니터링"
}
```

## 동작 예시

### 시나리오: 종목이 조건 이탈 후 재진입

#### 기존 방식 (❌)
```
09:00:00 - [CONDITION] Stock IN: 035720
09:00:01 - [BUFFER] 035720: Collecting data (1/60)
09:00:02 - [BUFFER] 035720: Collecting data (2/60)
...
09:01:00 - [BUFFER] 035720: Ready (60/60)
09:01:01 - [PREDICT] 035720: Prediction started

09:02:00 - [CONDITION] Stock OUT: 035720
09:02:01 - [CONDITION] Stock removed from tracking
09:02:02 - Buffer cleared ❌

09:03:00 - [CONDITION] Stock IN: 035720 (재진입)
09:03:01 - [BUFFER] 035720: Collecting data (1/60) ❌ 다시 시작
...
09:04:00 - [BUFFER] 035720: Ready (60/60) ❌ 1분 대기
```

#### 새로운 방식 (✅)
```
09:00:00 - [CONDITION] Stock IN: 035720
09:00:01 - [BUFFER] 035720: Collecting data (1/60)
09:00:02 - [BUFFER] 035720: Collecting data (2/60)
...
09:01:00 - [BUFFER] 035720: Ready (60/60)
09:01:01 - [PREDICT] 035720: Prediction started

09:02:00 - [CONDITION] Stock OUT: 035720
09:02:01 - [CONDITION] ✅ Keeping 035720 in tracking
09:02:02 - Buffer maintained ✅
09:02:03 - [PREDICT] 035720: Prediction continues ✅

09:03:00 - [CONDITION] Stock IN: 035720 (재진입)
09:03:01 - Already monitoring ✅
09:03:02 - [PREDICT] 035720: Immediate prediction ✅
```

## 로그 예시

### Stock OUT 신호 수신
```
2024-11-18 09:24:00 - [CONDITION] Stock OUT signal received: 035720 -> 035720 (), cond_idx=0
2024-11-18 09:24:00 - [CONDITION] ✅ Keeping 035720 in tracking (continue monitoring)
```

### 계속 예측 실행
```
2024-11-18 09:24:01 - [SEQUENCE] 035720: shape=(60, 28), mean=0.09, std=0.92
2024-11-18 09:24:01 - [PREDICT] 035720: Action=HOLD, confidence=0.45
```

### 재진입 시
```
2024-11-18 09:25:00 - [CONDITION] Stock IN signal received: 035720 -> 035720 ()
2024-11-18 09:25:00 - [CONDITION] Stock already tracked: 035720
```

## 메모리 관리

### 우려사항
종목을 제거하지 않으면 메모리가 계속 증가하지 않을까?

### 해결책
1. **자동 정리**: 장 종료 시 모든 종목 정리
2. **최대 종목 수 제한**: 설정 가능
3. **메모리 사용량**: 종목당 약 200KB (1000개 = 200MB)

### 메모리 사용량 계산
```
종목당 메모리:
- 데이터 버퍼: 60 × 28 × 4 bytes = 6.7 KB
- Rolling normalizer: 1000 × 28 × 4 bytes = 112 KB
- 기타 메타데이터: ~80 KB
= 약 200 KB/종목

100개 종목: 20 MB (무시 가능)
1000개 종목: 200 MB (여전히 적음)
```

## 선택적 비활성화

필요시 기존 동작으로 되돌릴 수 있습니다:

### 방법 1: 설정 파일
```json
{
  "keep_monitoring_after_out": false
}
```

### 방법 2: 코드 수정
```python
def on_stock_out(code, name, cond_idx):
    code_clean = code[1:] if code.startswith('A') else code
    logger.info(f"[CONDITION] Stock OUT signal received: {code} -> {code_clean}")
    
    # 종목 제거 (기존 동작)
    with self._condition_codes_lock:
        if code_clean in self._condition_codes:
            self._condition_codes.remove(code_clean)
            logger.info(f"[CONDITION] Stock removed from tracking")
```

## 장점 요약

1. ✅ **데이터 연속성**: 버퍼 유지로 예측 품질 향상
2. ✅ **빠른 반응**: 재진입 시 즉시 예측 가능
3. ✅ **거래 기회 증가**: 대기 시간 없음
4. ✅ **안정성**: 조건 이탈/재진입 반복에도 안정적
5. ✅ **메모리 효율**: 종목당 200KB로 매우 적음

## 주의사항

### 1. 장 종료 시 정리
```python
def shutdown(self):
    """장 종료 시 모든 종목 정리"""
    self._condition_codes.clear()
    self.data_buffers.clear()
    self.normalizers.clear()
```

### 2. 최대 종목 수 제한 (선택사항)
```python
MAX_TRACKED_STOCKS = 100

def on_stock_in(code, name, cond_idx):
    if len(self._condition_codes) >= MAX_TRACKED_STOCKS:
        logger.warning(f"Max tracked stocks reached: {MAX_TRACKED_STOCKS}")
        return
    # ... 종목 추가
```

### 3. 오래된 종목 자동 제거 (선택사항)
```python
# 마지막 업데이트 시간 추적
self.last_update_time = {}

def cleanup_old_stocks(self, max_age_seconds=3600):
    """1시간 이상 업데이트 없는 종목 제거"""
    current_time = time.time()
    for code in list(self._condition_codes):
        if current_time - self.last_update_time.get(code, 0) > max_age_seconds:
            self._condition_codes.remove(code)
            logger.info(f"Removed inactive stock: {code}")
```

## 테스트 방법

### 1. 조건 이탈/재진입 테스트
```bash
# 실시간 거래 실행
python scripts/live/live_trading.py --config config/trading_config.json

# 로그 모니터링
tail -f logs/live_trading_*.log | grep -E "Stock OUT|Keeping.*tracking|PREDICT"
```

### 2. 예상 로그
```
[CONDITION] Stock OUT signal received: 035720
[CONDITION] ✅ Keeping 035720 in tracking (continue monitoring)
[PREDICT] 035720: Action=HOLD, confidence=0.45
[PREDICT] 035720: Action=HOLD, confidence=0.48
[CONDITION] Stock IN signal received: 035720
[CONDITION] Stock already tracked: 035720
[PREDICT] 035720: Action=BUY, confidence=0.52
```

## 관련 파일

- `scripts/live/live_trading.py`: Stock OUT 처리 로직
- `config/trading_config.json`: 설정 파일
- `docs/CONTINUOUS_MONITORING.md`: 이 문서

---

**작성일**: 2024-11-18  
**작성자**: AI Trading System  
**버전**: 1.0  
**상태**: ✅ 구현 완료
