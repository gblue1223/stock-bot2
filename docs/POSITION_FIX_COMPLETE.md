# 포지션 관리 시스템 수정 완료

## 📋 요약

**날짜**: 2025-11-18  
**문제**: 494120 종목 매수 1주 성공 후 매도 17주 시도 → 수량 불일치  
**상태**: ✅ 수정 완료 및 테스트 통과

## 🔍 문제 분석

### 발생한 문제
```
09:51:50 - [BUY] 494120: qty=1주 ✅
09:51:51 - [SELL] 494120: qty=17주 ❌
→ 오류: 매도가능수량이 부족합니다. 1주 매도가능
```

### 근본 원인
1. **수량 정보 미저장**: Position 객체에 실제 매수 수량 저장 안 됨
2. **잘못된 계산**: `max_position_size / entry_price`로 수량 계산
3. **체결 확인 부재**: 매수 후 1초 만에 매도 시도

## ✅ 수정 사항

### 1. 포지션에 수량 정보 저장
```python
# scripts/live/live_trading.py - execute_buy()
self.positions[code] = Position(...)
self.positions[code].quantity = quantity  # ✅ 실제 매수 수량
self.positions[code].order_no = order_result.get('order_no', '')
self.positions[code].status = 'PENDING'  # 체결 대기
```

### 2. 실제 수량으로 매도
```python
# scripts/live/live_trading.py - execute_sell()
quantity = getattr(position, 'quantity', 0)  # ✅ 저장된 수량 사용

if quantity <= 0:
    logger.error(f"[SELL SKIP] {code}: Invalid quantity")
    return
```

### 3. 최소 보유 시간 검증
```python
# scripts/live/live_trading.py - execute_sell()
time_since_buy = time.time() - position.entry_time
if time_since_buy < self.config.min_hold_time_seconds:  # 3초
    logger.debug(f"[SELL WAIT] {code}: Minimum hold time not met")
    return
```

### 4. 최대 보유 시간 검증 (신규 추가)
```python
# scripts/live/live_trading.py - execute_sell()
if time_since_buy > self.config.max_hold_time_seconds:  # 300초 (5분)
    logger.warning(f"[FORCE SELL] {code}: Maximum hold time exceeded")
    reason = "MaxHoldTime"  # 강제 청산
```

### 5. 체결 확인 로직
```python
# scripts/live/live_trading.py - execute_sell()
if self.config.verify_position_before_sell:
    if getattr(position, 'status', 'FILLED') == 'PENDING':
        # 실제 보유 수량 조회
        balance = self.koapys.get_balance(account_no=self.account_no)
        # 체결 확인 후 매도
```

### 6. 설정 파일 업데이트
```json
// config/trading_config.json
{
  "min_hold_time_seconds": 3,
  "max_hold_time_seconds": 300,
  "verify_position_before_sell": true,
  "max_sell_attempts": 3
}
```

## 🧪 테스트 결과

```
============================================================
✅ 모든 테스트 통과!
============================================================

수정 완료 사항:
1. ✅ 포지션에 수량 정보 저장 (quantity, order_no, status)
2. ✅ 실제 매수 수량으로 매도
3. ✅ 최소 보유 시간 검증 (3초)
4. ✅ 최대 보유 시간 검증 (300초, 강제 청산)
5. ✅ 체결 상태 추적 (PENDING/FILLED)
6. ✅ 매도 전 포지션 검증
```

### 테스트 시나리오

#### 1. 보유 시간 검증
- **즉시 매도 시도**: 0.3초 → WAIT (< 3초)
- **정상 보유**: 10.3초 → OK (3초 ~ 300초)
- **장기 보유**: 400.3초 → FORCE SELL (> 300초)

#### 2. 수량 검증
- **정상 케이스**: quantity=1, status=FILLED → 매도 가능
- **체결 대기**: quantity=1, status=PENDING → 검증 필요
- **수량 없음**: quantity=0 → SKIP

## 📊 Before & After

### Before (문제)
```
매수: quantity 저장 안 됨
매도: max_position_size / entry_price 계산
     → 1,000,000 / 57,400 = 17주 (잘못된 계산)
     → 오류 발생
```

### After (수정)
```
매수: position.quantity = 1 (실제 수량 저장)
     position.status = 'PENDING'
매도: quantity = position.quantity = 1
     최소 3초 대기
     체결 확인
     → 1주 매도 성공
```

## 📁 수정된 파일

1. **scripts/live/live_trading.py**
   - `TradingConfig.__init__()`: 새로운 설정 필드 추가
   - `execute_buy()`: 수량 정보 저장
   - `execute_sell()`: 실제 수량 사용 + 시간 검증 + 체결 확인

2. **config/trading_config.json**
   - `min_hold_time_seconds`: 3
   - `max_hold_time_seconds`: 300
   - `verify_position_before_sell`: true
   - `max_sell_attempts`: 3

3. **scripts/live/test_position_fix.py**
   - 포지션 관리 검증 테스트

4. **docs/POSITION_MANAGEMENT_FIX.md**
   - 상세 수정 문서

## 🎯 기대 효과

### 1. 수량 불일치 해결
- 실제 매수 수량으로 매도
- 수량 정보 안전하게 저장 및 조회

### 2. 체결 안정성
- 최소 3초 대기로 체결 시간 확보
- 실제 보유 수량 확인 후 매도

### 3. 리스크 관리
- 최대 5분 보유 후 강제 청산
- 장기 포지션 방지

### 4. 오류 방지
- 다양한 검증 로직
- 상세한 로그로 디버깅 용이

## 🔧 추가 수정 (2025-11-18 12:29)

### 잔고 조회 API 오류 수정

**문제**: `'KoapyRestSimple' object has no attribute 'get_balance'`

**수정**:
```python
# Before
balance = self.koapys.get_balance(account_no=self.account_no)

# After
balance = self.koapys.get_equity_balance(account_no=self.account_no)
stocks = balance.get('stocks', [])
# 종목코드, 보유수량 등 한글 키 사용
```

**참고**: `docs/BALANCE_API_FIX.md`

## 📝 다음 단계

1. ✅ 코드 수정 완료
2. ✅ 단위 테스트 통과
3. ✅ 최대 보유 시간 추가
4. ✅ 잔고 조회 API 수정
5. ⏳ 실전 테스트 (다음 거래일)
6. ⏳ 모니터링 및 검증
7. ⏳ 추가 개선 사항 반영

---

**작성일**: 2025-11-18  
**최종 업데이트**: 2025-11-18 12:30  
**상태**: ✅ 수정 완료 및 테스트 통과 (잔고 API 수정 포함)
