# 주문 실행 오류 수정

## 문제

실시간 거래 시스템에서 BUY/SELL 신호는 발생하지만 실제 주문이 실패하는 문제가 발생했습니다.

### 오류 메시지
```
'return_msg': '[2000](502048:단가를 입력하지 않는 호가입니다)'
'return_code': 20
```

## 원인 분석

### 기존 코드
```python
order_result = self.koapys.send_order(
    rqname=f"BUY_{code}_{datetime.now().strftime('%H%M%S')}",
    account_no=self.account_no,
    order_type=OrderType.BUY,
    code=code,
    quantity=quantity,
    price=int(current_price),  # ❌ 문제: 시장가 주문에 가격 지정
    hoga=OrderBookType.MARKET_IOC  # ❌ 문제: IOC 타입
)
```

### 문제점
1. **시장가 주문에 가격 지정**: `MARKET_IOC`는 시장가 주문인데 `price`를 지정함
2. **키움증권 API 요구사항**: 시장가 주문 시 `price=0`으로 설정해야 함
3. **호가 타입**: `MARKET_IOC` 대신 `MARKET` 사용 권장

## 해결 방법

### 수정된 코드

#### BUY 주문
```python
# 시장가 주문 (가격 0으로 설정)
order_result = self.koapys.send_order(
    rqname=f"BUY_{code}_{datetime.now().strftime('%H%M%S')}",
    account_no=self.account_no,
    order_type=OrderType.BUY,
    code=code,
    quantity=quantity,
    price=0,  # ✅ 시장가는 0
    hoga=OrderBookType.MARKET  # ✅ MARKET 사용
)
```

#### SELL 주문
```python
# 시장가 주문 (가격 0으로 설정)
order_result = self.koapys.send_order(
    rqname=f"SELL_{code}_{datetime.now().strftime('%H%M%S')}",
    account_no=self.account_no,
    order_type=OrderType.SELL,
    code=code,
    quantity=quantity,
    price=0,  # ✅ 시장가는 0
    hoga=OrderBookType.MARKET  # ✅ MARKET 사용
)
```

## OrderBookType 옵션

키움증권 API에서 지원하는 호가 타입:

| 타입 | 코드 | 설명 | 가격 설정 |
|------|------|------|----------|
| LIMIT | "00" | 지정가 | 필수 |
| MARKET | "03" | 시장가 | 0 |
| CONDITIONAL_LIMIT | "05" | 조건부지정가 | 필수 |
| BEST_LIMIT | "06" | 최유리지정가 | 0 |
| PRIORITY_LIMIT | "07" | 최우선지정가 | 0 |
| LIMIT_IOC | "10" | 지정가IOC | 필수 |
| MARKET_IOC | "13" | 시장가IOC | 0 |
| BEST_IOC | "16" | 최유리IOC | 0 |
| LIMIT_FOK | "20" | 지정가FOK | 필수 |
| MARKET_FOK | "23" | 시장가FOK | 0 |
| BEST_FOK | "26" | 최유리FOK | 0 |

## 스캘핑 전략에 적합한 호가 타입

### 1. MARKET (시장가) - 현재 사용 ✅
**장점**:
- 즉시 체결 보장
- 빠른 진입/청산

**단점**:
- 슬리피지 발생 가능
- 불리한 가격에 체결될 수 있음

**권장**: 스캘핑에 적합 (속도 우선)

### 2. BEST_LIMIT (최유리지정가)
**장점**:
- 현재 최우선 호가로 주문
- 시장가보다 유리한 가격

**단점**:
- 체결 보장 안됨
- 호가 변동 시 미체결 가능

**권장**: 유동성 높은 종목에 적합

### 3. LIMIT_IOC (지정가IOC)
**장점**:
- 지정가로 즉시 체결
- 미체결 부분 자동 취소

**단점**:
- 가격 설정 필요
- 부분 체결 가능

**권장**: 가격 통제가 중요한 경우

## 테스트 방법

### 1. 시뮬레이션 모드 테스트
```bash
# .env 파일에서
SIMULATION=true

# 실행
python scripts/live/live_trading.py --config config/trading_config.json
```

### 2. 주문 결과 확인
```python
# 성공 시
{
    'status': '접수',
    'return_code': 0,
    'return_msg': '정상처리',
    'order_no': '0000123456'
}

# 실패 시
{
    'status': '접수',
    'return_code': 20,
    'return_msg': '[2000](502048:단가를 입력하지 않는 호가입니다)'
}
```

### 3. 로그 모니터링
```bash
# BUY 신호 확인
grep "BUY signal" logs/live_trading_*.log

# 주문 실행 확인
grep "Buy order executed" logs/live_trading_*.log

# 오류 확인
grep "502048" logs/live_trading_*.log
```

## 예상 결과

### 수정 전
```
[PREDICT] 085310 (): BUY signal, confidence=0.5004
[BUY SIGNAL] 085310 (): 현재가=1,706, 등락률=8.04%
[BUY] 085310: price=1,706원, qty=59주, cost=100,654원
[OK] Buy order executed: {'return_code': 20, 'return_msg': '[2000](502048:단가를 입력하지 않는 호가입니다)'}
❌ 주문 실패
```

### 수정 후
```
[PREDICT] 085310 (): BUY signal, confidence=0.5004
[BUY SIGNAL] 085310 (): 현재가=1,706, 등락률=8.04%
[BUY] 085310: price=1,706원, qty=59주, cost=100,654원
[OK] Buy order executed: {'return_code': 0, 'return_msg': '정상처리', 'order_no': '0000123456'}
✅ 주문 성공
```

## 추가 개선 사항

### 1. 호가 타입 설정 가능하게 변경
```python
# config/trading_config.json
{
  "order_hoga_type": "MARKET",  # MARKET, BEST_LIMIT, LIMIT_IOC 등
  "use_limit_price": false,     # true면 지정가 사용
  "limit_price_offset": 0       # 지정가 오프셋 (틱)
}
```

### 2. 슬리피지 모니터링
```python
# 주문 체결 후 실제 체결가 확인
actual_price = get_executed_price(order_no)
slippage = (actual_price - expected_price) / expected_price
logger.info(f"Slippage: {slippage:.2%}")
```

### 3. 부분 체결 처리
```python
# IOC/FOK 사용 시 부분 체결 확인
executed_qty = get_executed_quantity(order_no)
if executed_qty < requested_qty:
    logger.warning(f"Partial fill: {executed_qty}/{requested_qty}")
```

## 관련 파일

- `scripts/live/live_trading.py`: 주문 실행 로직
- `koapys/types.py`: OrderType, OrderBookType 정의
- `config/trading_config.json`: 거래 설정

## 참고 자료

- 키움증권 OpenAPI 문서
- `koapys` 라이브러리 문서
- 호가 유형별 체결 방식

---

**작성일**: 2024-11-18  
**작성자**: AI Trading System  
**버전**: 1.0  
**상태**: ✅ 수정 완료
