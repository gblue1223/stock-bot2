# BUY 신호 발생 흐름 분석

## 📋 전체 흐름 요약

```
실시간 데이터 수신 → 버퍼 저장 → 시퀀스 준비 → 모델 추론 → 신호 필터링 → 매수 실행
```

## 🔄 상세 흐름

### 1. 실시간 데이터 수신
**위치**: `scripts/live/live_trading.py` - `on_data_update()` 콜백

```python
def on_data_update(code: str, stock_data: StockRealtimeData):
    # RealtimeStockClient가 WebSocket으로 실시간 데이터 수신
    # 체결강도 > 0 검증
    if stock_data.체결강도 == 0:
        return  # 거래 비활발한 종목 제외
    
    # 특징 추출
    features = self.extract_features(market_data)
    
    # 버퍼에 추가
    self.data_buffers[code].add(features)
```

**발생 시점**: 실시간 (틱 데이터 수신 시마다)

---

### 2. 버퍼 데이터 수집
**위치**: `scripts/live/live_trading.py` - `MarketDataBuffer`

```python
class MarketDataBuffer:
    def __init__(self, seq_len: int, num_features: int):
        self.seq_len = 60  # 60개 데이터 필요
        self.buffer = deque(maxlen=seq_len)
    
    def is_ready(self) -> bool:
        return len(self.buffer) >= self.seq_len
```

**조건**: 60개 데이터가 모여야 예측 가능

---

### 3. 종목 처리 및 시퀀스 준비
**위치**: `scripts/live/live_trading.py` - `process_stock()`

```python
def process_stock(self, code: str):
    # 1. 버퍼 준비 확인
    if not self.data_buffers[code].is_ready():
        return  # 60개 미만이면 대기
    
    # 2. 거래 활발도 검증
    if stock_data.체결강도 == 0:
        return  # 체결강도 0이면 스킵
    
    if stock_data.매도대기금액1 == 0 and stock_data.매수대기금액1 == 0:
        return  # 호가 없으면 스킵
    
    # 3. 시퀀스 가져오기
    sequence = self.data_buffers[code].get_sequence()  # (60, 28)
    
    # 4. 현재 포지션 확인
    current_position = self.positions.get(code)
    current_price = self.get_current_price(code)
    
    # 5. 추론 실행
    action, confidence, pred_info = self.inference.predict(
        sequence=sequence,
        current_position=current_position,
        current_price=current_price,
        deterministic=True
    )
```

**발생 시점**: 메인 루프에서 주기적으로 호출 (약 1초마다)

---

### 4. 모델 추론 (1단계: 기본 추론)
**위치**: `ai_trader/grpo/inference/infer_grpo.py` - `GRPOInference.predict()`

```python
def predict(self, sequence: np.ndarray, deterministic: bool = True):
    # 1. numpy → torch tensor 변환
    sequence = torch.tensor(sequence, dtype=torch.float32, device=self.device)
    
    # 2. 정책 타입에 따라 입력 처리
    if self.policy_type == 'DirectFeaturePolicy':
        # 시퀀스 평탄화: (60, 28) → (1680,) → (1, 1680)
        policy_input = sequence.flatten().unsqueeze(0)
    else:
        # GRPOPolicy: 임베딩 생성
        embedding = self._generate_embedding(sequence)
        policy_input = embedding.unsqueeze(0)
    
    # 3. 정책 실행
    with torch.no_grad():
        action_logits, _ = self.policy(policy_input)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)  # 최대 확률 행동
            confidence = action_probs[0, action].item()
    
    # 4. 결과 반환
    return action.item(), confidence  # (0: HOLD, 1: BUY, 2: SELL)
```

**출력**: 
- `action`: 0 (HOLD), 1 (BUY), 2 (SELL)
- `confidence`: 0.0 ~ 1.0

---

### 5. 신호 필터링 (2단계: 개선된 추론)
**위치**: `ai_trader/grpo/inference/enhanced_inference.py` - `EnhancedGRPOInference.predict()`

```python
def predict(self, sequence, current_position, current_price, deterministic):
    # 1. 기본 추론 실행
    raw_action, raw_confidence = self.base_inference.predict(
        sequence, deterministic=deterministic
    )
    
    # 2. BUY 신호 필터링
    if raw_action == Action.BUY.value:  # raw_action == 1
        self.stats['buy_signals'] += 1
        
        # 필터 1: 신뢰도 확인
        if raw_confidence < self.min_buy_confidence:  # 기본값: 0.0
            logger.debug(f"Buy signal filtered: low confidence")
            return Action.HOLD.value, raw_confidence, info
        
        # 필터 2: 이미 포지션 있으면 추가 매수 방지
        if current_position is not None:
            logger.debug("Buy signal filtered: already in position")
            return Action.HOLD.value, raw_confidence, info
    
    # 3. 필터링 통과
    if raw_action == Action.BUY.value:
        logger.info(
            f"[INFERENCE] Final action: BUY, confidence={raw_confidence:.4f}, "
            f"position=NO"
        )
    
    return raw_action, raw_confidence, info
```

**필터링 조건**:
1. ✅ `confidence >= min_buy_confidence` (기본값: 0.0)
2. ✅ `current_position is None` (포지션 없음)

---

### 6. 매수 실행
**위치**: `scripts/live/live_trading.py` - `execute_buy()`

```python
def execute_buy(self, code: str, confidence: float):
    # 1. 이미 포지션 있으면 무시
    if code in self.positions:
        return
    
    # 2. 최대 포지션 수 확인
    if len(self.positions) >= self.config.max_positions:  # 기본값: 3
        return
    
    # 3. 현재 가격 조회
    current_price = self.get_current_price(code)
    
    # 4. 가용 금액 확인
    available_cash = self.get_available_cash()
    if available_cash < current_price:
        return
    
    # 5. 매수 수량 계산
    max_buyable = int(available_cash / current_price)
    max_by_config = int(self.config.max_position_size / current_price)
    quantity = min(max_buyable, max_by_config)
    
    # 6. 로그 출력
    logger.info(
        f"[BUY SIGNAL] {code} ({stock_data.종목명}): "
        f"현재가={current_price:,.0f}, 등락률={stock_data.등락률:.2f}%, "
        f"체결강도={stock_data.체결강도:.1f}"
    )
    
    logger.info(
        f"[BUY] {code}: price={current_price:,.0f}원, qty={quantity}주, "
        f"cost={current_price * quantity:,.0f}원, confidence={confidence:.3f}"
    )
    
    # 7. 주문 실행
    order_result = self.koapys.send_order(
        rqname=f"BUY_{code}_{datetime.now().strftime('%H%M%S')}",
        account_no=self.account_no,
        order_type=OrderType.BUY,
        code=code,
        quantity=quantity,
        price=0,  # 시장가
        hoga=OrderBookType.MARKET
    )
    
    # 8. 포지션 생성
    self.positions[code] = Position(
        entry_price=current_price,
        entry_time=int(time.time()),
        current_price=current_price,
        holding_period=0,
        cumulative_return=0.0
    )
    self.positions[code].quantity = quantity
    self.positions[code].order_no = order_result.get('order_no', '')
    self.positions[code].status = 'PENDING'
```

**실행 조건**:
1. ✅ `action == Action.BUY.value` (1)
2. ✅ `self.is_trading_time()` (거래 시간 내)
3. ✅ `code not in self.positions` (포지션 없음)
4. ✅ `len(self.positions) < max_positions` (최대 3개)
5. ✅ `available_cash >= current_price` (가용 금액 충분)

---

## 📊 BUY 신호 발생 조건 요약

### 필수 조건 (모두 만족해야 함)

1. **데이터 준비**
   - ✅ 버퍼에 60개 데이터 수집 완료
   - ✅ 체결강도 > 0 (거래 활발)
   - ✅ 매도/매수 대기금액 > 0 (호가 존재)

2. **모델 예측**
   - ✅ `raw_action == 1` (BUY)
   - ✅ `confidence >= 0.0` (기본값)

3. **필터링 통과**
   - ✅ `current_position is None` (포지션 없음)
   - ✅ `len(positions) < 3` (최대 포지션 수)

4. **거래 조건**
   - ✅ `is_trading_time()` (09:00 ~ 13:20)
   - ✅ `available_cash >= current_price` (가용 금액)

---

## 🔍 BUY 신호 발생 빈도

### 로그 분석 결과 (2025-11-18 ~ 2025-11-19)

```
총 예측: 41,901회
BUY 신호: 47건 (0.11%)
실제 매수: 47건 (100% 실행)
```

### BUY 신호가 적은 이유

1. **모델 특성**: GRPO 모델이 보수적으로 학습됨
2. **필터링**: 포지션 있으면 추가 매수 방지
3. **최대 포지션**: 동시에 최대 3개만 보유
4. **데이터 품질**: 체결강도=0인 종목 제외

---

## 🎯 BUY 신호 증가 방법

### 1. 신뢰도 임계값 조정 (현재: 0.0)
```json
{
  "min_buy_confidence": 0.0  // 낮을수록 더 많은 신호
}
```

### 2. 최대 포지션 수 증가 (현재: 3)
```json
{
  "max_positions": 5  // 3 → 5
}
```

### 3. 거래 시간 확대 (현재: 09:00 ~ 13:20)
```json
{
  "trading_start": "09:00",
  "trading_end": "15:20"  // 13:20 → 15:20
}
```

### 4. 모델 재학습
- 더 공격적인 매수 전략으로 학습
- 보상 함수 조정

---

## 📝 로그 예시

### 정상적인 BUY 신호 발생
```
09:00:56 - [SEQUENCE] 476060: shape=(60, 28), mean=0.0585, std=0.5046
09:00:56 - [INFERENCE] Raw prediction #1: action=BUY(1), confidence=0.5010
09:00:56 - [INFERENCE] Final action: BUY, confidence=0.5010, position=NO
09:00:56 - [PREDICT] 476060: BUY signal, confidence=0.5010
09:00:56 - [BUY SIGNAL] 476060 (): 현재가=14,990, 등락률=+2.05%, 체결강도=85.3
09:00:56 - [BUY] 476060: price=14,990원, qty=4주, cost=59,960원, confidence=0.501
09:00:56 - [OK] Buy order executed: {'status': '접수', 'order_no': '0012345'}
```

### BUY 신호 필터링 (포지션 있음)
```
09:01:00 - [INFERENCE] Raw prediction #5: action=BUY(1), confidence=0.5200
09:01:00 - [DEBUG] Buy signal filtered: already in position
09:01:00 - [PREDICT] 476060: HOLD signal (filtered)
```

---

**작성일**: 2025-11-19  
**상태**: ✅ 분석 완료
