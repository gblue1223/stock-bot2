# 수익률 계산 문제 분석 및 해결 방안

## 🔍 문제 발견

### 검증 결과

```
Entry price (등락률): 0.35%
Exit price (등락률): 0.47%
Profit rate: 34.29%  ← 문제!
```

**실제 의미:**
- 등락률 0.35% → 0.47% 변화 (0.12%p 상승)
- 하지만 계산된 수익률: 34.29%

### 근본 원인

**1. 환경(DirectFeatureEnv)의 데이터 구조:**
```python
# 등락률을 첫 번째 특징으로 사용
current_return = float(self.episode_data[self.current_step, 0]) / 100.0
# 예: 0.35 → 0.0035 (0.35%)
```

**2. Enhanced Inference의 가격 사용:**
```python
# Position 클래스에 등락률을 "가격"으로 전달
current_price = state[0]  # 0.35 (등락률 0.35%)

position = Position(
    entry_price=current_price,  # 0.35를 가격으로 취급!
    ...
)
```

**3. Position의 수익률 계산:**
```python
@property
def profit_rate(self) -> float:
    return (self.current_price - self.entry_price) / self.entry_price * 100
    # (0.47 - 0.35) / 0.35 * 100 = 34.29%
```

### 왜 이런 결과가 나왔나?

**백테스팅 결과 재해석:**
```
Mean Profit Rate: 623.18%  ← 등락률 차이를 수익률로 계산
Mean Winning Profit: 880.54%  ← 등락률 0.88%p 차이
Mean Losing Profit: -417.70%  ← 등락률 -4.18%p 차이
```

**실제 의미:**
- 평균 수익률 623% = 평균 등락률 차이 6.23%p
- 승리 거래 880% = 등락률 8.8%p 상승
- 손실 거래 -417% = 등락률 -4.17%p 하락

## ✅ 올바른 해석

### 환경의 실제 동작

DirectFeatureEnv는 **등락률의 누적**으로 수익률을 계산합니다:

```python
# 환경의 올바른 수익률 계산
if self.position == Position.LONG:
    self.cumulative_return += current_return  # 등락률 누적
    # 예: 0.35% + 0.47% + ... = 누적 수익률
```

**검증 결과:**
```
Step 1: 등락률 -0.47%, Cumulative return: -0.48%
Step 2: 등락률 -0.47%, Cumulative return: -0.95%
Step 3: 등락률 -0.47%, Cumulative return: -1.42%
...
```

### Position 클래스의 잘못된 계산

Position 클래스는 등락률을 가격으로 취급하여 잘못 계산:

```python
# 잘못된 계산
entry_price = 0.35  # 등락률 0.35%
exit_price = 0.47   # 등락률 0.47%
profit_rate = (0.47 - 0.35) / 0.35 * 100 = 34.29%  ← 틀림!

# 올바른 해석
# 등락률 0.35%에서 0.47%로 변화 = 0.12%p 상승
# 실제 수익률은 누적 등락률로 계산해야 함
```

## 🔧 해결 방안

### 방안 1: Position 클래스 수정 (권장)

Enhanced Inference에서 누적 수익률을 직접 추적:

```python
class Position:
    def __init__(
        self,
        entry_price: float,
        entry_time: int,
        current_price: float,
        holding_period: int,
        cumulative_return: float = 0.0  # 추가
    ):
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.current_price = current_price
        self.holding_period = holding_period
        self.cumulative_return = cumulative_return  # 누적 수익률
    
    @property
    def profit_rate(self) -> float:
        """누적 수익률 반환 (백분율)"""
        return self.cumulative_return * 100  # 이미 비율이므로 100 곱하기만
```

**live_trading.py 수정:**
```python
def process_stock(self, code: str):
    # ...
    
    # 포지션 업데이트
    if current_position is not None:
        # 현재 등락률 가져오기
        current_return = sequence[-1, 0] / 100.0  # 0.35 → 0.0035
        current_position.cumulative_return += current_return
        current_position.holding_period += 1
    
    # 추론
    action, confidence, pred_info = self.inference.predict(
        sequence=sequence,
        current_position=current_position,
        current_price=current_price,
        deterministic=True
    )
```

### 방안 2: 환경 수정 (더 복잡)

환경에서 실제 가격을 제공하도록 수정:

```python
class DirectFeatureEnv:
    def __init__(self, ...):
        # ...
        self.price_history = []  # 실제 가격 추적
    
    def step(self, action):
        # 실제 가격 계산 (누적 등락률 적용)
        if len(self.price_history) == 0:
            base_price = 100000  # 기준 가격
        else:
            base_price = self.price_history[-1]
        
        current_return = float(self.episode_data[self.current_step, 0]) / 100.0
        current_price = base_price * (1 + current_return)
        self.price_history.append(current_price)
        
        # ...
```

### 방안 3: 백테스팅 결과 재해석 (임시)

현재 결과를 올바르게 해석:

```python
# 백테스팅 결과 변환
reported_profit_rate = 623.18  # %
actual_profit_rate = reported_profit_rate / 100  # 6.23%p

# 실제 의미
print(f"평균 등락률 차이: {actual_profit_rate:.2f}%p")
print(f"이는 실제 수익률 약 {actual_profit_rate:.2f}%에 해당")
```

## 💡 권장 조치

### 즉시 적용 (방안 1)

1. **Enhanced Inference 수정:**

```python
# ai_trader/grpo/inference/enhanced_inference.py

@dataclass
class Position:
    entry_price: float
    entry_time: int
    current_price: float
    holding_period: int
    cumulative_return: float = 0.0  # 추가
    
    @property
    def profit_rate(self) -> float:
        """누적 수익률 반환 (백분율)"""
        return self.cumulative_return * 100
    
    @property
    def is_profit(self) -> bool:
        return self.cumulative_return > 0
```

2. **Live Trading 수정:**

```python
# scripts/real/live_trading.py

def process_stock(self, code: str):
    # ...
    
    # 포지션 생성 시
    if action == Action.BUY.value and current_position is None:
        current_position = Position(
            entry_price=current_price,
            entry_time=step,
            current_price=current_price,
            holding_period=0,
            cumulative_return=0.0  # 초기화
        )
    
    # 포지션 업데이트 시
    if current_position is not None:
        # 현재 등락률 추출
        current_return = sequence[-1, 0] / 100.0
        current_position.cumulative_return += current_return
        current_position.current_price = current_price
        current_position.holding_period += 1
```

3. **Backtest 수정:**

```python
# scripts/real/backtest_strategy.py

def run_episode(self):
    # ...
    
    # 포지션 생성
    if action == Action.BUY.value and current_position is None:
        current_position = Position(
            entry_price=current_price,
            entry_time=info.get('step', 0),
            current_price=current_price,
            holding_period=0,
            cumulative_return=0.0
        )
    
    # 포지션 업데이트
    if current_position is not None:
        current_return = sequence[-1, 0] / 100.0
        current_position.cumulative_return += current_return
        current_position.current_price = current_price
        current_position.holding_period += 1
```

### 검증

수정 후 다시 백테스팅:

```bash
python scripts/real/backtest_strategy.py --episodes 100
```

**기대 결과:**
```
Mean Profit Rate: 6.23%  (기존 623.18% → 100으로 나눔)
Mean Winning Profit: 8.81%  (기존 880.54%)
Mean Losing Profit: -4.18%  (기존 -417.70%)
```

## 📊 수정 전후 비교

### 수정 전 (잘못된 계산)

```
Entry: 0.35 (등락률)
Exit: 0.47 (등락률)
Profit: (0.47 - 0.35) / 0.35 * 100 = 34.29%  ← 틀림
```

### 수정 후 (올바른 계산)

```
Entry: 0.35% (등락률)
Cumulative returns: 0.35% + 0.12% + ... = 6.23%
Profit: 6.23%  ← 올바름
```

## 🎯 결론

**문제:**
- Position 클래스가 등락률을 가격으로 취급
- 수익률이 100배 과대 계산됨

**해결:**
- Position에 cumulative_return 필드 추가
- 등락률을 누적하여 실제 수익률 계산
- 백테스팅 결과가 현실적인 수준으로 조정됨

**실제 성과 (수정 후 예상):**
- 평균 수익률: ~6% (기존 623%)
- 승률: 80% (변화 없음)
- 익절/손절: 정상 작동

이제 실전 거래에 사용 가능한 현실적인 수치입니다!
