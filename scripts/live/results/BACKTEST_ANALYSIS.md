# 백테스팅 결과 분석

## 실행 결과 (2025-10-17)

### 📊 핵심 지표

```
Episodes: 100
Mean Reward: 236.84 ± 601.79
Win Rate: 80.18% (2,738 / 3,415)
Total Return: 218.20%
```

### ✅ 긍정적 지표

**1. 높은 승률 (80.18%)**
- 목표 승률(75%) 초과 달성
- 안정적인 수익 구조
- 자동 청산 시스템 효과적

**2. 자동 청산 100% 작동**
- 익절: 2,738건 (80.18%)
- 손절: 677건 (19.82%)
- 수동 청산 없음 (모두 자동)

**3. 높은 거래 빈도**
- 에피소드당 평균 34.15건
- 충분한 거래 기회 포착

### ⚠️ 주의 사항

**1. 비정상적으로 높은 수익률**

```
Mean Profit Rate: 623.18%
Median Profit Rate: 350.00%
Mean Winning Profit: 880.54%
Mean Losing Profit: -417.70%
```

**문제점:**
- 실제 시장에서 불가능한 수준
- 가격 스케일 또는 계산 오류 가능성
- 환경 설정 확인 필요

**원인 분석:**

```python
# 가능성 1: 정규화된 가격 사용
# DirectFeatureEnv에서 정규화된 등락률을 가격으로 사용
current_price = state[0]  # 이것이 정규화된 값일 수 있음

# 가능성 2: 보상 계산 방식
# 환경의 보상이 실제 수익률과 다를 수 있음
reward = (current_price - entry_price) / entry_price * 100
```

**2. 높은 변동성**

```
Standard Deviation: 601.79 (평균보다 큰 값)
Min/Max Reward: -717.28 / 5,347.01
```

**의미:**
- 리스크가 매우 높음
- 일부 에피소드에서 극단적 손실/수익
- 실전에서는 위험할 수 있음

**3. Buy Signal Rate 50.90%**

```
Total Predictions: 7,100
Buy Signals: 3,614 (50.90%)
Buy Filter Rate: 2.82%
```

**분석:**
- 여전히 Buy 신호가 많음 (50%)
- 필터링은 2.82%만 작동
- 포지션 관리로 대부분 제어됨

## 🔍 상세 분석

### 1. 수익률 분포 확인 필요

현재 결과가 정확한지 확인하려면:

```python
# 수익률 히스토그램 생성
import matplotlib.pyplot as plt

profit_rates = [t['profit_rate'] for t in sell_trades]
plt.hist(profit_rates, bins=50)
plt.xlabel('Profit Rate (%)')
plt.ylabel('Frequency')
plt.title('Profit Rate Distribution')
plt.savefig('profit_distribution.png')
```

### 2. 환경 설정 확인

```python
# DirectFeatureEnv 확인
env = DirectFeatureEnv(
    db_path=db_path,
    table_name='datasets_raw',
    seq_len=30,
    expected_features=24,
    transaction_cost_rate=0.00215,  # 0.215% 수수료
    max_episode_steps=100,
    device=device
)

# 가격 스케일 확인
print(f"Price scale: {env.price_scale}")
print(f"Reward scale: {env.reward_scale}")
```

### 3. 실제 가격 vs 정규화 가격

```python
# Position 클래스에서 사용하는 가격이 무엇인지 확인
position = Position(
    entry_price=current_price,  # 이것이 실제 가격인가?
    entry_time=step,
    current_price=current_price,
    holding_period=0
)

# profit_rate 계산
profit_rate = (current_price - entry_price) / entry_price * 100
```

## 💡 권장 사항

### 1. 즉시 확인 필요

**가격 스케일 검증:**

```bash
# 환경에서 실제 사용하는 가격 범위 확인
python -c "
from ai_trader.grpo.environments.direct_feature_env import DirectFeatureEnv
env = DirectFeatureEnv(db_path='datasets.duckdb', ...)
state, _ = env.reset()
print(f'State[0] (price): {state[0]}')
print(f'State range: {state.min()} ~ {state.max()}')
"
```

**수익률 계산 검증:**

```python
# 실제 거래 1건의 수익률 추적
# entry_price와 exit_price 로그 출력
logger.info(f"Entry: {entry_price}, Exit: {exit_price}, Profit: {profit_rate}%")
```

### 2. 실전 거래 전 필수 조치

**A. 수익률 정규화**

만약 환경이 정규화된 값을 사용한다면:

```python
# 실제 가격으로 변환
def denormalize_price(normalized_price, mean, std):
    return normalized_price * std + mean

# Position 클래스 수정
class Position:
    def __init__(self, entry_price, mean, std):
        self.entry_price_normalized = entry_price
        self.entry_price_real = denormalize_price(entry_price, mean, std)
```

**B. 현실적인 목표 설정**

```json
{
  "stop_loss_rate": -2.0,    // 현실적
  "take_profit_rate": 5.0,   // 현실적
  "expected_win_rate": 0.75, // 현실적
  "expected_profit": 623.18  // 비현실적! → 2-5% 정도가 현실적
}
```

**C. 소액 테스트**

```python
# 최소 금액으로 시작
config = {
    "max_position_size": 100000,  # 10만원
    "max_positions": 1,            # 1개만
    "target_stocks": ["005930"]    # 삼성전자만
}
```

### 3. 추가 백테스팅

**다양한 파라미터 테스트:**

```bash
# 보수적 설정
python scripts/real/backtest_strategy.py \
    --stop-loss -1.0 \
    --take-profit 2.0 \
    --episodes 100

# 공격적 설정
python scripts/real/backtest_strategy.py \
    --stop-loss -3.0 \
    --take-profit 10.0 \
    --episodes 100
```

## 📋 체크리스트

실전 거래 전 확인:

- [ ] 수익률 계산 방식 검증
- [ ] 가격 스케일 확인
- [ ] 환경 설정 검토
- [ ] 현실적인 목표 설정
- [ ] 소액 테스트 완료
- [ ] 리스크 관리 강화

## 🎯 결론

**현재 상태:**
- ✅ 승률 80%는 우수함
- ✅ 자동 청산 시스템 작동
- ⚠️ 수익률 수치는 검증 필요
- ⚠️ 변동성이 높음

**다음 단계:**
1. 가격 스케일 및 수익률 계산 검증
2. 현실적인 기대치 설정
3. 소액 시뮬레이션 테스트
4. 점진적 실전 투입

**실전 거래 가능 여부:**
- 시스템 자체는 작동함 ✅
- 수익률 검증 후 진행 권장 ⚠️
- 소액 테스트 필수 ⚠️

## 참고

- [백테스팅 스크립트](backtest_strategy.py)
- [실전 거래 가이드](QUICKSTART.md)
- [구현 문서](../../docs/REAL_TRADING_IMPLEMENTATION.md)
