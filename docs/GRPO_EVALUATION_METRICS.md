# GRPO 에이전트 평가 메트릭

## 개요

이 문서는 GRPO (Group Relative Policy Optimization) 에이전트의 성능을 평가하기 위한 메트릭 시스템을 설명합니다. 요구사항 7.2에 따라 다음 메트릭을 계산합니다:

- **승률 (Win Rate)**: 수익이 발생한 거래의 비율
- **거래당 평균 수익 (Average Profit per Trade)**: 모든 거래의 평균 수익률
- **최대 낙폭 (Maximum Drawdown)**: 누적 수익의 최대 하락폭
- **샤프 비율 (Sharpe Ratio)**: 위험 대비 수익률
- **평균 보유 시간 (Average Holding Time)**: 거래당 평균 포지션 보유 시간

## 설치 및 사용

### 기본 사용법

```python
from ai_trader.grpo import GRPOEvaluationMetrics, evaluate_grpo_agent

# 에피소드 데이터 (환경에서 수집)
episodes = [
    {
        'trades': [
            {'reward': 0.01, 'holding_time': 10.0, 'profit_rate': 0.015},
            {'reward': -0.005, 'holding_time': 5.0, 'profit_rate': -0.001},
            # ...
        ],
        'total_return': 0.005,
        'num_trades': 2
    },
    # ... 더 많은 에피소드
]

# 평가 수행
metrics = evaluate_grpo_agent(episodes, print_results=True)

# 개별 메트릭 접근
print(f"승률: {metrics['win_rate']:.2%}")
print(f"샤프 비율: {metrics['sharpe_ratio']:.4f}")
```

### 점진적 평가

```python
from ai_trader.grpo import GRPOEvaluationMetrics

# 평가기 초기화
evaluator = GRPOEvaluationMetrics()

# 에피소드를 하나씩 추가하면서 평가
for episode in episodes:
    evaluator.add_episode(episode)
    
    # 중간 평가
    metrics = evaluator.compute_metrics()
    print(f"현재 승률: {metrics['win_rate']:.2%}")
```

## 메트릭 상세 설명

### 1. 승률 (Win Rate)

**정의**: 수익이 발생한 거래의 비율

**계산식**:
```
승률 = (수익이 발생한 거래 수) / (총 거래 수)
```

**해석**:
- 0.0 ~ 1.0 범위의 값
- 0.5 이상이면 절반 이상의 거래에서 수익 발생
- 스캘핑 전략의 경우 일반적으로 0.5 ~ 0.6 목표

**코드**:
```python
def _compute_win_rate(self, trades):
    winning_trades = sum(1 for trade in trades if trade['reward'] > 0)
    return winning_trades / len(trades)
```

### 2. 거래당 평균 수익 (Average Profit per Trade)

**정의**: 모든 거래의 평균 수익률 (거래 비용 포함)

**계산식**:
```
거래당 평균 수익 = sum(rewards) / num_trades
```

**해석**:
- 양수면 평균적으로 수익 발생
- 거래 비용(0.43%)을 고려하여 최소 0.0043 이상이어야 실질적 수익
- 스캘핑의 경우 일반적으로 0.005 ~ 0.01 목표

**코드**:
```python
def _compute_avg_profit_per_trade(self, trades):
    total_profit = sum(trade['reward'] for trade in trades)
    return total_profit / len(trades)
```

### 3. 최대 낙폭 (Maximum Drawdown)

**정의**: 누적 수익의 최고점에서 최저점까지의 최대 하락폭

**계산식**:
```
최대 낙폭 = max((peak - trough) / peak)
```

여기서:
- peak: 각 시점까지의 누적 수익 최고점
- trough: 현재 누적 수익

**해석**:
- 0.0 ~ 1.0 범위의 값 (백분율로 표시)
- 0.1 = 10% 낙폭 (최고점 대비 10% 하락)
- 낮을수록 안정적인 전략
- 스캘핑의 경우 일반적으로 0.1 (10%) 미만 목표

**코드**:
```python
def _compute_max_drawdown(self, trades):
    # 복리 누적 수익 계산
    cumulative_returns = []
    cumulative_return = 1.0  # 초기 자본
    
    for trade in trades:
        cumulative_return *= (1.0 + trade['reward'])
        cumulative_returns.append(cumulative_return)
    
    # 최대 낙폭 계산
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    
    return np.max(drawdowns)
```

### 4. 샤프 비율 (Sharpe Ratio)

**정의**: 위험 대비 수익률 (평균 수익 / 수익 표준편차)

**계산식**:
```
샤프 비율 = (평균 수익 - 무위험 수익률) / 수익 표준편차
```

스캘핑의 경우 무위험 수익률은 0으로 가정:
```
샤프 비율 = 평균 수익 / 수익 표준편차
```

**해석**:
- 높을수록 위험 대비 수익이 좋음
- 1.0 이상이면 우수한 전략
- 2.0 이상이면 매우 우수한 전략
- 음수면 평균적으로 손실 발생

**코드**:
```python
def _compute_sharpe_ratio(self, trades):
    returns = np.array([trade['reward'] for trade in trades])
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    return mean_return / std_return
```

### 5. 평균 보유 시간 (Average Holding Time)

**정의**: 거래당 평균 포지션 보유 시간 (초)

**계산식**:
```
평균 보유 시간 = sum(holding_times) / num_trades
```

**해석**:
- 스캘핑 전략의 특성을 나타내는 지표
- 낮을수록 빠른 진입/청산 전략
- 스캘핑의 경우 일반적으로 30초 미만 목표
- 너무 짧으면 거래 비용 부담 증가

**코드**:
```python
def _compute_avg_holding_time(self, trades):
    total_holding_time = sum(trade['holding_time'] for trade in trades)
    return total_holding_time / len(trades)
```

## 성능 임계값

요구사항 7.2에 따른 권장 성능 임계값:

| 메트릭 | 임계값 | 조건 |
|--------|--------|------|
| 승률 | 50% | >= |
| 샤프 비율 | 1.0 | >= |
| 최대 낙폭 | 10% | < |
| 평균 보유 시간 | 30초 | < |

### 임계값 체크 예제

```python
from ai_trader.grpo import evaluate_grpo_agent

# 평가 수행
metrics = evaluate_grpo_agent(episodes, print_results=False)

# 성능 임계값 체크
thresholds = {
    'win_rate': (0.50, '>='),
    'sharpe_ratio': (1.0, '>='),
    'max_drawdown': (0.10, '<'),
    'avg_holding_time': (30.0, '<'),
}

all_passed = True
for metric_name, (threshold, operator) in thresholds.items():
    value = metrics[metric_name]
    
    if operator == '>=':
        passed = value >= threshold
    elif operator == '<':
        passed = value < threshold
    
    print(f"{metric_name}: {value:.4f} {operator} {threshold} - {'PASS' if passed else 'FAIL'}")
    
    if not passed:
        all_passed = False

if all_passed:
    print("모든 성능 임계값을 통과했습니다!")
else:
    print("일부 성능 임계값을 통과하지 못했습니다.")
```

## 환경과의 통합

GRPO 환경은 에피소드 종료 시 자동으로 메타데이터를 생성합니다:

```python
from ai_trader.grpo import GRPOScalpingEnv, GRPOEvaluationMetrics

# 환경 생성
env = GRPOScalpingEnv(...)

# 에피소드 실행
episodes = []
for _ in range(10):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # 에피소드 메타데이터 수집
    episode_metadata = info['episode']
    episodes.append(episode_metadata)

# 평가
metrics = evaluate_grpo_agent(episodes, print_results=True)
```

## API 레퍼런스

### GRPOEvaluationMetrics

평가 메트릭 계산기 클래스

**생성자**:
```python
GRPOEvaluationMetrics(episodes: Optional[List[Dict[str, Any]]] = None)
```

**메서드**:

- `add_episode(episode: Dict[str, Any])`: 에피소드 추가
- `add_episodes(episodes: List[Dict[str, Any]])`: 여러 에피소드 추가
- `clear()`: 에피소드 데이터 초기화
- `compute_metrics(force_recompute: bool = False) -> Dict[str, float]`: 메트릭 계산
- `print_metrics(metrics: Optional[Dict[str, float]] = None)`: 메트릭 출력

**반환 메트릭**:
```python
{
    'win_rate': float,              # 승률 (0.0 ~ 1.0)
    'avg_profit_per_trade': float,  # 거래당 평균 수익
    'max_drawdown': float,          # 최대 낙폭 (0.0 ~ 1.0)
    'sharpe_ratio': float,          # 샤프 비율
    'avg_holding_time': float,      # 평균 보유 시간 (초)
    'total_trades': int,            # 총 거래 횟수
    'total_episodes': int,          # 총 에피소드 수
    'total_return': float,          # 총 수익률
    'avg_episode_return': float     # 에피소드당 평균 수익률
}
```

### evaluate_grpo_agent

편의 함수

```python
evaluate_grpo_agent(
    episodes: List[Dict[str, Any]],
    print_results: bool = True
) -> Dict[str, float]
```

**인수**:
- `episodes`: 에피소드 데이터 리스트
- `print_results`: 결과를 출력할지 여부

**반환**: 평가 메트릭 딕셔너리

## 예제

### 예제 1: 기본 평가

```python
from ai_trader.grpo import evaluate_grpo_agent

episodes = [...]  # 환경에서 수집한 에피소드
metrics = evaluate_grpo_agent(episodes, print_results=True)
```

### 예제 2: 여러 에이전트 비교

```python
from ai_trader.grpo import evaluate_grpo_agent

# 에이전트 A 평가
metrics_a = evaluate_grpo_agent(episodes_a, print_results=False)

# 에이전트 B 평가
metrics_b = evaluate_grpo_agent(episodes_b, print_results=False)

# 비교
print(f"승률: A={metrics_a['win_rate']:.2%}, B={metrics_b['win_rate']:.2%}")
print(f"샤프 비율: A={metrics_a['sharpe_ratio']:.4f}, B={metrics_b['sharpe_ratio']:.4f}")
```

### 예제 3: 실시간 모니터링

```python
from ai_trader.grpo import GRPOEvaluationMetrics

evaluator = GRPOEvaluationMetrics()

# 훈련 중 실시간 평가
for iteration in range(num_iterations):
    # 에피소드 수집
    episodes = collect_episodes(...)
    
    # 평가기에 추가
    evaluator.add_episodes(episodes)
    
    # 현재 성능 확인
    metrics = evaluator.compute_metrics()
    
    # 로깅
    logger.info(f"Iteration {iteration}: win_rate={metrics['win_rate']:.2%}")
    
    # 성능 저하 감지
    if metrics['max_drawdown'] > 0.15:
        logger.warning("최대 낙폭이 15%를 초과했습니다!")
```

## 테스트

평가 메트릭 구현은 포괄적인 단위 테스트로 검증되었습니다:

```bash
# 모든 테스트 실행
python -m pytest tests/test_grpo_evaluation.py -v

# 특정 테스트 실행
python -m pytest tests/test_grpo_evaluation.py::TestGRPOEvaluationMetrics::test_win_rate_calculation -v
```

## 참고 자료

- [GRPO 환경 문서](./grpo_reward_calculation.md)
- [요구사항 문서](../.kiro/specs/grpo-scalping-embedding/requirements.md)
- [설계 문서](../.kiro/specs/grpo-scalping-embedding/design.md)

## 라이선스

이 코드는 프로젝트 라이선스를 따릅니다.
