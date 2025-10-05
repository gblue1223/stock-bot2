# GRPO 평가 메트릭 빠른 참조

## 한 줄 평가

```python
from ai_trader.grpo import evaluate_grpo_agent

metrics = evaluate_grpo_agent(episodes, print_results=True)
```

## 메트릭 요약

| 메트릭 | 키 | 목표 | 설명 |
|--------|-----|------|------|
| 승률 | `win_rate` | >= 50% | 수익 거래 비율 |
| 평균 수익 | `avg_profit_per_trade` | > 0.43% | 거래당 평균 수익률 |
| 최대 낙폭 | `max_drawdown` | < 10% | 최대 하락폭 |
| 샤프 비율 | `sharpe_ratio` | >= 1.0 | 위험 대비 수익 |
| 보유 시간 | `avg_holding_time` | < 30초 | 평균 포지션 보유 시간 |

## 임계값 체크

```python
metrics = evaluate_grpo_agent(episodes, print_results=False)

passed = (
    metrics['win_rate'] >= 0.5 and
    metrics['sharpe_ratio'] >= 1.0 and
    metrics['max_drawdown'] < 0.1 and
    metrics['avg_holding_time'] < 30.0
)

print("성능 평가:", "통과" if passed else "실패")
```

## 점진적 평가

```python
from ai_trader.grpo import GRPOEvaluationMetrics

evaluator = GRPOEvaluationMetrics()

for episode in episodes:
    evaluator.add_episode(episode)
    metrics = evaluator.compute_metrics()
    print(f"승률: {metrics['win_rate']:.2%}")
```

## 에이전트 비교

```python
metrics_a = evaluate_grpo_agent(episodes_a, print_results=False)
metrics_b = evaluate_grpo_agent(episodes_b, print_results=False)

print(f"승률: A={metrics_a['win_rate']:.2%} vs B={metrics_b['win_rate']:.2%}")
print(f"샤프: A={metrics_a['sharpe_ratio']:.2f} vs B={metrics_b['sharpe_ratio']:.2f}")
```

## 환경 통합

```python
from ai_trader.grpo import GRPOScalpingEnv, evaluate_grpo_agent

env = GRPOScalpingEnv(...)
episodes = []

for _ in range(10):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    episodes.append(info['episode'])

metrics = evaluate_grpo_agent(episodes, print_results=True)
```

## 반환 메트릭

```python
{
    'win_rate': 0.52,              # 52% 승률
    'avg_profit_per_trade': 0.0045, # 0.45% 평균 수익
    'max_drawdown': 0.08,          # 8% 최대 낙폭
    'sharpe_ratio': 1.2,           # 1.2 샤프 비율
    'avg_holding_time': 18.5,      # 18.5초 평균 보유
    'total_trades': 103,           # 103개 거래
    'total_episodes': 10,          # 10개 에피소드
    'total_return': 0.046,         # 4.6% 총 수익
    'avg_episode_return': 0.0046   # 0.46% 에피소드당 평균
}
```

## 테스트

```bash
# 모든 테스트 실행
python -m pytest tests/test_grpo_evaluation.py -v

# 예제 실행
python -m examples.grpo_evaluation_example
```

## 문서

- 상세 문서: `docs/GRPO_EVALUATION_METRICS.md`
- 구현 요약: `docs/TASK_11.2_IMPLEMENTATION_SUMMARY.md`
- 요구사항: `.kiro/specs/grpo-scalping-embedding/requirements.md` (7.2)
