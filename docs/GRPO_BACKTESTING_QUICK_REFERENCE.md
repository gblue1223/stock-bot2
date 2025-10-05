# GRPO 백테스팅 빠른 참조 가이드

## 빠른 시작

### 1. 기본 백테스팅 실행

```python
from ai_trader.grpo.backtest import run_backtest

results = run_backtest(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    db_path='datasets.duckdb',
    device='cuda'
)
```

### 2. CLI로 실행

```bash
python examples/grpo_backtest_example.py \
    --embedding-model models/embedding/checkpoint.pt \
    --policy models/grpo/checkpoint.pt \
    --db datasets.duckdb \
    --device cuda
```

## 주요 매개변수

| 매개변수 | 기본값 | 설명 |
|---------|--------|------|
| `transaction_cost_rate` | 0.00215 | 거래 비용 비율 (0.215%) |
| `slippage_rate` | 0.0001 | 슬리피지 비율 (0.01%) |
| `quick_exit_threshold` | 1.5 | 빠른 손절 시간 임계값 (초) |
| `quick_exit_penalty` | 0.01 | 빠른 손절 룰 위반 페널티 |
| `seq_len` | 60 | 시퀀스 길이 |

## 결과 해석

### 핵심 메트릭

```python
metrics = results['metrics']

# 승률 (목표: > 50%)
win_rate = metrics['win_rate']

# 샤프 비율 (목표: > 1.0)
sharpe_ratio = metrics['sharpe_ratio']

# 최대 낙폭 (목표: < 10%)
max_drawdown = metrics['max_drawdown']

# 평균 보유 시간 (목표: < 30초)
avg_holding_time = metrics['avg_holding_time']
```

### 비용 분석

```python
summary = results['summary']

# 총 거래 비용
total_cost = summary['total_transaction_cost']

# 총 슬리피지 비용
total_slippage = summary['total_slippage_cost']

# 빠른 손절 룰 위반 횟수
violations = summary['quick_exit_violations']
```

## 성능 기준

| 메트릭 | 기준 | 설명 |
|--------|------|------|
| 승률 | > 50% | 수익 거래 비율 |
| 샤프 비율 | > 1.0 | 위험 대비 수익률 |
| 최대 낙폭 | < 10% | 최대 손실 폭 |
| 평균 보유 시간 | < 30초 | 스캘핑 특성 |

## 비용 구조

### 거래 비용
- **수수료**: 0.015%
- **세금**: 0.2%
- **편도**: 0.215%
- **왕복**: 0.43%

### 슬리피지
- **매수**: +0.01% (가격 상승)
- **매도**: -0.01% (가격 하락)
- **총**: 0.02% (왕복)

### 총 비용
- **거래 비용 + 슬리피지**: 0.45% (왕복)
- **손익분기점**: 수익률 > 0.45%

## 일반적인 사용 사례

### 1. 특정 기간 백테스팅

```python
results = run_backtest(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    db_path='datasets.duckdb',
    test_start_date='2024-01-01',
    test_end_date='2024-12-31'
)
```

### 2. 특정 종목만 테스트

```python
results = run_backtest(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    db_path='datasets.duckdb',
    stock_codes=['005930', '000660']  # 삼성전자, SK하이닉스
)
```

### 3. 커스텀 비용 설정

```python
results = run_backtest(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    db_path='datasets.duckdb',
    transaction_cost_rate=0.003,  # 0.3% (더 높은 비용)
    slippage_rate=0.0005          # 0.05% (더 높은 슬리피지)
)
```

### 4. 결과 저장

```python
results = run_backtest(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    db_path='datasets.duckdb',
    output_path='backtest_results.json'
)
```

## 문제 해결

### Q: 백테스팅이 너무 느려요
A: GPU를 사용하세요 (`--device cuda`)

### Q: 메모리 부족 오류가 발생해요
A: 테스트 기간을 줄이거나 특정 종목만 선택하세요

### Q: 승률이 너무 낮아요
A: 모델 재훈련이 필요할 수 있습니다

### Q: 슬리피지가 너무 커요
A: `--slippage` 매개변수를 조정하세요

## 추가 리소스

- [상세 가이드](GRPO_BACKTESTING.md)
- [구현 요약](TASK_11.3_IMPLEMENTATION_SUMMARY.md)
- [GRPO 환경](../ai_trader/grpo/env.py)
- [평가 메트릭](../ai_trader/grpo/evaluation.py)
