# GRPO 백테스팅 시뮬레이터

## 개요

GRPO 백테스팅 시뮬레이터는 홀드아웃 테스트 데이터에서 GRPO 에이전트의 거래를 시뮬레이션하고, 현실적인 거래 비용 및 슬리피지를 적용하여 성능을 평가합니다.

## 주요 기능

### 1. 현실적인 거래 비용 적용

- **거래 비용**: 수수료(0.015%) + 세금(0.2%) = 0.215% (매수/매도 각각 적용)
- **왕복 거래 비용**: 0.43% (매수 + 매도)
- **양의 보상 조건**: 수익률이 0.43%를 초과해야 양의 보상 발생

### 2. 슬리피지 시뮬레이션

슬리피지는 주문 실행 시 예상 가격과 실제 체결 가격의 차이입니다:

- **매수 시**: 가격이 상승 (불리한 방향)
- **매도 시**: 가격이 하락 (불리한 방향)
- **기본 슬리피지**: 0.01% (설정 가능)

### 3. 빠른 손절 룰 적용

- 매수 후 설정 가능한 시간 임계값(기본값 1.5초) 이내에 가격이 상승하지 않으면 자동 매도
- 위반 시 설정 가능한 페널티(기본값 0.01) 적용
- 위반 횟수 추적 및 보고

### 4. 포괄적인 평가 메트릭

요구사항 7.2에 따라 다음 메트릭을 계산합니다:

- **승률**: 수익이 발생한 거래의 비율
- **거래당 평균 수익**: 모든 거래의 평균 수익률
- **최대 낙폭**: 누적 수익의 최대 하락폭
- **샤프 비율**: 위험 대비 수익률
- **평균 보유 시간**: 거래당 평균 포지션 보유 시간

## 사용 방법

### 기본 사용

```python
from ai_trader.grpo.backtest import run_backtest

# 백테스팅 실행
results = run_backtest(
    embedding_model_path='models/embedding/checkpoint_epoch50.pt',
    policy_path='models/grpo_scalping/checkpoint_step1000000.pt',
    db_path='datasets_norm_all.duckdb',
    table_name='datasets',
    test_start_date='2024-01-01',
    test_end_date='2024-12-31',
    device='cuda',
    output_path='backtest_results.json',
    verbose=True
)

# 결과 확인
print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### CLI 사용

```bash
python examples/grpo_backtest_example.py \
    --embedding-model models/embedding/checkpoint_epoch50.pt \
    --policy models/grpo_scalping/checkpoint_step1000000.pt \
    --db datasets_norm_all.duckdb \
    --test-start-date 2024-01-01 \
    --test-end-date 2024-12-31 \
    --transaction-cost 0.00215 \
    --slippage 0.0001 \
    --quick-exit-threshold 1.5 \
    --quick-exit-penalty 0.01 \
    --device cuda \
    --output backtest_results.json \
    --verbose
```

### 고급 사용

```python
from ai_trader.grpo.backtest import BacktestSimulator
from ai_trader.inference.infer_grpo import GRPOInference

# 추론 엔진 초기화
inference_engine = GRPOInference(
    embedding_model_path='models/embedding/checkpoint_epoch50.pt',
    policy_path='models/grpo_scalping/checkpoint_step1000000.pt',
    device='cuda'
)

# 백테스팅 시뮬레이터 초기화
simulator = BacktestSimulator(
    inference_engine=inference_engine,
    db_path='datasets_norm_all.duckdb',
    table_name='datasets',
    seq_len=60,
    transaction_cost_rate=0.00215,  # 0.215%
    slippage_rate=0.0001,           # 0.01%
    quick_exit_threshold=1.5,       # 1.5초
    quick_exit_penalty=0.01
)

# 특정 종목만 백테스팅
results = simulator.run_backtest(
    test_start_date='2024-01-01',
    test_end_date='2024-12-31',
    stock_codes=['005930', '000660'],  # 삼성전자, SK하이닉스
    deterministic=True,
    verbose=True
)

# 결과 저장
simulator.save_results(results, 'backtest_results.json')

# 리소스 정리
simulator.close()
```

## 결과 구조

백테스팅 결과는 다음과 같은 구조를 가집니다:

```python
{
    'episodes': [
        {
            'total_return': 0.0234,
            'num_trades': 15,
            'avg_holding_time': 25.3,
            'sharpe_ratio': 1.45,
            'quick_exit_violations': 2,
            'win_rate': 0.60,
            'avg_profit_per_trade': 0.00156,
            'trades': [
                {
                    'entry_price': 100.0,
                    'exit_price': 100.5,
                    'holding_time': 20.0,
                    'profit_rate': 0.005,
                    'reward': 0.0007,
                    'stock_code': '005930',
                    'date': '2024-01-01',
                    'time': 90000.0,
                    'confidence': 0.85
                },
                # ... 더 많은 거래
            ],
            'stock_code': '005930',
            'date': '2024-01-01'
        },
        # ... 더 많은 에피소드
    ],
    'metrics': {
        'win_rate': 0.58,
        'avg_profit_per_trade': 0.00123,
        'max_drawdown': 0.08,
        'sharpe_ratio': 1.32,
        'avg_holding_time': 24.5,
        'total_trades': 450,
        'total_episodes': 30,
        'total_return': 0.5535,
        'avg_episode_return': 0.01845
    },
    'trades': [
        # 모든 에피소드의 거래 데이터
    ],
    'summary': {
        'num_episodes': 30,
        'num_stocks': 2,
        'num_days': 15,
        'total_trades': 450,
        'total_slippage_cost': 0.09,
        'avg_slippage_per_trade': 0.0002,
        'total_transaction_cost': 1.935,
        'avg_transaction_cost_per_trade': 0.0043,
        'quick_exit_violations': 45,
        'quick_exit_violation_rate': 0.10
    }
}
```

## 성능 기준 검증

백테스팅 결과는 다음 성능 기준에 따라 평가됩니다 (요구사항 7.2, 7.3):

1. **승률 > 50%**: 수익 거래가 전체 거래의 50%를 초과해야 함
2. **샤프 비율 > 1.0**: 위험 대비 수익률이 1.0을 초과해야 함
3. **최대 낙폭 < 10%**: 누적 수익의 최대 하락폭이 10% 미만이어야 함
4. **평균 보유 시간 < 30초**: 스캘핑 특성상 평균 보유 시간이 30초 미만이어야 함

## 비용 분석

백테스팅 결과에는 다음과 같은 비용 분석이 포함됩니다:

### 거래 비용

- **총 거래 비용**: 모든 거래의 거래 비용 합계
- **거래당 평균 비용**: 0.43% (왕복)

### 슬리피지 비용

- **총 슬리피지 비용**: 모든 거래의 슬리피지 합계
- **거래당 평균 슬리피지**: 설정된 슬리피지 비율의 2배 (매수 + 매도)

### 빠른 손절 룰 위반

- **총 위반 횟수**: 모든 에피소드의 위반 횟수 합계
- **위반 비율**: 위반 횟수 / 총 거래 횟수

## 예제 출력

```
======================================================================
BACKTEST RESULTS
======================================================================
Test Period:             15 days
Stocks Tested:           2
Total Episodes:          30
Total Trades:            450
----------------------------------------------------------------------
PERFORMANCE METRICS
----------------------------------------------------------------------
Total Return:            0.5535
Avg Episode Return:      0.0185
Win Rate:                58.00%
Avg Profit per Trade:    0.0012
Max Drawdown:            8.00%
Sharpe Ratio:            1.3200
Avg Holding Time:        24.50 seconds
----------------------------------------------------------------------
COST ANALYSIS
----------------------------------------------------------------------
Total Transaction Cost:  1.9350 (0.430% per trade)
Total Slippage Cost:     0.0900 (0.020% per trade)
Quick Exit Violations:   45 (10.00% of trades)
======================================================================

Performance Criteria Check:
----------------------------------------------------------------------
✓ Win Rate > 50%: 58.00%
✓ Sharpe Ratio > 1.0: 1.3200
✓ Max Drawdown < 10%: 8.00%
✓ Avg Holding Time < 30s: 24.50s
----------------------------------------------------------------------
✓ All performance criteria met!
```

## 주의사항

1. **데이터 품질**: 백테스팅 결과는 입력 데이터의 품질에 크게 의존합니다. 정규화된 데이터를 사용하세요.

2. **슬리피지 설정**: 실제 시장 상황에 맞게 슬리피지 비율을 조정하세요. 유동성이 낮은 종목은 더 높은 슬리피지가 발생할 수 있습니다.

3. **거래 비용**: 거래 비용은 증권사와 계좌 유형에 따라 다를 수 있습니다. 실제 비용에 맞게 조정하세요.

4. **과최적화**: 백테스팅 결과가 좋다고 해서 실제 거래에서도 같은 성과를 보장하지 않습니다. 과최적화를 주의하세요.

5. **시장 변화**: 과거 데이터로 학습한 모델은 시장 상황이 변하면 성능이 저하될 수 있습니다. 정기적으로 재훈련하세요.

## 참고 자료

- [GRPO 환경 구현](../ai_trader/grpo/env.py)
- [GRPO 평가 메트릭](../ai_trader/grpo/evaluation.py)
- [GRPO 추론 엔진](../ai_trader/inference/infer_grpo.py)
- [요구사항 문서](../.kiro/specs/grpo-scalping-embedding/requirements.md)
- [설계 문서](../.kiro/specs/grpo-scalping-embedding/design.md)
