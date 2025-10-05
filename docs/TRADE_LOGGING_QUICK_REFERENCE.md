# 실시간 거래 로깅 빠른 참조 가이드

## 빠른 시작

```python
from ai_trader.inference import TradeLogger

# 1. 초기화
logger = TradeLogger(log_dir='logs/trading')

# 2. 거래 기록
logger.log_trade(
    stock_code='005930',
    entry_timestamp='2025-10-05T09:00:00',
    entry_price=70000.0,
    exit_timestamp='2025-10-05T09:00:30',
    exit_price=70500.0,
    holding_time=30.0,
    profit_rate=0.0071
)

# 3. 통계 조회
stats = logger.get_summary_stats()
print(f"Win Rate: {stats['win_rate']:.2%}")

# 4. 종료
logger.close()
```

## 주요 메서드

| 메서드 | 설명 | 예제 |
|--------|------|------|
| `log_trade()` | 거래 기록 | `logger.log_trade(stock_code='005930', ...)` |
| `get_trades()` | 거래 목록 반환 | `trades = logger.get_trades()` |
| `get_trades_df()` | DataFrame 반환 | `df = logger.get_trades_df()` |
| `get_summary_stats()` | 요약 통계 | `stats = logger.get_summary_stats()` |
| `save_summary()` | 요약 저장 | `logger.save_summary()` |
| `clear()` | 기록 초기화 | `logger.clear()` |
| `close()` | 로거 종료 | `logger.close()` |

## 기록되는 정보

✅ 타임스탬프 (거래 기록 시간, 진입 시간, 청산 시간)  
✅ 진입/청산 가격  
✅ 보유 기간 (초)  
✅ 수익률  
✅ 수익 금액  
✅ 거래 비용  
✅ 순수익  
✅ 포지션 크기  
✅ 메타데이터 (선택)

## 출력 파일

- `trades_{session}.json` - JSON 형식
- `trades_{session}.csv` - CSV 형식
- `trades_{session}.log` - 텍스트 로그
- `summary_{session}.json` - 요약 통계

## 요약 통계

```python
stats = logger.get_summary_stats()
# {
#   'total_trades': 150,
#   'win_rate': 0.58,
#   'avg_profit_rate': 0.0023,
#   'avg_holding_time': 28.5,
#   'total_profit': 45000.0,
#   'max_profit': 2500.0,
#   'max_loss': -1800.0
# }
```

## GRPO와 통합

```python
from ai_trader.inference import GRPOInference, TradeLogger

inference = GRPOInference(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt'
)

logger = TradeLogger(log_dir='logs/trading')

# 거래 루프
for sequence in data_stream:
    action, confidence = inference.predict(sequence)
    
    if action == 1:  # 매수
        entry_price = current_price
        entry_time = current_time
    
    elif action == 2:  # 매도
        logger.log_trade(
            stock_code='005930',
            entry_timestamp=entry_time,
            entry_price=entry_price,
            exit_timestamp=current_time,
            exit_price=current_price,
            holding_time=current_time - entry_time,
            profit_rate=(current_price - entry_price) / entry_price,
            metadata={'confidence': confidence}
        )

logger.close()
```

## 실시간 거래 예제 실행

```bash
python examples/grpo_live_trading_example.py \
    --embedding-model models/embedding/checkpoint.pt \
    --policy models/grpo/checkpoint.pt \
    --log-dir logs/live_trading \
    --max-trades 100
```

## 테스트 실행

```bash
python -m pytest tests/test_trade_logger.py -v
```

## 요구사항 7.4 충족

✅ 타임스탬프 기록  
✅ 진입/청산 가격 기록  
✅ 보유 기간 기록  
✅ 손익 기록  
✅ 모든 거래 로깅

## 자세한 문서

전체 문서는 [TRADE_LOGGING.md](./TRADE_LOGGING.md)를 참조하세요.
