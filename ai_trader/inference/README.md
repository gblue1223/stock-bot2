# 추론 모듈 (Inference Module)

실시간 매매를 위한 추론 엔진과 거래 로깅 시스템을 제공합니다.

## 주요 컴포넌트

### 1. GRPOInference

실시간 매매를 위한 GRPO 추론 엔진입니다.

**주요 기능:**
- 임베딩 모델과 정책을 로드하여 실시간 행동 예측
- TorchScript 컴파일로 추론 속도 최적화
- 임베딩 캐시로 중복 계산 방지
- 목표 지연 시간: < 10ms per sample

**사용 예제:**

```python
from ai_trader.inference import GRPOInference

# 추론 엔진 초기화
inference = GRPOInference(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    device='cuda',
    use_torchscript=True,
    cache_size=1000
)

# 행동 예측
action, confidence = inference.predict(sequence, deterministic=True)
# action: 0=보유, 1=매수, 2=매도
# confidence: 행동에 대한 신뢰도 (0.0 ~ 1.0)

# 성능 통계 확인
stats = inference.get_performance_stats()
print(f"Mean inference time: {stats['mean_inference_time_ms']:.2f}ms")
```

### 2. TradeLogger

실시간 거래 로깅 시스템입니다.

**주요 기능:**
- 모든 거래를 타임스탬프, 진입/청산 가격, 보유 기간, 손익과 함께 기록
- 다양한 형식으로 저장: JSON, CSV, 텍스트 로그
- 거래 요약 통계 자동 계산
- 세션별 거래 기록 관리

**사용 예제:**

```python
from ai_trader.inference import TradeLogger

# 거래 로거 초기화
logger = TradeLogger(
    log_dir='logs/live_trading',
    session_name='20251005_090000',
    enable_json=True,
    enable_csv=True,
    enable_text=True
)

# 거래 로깅
logger.log_trade(
    stock_code='005930',
    entry_timestamp='2025-10-05T09:00:00',
    entry_price=70000.0,
    exit_timestamp='2025-10-05T09:00:30',
    exit_price=70500.0,
    holding_time=30.0,
    profit_rate=0.0071,
    transaction_cost=0.0043,
    position_size=1.0,
    metadata={'confidence': 0.95}
)

# 거래 요약 통계
stats = logger.get_summary_stats()
print(f"Total trades: {stats['total_trades']}")
print(f"Win rate: {stats['win_rate']:.2%}")
print(f"Avg profit rate: {stats['avg_profit_rate']:.4f}")

# 로거 종료 (자동으로 요약 저장)
logger.close()
```

**로그 파일 형식:**

1. **JSON 파일** (`trades_{session_name}.json`):
   ```json
   [
     {
       "timestamp": "2025-10-05T09:00:30.123456",
       "stock_code": "005930",
       "entry_timestamp": "2025-10-05T09:00:00",
       "entry_price": 70000.0,
       "exit_timestamp": "2025-10-05T09:00:30",
       "exit_price": 70500.0,
       "holding_time_seconds": 30.0,
       "profit_rate": 0.0071,
       "profit_amount": 500.0,
       "transaction_cost": 0.0043,
       "net_profit": 199.0,
       "action": "SELL",
       "position_size": 1.0,
       "metadata": {"confidence": 0.95}
     }
   ]
   ```

2. **CSV 파일** (`trades_{session_name}.csv`):
   ```csv
   timestamp,stock_code,entry_timestamp,entry_price,exit_timestamp,exit_price,holding_time_seconds,profit_rate,profit_amount,transaction_cost,net_profit,action,position_size,metadata
   2025-10-05T09:00:30.123456,005930,2025-10-05T09:00:00,70000.0,2025-10-05T09:00:30,70500.0,30.0,0.0071,500.0,0.0043,199.0,SELL,1.0,"{""confidence"": 0.95}"
   ```

3. **텍스트 로그** (`trades_{session_name}.log`):
   ```
   2025-10-05 09:00:30 - INFO - Trade: 005930 | Entry: 70000.00 @ 2025-10-05T09:00:00 | Exit: 70500.00 @ 2025-10-05T09:00:30 | Holding: 30.00s | Profit Rate: 0.0071 | Net Profit: 199.00
   ```

4. **요약 파일** (`summary_{session_name}.json`):
   ```json
   {
     "session_name": "20251005_090000",
     "start_time": "2025-10-05T09:00:00.000000",
     "end_time": "2025-10-05T10:00:00.000000",
     "statistics": {
       "total_trades": 50,
       "win_rate": 0.58,
       "avg_profit_rate": 0.0023,
       "avg_holding_time": 28.5,
       "total_profit": 5750.0,
       "max_profit": 850.0,
       "max_loss": -420.0
     },
     "trades": [...]
   }
   ```

## 실시간 거래 시뮬레이션 예제

전체 실시간 거래 시뮬레이션 예제는 `examples/grpo_live_trading_example.py`를 참조하세요.

```bash
# 실시간 거래 시뮬레이션 실행
python examples/grpo_live_trading_example.py \
    --embedding-model models/embedding/checkpoint.pt \
    --policy models/grpo/checkpoint.pt \
    --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
    --log-dir logs/live_trading \
    --max-trades 100 \
    --device cuda
```

## 요구사항

- 요구사항 7.4: 실시간 거래 로깅
  - 모든 거래를 타임스탬프, 진입/청산 가격, 보유 기간, 손익과 함께 기록
  - 다양한 형식으로 저장 (JSON, CSV, 텍스트)
  - 거래 요약 통계 자동 계산

## 테스트

```bash
# TradeLogger 테스트 실행
python -m pytest tests/test_trade_logger.py -v
```

## 성능

- **추론 지연 시간**: < 10ms per sample (목표)
- **로깅 오버헤드**: < 1ms per trade
- **메모리 사용량**: 거래당 약 1KB (메모리 내 저장)

## 주의사항

1. **파일 I/O**: 거래가 많을 경우 파일 I/O가 병목이 될 수 있습니다. 필요시 비동기 로깅을 고려하세요.
2. **메모리 관리**: 장시간 실행 시 메모리 사용량이 증가할 수 있습니다. 주기적으로 `clear()`를 호출하여 메모리를 해제하세요.
3. **디스크 공간**: 거래가 많을 경우 로그 파일이 커질 수 있습니다. 주기적으로 로그를 아카이브하세요.
