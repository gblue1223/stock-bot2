# 실시간 거래 로깅 (Trade Logging)

## 개요

TradeLogger는 실시간 거래 중 모든 거래를 기록하는 컴포넌트입니다. 요구사항 7.4에 따라 타임스탬프, 진입/청산 가격, 보유 기간, 손익 등을 포함한 모든 거래 정보를 로깅합니다.

## 주요 기능

### 1. 다중 형식 로깅

TradeLogger는 세 가지 형식으로 거래를 기록합니다:

- **JSON 파일**: 구조화된 거래 데이터 (프로그래밍 방식 접근에 적합)
- **CSV 파일**: 표 형식의 거래 데이터 (스프레드시트 분석에 적합)
- **텍스트 로그**: 사람이 읽기 쉬운 형식 (실시간 모니터링에 적합)

### 2. 기록되는 정보

각 거래에 대해 다음 정보가 기록됩니다:

- **타임스탬프**: 거래 기록 시간
- **종목 코드**: 거래된 종목
- **진입 정보**: 진입 타임스탬프, 진입 가격
- **청산 정보**: 청산 타임스탬프, 청산 가격
- **보유 기간**: 초 단위 보유 시간
- **수익률**: 거래 수익률 (소수)
- **수익 금액**: 절대 수익 금액
- **거래 비용**: 적용된 거래 비용
- **순수익**: 거래 비용 차감 후 순수익
- **포지션 크기**: 거래 포지션 크기
- **메타데이터**: 추가 정보 (선택)

### 3. 요약 통계

TradeLogger는 다음 요약 통계를 제공합니다:

- 총 거래 횟수
- 승률 (win rate)
- 평균 수익률
- 평균 보유 시간
- 총 수익
- 최대 수익
- 최대 손실
- 평균 순수익
- 수익률 표준편차

## 사용 방법

### 기본 사용

```python
from ai_trader.inference import TradeLogger

# TradeLogger 초기화
trade_logger = TradeLogger(
    log_dir='logs/live_trading',
    session_name='session_20251005',
    enable_json=True,
    enable_csv=True,
    enable_text=True
)

# 거래 로깅
trade_logger.log_trade(
    stock_code='005930',
    entry_timestamp='2025-10-05T09:00:00',
    entry_price=70000.0,
    exit_timestamp='2025-10-05T09:00:30',
    exit_price=70500.0,
    holding_time=30.0,
    profit_rate=0.0071,
    transaction_cost=0.0043,
    position_size=1.0,
    metadata={'strategy': 'scalping'}
)

# 요약 통계 조회
stats = trade_logger.get_summary_stats()
print(f"Total Trades: {stats['total_trades']}")
print(f"Win Rate: {stats['win_rate']:.2%}")
print(f"Avg Profit Rate: {stats['avg_profit_rate']:.4f}")

# 로거 종료 (자동으로 요약 저장)
trade_logger.close()
```

### GRPO 추론 엔진과 통합

```python
from ai_trader.inference import GRPOInference, TradeLogger

# GRPO 추론 엔진 초기화
inference_engine = GRPOInference(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    device='cuda'
)

# TradeLogger 초기화
trade_logger = TradeLogger(
    log_dir='logs/live_trading',
    enable_json=True,
    enable_csv=True,
    enable_text=True
)

# 실시간 거래 루프
position = 0
entry_price = 0.0
entry_timestamp = ""

for sequence in data_stream:
    # 행동 예측
    action, confidence = inference_engine.predict(sequence)
    
    current_price = get_current_price()
    current_timestamp = get_current_timestamp()
    
    if action == 1 and position == 0:  # 매수
        position = 1
        entry_price = current_price
        entry_timestamp = current_timestamp
    
    elif action == 2 and position == 1:  # 매도
        holding_time = calculate_holding_time(entry_timestamp, current_timestamp)
        profit_rate = (current_price - entry_price) / entry_price
        
        # 거래 로깅
        trade_logger.log_trade(
            stock_code='005930',
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            exit_timestamp=current_timestamp,
            exit_price=current_price,
            holding_time=holding_time,
            profit_rate=profit_rate,
            metadata={'confidence': confidence}
        )
        
        position = 0

# 로거 종료
trade_logger.close()
```

### DataFrame으로 거래 분석

```python
import pandas as pd
import matplotlib.pyplot as plt

# 거래 기록을 DataFrame으로 변환
df = trade_logger.get_trades_df()

# 수익률 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(df['profit_rate'], bins=50, alpha=0.7)
plt.xlabel('Profit Rate')
plt.ylabel('Frequency')
plt.title('Profit Rate Distribution')
plt.show()

# 보유 시간 vs 수익률
plt.figure(figsize=(10, 6))
plt.scatter(df['holding_time_seconds'], df['profit_rate'], alpha=0.5)
plt.xlabel('Holding Time (seconds)')
plt.ylabel('Profit Rate')
plt.title('Holding Time vs Profit Rate')
plt.show()

# 시간대별 승률
df['hour'] = pd.to_datetime(df['entry_timestamp']).dt.hour
hourly_win_rate = df.groupby('hour').apply(lambda x: (x['profit_rate'] > 0).mean())
print(hourly_win_rate)
```

## 출력 파일 형식

### JSON 파일 (`trades_{session_name}.json`)

```json
[
  {
    "timestamp": "2025-10-05T09:00:35.123456",
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
    "metadata": {
      "confidence": 0.85,
      "strategy": "scalping"
    }
  }
]
```

### CSV 파일 (`trades_{session_name}.csv`)

```csv
timestamp,stock_code,entry_timestamp,entry_price,exit_timestamp,exit_price,holding_time_seconds,profit_rate,profit_amount,transaction_cost,net_profit,action,position_size,metadata
2025-10-05T09:00:35.123456,005930,2025-10-05T09:00:00,70000.0,2025-10-05T09:00:30,70500.0,30.0,0.0071,500.0,0.0043,199.0,SELL,1.0,"{""confidence"": 0.85}"
```

### 텍스트 로그 (`trades_{session_name}.log`)

```
2025-10-05 09:00:35 - INFO - Trade: 005930 | Entry: 70000.00 @ 2025-10-05T09:00:00 | Exit: 70500.00 @ 2025-10-05T09:00:30 | Holding: 30.00s | Profit Rate: 0.0071 | Net Profit: 199.00
```

### 요약 파일 (`summary_{session_name}.json`)

```json
{
  "session_name": "session_20251005",
  "start_time": "2025-10-05T09:00:00",
  "end_time": "2025-10-05T15:30:00",
  "statistics": {
    "total_trades": 150,
    "win_rate": 0.58,
    "avg_profit_rate": 0.0023,
    "avg_holding_time": 28.5,
    "total_profit": 45000.0,
    "max_profit": 2500.0,
    "max_loss": -1800.0,
    "avg_net_profit": 300.0,
    "std_profit_rate": 0.0045
  },
  "trades": [...]
}
```

## API 참조

### TradeLogger 클래스

#### `__init__(log_dir, session_name=None, enable_json=True, enable_csv=True, enable_text=True)`

TradeLogger 인스턴스를 초기화합니다.

**매개변수:**
- `log_dir` (str): 로그 파일을 저장할 디렉토리
- `session_name` (str, optional): 거래 세션 이름 (기본값: 현재 타임스탬프)
- `enable_json` (bool): JSON 로깅 활성화 (기본값: True)
- `enable_csv` (bool): CSV 로깅 활성화 (기본값: True)
- `enable_text` (bool): 텍스트 로깅 활성화 (기본값: True)

#### `log_trade(stock_code, entry_timestamp, entry_price, exit_timestamp, exit_price, holding_time, profit_rate, transaction_cost=0.0043, position_size=1.0, metadata=None)`

거래를 기록합니다.

**매개변수:**
- `stock_code` (str): 종목 코드
- `entry_timestamp` (str): 진입 타임스탬프
- `entry_price` (float): 진입 가격
- `exit_timestamp` (str): 청산 타임스탬프
- `exit_price` (float): 청산 가격
- `holding_time` (float): 보유 시간 (초)
- `profit_rate` (float): 수익률 (소수)
- `transaction_cost` (float): 거래 비용 비율 (기본값: 0.0043)
- `position_size` (float): 포지션 크기 (기본값: 1.0)
- `metadata` (dict, optional): 추가 메타데이터

#### `get_trades()`

모든 거래 기록을 반환합니다.

**반환값:**
- `List[Dict[str, Any]]`: 거래 기록 리스트

#### `get_trades_df()`

거래 기록을 pandas DataFrame으로 반환합니다.

**반환값:**
- `pd.DataFrame`: 거래 기록 DataFrame

#### `get_summary_stats()`

거래 요약 통계를 반환합니다.

**반환값:**
- `Dict[str, Any]`: 요약 통계 딕셔너리

#### `save_summary(output_path=None)`

거래 요약을 파일로 저장합니다.

**매개변수:**
- `output_path` (str, optional): 출력 파일 경로

#### `clear()`

거래 기록을 초기화합니다.

#### `close()`

로거를 종료하고 최종 요약을 저장합니다.

## 실시간 거래 예제

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

## 요구사항 충족

TradeLogger는 요구사항 7.4를 완전히 충족합니다:

> **요구사항 7.4**: WHEN 에이전트가 실시간으로 거래하면 THEN 시스템은 타임스탬프, 진입/청산 가격, 보유 기간, 손익과 함께 모든 거래를 로깅해야 합니다

### 충족 사항:

✅ **타임스탬프**: 거래 기록 시간, 진입 타임스탬프, 청산 타임스탬프 모두 기록  
✅ **진입/청산 가격**: entry_price, exit_price 필드로 기록  
✅ **보유 기간**: holding_time_seconds 필드로 초 단위 기록  
✅ **손익**: profit_rate (수익률), profit_amount (수익 금액), net_profit (순수익) 기록  
✅ **모든 거래**: 각 거래마다 log_trade() 호출로 완전한 기록 보장  
✅ **다중 형식**: JSON, CSV, 텍스트 로그로 다양한 분석 지원  
✅ **요약 통계**: 승률, 평균 수익률 등 종합 분석 제공

## 테스트

TradeLogger의 모든 기능은 `tests/test_trade_logger.py`에서 테스트됩니다:

```bash
# 테스트 실행
python -m pytest tests/test_trade_logger.py -v
```

테스트 커버리지:
- ✅ 초기화 테스트
- ✅ 거래 로깅 테스트
- ✅ 여러 거래 로깅 테스트
- ✅ JSON 로깅 테스트
- ✅ CSV 로깅 테스트
- ✅ 텍스트 로깅 테스트
- ✅ DataFrame 변환 테스트
- ✅ 요약 통계 테스트
- ✅ 요약 저장 테스트
- ✅ 거래 기록 초기화 테스트
- ✅ 빈 거래 기록 통계 테스트
- ✅ 수익 계산 테스트

## 성능 고려사항

### 파일 I/O 최적화

- **JSON**: 각 거래마다 파일을 다시 작성하므로 대량 거래 시 성능 영향 가능
- **CSV**: 추가 모드로 작성하므로 효율적
- **텍스트 로그**: 버퍼링된 파일 핸들러 사용으로 효율적

### 메모리 사용

- 모든 거래가 메모리에 저장됨 (`self.trades` 리스트)
- 장시간 실행 시 메모리 사용량 증가 가능
- 필요시 주기적으로 `clear()` 호출하여 메모리 해제

### 권장사항

- 대량 거래 시 JSON 로깅 비활성화 고려 (`enable_json=False`)
- CSV와 텍스트 로그만 사용하여 성능 최적화
- 주기적으로 요약 저장 및 거래 기록 초기화

## 문제 해결

### 파일 권한 오류

```python
# 로그 디렉토리가 존재하고 쓰기 권한이 있는지 확인
import os
log_dir = 'logs/live_trading'
os.makedirs(log_dir, exist_ok=True)
```

### 인코딩 오류

TradeLogger는 UTF-8 인코딩을 사용합니다. 한글 종목명이나 메타데이터가 올바르게 저장됩니다.

### 메모리 부족

```python
# 주기적으로 거래 기록 초기화
if len(trade_logger.get_trades()) > 10000:
    trade_logger.save_summary()
    trade_logger.clear()
```

## 참고 자료

- [GRPO 추론 엔진 문서](./GRPO_INFERENCE.md)
- [실시간 거래 예제](../examples/grpo_live_trading_example.py)
- [테스트 코드](../tests/test_trade_logger.py)
