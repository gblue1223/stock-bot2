# 주식 실시간 데이터 수신 모듈

키움증권의 **주식체결(0B)** 및 **주식호가(0C)** 실시간 데이터를 WebSocket으로 수신하는 헬퍼 클래스입니다.

## 주요 기능

- ✅ 주식체결(0B) 실시간 데이터 수신
- ✅ 주식호가(0C) 실시간 데이터 수신
- ✅ `FINAL_COLUMNS` 형식으로 정규화된 데이터 제공
- ✅ 매도/매수 대기금액 자동 계산 (호가 × 수량)
- ✅ 히스토리 데이터 저장 (Manager)
- ✅ 콜백 기반 비동기 처리

---

## FINAL_COLUMNS

데이터는 `normalize_datasets.py`의 `FINAL_COLUMNS`와 동일한 형식으로 제공됩니다:

```python
FINAL_COLUMNS = [
    "종목코드", "종목명", "시간", "등락률",
    "누적거래대금", "거래회전율", "체결강도",
    # 매도/매수 대기금액 1~10
    "매도대기금액1", "매도대기금액2", ..., "매도대기금액10",
    "매수대기금액1", "매수대기금액2", ..., "매수대기금액10",
]
```

### 필드 설명

| 필드명 | 타입 | 설명 | 출처 |
|--------|------|------|------|
| `종목코드` | str | 종목 코드 (예: '005930') | 0B/0C 필드 9001 |
| `종목명` | str | 종목명 (예: '삼성전자') | 0B/0C 필드 900 |
| `시간` | str | 체결/호가 시간 (HHMMSSmmm) | 0B 필드 569 |
| `등락률` | float | 등락률 (%) | 0B 필드 12 |
| `누적거래대금` | float | 누적 거래대금 (원) | 0B 필드 14 |
| `거래회전율` | float | 거래회전율 (%) | 0B 필드 31 |
| `체결강도` | float | 체결강도 (%) | 0B 필드 567 |
| `매도대기금액{1-10}` | float | 매도호가 × 매도수량 (백만원) | 0C 계산 |
| `매수대기금액{1-10}` | float | 매수호가 × 매수수량 (백만원) | 0C 계산 |

---

## 클래스

### 1. `RealtimeStockClient`

기본 실시간 데이터 수신 클라이언트

#### 생성자

```python
client = RealtimeStockClient(
    access_token="your_token_here",
    auto_translate=True
)
```

#### 주요 메서드

##### `start()`
WebSocket 연결을 시작합니다.

```python
await client.start()
```

##### `stop()`
WebSocket 연결을 종료합니다.

```python
await client.stop()
```

##### `register_stocks(stock_codes, include_trade=True, include_quote=True, group_no="1")`
종목을 실시간 등록합니다.

**매개변수:**
- `stock_codes` (List[str]): 종목코드 리스트 (예: `['005930', '000660']`)
- `include_trade` (bool): 주식체결(0B) 포함 여부 (기본: True)
- `include_quote` (bool): 주식호가(0C) 포함 여부 (기본: True)
- `group_no` (str): 그룹 번호 (기본: "1")

```python
await client.register_stocks(
    stock_codes=['005930', '000660'],
    include_trade=True,
    include_quote=True
)
```

##### `unregister_stocks(group_no="1")`
실시간 데이터 수신을 해지합니다.

```python
await client.unregister_stocks()
```

##### `get_stock_data(stock_code)`
특정 종목의 최신 데이터를 조회합니다.

**반환값:** `StockRealtimeData` 또는 `None`

```python
data = client.get_stock_data('005930')
if data:
    print(f"현재가: {data.현재가}, 등락률: {data.등락률}%")
```

##### `get_all_stock_data()`
모든 종목의 실시간 데이터를 조회합니다.

**반환값:** `Dict[str, StockRealtimeData]`

```python
all_data = client.get_all_stock_data()
for stock_code, data in all_data.items():
    print(f"{stock_code}: {data.종목명} - {data.현재가}")
```

#### 콜백 함수

##### `on_trade_data(stock_code: str, data: StockRealtimeData)`
주식체결(0B) 데이터 수신 시 호출됩니다.

```python
def on_trade_data(stock_code, data):
    print(f"[체결] {data.종목명}: 현재가={data.현재가}, 등락률={data.등락률}%")

client.on_trade_data = on_trade_data
```

##### `on_quote_data(stock_code: str, data: StockRealtimeData)`
주식호가(0C) 데이터 수신 시 호출됩니다.

```python
def on_quote_data(stock_code, data):
    print(f"[호가] {data.종목명}: 매도1={data.매도호가1}, 매수1={data.매수호가1}")

client.on_quote_data = on_quote_data
```

##### `on_data_update(stock_code: str, data: StockRealtimeData)`
모든 데이터 업데이트 시 호출됩니다.

```python
def on_data_update(stock_code, data):
    print(f"[업데이트] {stock_code}: {data.to_dict()}")

client.on_data_update = on_data_update
```

##### `on_error(error: Exception)`
오류 발생 시 호출됩니다.

```python
def on_error(error):
    print(f"오류 발생: {error}")

client.on_error = on_error
```

---

### 2. `RealtimeStockManager`

고급 기능을 제공하는 관리자 클래스 (히스토리 저장 등)

#### 생성자

```python
manager = RealtimeStockManager(access_token="your_token_here")
```

#### 주요 메서드

##### `start()`
매니저를 시작합니다.

```python
await manager.start()
```

##### `stop()`
매니저를 종료합니다.

```python
await manager.stop()
```

##### `add_stocks(stock_codes, include_trade=True, include_quote=True)`
종목을 추가하고 실시간 등록합니다.

```python
await manager.add_stocks(
    stock_codes=['005930', '000660'],
    include_trade=True,
    include_quote=True
)
```

##### `remove_stocks()`
모든 종목의 실시간 수신을 해지합니다.

```python
await manager.remove_stocks()
```

##### `get_latest_data(stock_code)`
특정 종목의 최신 데이터를 조회합니다.

```python
data = manager.get_latest_data('005930')
```

##### `get_history(stock_code, limit=None)`
특정 종목의 히스토리를 조회합니다.

**매개변수:**
- `stock_code` (str): 종목코드
- `limit` (Optional[int]): 최대 개수 (None이면 전체)

**반환값:** `List[Dict[str, Any]]`

```python
history = manager.get_history('005930', limit=10)
for data in history:
    print(data)
```

##### `clear_history(stock_code=None)`
히스토리를 초기화합니다.

```python
manager.clear_history('005930')  # 특정 종목
manager.clear_history()          # 전체
```

---

### 3. `StockRealtimeData`

실시간 데이터를 담는 데이터 클래스

#### 주요 메서드

##### `compute_waiting_amounts()`
호가와 수량을 곱해 대기금액을 계산합니다 (백만원 단위).

```python
data.compute_waiting_amounts()
```

##### `to_dict()`
`FINAL_COLUMNS` 형식의 딕셔너리로 변환합니다.

```python
data_dict = data.to_dict()
print(data_dict['등락률'])
```

##### `to_full_dict()`
모든 필드를 포함한 딕셔너리로 변환합니다.

```python
full_dict = data.to_full_dict()
```

---

## 사용 예제

### 예제 1: 간단한 실시간 데이터 수신

```python
import asyncio
from kiwoom_rest_api.koreanstock.realtime_stock import RealtimeStockClient

async def main():
    client = RealtimeStockClient(access_token="your_token")
    
    # 콜백 설정
    def on_trade(stock_code, data):
        print(f"체결: {data.종목명} - {data.현재가:,}원 ({data.등락률}%)")
    
    client.on_trade_data = on_trade
    
    # 시작 및 등록
    await client.start()
    await client.register_stocks(['005930', '000660'])
    
    # 30초간 실행
    await asyncio.sleep(30)
    await client.stop()

asyncio.run(main())
```

### 예제 2: Manager로 히스토리 저장

```python
import asyncio
from kiwoom_rest_api.koreanstock.realtime_stock import RealtimeStockManager

async def main():
    manager = RealtimeStockManager(access_token="your_token")
    
    await manager.start()
    await manager.add_stocks(['005930', '000660', '035720'])
    
    # 30초간 실행
    await asyncio.sleep(30)
    
    # 히스토리 출력
    history = manager.get_history('005930', limit=5)
    for data in history:
        print(f"시간: {data['시간']}, 등락률: {data['등락률']}%")
    
    await manager.stop()

asyncio.run(main())
```

### 예제 3: FINAL_COLUMNS 형식으로 데이터 추출

```python
import asyncio
from kiwoom_rest_api.koreanstock.realtime_stock import (
    RealtimeStockClient,
    FINAL_COLUMNS
)

async def main():
    client = RealtimeStockClient(access_token="your_token")
    
    def on_update(stock_code, data):
        # FINAL_COLUMNS 형식으로 변환
        data_dict = data.to_dict()
        
        print(f"종목: {data_dict['종목명']}")
        print(f"등락률: {data_dict['등락률']}%")
        print(f"체결강도: {data_dict['체결강도']}")
        print(f"매도대기금액1: {data_dict['매도대기금액1']:.2f} 백만원")
        print(f"매수대기금액1: {data_dict['매수대기금액1']:.2f} 백만원")
    
    client.on_data_update = on_update
    
    await client.start()
    await client.register_stocks(['005930'])
    
    await asyncio.sleep(30)
    await client.stop()

asyncio.run(main())
```

---

## 필드 매핑

### 주식체결(0B) 필드

| 필드 코드 | 필드명 | 데이터 클래스 속성 |
|-----------|--------|-------------------|
| 9001 | 종목코드 | `종목코드` |
| 900 | 종목명 | `종목명` |
| 10 | 현재가 | `현재가` |
| 11 | 전일대비 | `전일대비` |
| 12 | 등락율 | `등락률` |
| 27 | 매도호가 | `매도호가` |
| 28 | 매수호가 | `매수호가` |
| 15 | 거래량 | `거래량` |
| 13 | 누적거래량 | `누적거래량` |
| 14 | 누적거래대금 | `누적거래대금` |
| 16 | 시가 | `시가` |
| 17 | 고가 | `고가` |
| 18 | 저가 | `저가` |
| 31 | 거래회전율 | `거래회전율` |
| 567 | 체결강도 | `체결강도` |
| 569 | 체결시간 | `시간` |

### 주식호가(0C) 필드

| 필드 코드 | 필드명 | 데이터 클래스 속성 |
|-----------|--------|-------------------|
| 9001 | 종목코드 | `종목코드` |
| 900 | 종목명 | `종목명` |
| 27, 29, 31, ... 45 | 매도호가1~10 | `매도호가1` ~ `매도호가10` |
| 28, 30, 32, ... 46 | 매수호가1~10 | `매수호가1` ~ `매수호가10` |
| 47, 49, 51, ... 65 | 매도호가수량1~10 | `매도호가수량1` ~ `매도호가수량10` |
| 48, 50, 52, ... 66 | 매수호가수량1~10 | `매수호가수량1` ~ `매수호가수량10` |

---

## 참고

- 상세한 예제는 `examples/realtime_stock_example.py` 참고
- WebSocket 상수는 `websocket_constants.py` 참고
- 정규화 로직은 `scripts/data/normalize_datasets.py` 참고

---

## Import 방법

```python
# koapys 패키지에서 직접 import
from koapys import (
    RealtimeStockClient,
    RealtimeStockManager,
    StockRealtimeData,
    FINAL_COLUMNS
)

# 또는 하위 모듈에서 import
from kiwoom_rest_api.koreanstock.realtime_stock import (
    RealtimeStockClient,
    RealtimeStockManager,
    StockRealtimeData,
    FINAL_COLUMNS
)
```
