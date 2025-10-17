# 조건검색 WebSocket 사용 가이드

키움 REST API의 조건검색 기능을 WebSocket으로 실시간 사용하는 방법을 설명합니다.

## 개요

조건검색은 HTS에서 설정한 조건식에 맞는 종목을 실시간으로 조회하는 기능입니다.

- **REST API (ka10172, ka10173)**: 조건식 목록 조회 및 단순 조회
- **WebSocket**: 조건 충족 종목의 실시간 편입/이탈 알림

## 주요 기능

### 1. WebSocketClient에 추가된 메서드

#### `register_condition_search()`
조건검색 실시간 데이터 수신을 시작합니다.

```python
await client.register_condition_search(
    condition_index="000",      # 조건식 인덱스
    condition_name="상승추세",   # 조건식 이름
    group_no="1",               # 그룹 번호 (선택)
    refresh="1"                 # 기존 등록 유지 (선택)
)
```

#### `unregister_condition_search()`
조건검색 실시간 데이터 수신을 중지합니다.

```python
# 특정 조건식 해지
await client.unregister_condition_search(
    condition_index="000",
    group_no="1"
)

# 전체 조건식 해지
await client.unregister_condition_search(group_no="1")
```

### 2. WebSocket 메시지 형식

#### 등록 요청 (CNSRREQ)
```json
{
    "trnm": "CNSRREQ",
    "grp_no": "1",
    "refresh": "1",
    "data": [{
        "cond_idx": "000",
        "cond_nm": "상승추세종목",
        "rtime_flg": "1"
    }]
}
```

#### 실시간 데이터 수신 (COND_REAL)
```json
{
    "trnm": "COND_REAL",
    "return_code": 0,
    "return_msg": "정상",
    "data": [{
        "cond_idx": "000",
        "stk_cd": "005930",
        "stk_nm": "삼성전자",
        "action": "in"  // "in": 편입, "out": 이탈
    }]
}
```

## 사용 예제

### 기본 사용법

```python
import asyncio
from kiwoom_rest_api import WebSocketClient, RealTimeData
from kiwoom_rest_api.koreanstock.stockinfo import StockInfo
from kiwoom_rest_api.auth.token import TokenManager

async def main():
    # 1. 토큰 매니저 및 StockInfo 생성
    token_manager = TokenManager()
    stock_info = StockInfo(
        base_url="https://api.kiwoom.com", 
        token_manager=token_manager
    )
    
    # 2. 조건식 목록 조회 (WebSocket)
    # WebSocket 로그인 후 condition_list_request_ka10171() 호출
    # 응답은 on_data 콜백으로 수신됨
    
    if not conditions:
        print("등록된 조건식이 없습니다.")
        return
    
    # 3. WebSocket 클라이언트 생성
    access_token = token_manager.get_token()
    client = WebSocketClient(access_token=access_token)
    
    # 4. 데이터 수신 콜백 설정
    async def on_data(realtime_data: RealTimeData):
        if realtime_data.trnm == 'COND_REAL':
            for item in realtime_data.data:
                stock_code = item.get('stk_cd')
                stock_name = item.get('stk_nm')
                action = item.get('action')
                
                if action == 'in':
                    print(f"[편입] {stock_code} - {stock_name}")
                elif action == 'out':
                    print(f"[이탈] {stock_code} - {stock_name}")
    
    # 5. 로그인 후 조건검색 등록
    async def on_login():
        condition = conditions[0]
        await client.register_condition_search(
            condition_index=condition['cond_idx'],
            condition_name=condition['cond_nm']
        )
    
    client.on_data = on_data
    client.on_login = on_login
    
    # 6. 클라이언트 시작
    await client.start()
    
    # 7. 60초간 실행
    await asyncio.sleep(60)
    
    # 8. 정리
    await client.unregister_condition_search(
        condition_index=conditions[0]['cond_idx']
    )
    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 여러 조건식 동시 실행

```python
async def multiple_conditions():
    # ... 초기화 생략 ...
    
    # 여러 조건식 등록
    async def on_login():
        for i, condition in enumerate(conditions[:3]):  # 최대 3개
            await client.register_condition_search(
                condition_index=condition['cond_idx'],
                condition_name=condition['cond_nm'],
                group_no=str(i+1)  # 각 조건식을 별도 그룹으로
            )
            await asyncio.sleep(0.5)  # 등록 간격
    
    client.on_login = on_login
    await client.start()
    
    # 실행 후 모두 해지
    for i, condition in enumerate(conditions[:3]):
        await client.unregister_condition_search(
            condition_index=condition['cond_idx'],
            group_no=str(i+1)
        )
```

### 조건 충족 종목 저장

```python
# 조건별 종목 저장
condition_stocks = {}

async def on_data(realtime_data: RealTimeData):
    if realtime_data.trnm == 'COND_REAL':
        for item in realtime_data.data:
            cond_idx = item.get('cond_idx')
            stock_code = item.get('stk_cd')
            action = item.get('action')
            
            if cond_idx not in condition_stocks:
                condition_stocks[cond_idx] = set()
            
            if action == 'in':
                condition_stocks[cond_idx].add(stock_code)
            elif action == 'out':
                condition_stocks[cond_idx].discard(stock_code)

# 최종 결과 출력
for cond_idx, stocks in condition_stocks.items():
    print(f"조건식 {cond_idx}: {len(stocks)}개 종목")
    print(f"  종목: {', '.join(list(stocks)[:10])}")
```

## 주의사항

### 1. HTS 조건식 등록 필요
WebSocket 조건검색을 사용하려면 먼저 키움 HTS에서 조건식을 등록해야 합니다.

### 2. 동시 실행 제한
- 권장: 동시에 최대 3~5개 조건식
- 너무 많은 조건식을 등록하면 성능 저하 가능

### 3. 실시간 데이터 처리
- `action='in'`: 조건에 새로 편입된 종목
- `action='out'`: 조건에서 이탈한 종목
- 편입/이탈은 실시간으로 발생하므로 빠른 처리 필요

### 4. 연결 유지
- WebSocket 연결은 자동으로 유지됨 (auto_reconnect=True)
- 장시간 연결 시 주기적인 PING/PONG으로 연결 확인

## REST API vs WebSocket

| 기능 | REST API (ka10173) | WebSocket |
|------|-------------------|-----------|
| 조건 충족 종목 조회 | ✅ 가능 | ✅ 가능 |
| 실시간 편입/이탈 | ❌ 불가 | ✅ 가능 |
| 폴링 필요 | ✅ 필요 | ❌ 불필요 |
| 서버 부하 | 높음 | 낮음 |
| 권장 사용 사례 | 1회성 조회 | 실시간 모니터링 |

## 완전한 예제

전체 예제 코드는 다음 파일을 참고하세요:
```
kiwoom_rest_api/examples/condition_search_websocket_example.py
```

실행 방법:
```bash
# 환경 변수 설정
export KIWOOM_ACCESS_TOKEN="your_token_here"

# 예제 실행
python kiwoom_rest_api/examples/condition_search_websocket_example.py
```

## 트러블슈팅

### Q1. 조건식이 조회되지 않음
A: HTS에서 조건식을 먼저 등록하고 저장했는지 확인하세요.

### Q2. 실시간 데이터가 수신되지 않음
A: 
- WebSocket 로그인이 성공했는지 확인
- 조건식 등록 응답(CNSRREQ)의 return_code 확인
- 시장 시간 확인 (장 마감 후에는 데이터 없음)

### Q3. 연결이 자주 끊김
A:
- `auto_reconnect=True` 설정 확인
- 네트워크 상태 확인
- 토큰 유효기간 확인

## 참고 자료

- 키움 REST API 문서: 253페이지 (ka10173 조건검색 요청)
- WebSocket 예제: `kiwoom_rest_api/examples/websocket_example.py`
- StockInfo API: `kiwoom_rest_api/koreanstock/stockinfo.py`
