# koapys

Kiwoom REST API를 사용하면서 `koapys.KoaPySimple`의 핵심 인터페이스를 최대한 유사하게 제공하는 REST 기반 클라이언트입니다.

## 상태(Status)
- `koapys/endpoints.py`의 엔드포인트 경로들은 공식 문서(`koapys/docs/키움 REST API 문서.pdf`)에서 정확한 경로를 확인하기 전까지는 플레이스홀더입니다.
- `koapys/client.py`의 `KoapysSimple`은 가능한 범위 내에서 호환되는 메서드명과 시그니처를 제공합니다.

## 빠른 시작(Quickstart)
```python
from koapys import KoapysSimple, OrderType, OrderBookType

client = KoapysSimple(
    base_url="https://your-kiwoom-rest-host",   # 실제 REST 서버 주소로 교체
    api_key="<Your API Token>",                # 실제 API 토큰으로 교체
)
client.ensure_connected()

accounts = client.get_account_list()
acc = accounts[0]

# 예수금 상세
print(client.get_deposit(acc))

# 주식 잔고
print(client.get_equity_balance(acc))

# 시세/기초 정보
info = client.get_stock_basic_info("005930")
print("current:", client.get_current_price("005930"))
print("upper:", client.get_upper_limit_price("005930"))

# 주문 예시 (지정가 매수)
res = client.send_order(
    rqname="KOA_NORMAL_BUY",
    account_no=acc,
    order_type=OrderType.BUY,
    code="005930",
    quantity=10,
    price=70000,
    hoga=OrderBookType.LIMIT,
)
print(res)
```

## 설정(Configuration)
- 공식 PDF 확인 후 `koapys/endpoints.py`에 실제 엔드포인트 경로를 업데이트하세요.
- 인증 방식이 다를 경우(예: API Key 헤더, OAuth2 등) `KoapysSimple.__init__` 구현을 알맞게 수정하세요.

## 호환성(Compatibility) 안내
- `koapys`의 이벤트 기반 실시간 데이터 및 조건검색 기능은 데스크톱 OpenAPI에 의존합니다. REST API는 일반적으로 폴링 또는 스트리밍 엔드포인트를 제공합니다. 문서를 통해 확인이 되는 즉시, 가능한 범위에서 호환 레이어(래퍼)를 추가할 수 있습니다.
