# koapys

Kiwoom REST API를 사용하면서 `koapys.KoaPySimple`의 핵심 인터페이스를 최대한 유사하게 제공하는 REST 기반 클라이언트입니다.

## 상태(Status)
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
- 인증 방식이 다를 경우(예: API Key 헤더, OAuth2 등) `KoapysSimple.__init__` 구현을 알맞게 수정하세요.

## 호환성(Compatibility) 안내

### 지원되는 기능
- ✅ 계좌 조회: `get_deposit()`, `get_equity_balance()`, `get_equity_balance2()`
- ✅ 시세 조회: `get_stock_basic_info()`, `get_current_price()`, `get_upper_limit_price()`
- ✅ 주문: `send_order()`, `send_order_modify()`, `send_order_cancel()`
- ✅ **조건검색**: `get_condition_load()`, `get_condition_name_list()`, `get_stocks_by_condition()` (ka10171, ka10173)
- ✅ 한글 필드명: OpenAPI와 동일한 한글 키 이름 사용

### 미지원 기능
- ❌ **실시간 데이터**: 이벤트 기반 실시간 시세는 데스크톱 OpenAPI 전용
  - REST API는 폴링 방식으로 구현해야 합니다
- ⚠️ **계좌 목록**: `get_account_list()` 부분 지원 (시뮬레이션 모드만)

자세한 내용은 `MIGRATION_NOTES.md`를 참고하세요.
