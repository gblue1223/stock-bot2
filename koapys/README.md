# koapyRest

A REST-based client designed to mirror `koapys.KoaPySimple` key interfaces while using Kiwoom's REST API.

## Status
- Endpoints in `koapyRest/endpoints.py` are placeholders until we extract exact paths from the official PDF: `koapyRest/docs/키움 REST API 문서.pdf`.
- `KoapyRestSimple` in `koapyRest/client.py` implements compatible method names and signatures where feasible.

## Quickstart
```python
from koapyRest import KoapyRestSimple, OrderType, OrderBookType

client = KoapyRestSimple(
    base_url="https://your-kiwoom-rest-host",   # replace
    api_key="<Your API Token>",                # replace
)
client.ensure_connected()

accounts = client.get_account_list()
acc = accounts[0]

# Deposit detail
print(client.get_deposit(acc))

# Equity balance
print(client.get_equity_balance(acc))

# Market data
info = client.get_stock_basic_info("005930")
print("current:", client.get_current_price("005930"))
print("upper:", client.get_upper_limit_price("005930"))

# Order example (limit buy)
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

## Configuration
- Update concrete endpoint paths in `koapyRest/endpoints.py` after confirming from the PDF.
- If authentication uses a different scheme (e.g., API key header, OAuth2), update `KoapyRestSimple.__init__` accordingly.

## Compatibility notes
- Event-driven real-time data and condition search in `koapys` depend on the desktop OpenAPI. The REST API typically exposes polling or streaming endpoints; once we confirm via the docs, we can add compatible wrappers.
