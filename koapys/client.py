import logging
import threading
from typing import Any, Callable, Dict, Optional

import requests

from .endpoints import (
    DEFAULT_BASE_URL,
    AUTH_TOKEN,
    PING,
    ACCOUNTS_LIST,
    DEPOSIT_DETAIL,
    EQUITY_BALANCE,
    EQUITY_BALANCE2,
    STOCK_BASIC_INFO,
    CURRENT_PRICE,
    UPPER_LIMITS,
    SEND_ORDER,
    SEND_ORDER_MODIFY,
    SEND_ORDER_CANCEL,
)
from .types import OrderType, OrderBookType

OrderingListener = Callable[[str, Any, Any], None]
DataListener = Callable[[Dict[str, Any]], None]
StringListener = Callable[[str], None]

class KoapyRestSimple:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: int = 10,
        simulation: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._simulation = simulation
        self._session = requests.Session()
        self._logger = logger or logging.getLogger("koapyRest")
        self._is_connected = False

        if api_key:
            # Pre-auth header style, actual scheme TBD after PDF parsing
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    # --- Compatibility helpers ---
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_simulation_mode(self) -> bool:
        return self._simulation

    def ensure_connected(self):
        if self._simulation:
            self._is_connected = True
            return
        # Try a healthcheck; adjust endpoint after PDF parsing
        try:
            r = self._session.get(self._base_url + PING, timeout=self._timeout)
            r.raise_for_status()
            self._is_connected = True
        except Exception as e:
            self._logger.debug(f"Ping failed: {e}")
            self._is_connected = False
            raise ConnectionError("Unable to connect to Kiwoom REST API")

    def close(self):
        self._session.close()
        self._is_connected = False

    # --- Accounts ---
    def get_account_list(self):
        if self._simulation:
            return ["0000000000"]
        self.ensure_connected()
        r = self._session.get(self._base_url + ACCOUNTS_LIST, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        # Expect data to be a list or under a key, normalize
        if isinstance(data, dict) and "accounts" in data:
            return data["accounts"]
        return data

    def get_deposit(self, account_no: str) -> Dict[str, Any]:
        if self._simulation:
            return {
                "예수금": "100000000",
                "d+1추정예수금": "0",
                "d+2추정예수금": "0",
                "출금가능금액": "100000000",
                "d+1출금가능금액": "0",
                "d+2출금가능금액": "0",
            }
        self.ensure_connected()
        url = self._base_url + DEPOSIT_DETAIL.format(account_no=account_no)
        r = self._session.get(url, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def get_equity_balance(self, account_no: str) -> Dict[str, Any]:
        if self._simulation:
            return {
                "total": {
                    "총매입금액": "0", "총평가금액": "0", "총평가손익금액": "0",
                    "총수익률(%)": "0", "추정예탁자산": "0",
                },
                "stocks": [],
            }
        self.ensure_connected()
        url = self._base_url + EQUITY_BALANCE.format(account_no=account_no)
        r = self._session.get(url, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        # Normalize shape to { total: dict, stocks: list }
        if isinstance(data, dict) and ("total" in data or "stocks" in data):
            return data
        return {"total": data.get("summary", {}), "stocks": data.get("items", [])}

    def get_equity_balance2(self, account_no: str) -> Dict[str, Any]:
        if self._simulation:
            return {}
        self.ensure_connected()
        url = self._base_url + EQUITY_BALANCE2.format(account_no=account_no)
        r = self._session.get(url, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    # --- Market data ---
    def get_stock_basic_info(self, code: str, listener: Optional[Callable[[Dict[str, Any]], None]] = None):
        if self._simulation:
            data = {
                "종목명": "시뮬레이션",
                "기준가": "1000",
                "현재가": "1020",
                "시가": "1000",
                "고가": "1100",
                "저가": "990",
                "상한가": "1300",
                "하한가": "700",
            }
            if listener:
                threading.Thread(target=listener, args=(data,), daemon=True).start()
                return None
            return data
        self.ensure_connected()
        url = self._base_url + STOCK_BASIC_INFO.format(code=code)
        r = self._session.get(url, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        if listener:
            threading.Thread(target=listener, args=(data,), daemon=True).start()
            return None
        return data

    def get_upper_limit_price(self, code: str, listener: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
        data = self.get_stock_basic_info(code, listener=None)
        if not data:
            return -1
        price = data.get("상한가") or data.get("upperLimit") or data.get("upper_limit")
        try:
            return int(str(price))
        except Exception:
            return -1

    def get_current_price(self, code: str, listener: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
        if self._simulation:
            return 1020
        self.ensure_connected()
        # Some APIs include current price in basic info; try dedicated endpoint, fallback
        try:
            url = self._base_url + CURRENT_PRICE.format(code=code)
            r = self._session.get(url, timeout=self._timeout)
            r.raise_for_status()
            data = r.json()
            price = data.get("현재가") or data.get("price") or data.get("current_price")
            return int(str(price))
        except Exception:
            data = self.get_stock_basic_info(code)
            price = data.get("현재가") or data.get("price") or data.get("current_price")
            return int(str(price)) if price is not None else -1

    def get_stocks_of_upper_limit_reached(self, listener: Optional[Callable[[Dict[str, Any]], None]] = None):
        if self._simulation:
            data = []
            if listener:
                threading.Thread(target=listener, args=(data,), daemon=True).start()
                return None
            return data
        self.ensure_connected()
        url = self._base_url + UPPER_LIMITS
        r = self._session.get(url, timeout=self._timeout, params={"prevDay": True})
        r.raise_for_status()
        data = r.json()
        if listener:
            threading.Thread(target=listener, args=(data,), daemon=True).start()
            return None
        return data

    # --- Orders ---
    def send_order(
        self,
        rqname: str,
        account_no: str,
        order_type: OrderType,
        code: str,
        quantity: int,
        price: int,
        hoga: OrderBookType,
        origin_order_no: str = "",
    ) -> Dict[str, Any]:
        if self._simulation:
            return {"status": "접수", "rqname": rqname, "order_no": "SIM-ORDER-0001", "code": code}
        self.ensure_connected()
        payload = {
            "rqname": rqname,
            "account_no": account_no,
            "order_type": order_type.value,
            "code": code,
            "quantity": quantity,
            "price": price,
            "hoga": hoga.value,
        }
        if origin_order_no:
            payload["origin_order_no"] = origin_order_no
        r = self._session.post(self._base_url + SEND_ORDER, json=payload, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def send_order_modify(self, order_no: str, price: Optional[int] = None, quantity: Optional[int] = None) -> Dict[str, Any]:
        if self._simulation:
            return {"status": "확인", "order_no": order_no}
        self.ensure_connected()
        payload: Dict[str, Any] = {}
        if price is not None:
            payload["price"] = price
        if quantity is not None:
            payload["quantity"] = quantity
        r = self._session.patch(self._base_url + SEND_ORDER_MODIFY.format(order_no=order_no), json=payload, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def send_order_cancel(self, order_no: str) -> Dict[str, Any]:
        if self._simulation:
            return {"status": "취소", "order_no": order_no}
        self.ensure_connected()
        r = self._session.delete(self._base_url + SEND_ORDER_CANCEL.format(order_no=order_no), timeout=self._timeout)
        r.raise_for_status()
        return r.json() if r.content else {"status": "취소"}
