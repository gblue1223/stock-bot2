import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add kiwoom_rest_api to path
_kiwoom_rest_api_path = Path(__file__).parent / "kiwoom_rest_api" / "src"
if str(_kiwoom_rest_api_path) not in sys.path:
    sys.path.insert(0, str(_kiwoom_rest_api_path))

from kiwoom_rest_api.auth.token import TokenManager
from kiwoom_rest_api.koreanstock.stockinfo import StockInfo
from kiwoom_rest_api.koreanstock.account import Account
from kiwoom_rest_api.koreanstock.order import Order
from kiwoom_rest_api.config import get_base_url

from .types import OrderType, OrderBookType

OrderingListener = Callable[[str, Any, Any], None]
DataListener = Callable[[Dict[str, Any]], None]
StringListener = Callable[[str], None]

class KoapyRestSimple:
    def __init__(
        self,
        timeout: int = 10,
        simulation: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self._base_url = get_base_url()
        self._timeout = timeout
        self._simulation = simulation
        self._logger = logger or logging.getLogger("koapyRest")
        self._is_connected = False
        
        # Initialize kiwoom_rest_api components
        if not simulation:
            self._token_manager = TokenManager()
            self._stock_info = StockInfo(base_url=self._base_url, token_manager=self._token_manager)
            self._account = Account(base_url=self._base_url, token_manager=self._token_manager)
            self._order = Order(base_url=self._base_url, token_manager=self._token_manager)
        else:
            self._token_manager = None
            self._stock_info = None
            self._account = None
            self._order = None

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
        # Check connection by getting token
        try:
            if self._token_manager:
                token = self._token_manager.get_token()
                self._is_connected = bool(token)
                if not self._is_connected:
                    raise ConnectionError("Unable to get access token")
            else:
                raise ConnectionError("Token manager not initialized")
        except Exception as e:
            self._logger.debug(f"Connection failed: {e}")
            self._is_connected = False
            raise ConnectionError(f"Unable to connect to Kiwoom REST API: {e}")

    def close(self):
        self._is_connected = False

    # --- Accounts ---
    def get_account_list(self):
        if self._simulation:
            return ["0000000000"]
        self.ensure_connected()
        # Note: Account list API may need to be implemented in kiwoom_rest_api
        # For now, return a placeholder or raise NotImplementedError
        self._logger.warning("get_account_list not yet implemented with kiwoom_rest_api")
        return []

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
        try:
            # Use kt00001 - deposit_detail_status_request
            result = self._account.deposit_detail_status_request_kt00001(qry_tp="0")
            return result
        except Exception as e:
            self._logger.error(f"Failed to get deposit: {e}")
            raise

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
        try:
            # Use kt00004 - account_evaluation_status_request
            result = self._account.account_evaluation_status_request_kt00004(
                qry_tp="0",  # Query type
                dmst_stex_tp="KRX"  # Domestic stock exchange type
            )
            # Normalize shape to { total: dict, stocks: list }
            if isinstance(result, dict):
                return {
                    "total": result.get("summary", result.get("total", {})),
                    "stocks": result.get("items", result.get("stocks", []))
                }
            return {"total": {}, "stocks": []}
        except Exception as e:
            self._logger.error(f"Failed to get equity balance: {e}")
            raise

    def get_equity_balance2(self, account_no: str) -> Dict[str, Any]:
        if self._simulation:
            return {}
        self.ensure_connected()
        try:
            # Use kt00005 - filled_position_request for detailed balance
            result = self._account.filled_position_request_kt00005(dmst_stex_tp="KRX")
            return result
        except Exception as e:
            self._logger.error(f"Failed to get equity balance2: {e}")
            raise

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
        try:
            # Use ka10001 - basic_stock_information_request
            result = self._stock_info.basic_stock_information_request_ka10001(stock_code=code)
            data = result if isinstance(result, dict) else {}
            if listener:
                threading.Thread(target=listener, args=(data,), daemon=True).start()
                return None
            return data
        except Exception as e:
            self._logger.error(f"Failed to get stock basic info: {e}")
            if listener:
                threading.Thread(target=listener, args=({},), daemon=True).start()
                return None
            return {}

    def get_upper_limit_price(self, code: str, listener: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
        data = self.get_stock_basic_info(code, listener=None)
        if not data:
            return -1
        price = data.get("상한가") or data.get("upperLimit") or data.get("upper_limit") or data.get("uplmt_prc")
        try:
            return int(str(price).replace(",", ""))
        except Exception:
            return -1

    def get_current_price(self, code: str, listener: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
        if self._simulation:
            return 1020
        self.ensure_connected()
        try:
            data = self.get_stock_basic_info(code)
            price = data.get("현재가") or data.get("price") or data.get("current_price") or data.get("cur_prc")
            return int(str(price).replace(",", "")) if price is not None else -1
        except Exception as e:
            self._logger.error(f"Failed to get current price: {e}")
            return -1

    def get_stocks_of_upper_limit_reached(self, listener: Optional[Callable[[Dict[str, Any]], None]] = None):
        if self._simulation:
            data = []
            if listener:
                threading.Thread(target=listener, args=(data,), daemon=True).start()
                return None
            return data
        self.ensure_connected()
        # Note: This API might not be directly available in kiwoom_rest_api
        # You may need to implement using rank_info or other appropriate APIs
        self._logger.warning("get_stocks_of_upper_limit_reached not yet implemented with kiwoom_rest_api")
        data = []
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
        
        try:
            # Map OrderBookType to trde_tp (trading type)
            trde_tp_map = {
                OrderBookType.LIMIT: "0",
                OrderBookType.MARKET: "3",
                OrderBookType.CONDITIONAL_LIMIT: "5",
                OrderBookType.BEST_LIMIT: "6",
                OrderBookType.PRIORITY_LIMIT: "7",
                OrderBookType.LIMIT_IOC: "10",
                OrderBookType.MARKET_IOC: "13",
                OrderBookType.BEST_IOC: "16",
                OrderBookType.LIMIT_FOK: "20",
                OrderBookType.MARKET_FOK: "23",
                OrderBookType.BEST_FOK: "26",
                OrderBookType.PRE_MARKET_CLOSE: "61",
                OrderBookType.AFTER_HOURS_SINGLE: "62",
                OrderBookType.POST_MARKET_CLOSE: "81",
            }
            trde_tp = trde_tp_map.get(hoga, "0")
            
            # Determine if buy or sell
            if order_type == OrderType.BUY:
                result = self._order.stock_buy_order_request_kt10000(
                    dmst_stex_tp="KRX",
                    stk_cd=code,
                    ord_qty=str(quantity),
                    trde_tp=trde_tp,
                    ord_uv=str(price) if price > 0 else ""
                )
            elif order_type == OrderType.SELL:
                result = self._order.stock_sell_order_request_kt10001(
                    dmst_stex_tp="KRX",
                    stk_cd=code,
                    ord_qty=str(quantity),
                    trde_tp=trde_tp,
                    ord_uv=str(price) if price > 0 else ""
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return {
                "status": "접수",
                "rqname": rqname,
                "order_no": result.get("ord_no", ""),
                "code": code,
                **result
            }
        except Exception as e:
            self._logger.error(f"Failed to send order: {e}")
            raise

    def send_order_modify(self, order_no: str, price: Optional[int] = None, quantity: Optional[int] = None) -> Dict[str, Any]:
        if self._simulation:
            return {"status": "확인", "order_no": order_no}
        self.ensure_connected()
        
        try:
            # Use kt10002 - stock_modify_order_request
            result = self._order.stock_modify_order_request_kt10002(
                dmst_stex_tp="KRX",
                orig_ord_no=order_no,
                stk_cd="",  # Stock code may need to be retrieved or stored
                mdfy_qty=str(quantity) if quantity is not None else "0",
                mdfy_uv=str(price) if price is not None else "0"
            )
            return {"status": "확인", "order_no": order_no, **result}
        except Exception as e:
            self._logger.error(f"Failed to modify order: {e}")
            raise

    def send_order_cancel(self, order_no: str) -> Dict[str, Any]:
        if self._simulation:
            return {"status": "취소", "order_no": order_no}
        self.ensure_connected()
        
        try:
            # Use kt10003 - stock_cancel_order_request
            result = self._order.stock_cancel_order_request_kt10003(
                dmst_stex_tp="KRX",
                orig_ord_no=order_no,
                stk_cd="",  # Stock code may need to be retrieved or stored
                cncl_qty="0"  # 0 means cancel all remaining quantity
            )
            return {"status": "취소", "order_no": order_no, **result}
        except Exception as e:
            self._logger.error(f"Failed to cancel order: {e}")
            raise
