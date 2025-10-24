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
from kiwoom_rest_api.websocket import WebSocketClient

from .types import OrderType, OrderBookType

OrderingListener = Callable[[str, Any, Any], None]
DataListener = Callable[[Dict[str, Any]], None]
StringListener = Callable[[str], None]

class KoapyRestSimple:
    def __init__(
        self,
        timeout: int = 10,
        simulation: bool = False, # 미완성
        logger: Optional[logging.Logger] = None,
    ):
        self._base_url = get_base_url()
        self._timeout = timeout
        self._simulation = simulation
        self._logger = logger or logging.getLogger("koapyRest")
        self._is_connected = False
        self._token_manager = TokenManager()
        self._websocket_client: Optional[WebSocketClient] = None
        
        # Initialize kiwoom_rest_api components
        if not simulation:
            self._stock_info = StockInfo(base_url=self._base_url, token_manager=self._token_manager)
            self._account = Account(base_url=self._base_url, token_manager=self._token_manager)
            self._order = Order(base_url=self._base_url, token_manager=self._token_manager)
        else:
            self._stock_info = None
            self._account = None
            self._order = None

    @property
    def access_token(self) -> str:
        return self._token_manager.get_token()
    
    @property
    def websocket(self) -> Optional[WebSocketClient]:
        """WebSocketClient 인스턴스
        
        Returns:
            WebSocketClient 또는 None (start_websocket() 호출 전)
        """
        return self._websocket_client

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

    async def start_websocket(
        self,
        ws_url: Optional[str] = None,
        auto_reconnect: bool = True,
        reconnect_interval: int = 5,
        ping_interval: int = 30
    ) -> WebSocketClient:
        """WebSocket 클라이언트를 생성하고 시작합니다.
        
        Args:
            ws_url: 웹소켓 URL (None이면 설정에서 자동 선택)
            auto_reconnect: 자동 재연결 여부
            reconnect_interval: 재연결 간격 (초)
            ping_interval: PING 간격 (초)
            
        Returns:
            WebSocketClient 인스턴스
            
        Example:
            >>> client = KoapyRestSimple()
            >>> client.ensure_connected()
            >>> ws_client = await client.start_websocket()
            >>> # 웹소켓 사용
            >>> await client.stop_websocket()
        """
        if self._simulation:
            self._logger.warning("시뮬레이션 모드에서는 WebSocket을 사용할 수 없습니다")
            return None
        
        self.ensure_connected()
        
        if self._websocket_client is not None:
            self._logger.warning("WebSocket 클라이언트가 이미 생성되어 있습니다. 기존 클라이언트를 반환합니다.")
            return self._websocket_client
        
        # WebSocketClient 생성
        self._websocket_client = WebSocketClient(
            access_token=self.access_token,
            ws_url=ws_url,
            auto_reconnect=auto_reconnect,
            reconnect_interval=reconnect_interval,
            ping_interval=ping_interval
        )
        
        # WebSocket 연결 및 로그인
        await self._websocket_client.start()
        
        self._logger.info("WebSocket 클라이언트 시작 완료")
        return self._websocket_client
    
    async def stop_websocket(self):
        """WebSocket 클라이언트를 중지합니다.
        
        Example:
            >>> await client.stop_websocket()
        """
        if self._websocket_client is None:
            self._logger.warning("WebSocket 클라이언트가 시작되지 않았습니다")
            return
        
        await self._websocket_client.stop()
        self._websocket_client = None
        self._logger.info("WebSocket 클라이언트 중지 완료")
    
    def close(self):
        """REST API 클라이언트를 닫습니다.
        
        Note:
            WebSocket은 별도로 stop_websocket()을 호출해야 합니다.
        """
        self._is_connected = False

    # --- Accounts ---
    def get_account_list(self):
        if self._simulation:
            return ["0000000000"]
        self.ensure_connected()
        # Note: Account list API may need to be implemented in kiwoom_rest_api
        # For now, return a placeholder or raise NotImplementedError
        self._logger.warning("kiwoom_rest_api not support")
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
            
            # Element 이름을 한글명으로 변환
            element_to_korean = {
                "entr": "예수금",
                "profa_ch": "주식증거금현금",
                "bncr_profa_ch": "수익증권증거금현금",
                "nxdy_bncr_sell_exct": "익일수익증권매도정산대금",
                "fc_stk_krw_repl_set_amt": "해외주식원화대용설정금",
                "crd_grnta_ch": "신용보증금현금",
                "crd_grnt_ch": "신용담보금현금",
                "add_grnt_ch": "추가담보금현금",
                "etc_profa": "기타증거금",
                "uncl_stk_amt": "미수확보금",
                "shrts_prica": "공매도대금",
                "crd_set_grnta": "신용설정평가금",
                "chck_ina_amt": "수표입금액",
                "etc_chck_ina_amt": "기타수표입금액",
                "crd_grnt_ruse": "신용담보재사용",
                "knx_asset_evltv": "코넥스기본예탁금",
                "elwdpst_evlta": "ELW예탁평가금",
                "crd_ls_rght_frcs_amt": "신용대주권리예정금액",
                "lvlh_join_amt": "생계형가입금액",
                "lvlh_trns_alowa": "생계형입금가능금액",
                "repl_amt": "대용금평가금액(합계)",
                "remn_repl_evlta": "잔고대용평가금액",
                "trst_remn_repl_evlta": "위탁대용잔고평가금액",
                "bncr_remn_repl_evlta": "수익증권대용평가금액",
                "profa_repl": "위탁증거금대용",
                "crd_grnta_repl": "신용보증금대용",
                "crd_grnt_repl": "신용담보금대용",
                "add_grnt_repl": "추가담보금대용",
                "rght_repl_amt": "권리대용금",
                "pymn_alow_amt": "출금가능금액",
                "wrap_pymn_alow_amt": "랩출금가능금액",
                "ord_alow_amt": "주문가능금액",
                "bncr_buy_alowa": "수익증권매수가능금액",
                "20stk_ord_alow_amt": "20%종목주문가능금액",
                "30stk_ord_alow_amt": "30%종목주문가능금액",
                "40stk_ord_alow_amt": "40%종목주문가능금액",
                "100stk_ord_alow_amt": "100%종목주문가능금액",
                "ch_uncla": "현금미수금",
                "ch_uncla_dlfe": "현금미수연체료",
                "ch_uncla_tot": "현금미수금합계",
                "crd_int_npay": "신용이자미납",
                "int_npay_amt_dlfe": "신용이자미납연체료",
                "int_npay_amt_tot": "신용이자미납합계",
                "etc_loana": "기타대여금",
                "etc_loana_dlfe": "기타대여금연체료",
                "etc_loan_tot": "기타대여금합계",
                "nrpy_loan": "미상환융자금",
                "loan_sum": "융자금합계",
                "ls_sum": "대주금합계",
                "crd_grnt_rt": "신용담보비율",
                "mdstrm_usfe": "중도이용료",
                "min_ord_alow_yn": "최소주문가능금액",
                "loan_remn_evlt_amt": "대출총평가금액",
                "dpst_grntl_remn": "예탁담보대출잔고",
                "sell_grntl_remn": "매도담보대출잔고",
                "d1_entra": "d+1추정예수금",
                "d1_slby_exct_amt": "d+1매도매수정산금",
                "d1_buy_exct_amt": "d+1매수정산금",
                "d1_out_rep_mor": "d+1미수변제소요금",
                "d1_sel_exct_amt": "d+1매도정산금",
                "d1_pymn_alow_amt": "d+1출금가능금액",
                "d2_entra": "d+2추정예수금",
                "d2_slby_exct_amt": "d+2매도매수정산금",
                "d2_buy_exct_amt": "d+2매수정산금",
                "d2_out_rep_mor": "d+2미수변제소요금",
                "d2_sel_exct_amt": "d+2매도정산금",
                "d2_pymn_alow_amt": "d+2출금가능금액",
                "50stk_ord_alow_amt": "50%종목주문가능금액",
                "60stk_ord_alow_amt": "60%종목주문가능금액",
            }
            
            # 변환된 결과 생성
            converted_result = {}
            for key, value in result.items():
                korean_key = element_to_korean.get(key, key)
                converted_result[korean_key] = value
            
            return converted_result
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
            
            # Element 이름을 한글명으로 변환
            total_element_to_korean = {
                "acnt_nm": "계좌명",
                "brch_nm": "지점명",
                "entr": "예수금",
                "d2_entra": "D+2추정예수금",
                "tot_est_amt": "유가잔고평가액",
                "aset_evlt_amt": "예탁자산평가액",
                "tot_pur_amt": "총매입금액",
                "prsm_dpst_aset_amt": "추정예탁자산",
                "tot_grnt_sella": "매도담보대출금",
                "tdy_lspft_amt": "당일투자원금",
                "invt_bsamt": "당월투자원금",
                "lspft_amt": "누적투자원금",
                "tdy_lspft": "당일투자손익",
                "lspft2": "당월투자손익",
                "lspft": "누적투자손익",
                "tdy_lspft_rt": "당일손익율",
                "lspft_ratio": "당월손익율",
                "lspft_rt": "누적손익율",
            }
            
            stock_element_to_korean = {
                "stk_cd": "종목코드",
                "stk_nm": "종목명",
                "rmnd_qty": "보유수량",
                "avg_prc": "평균단가",
                "cur_prc": "현재가",
                "evlt_amt": "평가금액",
                "pl_amt": "손익금액",
                "pl_rt": "손익율",
                "loan_dt": "대출일",
                "pur_amt": "매입금액",
                "setl_remn": "결제잔고",
                "pred_buyq": "전일매수수량",
                "pred_sellq": "전일매도수량",
                "tdy_buyq": "금일매수수량",
                "tdy_sellq": "금일매도수량",
            }
            
            # 변환된 결과 생성
            if isinstance(result, dict):
                # total 정보 변환
                total = {}
                for key, value in result.items():
                    if key not in ["stk_acnt_evlt_prst", "return_code", "return_msg"]:
                        korean_key = total_element_to_korean.get(key, key)
                        total[korean_key] = value
                
                # stocks 정보 변환
                stocks = []
                stock_list = result.get("stk_acnt_evlt_prst", [])
                for stock in stock_list:
                    converted_stock = {}
                    for key, value in stock.items():
                        korean_key = stock_element_to_korean.get(key, key)
                        converted_stock[korean_key] = value
                    stocks.append(converted_stock)
                
                return {
                    "total": total,
                    "stocks": stocks
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
            
            # Element 이름을 한글명으로 변환
            total_element_to_korean = {
                "entr": "예수금",
                "entr_d1": "예수금D+1",
                "entr_d2": "예수금D+2",
                "pymn_alow_amt": "출금가능금액",
                "uncl_stk_amt": "미수확보금",
                "repl_amt": "대용금",
                "rght_repl_amt": "권리대용금",
                "ord_alowa": "주문가능현금",
                "ch_uncla": "현금미수금",
                "crd_int_npay_gold": "신용이자미납금",
                "etc_loana": "기타대여금",
                "nrpy_loan": "미상환융자금",
                "profa_ch": "증거금현금",
                "repl_profa": "증거금대용",
                "stk_buy_tot_amt": "주식매수총액",
                "evlt_amt_tot": "평가금액합계",
                "tot_pl_tot": "총손익합계",
                "tot_pl_rt": "총손익률",
                "tot_re_buy_alowa": "총재매수가능금액",
                "20ord_alow_amt": "20%주문가능금액",
                "30ord_alow_amt": "30%주문가능금액",
                "40ord_alow_amt": "40%주문가능금액",
                "50ord_alow_amt": "50%주문가능금액",
                "60ord_alow_amt": "60%주문가능금액",
                "100ord_alow_amt": "100%주문가능금액",
                "crd_loan_tot": "신용융자합계",
                "crd_loan_ls_tot": "신용융자대주합계",
                "crd_grnt_rt": "신용담보비율",
                "dpst_grnt_use_amt_amt": "예탁담보대출금액",
                "grnt_loan_amt": "매도담보대출금액",
            }
            
            stock_element_to_korean = {
                "crd_tp": "신용구분",
                "loan_dt": "대출일",
                "expr_dt": "만기일",
                "stk_cd": "종목번호",
                "stk_nm": "종목명",
                "setl_remn": "결제잔고",
                "cur_qty": "현재잔고",
                "cur_prc": "현재가",
                "buy_uv": "매입단가",
                "pur_amt": "매입금액",
                "evlt_amt": "평가금액",
                "evltv_prft": "평가손익",
                "pl_rt": "손익률",
            }
            
            # 변환된 결과 생성
            converted_result = {}
            for key, value in result.items():
                if key == "stk_cntr_remn":
                    # 종목별 체결잔고 배열 변환
                    converted_stocks = []
                    for stock in value:
                        converted_stock = {}
                        for stock_key, stock_value in stock.items():
                            korean_key = stock_element_to_korean.get(stock_key, stock_key)
                            converted_stock[korean_key] = stock_value
                        converted_stocks.append(converted_stock)
                    converted_result["종목별체결잔고"] = converted_stocks
                else:
                    korean_key = total_element_to_korean.get(key, key)
                    converted_result[korean_key] = value
            
            return converted_result
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
            
            # Element 이름을 한글명으로 변환
            # OpenAPI의 "주식기본정보요청" (opt10001)과 동일한 필드명 사용
            element_to_korean = {
                # 기본 정보
                "stk_cd": "종목코드",
                "stk_nm": "종목명",
                "base_pric": "기준가",
                "cur_prc": "현재가",
                "open_pric": "시가",
                "high_pric": "고가",
                "low_pric": "저가",
                "upl_pric": "상한가",
                "lst_pric": "하한가",
                "trde_qty": "거래량",
                "trde_pre": "거래대금",
                
                # 등락 정보
                "pre_sig": "전일대비기호",
                "pred_pre": "전일대비",
                "flu_rt": "등락율",
                
                # 52주/250일 고저가
                "oyr_hgst": "52주최고가",
                "oyr_lwst": "52주최저가",
                "250hgst": "250일최고가",
                "250lwst": "250일최저가",
                "250hgst_pric_dt": "250일최고가일자",
                "250hgst_pric_pre_rt": "250일최고가대비율",
                "250lwst_pric_dt": "250일최저가일자",
                "250lwst_pric_pre_rt": "250일최저가대비율",
                
                # 시장 정보
                "mac": "시가총액",
                "mac_wght": "시가총액비중",
                "for_exh_rt": "외국인소진율",
                "flo_stk": "유통주식",
                "dstr_stk": "배당주식수",
                "dstr_rt": "배당율",
                
                # 재무 정보
                "fav": "액면가",
                "fav_unit": "액면가단위",
                "cap": "자본금",
                "sale_amt": "매출액",
                "bus_pro": "영업이익",
                "cup_nga": "당기순이익",
                "crd_rt": "신용비율",
                
                # 투자지표
                "per": "PER",
                "pbr": "PBR",
                "eps": "EPS",
                "bps": "BPS",
                "roe": "ROE",
                "ev": "EV",
                
                # 예상 체결
                "exp_cntr_pric": "예상체결가",
                "exp_cntr_qty": "예상체결량",
                
                # 기타
                "setl_mm": "결제월",
                "repl_pric": "대용가",
            }
            
            # 변환된 결과 생성
            converted_data = {}
            if isinstance(result, dict):
                for key, value in result.items():
                    korean_key = element_to_korean.get(key, key)
                    converted_data[korean_key] = value
            
            data = converted_data if converted_data else {}
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
        # get_stock_basic_info가 이미 한글명으로 변환하므로 한글 키로 조회
        price = data.get("상한가")
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
            # get_stock_basic_info가 이미 한글명으로 변환하므로 한글 키로 조회
            price = data.get("현재가")
            return int(str(price).replace(",", "")) if price is not None else -1
        except Exception as e:
            self._logger.error(f"Failed to get current price: {e}")
            return -1

    def get_stocks_of_upper_limit_reached(self, listener: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        전일 상한가 종목들
        :param listener: 데이터를 처리할 리스너 None 이라면 동기식(블러킹)으로 처리
        """
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

    # --- Condition Search ---
    def get_condition_load(self) -> bool:
        """
        조건검색 목록을 로드합니다.
        
        Note:
            조건식 목록 조회는 WebSocket API (ka10171)를 사용해야 합니다.
            WebSocketClient.condition_list_request_ka10171()를 사용하세요.
            
        Returns:
            bool: 성공 시 True (항상 True 반환, 실제 조회는 WebSocket 사용)
        """
        if self._simulation:
            return True
        
        self.ensure_connected()
        
        # 조건식 목록 조회는 WebSocket을 통해서만 가능
        self._logger.warning(
            "조건식 목록 조회는 WebSocket API를 사용하세요. "
            "예제: kiwoom_rest_api/examples/condition_search_websocket_example.py"
        )
        return True

    def get_condition_name_list(self):
        """
        조건검색 목록을 조회합니다.
        
        Note:
            조건식 목록 조회는 WebSocket API (ka10171)를 사용해야 합니다.
            WebSocketClient.condition_list_request_ka10171()를 사용하세요.
        
        Returns:
            list: 빈 목록 (실제 조회는 WebSocket 사용)
        """
        if self._simulation:
            return []
        
        self.ensure_connected()
        
        # 조건식 목록 조회는 WebSocket을 통해서만 가능
        self._logger.warning(
            "조건식 목록 조회는 WebSocket API를 사용하세요. "
            "예제: kiwoom_rest_api/examples/condition_search_websocket_example.py"
        )
        return []

    def get_stocks_by_condition(self, condition_name: str, listener: Optional[Callable[[Any], None]] = None):
        """
        조건검색으로 종목을 조회합니다.
        
        Args:
            condition_name (str): 조건식 이름
            listener (Optional[Callable]): 비동기 처리를 위한 리스너
            
        Note:
            REST API ka10173을 사용하여 조건검색을 수행합니다.
            조건식 목록은 get_condition_name_list()로 먼저 조회해야 합니다.
            
        Returns:
            list or None: listener가 None이면 종목코드 리스트 반환,
                         listener가 있으면 None 반환 (비동기)
        """
        if self._simulation:
            data = []
            if listener:
                threading.Thread(target=listener, args=(data,), daemon=True).start()
                return None
            return data
        
        self.ensure_connected()
        
        try:
            # 조건식 목록에서 해당 조건식의 인덱스를 찾습니다
            condition_list = self.get_condition_name_list()
            condition_index = None
            
            for index, name in condition_list:
                if name == condition_name:
                    condition_index = index
                    break
            
            if condition_index is None:
                self._logger.warning(f"조건식 '{condition_name}'을 찾을 수 없습니다.")
                data = []
                if listener:
                    threading.Thread(target=listener, args=(data,), daemon=True).start()
                    return None
                return data
            
            # ka10173 API 호출
            result = self._stock_info.condition_search_realtime_request_ka10173(
                condition_index=condition_index,
                condition_name=condition_name,
                realtime_flag="1"  # 실시간 조회
            )
            
            # 결과에서 종목코드 리스트 추출
            # 실제 응답 구조에 맞게 수정 필요
            if isinstance(result, dict):
                # 예상되는 응답 형식에서 종목코드 추출
                stock_codes = result.get("stock_codes", [])
                # 또는 배열 형태로 온다면
                if not stock_codes and "stocks" in result:
                    stock_codes = [item.get("stk_cd", "") for item in result.get("stocks", [])]
                
                data = stock_codes
            else:
                data = []
            
            if listener:
                threading.Thread(target=listener, args=(data,), daemon=True).start()
                return None
            return data
            
        except Exception as e:
            self._logger.error(f"조건검색 중 오류 발생: {e}")
            data = []
            if listener:
                threading.Thread(target=listener, args=(data,), daemon=True).start()
                return None
            return data
