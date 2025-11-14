"""
주식 실시간 데이터 수신 모듈

키움증권의 주식체결(0B) 및 주식호가(0D) 실시간 WebSocket API를 위한 헬퍼 클래스
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from ..websocket import WebSocketClient, RealTimeData

logger = logging.getLogger(__name__)


# 최종 컬럼 정의 (normalize_datasets.py의 FINAL_COLUMNS 참고)
FINAL_COLUMNS: List[str] = [
    "종목코드", "종목명", "시간",
    # 파생 피처 (종목명과 시간에서 생성)
    "종목명_scalar", "시간_sin", "시간_cos", "시간_scalar",
    # 기본 지표
    "등락률", "누적거래대금", "거래회전율", "체결강도",
    # 매도/매수 호가 및 수량 1~10
    *[f"매도호가{i}" for i in range(1, 11)],
    *[f"매도호가수량{i}" for i in range(1, 11)],
    *[f"매수호가{i}" for i in range(1, 11)],
    *[f"매수호가수량{i}" for i in range(1, 11)],
    # 매도/매수 대기금액 1~10
    *[f"매도대기금액{i}" for i in range(1, 11)],
    *[f"매수대기금액{i}" for i in range(1, 11)],
]


@dataclass
class StockRealtimeData:
    """주식 실시간 데이터"""
    종목코드: str = ""
    종목명: str = ""
    시간: str = ""
    
    # 파생 피처 (종목명과 시간에서 계산)
    종목명_scalar: float = 0.0
    시간_sin: float = 0.0
    시간_cos: float = 0.0
    시간_scalar: float = 0.0
    
    # 체결 정보 (0B)
    현재가: float = 0.0
    전일대비: float = 0.0
    등락률: float = 0.0
    매도호가: float = 0.0
    매수호가: float = 0.0
    거래량: float = 0.0
    누적거래량: float = 0.0
    누적거래대금: float = 0.0
    시가: float = 0.0
    고가: float = 0.0
    저가: float = 0.0
    거래회전율: float = 0.0
    체결강도: float = 0.0
    시가총액: float = 0.0
    
    # 호가 정보 (0D) - 호가
    매도호가1: float = 0.0
    매도호가2: float = 0.0
    매도호가3: float = 0.0
    매도호가4: float = 0.0
    매도호가5: float = 0.0
    매도호가6: float = 0.0
    매도호가7: float = 0.0
    매도호가8: float = 0.0
    매도호가9: float = 0.0
    매도호가10: float = 0.0
    
    매수호가1: float = 0.0
    매수호가2: float = 0.0
    매수호가3: float = 0.0
    매수호가4: float = 0.0
    매수호가5: float = 0.0
    매수호가6: float = 0.0
    매수호가7: float = 0.0
    매수호가8: float = 0.0
    매수호가9: float = 0.0
    매수호가10: float = 0.0
    
    매도호가수량1: float = 0.0
    매도호가수량2: float = 0.0
    매도호가수량3: float = 0.0
    매도호가수량4: float = 0.0
    매도호가수량5: float = 0.0
    매도호가수량6: float = 0.0
    매도호가수량7: float = 0.0
    매도호가수량8: float = 0.0
    매도호가수량9: float = 0.0
    매도호가수량10: float = 0.0
    
    매수호가수량1: float = 0.0
    매수호가수량2: float = 0.0
    매수호가수량3: float = 0.0
    매수호가수량4: float = 0.0
    매수호가수량5: float = 0.0
    매수호가수량6: float = 0.0
    매수호가수량7: float = 0.0
    매수호가수량8: float = 0.0
    매수호가수량9: float = 0.0
    매수호가수량10: float = 0.0
    
    # 파생 필드 (대기금액)
    매도대기금액1: float = 0.0
    매도대기금액2: float = 0.0
    매도대기금액3: float = 0.0
    매도대기금액4: float = 0.0
    매도대기금액5: float = 0.0
    매도대기금액6: float = 0.0
    매도대기금액7: float = 0.0
    매도대기금액8: float = 0.0
    매도대기금액9: float = 0.0
    매도대기금액10: float = 0.0
    
    매수대기금액1: float = 0.0
    매수대기금액2: float = 0.0
    매수대기금액3: float = 0.0
    매수대기금액4: float = 0.0
    매수대기금액5: float = 0.0
    매수대기금액6: float = 0.0
    매수대기금액7: float = 0.0
    매수대기금액8: float = 0.0
    매수대기금액9: float = 0.0
    매수대기금액10: float = 0.0
    
    def compute_derived_features(self):
        """파생 피처 계산 (종목명_scalar, 시간_sin, 시간_cos, 시간_scalar)"""
        # lib.normalization의 중앙화된 함수 사용
        try:
            # 모든 파생 피처를 한 번에 계산
            from lib.normalization import compute_all_derived_features
            
            self.종목명_scalar, self.시간_sin, self.시간_cos, self.시간_scalar = \
                compute_all_derived_features(self.종목명, self.시간)
            
            # 로그 (주기적으로만 - 30초마다)
            if self.시간:
                try:
                    time_str = str(self.시간).zfill(9)
                    hh = int(time_str[0:2])
                    mm = int(time_str[2:4])
                    ss = int(time_str[4:6])
                    secs = hh * 3600 + mm * 60 + ss
                    
                    if secs % 30 < 1:  # 30초 근처에서만
                        logger.debug(
                            f"[DERIVED] {self.종목코드} 파생피처: 종목명_scalar={self.종목명_scalar:.4f}, "
                            f"시간_sin={self.시간_sin:.4f}, 시간_cos={self.시간_cos:.4f}, 시간_scalar={self.시간_scalar:.4f}"
                        )
                except Exception:
                    pass
        
        except Exception as e:
            logger.warning(f"[DERIVED] {self.종목코드} 파생피처 계산 실패: {e}")
            self.종목명_scalar = 0.0
            self.시간_sin = 0.0
            self.시간_cos = 0.0
            self.시간_scalar = 0.0
    
    def compute_waiting_amounts(self):
        """호가와 수량을 곱해 대기금액 계산 (백만원 단위)"""
        total_ask_amount = 0.0
        total_bid_amount = 0.0
        
        # 디버그: 처음 몇 번만 상세 로깅
        if not hasattr(self, '_waiting_debug_count'):
            self._waiting_debug_count = 0
        do_debug = self._waiting_debug_count < 3
        
        for i in range(1, 11):
            # 매도대기금액
            ask_price = getattr(self, f"매도호가{i}", 0.0)
            ask_qty = getattr(self, f"매도호가수량{i}", 0.0)
            ask_amount = (ask_price * ask_qty) / 1_000_000
            setattr(self, f"매도대기금액{i}", ask_amount)
            total_ask_amount += ask_amount
            
            # 매수대기금액
            bid_price = getattr(self, f"매수호가{i}", 0.0)
            bid_qty = getattr(self, f"매수호가수량{i}", 0.0)
            bid_amount = (bid_price * bid_qty) / 1_000_000
            setattr(self, f"매수대기금액{i}", bid_amount)
            total_bid_amount += bid_amount
            
            # 상세 디버그 로그
            if do_debug and i == 1:
                logger.debug(
                    f"[WAITING_CALC] {self.종목코드} 매도1: 호가={ask_price}, 수량={ask_qty}, 금액={ask_amount:.2f}백만 | "
                    f"매수1: 호가={bid_price}, 수량={bid_qty}, 금액={bid_amount:.2f}백만"
                )
        
        if do_debug:
            self._waiting_debug_count += 1
        
        # 로그 (1분마다 한 번만 - 시간이 정각일 때)
        if self.시간 and len(str(self.시간)) >= 4:
            time_str = str(self.시간).zfill(9)
            mm = int(time_str[2:4])
            ss = int(time_str[4:6])
            if ss < 1:  # 정각 근처
                logger.debug(
                    f"[WAITING] {self.종목코드} 대기금액: 총매도={total_ask_amount:.2f}백만, "
                    f"총매수={total_bid_amount:.2f}백만, 매도1={getattr(self, '매도대기금액1', 0):.2f}, "
                    f"매수1={getattr(self, '매수대기금액1', 0):.2f}"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (FINAL_COLUMNS 기준, 호가/수량 제외)"""
        result = {}
        for col in FINAL_COLUMNS:
            # 호가 및 호가수량 필드 제외 (매도호가1~10, 매도호가수량1~10, 매수호가1~10, 매수호가수량1~10)
            if col.startswith('매도호가') or col.startswith('매수호가'):
                continue
            
            default_val = "" if col in ["종목코드", "종목명", "시간"] else 0.0
            result[col] = getattr(self, col, default_val)
        return result
    
    def to_full_dict(self) -> Dict[str, Any]:
        """모든 필드를 포함한 딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class RealtimeStockClient:
    """주식 실시간 데이터 WebSocket 클라이언트
    
    주식체결(0B) 및 주식호가(0D) 실시간 데이터를 수신하는 헬퍼 클래스
    """
    
    def __init__(self, client: WebSocketClient, auto_translate: bool = True):
        """
        Args:
            client: WebSocketClient 인스턴스 (외부에서 생성 및 관리)
            auto_translate: 필드명 자동 변환 여부
        """
        self.client = client
        self.auto_translate = auto_translate
        self.stock_data: Dict[str, StockRealtimeData] = {}  # stock_code -> StockRealtimeData
        self._login_event = asyncio.Event()
        
        # 이미 로그인된 경우 이벤트 설정
        if self.client.is_logged_in:
            self._login_event.set()
        
        # 사용자 정의 콜백
        self.on_trade_data: Optional[Callable[[str, StockRealtimeData], None]] = None  # 체결 데이터
        self.on_quote_data: Optional[Callable[[str, StockRealtimeData], None]] = None  # 호가 데이터
        self.on_data_update: Optional[Callable[[str, StockRealtimeData], None]] = None  # 전체 데이터 업데이트
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # 이벤트 리스너 등록
        self.client.on('data', self._handle_data)
        self.client.on('login', self._on_login)
    
    def cleanup(self):
        """이벤트 리스너 제거"""
        self.client.remove_listener('data', self._handle_data)
        self.client.remove_listener('login', self._on_login)
    
    async def register_stocks(
        self,
        stock_codes: List[str],
        include_trade: bool = True,
        include_quote: bool = True,
        group_no: str = "1",
        refresh: str = "1"
    ):
        """주식 실시간 데이터 등록
        
        Args:
            stock_codes: 종목코드 리스트 (예: ['005930', '000660'])
            include_trade: 주식체결(0B) 포함 여부
            include_quote: 주식호가잔량(0D) 포함 여부 (호가+수량)
            group_no: 그룹 번호
            refresh: 기존등록유지여부 (0: 기존유지안함, 1: 기존유지)
        """
        # 로그인 완료 대기 (타임아웃 포함)
        try:
            await asyncio.wait_for(self._login_event.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            # 이미 로그인된 상태일 수 있으므로 확인
            if not self.client.is_logged_in:
                logger.error("로그인 대기 중 타임아웃 발생")
                raise
            else:
                logger.warning("로그인 이벤트 대기 타임아웃, 하지만 이미 로그인된 상태이므로 계속 진행")
                self._login_event.set()  # 이벤트 설정
        
        # 실시간 타입 리스트 생성
        type_list = []
        if include_trade:
            type_list.append('0B')
        if include_quote:
            type_list.append('0D')  # 주식호가잔량 (호가 가격 41-60 + 수량 61-80)
        
        if not type_list:
            logger.warning("등록할 실시간 타입이 없습니다")
            return
        
        # 종목별 데이터 초기화
        for code in stock_codes:
            if code not in self.stock_data:
                self.stock_data[code] = StockRealtimeData(종목코드=code)
        
        # 실시간 등록
        await self.client.register_realtime(
            group_no=group_no,
            type_list=type_list,
            item_list=stock_codes,
            refresh=refresh
        )
        
        logger.info(f"주식 실시간 데이터 등록: {len(stock_codes)}개 종목, 타입={type_list}")
    
    async def unregister_stocks(self, group_no: str = "1"):
        """주식 실시간 데이터 해지
        
        Args:
            group_no: 그룹 번호
        """
        await self.client.unregister_realtime(group_no=group_no)
        logger.info("주식 실시간 데이터 해지")
    
    def get_stock_data(self, stock_code: str) -> Optional[StockRealtimeData]:
        """특정 종목의 실시간 데이터 조회
        
        Args:
            stock_code: 종목코드
            
        Returns:
            StockRealtimeData 또는 None
        """
        return self.stock_data.get(stock_code)
    
    def get_all_stock_data(self) -> Dict[str, StockRealtimeData]:
        """모든 종목의 실시간 데이터 조회
        
        Returns:
            {stock_code: StockRealtimeData} 딕셔너리
        """
        return self.stock_data.copy()
    
    async def _on_login(self):
        """로그인 완료 시 호출"""
        logger.info("WebSocket 로그인 완료")
        self._login_event.set()
    
    async def _handle_data(self, realtime_data: RealTimeData):
        """실시간 데이터 처리"""
        try:
            # 디버깅: 수신된 메시지 타입 로그
            logger.debug(f"[RealtimeStockClient] trnm={realtime_data.trnm}, data_count={len(realtime_data.data) if realtime_data.data else 0}")
            
            if realtime_data.trnm != 'REAL':
                return
            
            for item_data in realtime_data.data:
                if not isinstance(item_data, dict):
                    continue
                
                type_code = item_data.get('type')
                if type_code not in ['0B', '0D']:
                    continue
                
                item_code = item_data.get('item')  # 종목코드
                if not item_code:
                    continue
                
                values = item_data.get('values', {})
                
                # 종목 데이터 가져오기 또는 생성
                if item_code not in self.stock_data:
                    self.stock_data[item_code] = StockRealtimeData(종목코드=item_code)
                
                stock = self.stock_data[item_code]
                
                # 타입별 데이터 처리
                if type_code == '0B':
                    # 주식체결 데이터
                    self._process_trade_data(stock, values)
                    if self.on_trade_data:
                        self.on_trade_data(item_code, stock)
                
                elif type_code == '0D':
                    # 주식호가 데이터 (0D: 호가잔량)
                    self._process_quote_data(stock, values)
                    if self.on_quote_data:
                        self.on_quote_data(item_code, stock)
                
                # 파생 피처 계산 (종목명_scalar, 시간_sin, 시간_cos, 시간_scalar)
                stock.compute_derived_features()
                
                # 대기금액 계산
                stock.compute_waiting_amounts()
                
                # 전체 데이터 업데이트 콜백
                if self.on_data_update:
                    self.on_data_update(item_code, stock)
        
        except Exception as e:
            logger.exception(f"데이터 처리 중 오류: {e}")
            if self.on_error:
                self.on_error(e)
    
    def _process_trade_data(self, stock: StockRealtimeData, values: Dict[str, Any]):
        """주식체결(0B) 데이터 처리"""
        # 디버그: 받은 필드 출력 (처음 몇 번만)
        if not hasattr(self, '_trade_debug_count'):
            self._trade_debug_count = 0
        if self._trade_debug_count < 5:
            logger.debug(f"[0B TRADE] {stock.종목코드} received fields: {list(values.keys())}")
            if '567' in values:
                logger.debug(f"[0B TRADE] {stock.종목코드} 체결강도 raw value: {values.get('228')}")
            self._trade_debug_count += 1
        
        # 필드 매핑 (websocket_constants.py 참고)
        stock.종목코드 = self._safe_str(values.get('9001', stock.종목코드))
        stock.종목명 = self._safe_str(values.get('900', stock.종목명))
        stock.현재가 = self._safe_float(values.get('10', stock.현재가))
        stock.전일대비 = self._safe_float(values.get('11', stock.전일대비))
        stock.등락률 = self._safe_float(values.get('12', stock.등락률))
        stock.매도호가 = self._safe_float(values.get('27', stock.매도호가))
        stock.매수호가 = self._safe_float(values.get('28', stock.매수호가))
        stock.거래량 = self._safe_float(values.get('15', stock.거래량))
        stock.누적거래량 = self._safe_float(values.get('13', stock.누적거래량))
        stock.누적거래대금 = self._safe_float(values.get('14', stock.누적거래대금))
        stock.시가 = self._safe_float(values.get('16', stock.시가))
        stock.고가 = self._safe_float(values.get('17', stock.고가))
        stock.저가 = self._safe_float(values.get('18', stock.저가))
        stock.거래회전율 = self._safe_float(values.get('31', stock.거래회전율))
        stock.체결강도 = self._safe_float(values.get('228', stock.체결강도))
        stock.시가총액 = self._safe_float(values.get('311', stock.시가총액))
        
        # 시간 정보 (체결시간)
        if '20' in values:
            stock.시간 = self._safe_str(values.get('20', stock.시간))
    
    def _process_quote_data(self, stock: StockRealtimeData, values: Dict[str, Any]):
        """주식호가(0D) 데이터 처리"""
        # 디버그: 받은 필드 출력 (처음 몇 번만)
        if not hasattr(self, '_quote_debug_count'):
            self._quote_debug_count = 0
        if self._quote_debug_count < 5:
            logger.debug(f"[QUOTE] {stock.종목코드} 0D 타입 - 매도호가1={values.get('41')}, 매도수량1={values.get('61')}, 매수호가1={values.get('51')}, 매수수량1={values.get('71')}")
            self._quote_debug_count += 1
        
        # 종목 정보
        stock.종목코드 = self._safe_str(values.get('9001', stock.종목코드))
        stock.종목명 = self._safe_str(values.get('900', stock.종목명))
        
        # 매도호가 1~10 (필드: 41-50)
        for i in range(1, 11):
            field = str(40 + i)
            if field in values:
                setattr(stock, f"매도호가{i}", self._safe_float(values.get(field)))
        
        # 매수호가 1~10 (필드: 51-60)
        for i in range(1, 11):
            field = str(50 + i)
            if field in values:
                setattr(stock, f"매수호가{i}", self._safe_float(values.get(field)))
        
        # 매도호가수량 1~10 (필드: 61-70)
        for i in range(1, 11):
            field = str(60 + i)
            if field in values:
                setattr(stock, f"매도호가수량{i}", self._safe_float(values.get(field)))
        
        # 매수호가수량 1~10 (필드: 71-80)
        for i in range(1, 11):
            field = str(70 + i)
            if field in values:
                setattr(stock, f"매수호가수량{i}", self._safe_float(values.get(field)))
        
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """안전한 float 변환"""
        try:
            if value is None or value == '':
                return default
            # 부호 제거 (+ 또는 -)
            if isinstance(value, str):
                value = value.strip().lstrip('+').lstrip('-')
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_str(value: Any, default: str = "") -> str:
        """안전한 str 변환"""
        try:
            if value is None:
                return default
            return str(value).strip()
        except (ValueError, TypeError):
            return default


class RealtimeStockManager:
    """주식 실시간 데이터 관리자 (고급 기능)
    
    여러 종목을 관리하고 데이터 저장 등의 고급 기능 제공
    """
    
    def __init__(self, realtime_client: RealtimeStockClient):
        """
        Args:
            realtime_client: RealtimeStockClient 인스턴스 (외부에서 생성 및 관리)
        """
        self.client = realtime_client
        self.registered_stocks: List[str] = []
        self.data_history: Dict[str, List[Dict[str, Any]]] = {}  # stock_code -> [data_dict]
        self.max_history_size: int = 1000
        
        # 데이터 업데이트 콜백 등록 (기존 콜백을 래핑)
        self._original_on_data_update = self.client.on_data_update
        self.client.on_data_update = self._on_data_update
    
    def cleanup(self):
        """콜백 복원"""
        if hasattr(self, '_original_on_data_update'):
            self.client.on_data_update = self._original_on_data_update
    
    async def add_stocks(
        self,
        stock_codes: List[str],
        include_trade: bool = True,
        include_quote: bool = True
    ):
        """종목 추가 및 실시간 등록
        
        Args:
            stock_codes: 종목코드 리스트
            include_trade: 주식체결(0B) 포함 여부
            include_quote: 주식호가(0D) 포함 여부
        """
        await self.client.register_stocks(
            stock_codes=stock_codes,
            include_trade=include_trade,
            include_quote=include_quote
        )
        
        for code in stock_codes:
            if code not in self.registered_stocks:
                self.registered_stocks.append(code)
            if code not in self.data_history:
                self.data_history[code] = []
    
    async def remove_stocks(self):
        """모든 종목 실시간 해지"""
        await self.client.unregister_stocks()
        self.registered_stocks.clear()
    
    def get_latest_data(self, stock_code: str) -> Optional[StockRealtimeData]:
        """특정 종목의 최신 데이터 조회
        
        Args:
            stock_code: 종목코드
            
        Returns:
            StockRealtimeData 또는 None
        """
        return self.client.get_stock_data(stock_code)
    
    def get_history(self, stock_code: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """특정 종목의 히스토리 조회
        
        Args:
            stock_code: 종목코드
            limit: 최대 개수 (None이면 전체)
            
        Returns:
            데이터 딕셔너리 리스트
        """
        history = self.data_history.get(stock_code, [])
        if limit:
            return history[-limit:]
        return history.copy()
    
    def clear_history(self, stock_code: Optional[str] = None):
        """히스토리 초기화
        
        Args:
            stock_code: 종목코드 (None이면 전체)
        """
        if stock_code:
            self.data_history[stock_code] = []
        else:
            self.data_history.clear()
    
    async def _on_data_update(self, stock_code: str, data: StockRealtimeData):
        """데이터 업데이트 시 히스토리에 저장"""
        if stock_code not in self.data_history:
            self.data_history[stock_code] = []
        
        # FINAL_COLUMNS 기준으로 저장
        data_dict = data.to_dict()
        data_dict['timestamp'] = datetime.now().isoformat()
        
        self.data_history[stock_code].append(data_dict)
        
        # 히스토리 크기 제한
        if len(self.data_history[stock_code]) > self.max_history_size:
            self.data_history[stock_code] = self.data_history[stock_code][-self.max_history_size:]
