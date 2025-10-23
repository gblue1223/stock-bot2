"""
주식 실시간 데이터 수신 모듈

키움증권의 주식체결(0B) 및 주식호가(0C) 실시간 WebSocket API를 위한 헬퍼 클래스
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
    
    # 호가 정보 (0C) - 호가
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
    
    # 호가 정보 (0C) - 수량
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
        import numpy as np
        
        # 종목명_scalar: 문자 레벨 스칼라 인코딩
        if self.종목명:
            char_sum = sum(ord(c) for c in self.종목명)
            self.종목명_scalar = float(char_sum % 10000) / 10000.0
        else:
            self.종목명_scalar = 0.0
        
        # 시간 파생 피처
        if self.시간:
            try:
                # HHMMSSmmm -> seconds
                time_str = str(self.시간).zfill(9)
                hh = int(time_str[0:2])
                mm = int(time_str[2:4])
                ss = int(time_str[4:6])
                secs = hh * 3600 + mm * 60 + ss
                
                # sin/cos 주기 변환
                SECONDS_IN_DAY = 24 * 60 * 60
                self.시간_sin = float(np.sin(2 * np.pi * secs / SECONDS_IN_DAY))
                self.시간_cos = float(np.cos(2 * np.pi * secs / SECONDS_IN_DAY))
                
                # 장 시작 후 경과 시간 (표준화)
                MARKET_OPEN_SECONDS = 9 * 3600  # 09:00:00
                sec_from_open = max(0, secs - MARKET_OPEN_SECONDS)
                # 간단한 표준화: 0~6시간(21600초) 범위를 -1~1로 매핑
                self.시간_scalar = float((sec_from_open - 10800) / 10800.0)  # 3시간 중심, ±3시간
            except Exception:
                self.시간_sin = 0.0
                self.시간_cos = 0.0
                self.시간_scalar = 0.0
        else:
            self.시간_sin = 0.0
            self.시간_cos = 0.0
            self.시간_scalar = 0.0
    
    def compute_waiting_amounts(self):
        """호가와 수량을 곱해 대기금액 계산 (백만원 단위)"""
        for i in range(1, 11):
            # 매도대기금액
            ask_price = getattr(self, f"매도호가{i}", 0.0)
            ask_qty = getattr(self, f"매도호가수량{i}", 0.0)
            setattr(self, f"매도대기금액{i}", (ask_price * ask_qty) / 1_000_000)
            
            # 매수대기금액
            bid_price = getattr(self, f"매수호가{i}", 0.0)
            bid_qty = getattr(self, f"매수호가수량{i}", 0.0)
            setattr(self, f"매수대기금액{i}", (bid_price * bid_qty) / 1_000_000)
    
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
    
    주식체결(0B) 및 주식호가(0C) 실시간 데이터를 수신하는 헬퍼 클래스
    """
    
    def __init__(self, access_token: str, auto_translate: bool = True):
        """
        Args:
            access_token: 키움 API 액세스 토큰
            auto_translate: 필드명 자동 변환 여부
        """
        self.client = WebSocketClient(access_token=access_token)
        self.auto_translate = auto_translate
        self.stock_data: Dict[str, StockRealtimeData] = {}  # stock_code -> StockRealtimeData
        self._login_event = asyncio.Event()
        
        # 사용자 정의 콜백
        self.on_trade_data: Optional[Callable[[str, StockRealtimeData], None]] = None  # 체결 데이터
        self.on_quote_data: Optional[Callable[[str, StockRealtimeData], None]] = None  # 호가 데이터
        self.on_data_update: Optional[Callable[[str, StockRealtimeData], None]] = None  # 전체 데이터 업데이트
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # 내부 콜백 등록
        self.client.on_data = self._handle_data
        self.client.on_login = self._on_login
    
    async def start(self):
        """WebSocket 연결 시작"""
        await self.client.start()
    
    async def stop(self):
        """WebSocket 연결 종료"""
        await self.client.stop()
    
    async def register_stocks(
        self,
        stock_codes: List[str],
        include_trade: bool = True,
        include_quote: bool = True,
        group_no: str = "1"
    ):
        """주식 실시간 데이터 등록
        
        Args:
            stock_codes: 종목코드 리스트 (예: ['005930', '000660'])
            include_trade: 주식체결(0B) 포함 여부
            include_quote: 주식호가(0C) 포함 여부
            group_no: 그룹 번호
        """
        # 로그인 완료 대기
        await self._login_event.wait()
        
        # 실시간 타입 리스트 생성
        type_list = []
        if include_trade:
            type_list.append('0B')
        if include_quote:
            type_list.append('0C')
        
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
            refresh="1"
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
            if realtime_data.trnm != 'REAL':
                return
            
            for item_data in realtime_data.data:
                if not isinstance(item_data, dict):
                    continue
                
                type_code = item_data.get('type')
                item_code = item_data.get('item')  # 종목코드
                values = item_data.get('values', {})
                
                if not item_code:
                    continue
                
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
                
                elif type_code == '0C':
                    # 주식호가 데이터
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
        stock.체결강도 = self._safe_float(values.get('567', stock.체결강도))
        
        # 시간 정보 (체결시간)
        if '569' in values:
            stock.시간 = self._safe_str(values.get('569', stock.시간))
    
    def _process_quote_data(self, stock: StockRealtimeData, values: Dict[str, Any]):
        """주식호가(0C) 데이터 처리"""
        # 종목 정보
        stock.종목코드 = self._safe_str(values.get('9001', stock.종목코드))
        stock.종목명 = self._safe_str(values.get('900', stock.종목명))
        
        # 매도호가 1~10 (필드: 27, 29, 31, 33, 35, 37, 39, 41, 43, 45)
        ask_price_fields = ['27', '29', '31', '33', '35', '37', '39', '41', '43', '45']
        for i, field in enumerate(ask_price_fields, 1):
            if field in values:
                setattr(stock, f"매도호가{i}", self._safe_float(values.get(field)))
        
        # 매수호가 1~10 (필드: 28, 30, 32, 34, 36, 38, 40, 42, 44, 46)
        bid_price_fields = ['28', '30', '32', '34', '36', '38', '40', '42', '44', '46']
        for i, field in enumerate(bid_price_fields, 1):
            if field in values:
                setattr(stock, f"매수호가{i}", self._safe_float(values.get(field)))
        
        # 매도호가수량 1~10 (필드: 47, 49, 51, 53, 55, 57, 59, 61, 63, 65)
        ask_qty_fields = ['47', '49', '51', '53', '55', '57', '59', '61', '63', '65']
        for i, field in enumerate(ask_qty_fields, 1):
            if field in values:
                setattr(stock, f"매도호가수량{i}", self._safe_float(values.get(field)))
        
        # 매수호가수량 1~10 (필드: 48, 50, 52, 54, 56, 58, 60, 62, 64, 66)
        bid_qty_fields = ['48', '50', '52', '54', '56', '58', '60', '62', '64', '66']
        for i, field in enumerate(bid_qty_fields, 1):
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
    
    def __init__(self, access_token: str):
        self.client = RealtimeStockClient(access_token=access_token)
        self.registered_stocks: List[str] = []
        self.data_history: Dict[str, List[Dict[str, Any]]] = {}  # stock_code -> [data_dict]
        self.max_history_size: int = 1000
    
    async def start(self):
        """시작"""
        # 데이터 업데이트 콜백 등록
        self.client.on_data_update = self._on_data_update
        await self.client.start()
    
    async def stop(self):
        """종료"""
        await self.client.stop()
    
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
            include_quote: 주식호가(0C) 포함 여부
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
