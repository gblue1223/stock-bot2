"""
조건검색 WebSocket API 모듈

키움증권의 조건검색 실시간 WebSocket API를 위한 헬퍼 클래스
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..websocket import WebSocketClient, RealTimeData

logger = logging.getLogger(__name__)


@dataclass
class ConditionInfo:
    """조건식 정보"""
    index: str
    name: str


class ConditionSearchClient:
    """조건검색 WebSocket 클라이언트
    
    조건검색 관련 작업을 간편하게 수행할 수 있는 헬퍼 클래스
    """
    
    def __init__(self, client: WebSocketClient, auto_translate: bool = True):
        """
        Args:
            client: WebSocketClient 인스턴스 (외부에서 생성 및 관리)
            auto_translate: 필드명 자동 변환 여부
        """
        self.client = client
        self.auto_translate = auto_translate
        self.conditions: List[ConditionInfo] = []
        self.received_stocks: Dict[str, List[str]] = {}  # condition_index -> [stock_codes]
        self._login_event = asyncio.Event()  # 로그인 완료 이벤트
        
        # 이미 로그인된 경우 이벤트 설정
        if self.client.is_logged_in:
            self._login_event.set()
        
        # 사용자 정의 콜백
        self.on_condition_list: Optional[Callable[[List[ConditionInfo]], None]] = None
        self.on_condition_registered: Optional[Callable[[str, str], None]] = None
        self.on_stock_in: Optional[Callable[[str, str, str], None]] = None  # code, name, condition_idx
        self.on_stock_out: Optional[Callable[[str, str, str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # 이벤트 리스너 등록
        self.client.on('data', self._handle_data)
        self.client.on('login', self._on_login)
        
    def cleanup(self):
        """이벤트 리스너 제거"""
        self.client.remove_listener('data', self._handle_data)
        self.client.remove_listener('login', self._on_login)
        
    async def load_conditions(self) -> List[ConditionInfo]:
        """조건식 목록 로드
        
        Returns:
            조건식 정보 리스트
        """
        # 로그인 완료 대기
        await self._login_event.wait()
        
        await self.client.condition_list_request_ka10171()
        # 응답을 기다리기 위해 잠시 대기
        await asyncio.sleep(1)
        return self.conditions.copy()
        
    async def register_condition(
        self,
        condition_index: str,
        condition_name: str,
        group_no: Optional[str] = None
    ):
        """조건검색 실시간 등록
        
        Args:
            condition_index: 조건식 인덱스
            condition_name: 조건식 이름
            group_no: 그룹 번호 (현재 WebSocket API에서 미지원, 호환성 유지용)
        """
        await self.client.register_condition_search_ka10173(
            condition_index=condition_index,
            condition_name=condition_name
        )
        
        # 조건별 종목 리스트 초기화
        if condition_index not in self.received_stocks:
            self.received_stocks[condition_index] = []
            
    async def unregister_condition(
        self,
        condition_index: str,
        group_no: Optional[str] = "1"
    ):
        """조건검색 해지
        
        Args:
            condition_index: 조건식 인덱스
            group_no: 그룹 번호 (기본값: "1")
        """
        await self.client.unregister_condition_search_ka10174(
            condition_index=condition_index,
            group_no=group_no
        )
        
    async def register_multiple_conditions(
        self,
        conditions: List[ConditionInfo],
        delay: float = 0.5
    ):
        """여러 조건식 동시 등록
        
        Args:
            conditions: 등록할 조건식 리스트
            delay: 등록 간 대기 시간 (초)
        """
        for condition in conditions:
            await self.register_condition(
                condition_index=condition.index,
                condition_name=condition.name
            )
            await asyncio.sleep(delay)
            
    async def unregister_multiple_conditions(
        self,
        conditions: List[ConditionInfo]
    ):
        """여러 조건식 해지
        
        Args:
            conditions: 해지할 조건식 리스트
        """
        for condition in conditions:
            await self.unregister_condition(
                condition_index=condition.index
            )
            
    def get_received_stocks(self, condition_index: Optional[str] = None) -> Dict[str, List[str]]:
        """수신된 종목 조회
        
        Args:
            condition_index: 특정 조건식 인덱스 (None이면 전체)
            
        Returns:
            조건식별 종목 코드 리스트
        """
        if condition_index:
            return {condition_index: self.received_stocks.get(condition_index, [])}
        return self.received_stocks.copy()
    
    async def _on_login(self):
        """로그인 완료 시 호출"""
        logger.info("WebSocket 로그인 완료")
        self._login_event.set()
        
    async def _handle_data(self, realtime_data: RealTimeData):
        """실시간 데이터 처리"""
        try:
            # 디버깅: 모든 메시지 타입 로그
            logger.debug(f"[ConditionSearchClient] Received trnm={realtime_data.trnm}, return_code={realtime_data.return_code}")
            
            # REAL 데이터는 다른 리스너들에게도 전달됨 (PyEventEmitter가 자동 처리)
            
            # 1. 조건식 목록 응답
            if realtime_data.trnm == 'CNSRLST':
                if realtime_data.return_code == 0:
                    self.conditions.clear()
                    # 데이터 형식: [['0', 'koa-시가베팅'], ['1', 'koa-관종-3%이상'], ...]
                    for item in realtime_data.data:
                        if isinstance(item, list) and len(item) >= 2:
                            cond_idx = item[0]
                            cond_nm = item[1]
                            self.conditions.append(ConditionInfo(index=cond_idx, name=cond_nm))
                    
                    # 사용자 콜백 호출
                    if self.on_condition_list:
                        self.on_condition_list(self.conditions.copy())
                else:
                    logger.error(f"조건식 목록 조회 실패: {realtime_data.return_msg}")
                    
            # 2. 조건검색 등록 응답
            elif realtime_data.trnm == 'CNSRREQ':
                if realtime_data.return_code == 0:
                    # seq에서 조건식 인덱스 추출
                    cond_idx = realtime_data.raw_data.get('seq', '')
                    
                    # 조건별 종목 리스트 초기화
                    if cond_idx and cond_idx not in self.received_stocks:
                        self.received_stocks[cond_idx] = []
                    
                    stock_count = 0
                    data = realtime_data.data if isinstance(realtime_data.data, list) else []
                    # 등록 성공 시 초기 종목 데이터 처리
                    for item in data:
                        if isinstance(item, dict):
                            # values 딕셔너리에서 데이터 추출
                            if 'values' in item and isinstance(item['values'], dict):
                                values = item['values']
                                # 필드 코드: 9001=종목코드, 302=종목명
                                stock_code = values.get('9001', '')
                                stock_name = values.get('302', '')
                            else:
                                # 직접 키로 접근하거나 숫자 키 처리
                                stock_code = item.get('stk_cd') or item.get('9001', '')
                                stock_name = item.get('stk_nm') or item.get('302', '')
                            
                            if stock_code and cond_idx:
                                if stock_code not in self.received_stocks[cond_idx]:
                                    self.received_stocks[cond_idx].append(stock_code)
                                    stock_count += 1
                                    
                                if self.on_stock_in:
                                    self.on_stock_in(stock_code, stock_name, cond_idx)
                    
                    logger.info(f"조건검색 등록 성공: 조건식[{cond_idx}], 초기 종목 {stock_count}개 수신")
                    
                    # 등록 완료 콜백
                    if self.on_condition_registered:
                        cond_name = next((c.name for c in self.conditions if c.index == cond_idx), '')
                        self.on_condition_registered(cond_idx, cond_name)
                else:
                    logger.error(f"조건검색 등록 실패: {realtime_data.return_msg}")
                    
            # 3. 실시간 조건검색 데이터 (COND_REAL)
            elif realtime_data.trnm == 'COND_REAL':
                logger.debug(f"[ConditionSearchClient] COND_REAL data received, items={len(realtime_data.data) if realtime_data.data else 0}")
                for item in realtime_data.data:
                    if not isinstance(item, dict):
                        continue
                    
                    # 문서 형식에 따라 데이터 추출
                    stock_code = item.get('stk_cd', '')
                    stock_name = item.get('stk_nm', '')
                    action = item.get('action', '')  # 'in' or 'out'
                    cond_idx = item.get('cond_idx', '')
                    
                    logger.debug(f"[ConditionSearchClient] COND_REAL: code={stock_code}, name={stock_name}, action={action}, cond_idx={cond_idx}")
                    
                    if not stock_code:
                        continue
                        
                    # 조건 인덱스가 없는 경우 첫 번째 조건으로 간주
                    if not cond_idx and self.received_stocks:
                        cond_idx = list(self.received_stocks.keys())[0]
                    
                    # 종목 리스트 초기화
                    if cond_idx not in self.received_stocks:
                        self.received_stocks[cond_idx] = []
                    
                    if action == 'in':
                        # 편입
                        if stock_code not in self.received_stocks[cond_idx]:
                            self.received_stocks[cond_idx].append(stock_code)
                        logger.info(f"[CONDITION] 종목 편입: {stock_code} ({stock_name})")
                        if self.on_stock_in:
                            self.on_stock_in(stock_code, stock_name, cond_idx)
                            
                    elif action == 'out':
                        # 이탈
                        if stock_code in self.received_stocks[cond_idx]:
                            self.received_stocks[cond_idx].remove(stock_code)
                        logger.info(f"[CONDITION] 종목 이탈: {stock_code} ({stock_name})")
                        if self.on_stock_out:
                            self.on_stock_out(stock_code, stock_name, cond_idx)
                    else:
                        logger.warning(f"[CONDITION] Unknown action '{action}' for {stock_code}")
                            
        except Exception as e:
            logger.exception(f"데이터 처리 중 오류: {e}")
            if self.on_error:
                self.on_error(e)
