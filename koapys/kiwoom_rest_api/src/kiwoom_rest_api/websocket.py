import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from pyee.asyncio import AsyncIOEventEmitter

from .config import get_ws_url, WS_TIMEOUT

logger = logging.getLogger(__name__)

class WebSocketError(Exception):
    """Custom exception for WebSocket errors"""
    def __init__(self, message: str, error_data: dict = None):
        self.message = message
        self.error_data = error_data or {}
        super().__init__(message)

class RealTimeData:
    """실시간 데이터를 담는 클래스"""
    def __init__(self, data: Dict[str, Any]):
        self.raw_data = data
        self.trnm = data.get('trnm')
        self.return_code = data.get('return_code')
        self.return_msg = data.get('return_msg')
        self.data = data.get('data', [])

class WebSocketClient(AsyncIOEventEmitter):
    """키움증권 실시간 웹소켓 클라이언트"""
    
    def __init__(
        self, 
        access_token: str,
        ws_url: Optional[str] = None,
        auto_reconnect: bool = True,
        reconnect_interval: int = 5,
        ping_interval: int = 30
    ):
        """
        웹소켓 클라이언트 초기화
        
        Args:
            access_token: 액세스 토큰
            ws_url: 웹소켓 URL (None이면 설정에서 자동 선택)
            auto_reconnect: 자동 재연결 여부
            reconnect_interval: 재연결 간격 (초)
            ping_interval: PING 간격 (초)
            
        Events:
            connect: 웹소켓 연결 성공 시 발생
            disconnect: 웹소켓 연결 종료 시 발생
            login: 로그인 성공 시 발생
            data: 실시간 데이터 수신 시 발생 (RealTimeData 전달)
            error: 오류 발생 시 발생 (Exception 전달)
        """
        super().__init__()
        
        self.access_token = access_token
        self.ws_url = ws_url or get_ws_url()
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.ping_interval = ping_interval
        
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.keep_running = True
        self.is_logged_in = False
        
        # 태스크들
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """웹소켓 서버에 연결"""
        try:
            if self.access_token:
                logger.info(f"웹소켓 서버에 연결 중 (토큰 길이: {len(self.access_token)}): {self.ws_url}")
            else:
                logger.info(f"웹소켓 서버에 연결 중: {self.ws_url}")
            
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=WS_TIMEOUT
            )
            self.connected = True
            logger.info("웹소켓 서버 연결 성공")
            
            self.emit('connect')
                
        except Exception as e:
            logger.error(f"웹소켓 연결 실패: {e}")
            self.connected = False
            self.emit('error', e)
            raise WebSocketError(f"연결 실패: {e}")

    async def login(self) -> None:
        """웹소켓 서버에 로그인
        
        Note:
            액세스 토큰은 LOGIN 메시지의 'token' 필드로 전송됩니다.
        """
        if not self.connected:
            await self.connect()
            
        login_data = {
            'trnm': 'LOGIN',
            'token': self.access_token
        }
        
        await self.send(login_data)
        logger.info(f"로그인 메시지 전송 (토큰 길이: {len(self.access_token) if self.access_token else 0})")

    async def send(self, message: Union[str, dict]) -> None:
        """메시지 전송"""
        if not self.connected:
            if self.auto_reconnect:
                await self.connect()
            else:
                raise WebSocketError("웹소켓이 연결되지 않았습니다")
        
        try:
            if isinstance(message, dict):
                message_str = json.dumps(message)
            else:
                message_str = message
                
            await self.websocket.send(message_str)
            logger.debug(f"송신: {message_str[:100]}...")
            
        except Exception as e:
            logger.error(f"메시지 전송 실패: {e}")
            self.emit('error', e)
            raise WebSocketError(f"메시지 전송 실패: {e}")

    async def register_realtime(
        self, 
        group_no: str = "1",
        type_list: List[str] = None,
        item_list: List[str] = None,
        refresh: str = "1"
    ) -> None:
        """
        실시간 데이터 등록
        
        Args:
            group_no: 그룹 번호
            type_list: 실시간 항목 리스트 (예: ['04', '0A', '0B'])
            item_list: 종목코드 리스트 (빈 리스트면 전체)
            refresh: 기존등록유지여부 (0: 기존유지안함, 1: 기존유지)
        """
        if type_list is None:
            type_list = ['04']  # 기본값: 잔고
        if item_list is None:
            item_list = ['']  # 빈 문자열은 전체 종목
            
        register_data = {
            'trnm': 'REG',
            'grp_no': group_no,
            'refresh': refresh,
            'data': [{
                'item': item_list,
                'type': type_list
            }]
        }
        
        await self.send(register_data)
        logger.info(f"실시간 데이터 등록: {type_list}")

    async def unregister_realtime(self, group_no: str = "1") -> None:
        """실시간 데이터 해지"""
        unregister_data = {
            'trnm': 'REMOVE',
            'grp_no': group_no
        }
        
        await self.send(unregister_data)
        logger.info("실시간 데이터 해지")

    async def condition_list_request_ka10171(
        self,
    ) -> None:
        """
        조건식 목록 조회
        
        API ID: ka10171
        
        Note:
            사용자가 HTS에서 등록한 조건검색 목록을 조회합니다.
            응답은 on_data 콜백을 통해 수신됩니다.
            
        Returns:
            None (응답은 on_data 콜백으로 수신)
            
        Response Format (via on_data callback):
            {
                "trnm": "CNSRLST",
                "return_code": 0,
                "return_msg": "성공",
                "data": [
                    {
                        "cond_idx": "000",
                        "cond_nm": "상승추세종목"
                    },
                    {
                        "cond_idx": "001",
                        "cond_nm": "거래량급증"
                    }
                ]
            }
            
        Example:
            >>> client = WebSocketClient(access_token=token)
            >>> await client.start()
            >>> await client.condition_list_request_ka10171()
        """
        list_request_data = {
            'trnm': 'CNSRLST'
        }
        
        await self.send(list_request_data)
        logger.info("조건식 목록 조회 요청")

    async def register_condition_search_ka10173(
        self,
        condition_index: str,
        condition_name: str,
    ) -> None:
        """
        조건검색 실시간 등록
        
        Args:
            condition_index: 조건식 인덱스 (예: "000", "001")
            condition_name: 조건식 이름
            group_no: 그룹 번호
            refresh: 기존등록유지여부 (0: 기존유지안함, 1: 기존유지)
        """
        register_data = {
            'trnm': 'CNSRREQ',  # 조건검색 등록
            "seq" : condition_index,# 조건검색식 일련번호
            "search_type" : "0",    # 0:조건검색 
            "stex_tp" : "K",        # K:KRX
            "cont_yn" : "N",        # 연속조회여부 (Y:연속조회, N:단순조회)
            "next_key" : ""         # 연속조회키 (연속조회여부가 'Y'인 경우 필수 세팅)
        }
        
        await self.send(register_data)
        logger.info(f"조건검색 실시간 등록: [{condition_index}] {condition_name}")

    async def unregister_condition_search_ka10174(
        self,
        condition_index: str = None,
        group_no: str = "1"
    ) -> None:
        """
        조건검색 실시간 해지
        
        Args:
            condition_index: 조건식 인덱스 (None이면 그룹 전체 해지)
            group_no: 그룹 번호
        """
        if condition_index:
            unregister_data = {
                'trnm': 'COND_REMOVE',
                'grp_no': group_no,
                'cond_idx': condition_index
            }
        else:
            unregister_data = {
                'trnm': 'COND_REMOVE_ALL',
                'grp_no': group_no
            }
        
        await self.send(unregister_data)
        logger.info(f"조건검색 실시간 해지: {condition_index or '전체'}")

    async def _handle_message(self, message: str) -> None:
        """메시지 처리"""
        try:
            data = json.loads(message)
            realtime_data = RealTimeData(data)
            
            trnm = realtime_data.trnm
            logger.debug(f"수신: trnm={trnm}, return_code={realtime_data.return_code}")
            
            if trnm == 'LOGIN':
                if realtime_data.return_code == 0:
                    self.is_logged_in = True
                    logger.info("로그인 성공")
                    self.emit('login')
                else:
                    error_msg = realtime_data.return_msg or "로그인 실패"
                    logger.error(f"로그인 실패: {error_msg}")
                    self.emit('error', WebSocketError(error_msg))
                        
            elif trnm == 'PING':
                # PING에 PONG으로 응답
                await self.send(data)
                logger.debug("PING-PONG 응답")
                
            elif trnm == 'REAL':
                # 실시간 데이터 수신
                data_count = len(realtime_data.data) if realtime_data.data else 0
                logger.debug(f"[WEBSOCKET] 실시간 데이터 수신, data 항목 수: {data_count}")
                self.emit('data', realtime_data)
                    
            else:
                # 기타 응답 (조건식 목록, 조건검색 응답 등)
                data_len = len(realtime_data.data) if realtime_data.data is not None else 0
                logger.debug(f"응답 수신 - trnm: {trnm}, data 항목 수: {data_len}")
                self.emit('data', realtime_data)
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}, 메시지: {message[:200]}")
            self.emit('error', e)
        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}", exc_info=True)
            self.emit('error', e)

    async def _receive_messages(self) -> None:
        """메시지 수신 루프"""
        while self.keep_running:
            try:
                if not self.connected or not self.websocket:
                    break
                    
                message = await self.websocket.recv()
                await self._handle_message(message)
                
            except ConnectionClosed:
                logger.warning("웹소켓 연결이 종료되었습니다")
                self.connected = False
                self.is_logged_in = False
                
                self.emit('disconnect')
                    
                if self.auto_reconnect and self.keep_running:
                    logger.info(f"{self.reconnect_interval}초 후 재연결을 시도합니다")
                    await asyncio.sleep(self.reconnect_interval)
                    try:
                        await self.connect()
                        await self.login()
                    except Exception as e:
                        logger.error(f"재연결 실패: {e}")
                else:
                    break
                    
            except Exception as e:
                logger.error(f"메시지 수신 오류: {e}")
                self.emit('error', e)

    async def _ping_loop(self) -> None:
        """PING 루프 (연결 유지)"""
        while self.keep_running and self.connected:
            try:
                await asyncio.sleep(self.ping_interval)
                if self.connected and self.websocket:
                    await self.websocket.ping()
                    logger.debug("PING 전송")
            except Exception as e:
                logger.error(f"PING 오류: {e}")

    async def start(self) -> None:
        """웹소켓 클라이언트 시작"""
        try:
            await self.connect()
            await self.login()
            
            # 수신 및 PING 태스크 시작
            self._receive_task = asyncio.create_task(self._receive_messages())
            self._ping_task = asyncio.create_task(self._ping_loop())
            
            logger.info("웹소켓 클라이언트 시작됨")
            
        except Exception as e:
            logger.error(f"웹소켓 클라이언트 시작 실패: {e}")
            raise

    async def stop(self) -> None:
        """웹소켓 클라이언트 중지"""
        logger.info("웹소켓 클라이언트 중지 중...")
        self.keep_running = False
        
        # 태스크들 취소
        if self._receive_task:
            self._receive_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()
            
        # 웹소켓 연결 종료
        if self.websocket:
            await self.websocket.close()
            
        self.connected = False
        self.is_logged_in = False
        logger.info("웹소켓 클라이언트 중지됨")

    async def run_forever(self) -> None:
        """무한 루프로 실행"""
        try:
            await self.start()
            # 태스크들이 완료될 때까지 대기
            if self._receive_task:
                await self._receive_task
        except asyncio.CancelledError:
            logger.info("웹소켓 클라이언트가 취소되었습니다")
        except Exception as e:
            logger.error(f"웹소켓 클라이언트 실행 오류: {e}")
        finally:
            await self.stop()

    def run_sync(self) -> None:
        """동기적으로 실행 (새로운 이벤트 루프에서)"""
        try:
            asyncio.run(self.run_forever())
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단되었습니다")
        except Exception as e:
            logger.error(f"동기 실행 오류: {e}")
            raise 