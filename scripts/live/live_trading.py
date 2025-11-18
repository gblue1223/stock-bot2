#!/usr/bin/env python3
"""
실전 거래 스크립트

EnhancedGRPOInference와 Koapys API를 사용한 실시간 자동 매매

주요 기능:
1. 실시간 시장 데이터 수집
2. GRPO 모델 기반 매매 신호 생성
3. 자동 손절/익절 관리
4. 포지션 및 리스크 관리
5. 거래 로깅 및 모니터링

사용법:
    python scripts/real/live_trading.py --config config/trading_config.json
"""

import os
import sys
import json
import logging
import argparse
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, Optional, List, Set
from datetime import datetime, time as dt_time
from collections import deque

import numpy as np
import torch
from dotenv import load_dotenv

# ⚠️ 중요: 다른 모듈을 import하기 전에 환경 변수를 먼저 로드해야 합니다
# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드 (모든 import 전에!)
load_dotenv()

from koapys.types import OrderType, OrderBookType
from koapys import KoapyRestSimple
from koapys import ConditionSearchClient, RealtimeStockClient, StockRealtimeData
from koapys import FINAL_COLUMNS
from ai_trader.grpo.inference.infer_grpo import GRPOInference
from ai_trader.grpo.inference.enhanced_inference import EnhancedGRPOInference, Position, Action

from lib.rolling_normalization import RollingNormalizer
from lib.normalization import FEATURE_NAMES
        
# 로깅 설정
import os
from datetime import datetime

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 타임스탬프 포함 로그 파일명
log_filename = os.path.join(log_dir, f'live_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 로그 포맷
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# 파일 핸들러 (상세 로그)
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 콘솔 핸들러 (요약 로그)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 루트 로거 설정
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_filename}")

# 오후장 디버깅을 위해 관련 모듈의 로그 레벨을 DEBUG로 설정
# logging.getLogger('koapys.kiwoom_rest_api.src.kiwoom_rest_api.koreanstock.realtime_stock').setLevel(logging.DEBUG)
logging.getLogger('koapys.kiwoom_rest_api.src.kiwoom_rest_api.koreanstock.condition_search').setLevel(logging.DEBUG)
# logging.getLogger('kiwoom_rest_api.websocket').setLevel(logging.DEBUG)

# 메인 모듈 DEBUG 로그 활성화 (버퍼 디버깅용)
logging.getLogger(__name__).setLevel(logging.DEBUG)

# Windows 콘솔 인코딩 설정
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


class TradingConfig:
    """거래 설정"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로 (JSON)
        """
        # 기본 설정
        self.model_path = None
        self.embedding_model_path = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 거래 설정
        self.target_stocks = []  # 거래 대상 종목 코드 리스트
        self.max_position_size = 1000000  # 최대 포지션 크기 (원)
        self.max_positions = 3  # 최대 동시 포지션 수
        
        # 추론 엔진 설정
        self.min_buy_confidence = 0.0  # Buy 최소 신뢰도 (검증 결과: 0.0이 최적)
        self.min_sell_confidence = 0.0  # Sell 최소 신뢰도
        self.stop_loss_rate = -2.0  # 손절 비율 (%)
        self.take_profit_rate = 5.0  # 익절 비율 (%)
        self.max_holding_period = 100  # 최대 보유 기간 (틱)
        
        # 시장 시간 설정 (문자열로 저장)
        self.market_start = "09:00"  # 장 시작
        self.market_end = "15:30"  # 장 종료
        self.trading_start = "09:00"  # 거래 시작
        self.trading_end = "11:00"  # 거래 종료 (스캘핑은 오전만)
        
        # 데이터 수집 설정
        self.seq_len = 60  # 시퀀스 길이
        self.num_features = 28  # 특징 수 (파생 피처 4개 + 기본 지표 4개 + 대기금액 20개)
        self.update_interval = 1.0  # 데이터 업데이트 간격 (초)
        
        # 정규화 설정 (PerStockNormalizer 고정)
        self.online_window_size = 200  # Rolling window 크기
        self.online_warmup_samples = 50  # 워밍업 샘플 수
        self.normalization_stats_path = None  # ✅ 학습 데이터 정규화 통계 파일 경로
        
        # 조건검색 사용 설정
        self.use_condition_monitor = True
        self.condition_collect_delay = 2.0  # 최초 등록 직후 딜레이(초)
        
        # 설정 파일에서 로드
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # 시간 문자열을 dt_time 객체로 변환 (내부 사용)
        self._market_start_time = None
        self._market_end_time = None
        self._trading_start_time = None
        self._trading_end_time = None
    
    def load_from_file(self, config_path: str):
        """JSON 파일에서 설정 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 설정 업데이트
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"Configuration loaded from {config_path}")
    
    def save_to_file(self, config_path: str):
        """설정을 JSON 파일로 저장"""
        config = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {config_path}")


class MarketDataBuffer:
    """시장 데이터 버퍼"""
    
    def __init__(self, seq_len: int, num_features: int):
        """
        Args:
            seq_len: 시퀀스 길이
            num_features: 특징 수
        """
        self.seq_len = seq_len
        self.num_features = num_features
        self.buffer = deque(maxlen=seq_len)
    
    def add(self, features: np.ndarray):
        """
        특징 추가
        
        Args:
            features: 특징 벡터 (num_features,)
        """
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(features)}")
        
        self.buffer.append(features)
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        현재 시퀀스 반환
        
        Returns:
            시퀀스 (seq_len, num_features) 또는 None (데이터 부족)
        """
        if len(self.buffer) < self.seq_len:
            return None
        
        return np.array(list(self.buffer))
    
    def is_ready(self) -> bool:
        """시퀀스가 준비되었는지 확인"""
        return len(self.buffer) >= self.seq_len
    
    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear()


class LiveTrader:
    """실전 거래 시스템"""
    
    def __init__(self, config: TradingConfig):
        """
        Args:
            config: 거래 설정
        """
        self.config = config
        
        # Koapys 클라이언트 초기화
        logger.info("Initializing Koapys client...")
        self.koapys = KoapyRestSimple(
            simulation=False
        )
        
        # 연결 확인
        try:
            self.koapys.ensure_connected()
            logger.info("[OK] Koapys connected successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Koapys: {e}")
            raise
        
        # 계좌 정보
        accounts = self.koapys.get_account_list()
        self.account_no = accounts[0] if accounts else None
        logger.info(f"Using account: {self.account_no}")
        
        # GRPO 추론 엔진 초기화
        logger.info("Initializing GRPO inference engine...")
        
        # ✅ Rolling Window Normalizer 초기화 (분포 이동 문제 해결)
        # 종목별 normalizer 딕셔너리
        self.normalizers = {}
        
        # Rolling window 설정
        self.rolling_window_size = 1000  # 최근 1000개 데이터
        self.rolling_min_samples = 100   # 최소 100개 수집 후 정규화
        
        logger.info(f"✅ Rolling Window Normalizer initialized")
        logger.info(f"   Window size: {self.rolling_window_size}")
        logger.info(f"   Min samples: {self.rolling_min_samples}")
        logger.info(f"   Strategy: Adaptive to market changes")
        
        # GRPOInference는 정규화 통계 없이 초기화
        # 정규화는 PerStockNormalizer가 처리 (학습 통계 또는 온라인 통계)
        base_inference = GRPOInference(
            policy_path=config.model_path,
            embedding_model_path=config.embedding_model_path,
            device=config.device,
            use_torchscript=False,
            normalization_stats=None  # PerStockNormalizer가 처리
        )
        
        self.inference = EnhancedGRPOInference(
            base_inference=base_inference,
            min_buy_confidence=config.min_buy_confidence,
            min_sell_confidence=config.min_sell_confidence,
            stop_loss_rate=config.stop_loss_rate,
            take_profit_rate=config.take_profit_rate,
            max_holding_period=config.max_holding_period,
            enable_auto_exit=True
        )
        
        # PerStockNormalizer를 사용하므로 사전 통계 불필요
        
        logger.info("[OK] Inference engine initialized")
        
        # 데이터 버퍼 (종목별)
        self.data_buffers: Dict[str, MarketDataBuffer] = {}
        for code in config.target_stocks:
            self.data_buffers[code] = MarketDataBuffer(
                seq_len=config.seq_len,
                num_features=config.num_features
            )
        
        # 포지션 관리
        self.positions: Dict[str, Position] = {}  # code -> Position
        self.order_history: List[Dict] = []
        
        # 통계
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'predictions': 0  # ✅ 예측 횟수 추가
        }
        
        # 실행 상태
        self.is_running = False
        self.last_update_time = {}  # code -> timestamp
        
        # 실시간 데이터 클라이언트
        self.realtime_client: Optional[RealtimeStockClient] = None
        self.condition_client: Optional[ConditionSearchClient] = None
        self.realtime_thread: Optional[threading.Thread] = None
        self.realtime_stop_event = threading.Event()
        self.realtime_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 조건검색으로 받은 종목 코드 저장
        self._condition_codes: Set[str] = set()
        self._condition_codes_lock = threading.Lock()
        
        # 실시간 데이터 클라이언트 시작 (WebSocket 및 조건검색 포함)
        self._start_realtime_client()
    
    def get_condition_codes(self) -> List[str]:
        """조건검색으로 받은 종목 코드 반환"""
        with self._condition_codes_lock:
            codes = sorted(self._condition_codes)
            if not hasattr(self, '_last_logged_codes') or self._last_logged_codes != codes:
                logger.info(f"[CONDITION] Current tracked stocks: {len(codes)} - {codes[:5]}{'...' if len(codes) > 5 else ''}")
                self._last_logged_codes = codes
            return codes
    
    def is_market_open(self) -> bool:
        """장이 열려있는지 확인"""
        now = datetime.now().time()
        market_start = dt_time(*map(int, self.config.market_start.split(':')))
        market_end = dt_time(*map(int, self.config.market_end.split(':')))
        return market_start <= now <= market_end
    
    def is_trading_time(self) -> bool:
        """거래 시간인지 확인"""
        now = datetime.now().time()
        trading_start = dt_time(*map(int, self.config.trading_start.split(':')))
        trading_end = dt_time(*map(int, self.config.trading_end.split(':')))
        return trading_start <= now <= trading_end
    
    def _start_realtime_client(self):
        """실시간 데이터 클라이언트 시작"""
        try:
            access_token = self.koapys.access_token
            if not access_token:
                logger.warning("No access token; realtime client disabled")
                return
            
            # 백그라운드 스레드에서 실행 (WebSocket도 내부에서 시작)
            self.realtime_stop_event.clear()
            self.realtime_thread = threading.Thread(
                target=self._run_realtime_client,
                name="RealtimeStockClient",
                daemon=True
            )
            self.realtime_thread.start()
            logger.info("[OK] Realtime stock client thread started")
            
        except Exception as e:
            logger.error(f"Failed to start realtime client: {e}", exc_info=True)
    
    def _run_realtime_client(self):
        """실시간 클라이언트 실행 (백그라운드 스레드)"""
        try:
            # 새로운 이벤트 루프 생성 및 저장
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.realtime_loop = loop
            
            # 비동기 메인 실행
            loop.run_until_complete(self._run_realtime_async())
        except Exception as e:
            logger.error(f"Realtime client error: {e}", exc_info=True)
        finally:
            self.realtime_loop = None
    
    async def _run_realtime_async(self):
        """실시간 클라이언트 비동기 실행"""
        try:
            # WebSocket 시작
            logger.info("Starting WebSocket client in realtime thread...")
            await self.koapys.start_websocket()
            
            # RealtimeStockClient 생성
            self.realtime_client = RealtimeStockClient(client=self.koapys.websocket)
            
            # 이미 로그인되어 있으므로 이벤트 설정
            if self.koapys.websocket.is_logged_in:
                self.realtime_client._login_event.set()
            
            # 실시간 데이터 수신 카운터 (디버깅용)
            data_receive_count = {'count': 0, 'last_log_time': time.time()}
            
            # 콜백 설정: 데이터 업데이트 시 자동으로 버퍼에 추가
            def on_data_update(code: str, stock_data: StockRealtimeData):
                try:
                    # 실시간 데이터 수신 모니터링 (5초마다 로그)
                    data_receive_count['count'] += 1
                    current_time = time.time()
                    if current_time - data_receive_count['last_log_time'] >= 5.0:
                        logger.info(f"[REALTIME] Received {data_receive_count['count']} data updates in last 5 seconds")
                        data_receive_count['count'] = 0
                        data_receive_count['last_log_time'] = current_time
                    
                    # ✅ 거래 활발도 검증 (체결강도 > 0)
                    if stock_data.체결강도 == 0:
                        # 체결강도=0인 종목은 거래가 활발하지 않음
                        # 1분마다 한 번씩만 로그
                        warn_log_attr = f'_inactive_stock_{code}'
                        if not hasattr(self, warn_log_attr) or current_time - getattr(self, warn_log_attr) >= 60:
                            setattr(self, warn_log_attr, current_time)
                            logger.debug(
                                f"[INACTIVE] {code}: 체결강도=0 (거래 비활발), "
                                f"등락률={stock_data.등락률:.2f}%, 거래회전율={stock_data.거래회전율:.2f}%"
                            )
                        return
                    
                    if code not in self.data_buffers:
                        logger.debug(f"[REALTIME] Skipping {code}: buffer not found")
                        return
                    
                    # 첫 데이터 수신 로그
                    is_first_data = code not in self.last_update_time
                    
                    # StockRealtimeData를 dict로 변환
                    market_data = stock_data.to_dict()
                    
                    # 주기적으로 raw 데이터 로깅 (30초마다)
                    raw_log_attr = f'_raw_data_log_{code}'
                    if not hasattr(self, raw_log_attr) or current_time - getattr(self, raw_log_attr) >= 30:
                        setattr(self, raw_log_attr, current_time)
                        logger.debug(
                            f"[RAW DATA] {code}: 등락률={market_data.get('등락률', 0):.4f}, "
                            f"누적거래대금={market_data.get('누적거래대금', 0):.2e}, "
                            f"거래회전율={market_data.get('거래회전율', 0):.4f}, "
                            f"체결강도={market_data.get('체결강도', 0):.4f}, "
                            f"매도대기금액1={market_data.get('매도대기금액1', 0):.2f}, "
                            f"매수대기금액1={market_data.get('매수대기금액1', 0):.2f}"
                        )
                    
                    # 특징 추출
                    features = self.extract_features(market_data)
                    logger.debug(f"[REALTIME] {code}: Extracted {len(features)} features")
                    
                    # 버퍼에 추가 (체결강도 > 0 검증은 이미 완료)
                    before_len = len(self.data_buffers[code].buffer)
                    self.data_buffers[code].add(features)
                    after_len = len(self.data_buffers[code].buffer)
                    logger.debug(f"[REALTIME] {code}: Buffer {before_len} -> {after_len}/{self.config.seq_len}")
                    
                    # 업데이트 시간 기록
                    self.last_update_time[code] = time.time()
                    
                    # 첫 데이터 수신 시 로그
                    if is_first_data:
                        logger.info(
                            f"[REALTIME] First data received for {code} ({stock_data.종목명}): "
                            f"현재가={stock_data.현재가:,.0f}원, 등락률={stock_data.등락률:+.2f}%"
                        )
                    
                except Exception as e:
                    logger.error(f"[REALTIME] Failed to process data for {code}: {e}", exc_info=True)
            
            self.realtime_client.on_data_update = on_data_update
            logger.info("[OK] RealtimeStockClient initialized")
            logger.info(f"[DEBUG] Current data buffers: {list(self.data_buffers.keys())}")
            
            # 조건검색 클라이언트 생성 (use_condition_monitor가 True인 경우)
            if self.config.use_condition_monitor:
                self.condition_client = ConditionSearchClient(client=self.koapys.websocket)
                
                # 조건검색 콜백 설정
                def on_stock_in(code, name, cond_idx):
                    """종목 편입 시 호출 - 거래 활발도 검증 추가"""
                    code_clean = code[1:] if code.startswith('A') else code
                    logger.info(f"[CONDITION] Stock IN signal received: {code} -> {code_clean} ({name}), cond_idx={cond_idx}")
                    
                    # ✅ 거래 활발도 사전 검증
                    # 실시간 데이터에서 체결강도를 확인하여 활발한 종목만 추가
                    # (실제 검증은 on_data_update에서 수행)
                    
                    with self._condition_codes_lock:
                        if code_clean not in self._condition_codes:
                            self._condition_codes.add(code_clean)
                            logger.info(f"[CONDITION] Stock added to tracking: {code_clean} ({name}), Total: {len(self._condition_codes)}")
                        else:
                            logger.debug(f"[CONDITION] Stock {code_clean} already tracked")
                
                def on_stock_out(code, name, cond_idx):
                    """
                    종목 이탈 시 호출
                    
                    ✅ 변경: 한번 포착된 종목은 계속 모니터링
                    조건검색에서 이탈해도 실시간 데이터는 계속 수신
                    """
                    code_clean = code[1:] if code.startswith('A') else code
                    logger.info(f"[CONDITION] Stock OUT signal received: {code} -> {code_clean} ({name}), cond_idx={cond_idx}")
                    logger.info(f"[CONDITION] ✅ Keeping {code_clean} in tracking (continue monitoring)")
                    
                    # ✅ 변경: 종목을 제거하지 않음 (계속 모니터링)
                    # with self._condition_codes_lock:
                    #     if code_clean in self._condition_codes:
                    #         self._condition_codes.remove(code_clean)
                    #         logger.info(f"[CONDITION] Stock removed from tracking: {code_clean} ({name}), Remaining: {len(self._condition_codes)}")
                    #     else:
                    #         logger.debug(f"[CONDITION] Stock {code_clean} was not tracked")
                
                self.condition_client.on_stock_in = on_stock_in
                self.condition_client.on_stock_out = on_stock_out
                
                # 조건식 로드 및 등록
                logger.info("[CONDITION] Loading condition list...")
                conditions = await self.condition_client.load_conditions()
                if conditions:
                    logger.info(f"[CONDITION] Found {len(conditions)} conditions")
                    selected = conditions[0]
                    logger.info(f"[CONDITION] Registering condition: [{selected.index}] {selected.name}")
                    await self.condition_client.register_condition(
                        condition_index=selected.index,
                        condition_name=selected.name
                    )
                    logger.info(f"[OK] Condition monitor registered: [{selected.index}] {selected.name}")
                else:
                    logger.warning("[CONDITION] No conditions found for monitoring")
            
            # 초기 종목 등록
            if self.config.target_stocks:
                await self.realtime_client.register_stocks(
                    stock_codes=self.config.target_stocks,
                    include_trade=True,
                    include_quote=True
                )
                logger.info(f"Registered {len(self.config.target_stocks)} stocks for realtime data")
            
            # 종료 신호까지 대기
            while not self.realtime_stop_event.is_set():
                await asyncio.sleep(0.5)
        
        finally:
            # 정리
            if self.condition_client:
                self.condition_client.cleanup()
            if self.realtime_client:
                self.realtime_client.cleanup()
            await self.koapys.stop_websocket()
            logger.info("WebSocket stopped in realtime thread")
    
    async def _register_realtime_stocks(self, stock_codes: List[str]):
        """실시간 데이터 종목 등록"""
        if not self.realtime_client:
            logger.warning("Realtime client not available")
            return
        
        try:
            logger.info(f"[REGISTER] Starting registration of {len(stock_codes)} stocks: {stock_codes}")
            await self.realtime_client.register_stocks(
                stock_codes=stock_codes,
                include_trade=True,
                include_quote=True,
                refresh="0"
            )
            logger.info(f"[REGISTER] ✓ Successfully registered {len(stock_codes)} stocks for realtime data")
        except Exception as e:
            logger.error(f"[REGISTER] ✗ Failed to register stocks: {e}", exc_info=True)
            raise  # Re-raise to propagate to caller
    
    def extract_features(self, market_data: Dict) -> np.ndarray:
        """
        시장 데이터에서 특징 추출 (to_dict와 동일한 로직)
        
        Args:
            market_data: Koapys에서 받은 시장 데이터 (dict 형태)
            
        Returns:
            특징 벡터 (28,) - FINAL_COLUMNS에서 메타데이터 및 호가/수량 제외
            
        추출되는 특징 (28개):
        - 파생 피처: 종목명_scalar, 시간_sin, 시간_cos, 시간_scalar (4개)
        - 기본 지표: 등락률, 누적거래대금, 거래회전율, 체결강도 (4개)
        - 매도대기금액1~10 (10개)
        - 매수대기금액1~10 (10개)
        
        제외되는 필드:
        - 메타데이터: 종목코드, 종목명, 시간
        - 호가/수량: 매도호가1~10, 매도호가수량1~10, 매수호가1~10, 매수호가수량1~10
        """
        try:
            # 종목코드 추출
            code = market_data.get('종목코드', 'UNKNOWN')
            
            features = []
            feature_names = []
            
            # FINAL_COLUMNS를 순회하면서 to_dict()와 동일한 로직 적용
            for col in FINAL_COLUMNS:
                # 메타데이터 제외
                if col in ["종목코드", "종목명", "시간"]:
                    continue
                
                # 호가 및 호가수량 제외 (to_dict와 동일)
                if col.startswith('매도호가') or col.startswith('매수호가'):
                    continue
                
                # 특징 추출
                value = market_data.get(col, 0.0)
                features.append(float(value))
                feature_names.append(col)
            
            # numpy array로 변환
            features_array = np.array(features, dtype=np.float32)
            
            # ✅ 체결강도 검증 (정규화 전) - 경고만 출력
            # 인덱스 7 = 체결강도 (FINAL_COLUMNS 순서 기준)
            # on_data_update에서 이미 필터링되므로 여기서는 경고만
            if len(features_array) > 7 and features_array[7] == 0.0:
                current_time = int(time.time())
                warn_log_attr = f'_zero_intensity_{code}'
                if not hasattr(self, warn_log_attr) or current_time - getattr(self, warn_log_attr) >= 60:
                    setattr(self, warn_log_attr, current_time)
                    logger.debug(
                        f"[FEATURE INFO] {code}: 체결강도=0 in features "
                        f"(이미 on_data_update에서 필터링됨)"
                    )
            
            # ✅ 정규화 전 원본 데이터 로깅 (디버그용, 10번마다)
            if self.stats['predictions'] % 10 == 0:
                logger.debug(
                    f"[BEFORE NORM] {code}: mean={features_array.mean():.2f}, "
                    f"std={features_array.std():.2f}, "
                    f"range=[{features_array.min():.2f}, {features_array.max():.2f}]"
                )
                logger.debug(
                    f"[BEFORE NORM] {code} 주요값: "
                    f"등락률={features_array[4]:.4f}, "
                    f"누적거래대금={features_array[5]:.2e}, "
                    f"거래회전율={features_array[6]:.4f}, "
                    f"체결강도={features_array[7]:.4f}"
                )
            
            # ✅ Rolling Window 정규화 적용 (분포 이동 문제 해결)
            # 종목별 normalizer 가져오기 (없으면 생성)
            if code not in self.normalizers:
                self.normalizers[code] = RollingNormalizer(
                    window_size=self.rolling_window_size,
                    min_samples=self.rolling_min_samples,
                    feature_names=FEATURE_NAMES
                )
                logger.info(f"[NORMALIZER] Created rolling normalizer for {code}")
            
            normalizer = self.normalizers[code]
            features_array = normalizer.normalize(features_array, update=True)
            
            # 워밍업 상태 로그 (종목별 1회만)
            warmup_log_attr = f'_warmup_logged_{code}'
            stats = normalizer.get_stats()
            
            if stats['n_samples'] < self.rolling_min_samples:
                if not hasattr(self, warmup_log_attr):
                    logger.info(f"[NORMALIZER] {code} warming up... ({stats['n_samples']}/{self.rolling_min_samples})")
                    setattr(self, warmup_log_attr, True)
            else:
                # 준비 완료 시 한 번 로그
                ready_log_attr = f'_ready_logged_{code}'
                if not hasattr(self, ready_log_attr):
                    logger.info(f"[NORMALIZER] {code} ready for inference (samples={stats['n_samples']})")
                    setattr(self, ready_log_attr, True)
            
            # ✅ 정규화 후 데이터 로깅 (디버그용, 10번마다)
            if self.stats['predictions'] % 10 == 0:
                logger.debug(
                    f"[AFTER NORM] {code}: mean={features_array.mean():.4f}, "
                    f"std={features_array.std():.4f}, "
                    f"range=[{features_array.min():.4f}, {features_array.max():.4f}]"
                )
            
            # 길이 확인
            if len(features_array) != self.config.num_features:
                logger.warning(
                    f"Feature length mismatch: expected {self.config.num_features}, "
                    f"got {len(features_array)}"
                )
                # 패딩 또는 자르기
                if len(features_array) < self.config.num_features:
                    features_array = np.pad(
                        features_array,
                        (0, self.config.num_features - len(features_array)),
                        mode='constant'
                    )
                else:
                    features_array = features_array[:self.config.num_features]
            
            # 주기적으로 특징 값 샘플 로그 (5분마다)
            current_time = int(time.time())
            log_attr = f'_feature_log_{code}'
            if not hasattr(self, log_attr) or current_time - getattr(self, log_attr) >= 300:  # 5분
                setattr(self, log_attr, current_time)
                
                # 통계 계산
                non_zero_count = np.count_nonzero(features_array)
                mean_val = np.mean(features_array)
                std_val = np.std(features_array)
                min_val = np.min(features_array)
                max_val = np.max(features_array)
                
                logger.info(
                    f"[FEATURES] {code} 특징 추출 샘플: "
                    f"길이={len(features_array)}, 비영={non_zero_count}, "
                    f"평균={mean_val:.4f}, 표준편차={std_val:.4f}, 범위=[{min_val:.4f}, {max_val:.4f}]"
                )
                
                # 주요 특징 값 출력
                logger.debug(
                    f"[FEATURES] {code} 주요값: "
                    f"등락률={features_array[4]:.4f}, 누적거래대금={features_array[5]:.2e}, "
                    f"거래회전율={features_array[6]:.4f}, 체결강도={features_array[7]:.4f}"
                )
            
            return features_array
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}", exc_info=True)
            # 오류 시 제로 벡터 반환
            return np.zeros(self.config.num_features, dtype=np.float32)
    
    def update_market_data(self, code: str):
        """
        시장 데이터 업데이트 (실시간 클라이언트가 콜백으로 자동 처리)
        
        Args:
            code: 종목 코드
        
        Note:
            실시간 클라이언트가 활성화되어 있으면 콜백에서 자동으로 버퍼가 업데이트됩니다.
            이 메서드는 호환성을 위해 유지되지만, 실제로는 아무 작업도 하지 않습니다.
        """
        # 실시간 클라이언트가 활성화되어 있으면 콜백에서 자동 처리
        if self.realtime_client:
            return
    
    def get_current_price(self, code: str) -> float:
        """
        현재 가격 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            현재 가격
        """
        try:
            # 실시간 클라이언트에서 먼저 시도
            if self.realtime_client:
                stock_data = self.realtime_client.get_stock_data(code)
                if stock_data and stock_data.현재가 > 0:
                    return float(stock_data.현재가)
            
            # 폴백: REST API 사용
            price = self.koapys.get_current_price(code)
            return float(price)
        except Exception as e:
            logger.error(f"Failed to get current price for {code}: {e}")
            return 0.0
    
    def get_available_cash(self) -> float:
        """
        주문 가능 금액 조회
        
        Returns:
            주문 가능 금액 (원)
        """
        try:
            deposit_info = self.koapys.get_deposit(self.account_no)
            available = deposit_info.get('주문가능금액', 0)
            return float(available) if available else 0.0
        except Exception as e:
            logger.error(f"Failed to get available cash: {e}")
            return 0.0
    
    def execute_buy(self, code: str, confidence: float):
        """
        매수 실행
        
        Args:
            code: 종목 코드
            confidence: 신뢰도
        """
        # 이미 포지션이 있으면 무시
        if code in self.positions:
            logger.debug(f"Already in position for {code}")
            return
        
        # 최대 포지션 수 확인
        if len(self.positions) >= self.config.max_positions:
            logger.debug(f"Max positions reached ({self.config.max_positions})")
            return
        
        try:
            # 현재 가격 조회
            current_price = self.get_current_price(code)
            if current_price <= 0:
                logger.warning(f"[BUY SKIP] {code}: Invalid price {current_price}")
                return
            
            # 주문 가능 금액 확인
            available_cash = self.get_available_cash()
            required_cash = current_price * 1  # 최소 1주
            
            if available_cash < required_cash:
                logger.warning(
                    f"[BUY SKIP] {code}: Insufficient cash. "
                    f"Available: {available_cash:,.0f}원, Required: {required_cash:,.0f}원"
                )
                return
            
            # 매수 수량 계산 (가용 금액과 설정된 최대 포지션 크기 중 작은 값)
            max_buyable = int(available_cash / current_price)
            max_by_config = int(self.config.max_position_size / current_price)
            quantity = min(max_buyable, max_by_config)
            
            if quantity <= 0:
                logger.warning(f"[BUY SKIP] {code}: Invalid quantity {quantity}")
                return
            
            # 실제 매수 금액
            total_cost = current_price * quantity
            
            # 실시간 종목 정보 로그
            if self.realtime_client:
                stock_data = self.realtime_client.get_stock_data(code)
                if stock_data:
                    logger.info(
                        f"[BUY SIGNAL] {code} ({stock_data.종목명}): "
                        f"현재가={current_price:,.0f}, 등락률={stock_data.등락률:.2f}%, "
                        f"체결강도={stock_data.체결강도:.1f}, 거래회전율={stock_data.거래회전율:.2f}%"
                    )
            
            # 주문 실행
            logger.info(
                f"[BUY] {code}: price={current_price:,.0f}원, qty={quantity}주, "
                f"cost={total_cost:,.0f}원, available={available_cash:,.0f}원, confidence={confidence:.3f}"
            )
            
            # 시장가 주문 (가격 0으로 설정)
            order_result = self.koapys.send_order(
                rqname=f"BUY_{code}_{datetime.now().strftime('%H%M%S')}",
                account_no=self.account_no,
                order_type=OrderType.BUY,
                code=code,
                quantity=quantity,
                price=0,  # 시장가는 0
                hoga=OrderBookType.MARKET
            )
            
            # 포지션 생성
            self.positions[code] = Position(
                entry_price=current_price,
                entry_time=int(time.time()),
                current_price=current_price,
                holding_period=0,
                cumulative_return=0.0  # 누적 수익률 초기화
            )
            
            # 주문 기록
            self.order_history.append({
                'timestamp': datetime.now().isoformat(),
                'code': code,
                'action': 'BUY',
                'price': current_price,
                'quantity': quantity,
                'confidence': confidence,
                'order_result': order_result
            })
            
            self.stats['total_trades'] += 1
            
            logger.info(f"[OK] Buy order executed: {order_result}")
            
        except Exception as e:
            logger.error(f"Failed to execute buy for {code}: {e}", exc_info=True)
    
    def execute_sell(self, code: str, confidence: float, reason: str = "Signal"):
        """
        매도 실행
        
        Args:
            code: 종목 코드
            confidence: 신뢰도
            reason: 매도 이유
        """
        # 포지션이 없으면 무시
        if code not in self.positions:
            logger.debug(f"No position for {code}")
            return
        
        try:
            position = self.positions[code]
            
            # 현재 가격
            current_price = self.get_current_price(code)
            if current_price <= 0:
                logger.warning(f"Invalid price for {code}: {current_price}")
                return
            
            # 수익률 계산
            position.current_price = current_price
            profit_rate = position.profit_rate
            
            # 매도 수량 (전량 매도)
            # TODO: 실제로는 보유 수량을 조회해야 함
            quantity = int(self.config.max_position_size / position.entry_price)
            
            # 실시간 종목 정보 로그
            if self.realtime_client:
                stock_data = self.realtime_client.get_stock_data(code)
                if stock_data:
                    logger.info(
                        f"[SELL SIGNAL] {code} ({stock_data.종목명}): "
                        f"현재가={current_price:,.0f}, 등락률={stock_data.등락률:.2f}%, "
                        f"체결강도={stock_data.체결강도:.1f}, 거래회전율={stock_data.거래회전율:.2f}%"
                    )
            
            # 주문 실행
            profit_amount = (current_price - position.entry_price) * quantity
            logger.info(
                f"[SELL] {code}: entry={position.entry_price:,.0f}원, current={current_price:,.0f}원, "
                f"qty={quantity}주, profit={profit_rate:+.2f}% ({profit_amount:+,.0f}원), "
                f"confidence={confidence:.3f}, reason={reason}"
            )
            
            # 시장가 주문 (가격 0으로 설정)
            order_result = self.koapys.send_order(
                rqname=f"SELL_{code}_{datetime.now().strftime('%H%M%S')}",
                account_no=self.account_no,
                order_type=OrderType.SELL,
                code=code,
                quantity=quantity,
                price=0,  # 시장가는 0
                hoga=OrderBookType.MARKET
            )
            
            # 포지션 제거
            del self.positions[code]
            
            # 주문 기록
            self.order_history.append({
                'timestamp': datetime.now().isoformat(),
                'code': code,
                'action': 'SELL',
                'price': current_price,
                'quantity': quantity,
                'confidence': confidence,
                'profit_rate': profit_rate,
                'reason': reason,
                'order_result': order_result
            })
            
            # 통계 업데이트
            if profit_rate > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
            
            self.stats['total_profit'] += profit_rate
            
            logger.info(f"[OK] Sell order executed: {order_result}")
            
        except Exception as e:
            logger.error(f"Failed to execute sell for {code}: {e}", exc_info=True)
    
    def process_stock(self, code: str):
        """
        종목 처리 (데이터 수집 및 매매 신호 생성)
        
        Args:
            code: 종목 코드
        """
        # 데이터 업데이트 (실시간 클라이언트가 없으면 폴링)
        self.update_market_data(code)
        
        # 버퍼 확인
        if code not in self.data_buffers:
            # 첫 체크 시에만 로그 (중복 방지)
            if not hasattr(self, f'_buffer_not_found_{code}'):
                setattr(self, f'_buffer_not_found_{code}', True)
                logger.info(f"[SKIP] {code}: Buffer not found")
            return
        
        # 시퀀스가 준비되지 않았으면 대기
        buffer = self.data_buffers[code]
        if not buffer.is_ready():
            buffer_len = len(buffer.buffer)
            # 10초마다 한 번씩만 로그 (중복 방지)
            current_time = int(time.time())
            last_log_attr = f'_buffer_log_{code}'
            if not hasattr(self, last_log_attr) or current_time - getattr(self, last_log_attr) >= 10:
                setattr(self, last_log_attr, current_time)
                logger.info(f"[BUFFER] {code}: Collecting data ({buffer_len}/{self.config.seq_len})")
            return
        
        # ✅ 거래 활발도 검증 (체결강도 > 0)
        # on_data_update에서 이미 필터링되므로 여기서는 추가 검증만 수행
        if self.realtime_client:
            stock_data = self.realtime_client.get_stock_data(code)
            if stock_data:
                # 체결강도가 0이면 거래 비활발 (이미 on_data_update에서 필터링됨)
                if stock_data.체결강도 == 0:
                    return
                
                # 추가 검증: 매도/매수 대기금액이 모두 0인 경우
                if (stock_data.매도대기금액1 == 0 and stock_data.매수대기금액1 == 0):
                    current_time = int(time.time())
                    warn_log_attr = f'_quote_warn_{code}'
                    if not hasattr(self, warn_log_attr) or current_time - getattr(self, warn_log_attr) >= 30:
                        setattr(self, warn_log_attr, current_time)
                        logger.debug(
                            f"[DATA QUALITY] {code}: 매도/매수대기금액=0, 예측 스킵"
                        )
                    return
        
        # 시퀀스 가져오기
        sequence = self.data_buffers[code].get_sequence()
        
        # 주기적으로 시퀀스 통계 로깅 (1분마다)
        current_time = int(time.time())
        seq_log_attr = f'_seq_log_{code}'
        if not hasattr(self, seq_log_attr) or current_time - getattr(self, seq_log_attr) >= 60:
            setattr(self, seq_log_attr, current_time)
            seq_mean = np.mean(sequence)
            seq_std = np.std(sequence)
            seq_min = np.min(sequence)
            seq_max = np.max(sequence)
            logger.info(
                f"[SEQUENCE] {code}: shape={sequence.shape}, mean={seq_mean:.4f}, "
                f"std={seq_std:.4f}, range=[{seq_min:.4f}, {seq_max:.4f}]"
            )
        
        # 현재 포지션
        current_position = self.positions.get(code)
        
        # 현재 가격
        current_price = self.get_current_price(code)
        
        # 포지션 업데이트
        if current_position is not None:
            current_position.current_price = current_price
            current_position.holding_period += 1
        
        # 추론
        action, confidence, pred_info = self.inference.predict(
            sequence=sequence,
            current_position=current_position,
            current_price=current_price,
            deterministic=True
        )
        
        # 예측 결과 로깅 (HOLD가 아닌 경우만)
        if action != Action.HOLD.value:
            action_name = 'BUY' if action == Action.BUY.value else 'SELL'
            stock_name = ''
            if self.realtime_client:
                stock_data = self.realtime_client.get_stock_data(code)
                if stock_data:
                    stock_name = f" ({stock_data.종목명})"
            
            logger.info(
                f"[PREDICT] {code}{stock_name}: {action_name} signal, "
                f"confidence={confidence:.4f}, raw_action={pred_info.get('raw_action')}, "
                f"filtered={pred_info.get('filtered')}, reason={pred_info.get('filter_reason') or pred_info.get('exit_reason')}"
            )
        
        # 행동 실행
        if action == Action.BUY.value and self.is_trading_time():
            self.execute_buy(code, confidence)
        elif action == Action.SELL.value:
            reason = pred_info.get('exit_reason', 'Signal')
            self.execute_sell(code, confidence, reason)
    
    def run(self):
        """거래 시스템 실행"""
        logger.info("=" * 80)
        logger.info("[START] Live Trading System")
        logger.info("=" * 80)
        logger.info(f"Target stocks (initial): {self.config.target_stocks}")
        logger.info(f"Device: {self.config.device}")
        logger.info("=" * 80)
        
        self.is_running = True
        market_check_count = 0
        
        try:
            while self.is_running:
                # 장 시간 확인
                if self.is_market_open():
                    market_check_count = 0
                else:
                    market_check_count += 1
                    if market_check_count == 1:
                        logger.info("Market is closed. Waiting...")
                    time.sleep(1)
                    continue
                
                # 조건검색 실시간으로부터 종목 동기화
                dynamic_codes: List[str] = self.config.target_stocks
                if self.config.use_condition_monitor:
                    prev_count = len(dynamic_codes)
                    dynamic_codes = self.get_condition_codes()
                    
                    # 종목 변화 감지
                    if prev_count == 0 and len(dynamic_codes) == 0:
                        if not hasattr(self, '_no_codes_warned'):
                            logger.warning("[CONDITION] No stocks from condition search yet. Waiting...")
                            self._no_codes_warned = True
                    elif len(dynamic_codes) > 0:
                        self._no_codes_warned = False

                    # 신규 코드 발견 시 버퍼 및 실시간 등록
                    new_codes = []
                    for code in dynamic_codes:
                        if code not in self.data_buffers:
                            self.data_buffers[code] = MarketDataBuffer(
                                seq_len=self.config.seq_len,
                                num_features=self.config.num_features,
                            )
                            new_codes.append(code)
                            logger.info(f"[CONDITION] New stock detected: {code}")
                    
                    # 신규 종목을 실시간 데이터에 등록
                    if new_codes:
                        if not self.realtime_client:
                            logger.warning(f"[NEW STOCKS] Cannot register {len(new_codes)} stocks: realtime_client is None")
                        elif not self.realtime_loop:
                            logger.warning(f"[NEW STOCKS] Cannot register {len(new_codes)} stocks: realtime_loop is None")
                        else:
                            logger.info(f"[NEW STOCKS] Registering {len(new_codes)} new stocks: {', '.join(new_codes)}")
                            try:
                                future = asyncio.run_coroutine_threadsafe(
                                    self._register_realtime_stocks(new_codes),
                                    self.realtime_loop
                                )
                                # 타임아웃과 함께 결과 대기 (충분한 시간 제공)
                                future.result(timeout=5.0)
                                logger.info(f"[OK] Successfully registered {len(new_codes)} stocks for realtime data")
                            except TimeoutError:
                                logger.warning(f"Timeout while registering new stocks in realtime (waited 5 seconds)")
                            except Exception as e:
                                logger.warning(f"Failed to register new stocks in realtime: {e}", exc_info=True)
                    # 제거된 코드 버퍼는 남겨두어도 무방(메모리 사용 적음). 필요 시 정리 가능.

                # 각 종목 처리
                if not dynamic_codes:
                    if int(time.time()) % 10 == 0:  # 10초마다 한 번만 로그
                        logger.warning("[SKIP] No dynamic codes to process")
                else:
                    for code in dynamic_codes:
                        try:
                            self.process_stock(code)
                        except Exception as e:
                            logger.error(f"Error processing {code}: {e}", exc_info=True)
                
                # 통계 출력 (1분마다)
                current_time = int(time.time())
                if current_time % 60 == 0:
                    # 동일한 시간에 중복 출력 방지
                    if not hasattr(self, '_last_stats_time') or current_time != self._last_stats_time:
                        self._last_stats_time = current_time
                        self.print_stats()
                
                # 대기
                time.sleep(self.config.update_interval)
                
        except KeyboardInterrupt:
            logger.info("\n[STOP] Interrupted by user")
        except Exception as e:
            logger.error(f"[ERROR] Fatal error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def print_stats(self):
        """통계 출력"""
        logger.info("=" * 80)
        logger.info("[STATS] Trading Statistics")
        logger.info("=" * 80)
        logger.info(f"Total Trades: {self.stats['total_trades']}")
        logger.info(f"Winning Trades: {self.stats['winning_trades']}")
        logger.info(f"Losing Trades: {self.stats['losing_trades']}")
        
        if self.stats['total_trades'] > 0:
            win_rate = self.stats['winning_trades'] / self.stats['total_trades'] * 100
            logger.info(f"Win Rate: {win_rate:.2f}%")
        
        logger.info(f"Total Profit: {self.stats['total_profit']:.2f}%")
        logger.info(f"Active Positions: {len(self.positions)}")
        
        # 현재 포지션 상세 정보
        if self.positions:
            logger.info(f"\n[POSITIONS] Current Holdings:")
            for code, position in self.positions.items():
                current_price = self.get_current_price(code)
                position.current_price = current_price
                profit_rate = position.profit_rate
                profit_amount = (current_price - position.entry_price) * int(self.config.max_position_size / position.entry_price)
                
                stock_name = ""
                if self.realtime_client:
                    stock_data = self.realtime_client.get_stock_data(code)
                    if stock_data:
                        stock_name = f" ({stock_data.종목명})"
                
                logger.info(
                    f"  {code}{stock_name}: entry={position.entry_price:,.0f}원, "
                    f"current={current_price:,.0f}원, profit={profit_rate:+.2f}% ({profit_amount:+,.0f}원), "
                    f"holding={position.holding_period}틱"
                )
        
        # 현재 모니터링 중인 종목
        if self.config.use_condition_monitor:
            monitored_codes = self.get_condition_codes()
            
            logger.info("-" * 80)
            logger.info(f"[MONITOR] Tracked Stocks: {len(self.get_condition_codes())}")
            # 최대 10개까지만 표시
            display_codes = monitored_codes[:10]
            logger.info(f"  Codes: {', '.join(display_codes)}")
            if len(monitored_codes) > 10:
                logger.info(f"  ... and {len(monitored_codes) - 10} more")
                
                # 실시간 데이터가 있으면 주요 종목 정보 출력
                if self.realtime_client:
                    logger.info(f"\n[REALTIME DATA] Top 5 Stocks:")
                    for i, code in enumerate(display_codes[:5], 1):
                        stock_data = self.realtime_client.get_stock_data(code)
                        if stock_data and stock_data.현재가 > 0:
                            logger.info(
                                f"  {i}. {code} ({stock_data.종목명}): "
                                f"현재가={stock_data.현재가:,.0f}원, "
                                f"등락률={stock_data.등락률:+.2f}%, "
                                f"체결강도={stock_data.체결강도:.1f}, "
                                f"거래회전율={stock_data.거래회전율:.2f}%"
                            )
        
        # 예수금 정보
        try:
            available_cash = self.get_available_cash()
            logger.info(f"\n[ACCOUNT] Available Cash: {available_cash:,.0f}원")
        except Exception as e:
            logger.debug(f"Failed to get cash info: {e}")
        
        # 추론 엔진 통계
        inference_stats = self.inference.get_stats()
        logger.info(f"\n[AI] Inference Stats:")
        logger.info(f"Total Predictions: {inference_stats['total_predictions']}")
        logger.info(f"Buy Signal Rate: {inference_stats.get('buy_signal_rate', 0):.2%}")
        logger.info(f"Buy Filter Rate: {inference_stats.get('buy_filter_rate', 0):.2%}")
        logger.info(f"Auto Exit Rate: {inference_stats.get('auto_exit_rate', 0):.2%}")
        
        # 정규화 통계 (Rolling Window)
        logger.info(f"\n[NORMALIZER] Per-Stock Statistics:")
        tracked_codes = self.get_condition_codes() if self.config.use_condition_monitor else self.config.target_stocks
        for code in tracked_codes[:5]:  # 최대 5개만 출력
            if code in self.normalizers:
                stats = self.normalizers[code].get_stats()
                logger.info(
                    f"  {code}: samples={stats['n_samples']}, window={stats['window_size']}"
                )
        
        logger.info("=" * 80)
    
    def shutdown(self):
        """시스템 종료"""
        logger.info("=" * 80)
        logger.info("[SHUTDOWN] Trading system...")
        logger.info("=" * 80)
        
        # 모든 포지션 청산
        if self.positions:
            logger.info(f"Closing {len(self.positions)} open positions...")
            for code in list(self.positions.keys()):
                self.execute_sell(code, 1.0, "Shutdown")
        
        # 최종 통계
        self.print_stats()
        
        # 주문 기록 저장
        if self.order_history:
            log_path = f"logs/orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.order_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Order history saved to {log_path}")
        
        # 실시간 클라이언트 중지 (WebSocket 및 조건검색도 내부에서 중지됨)
        if self.realtime_thread:
            try:
                self.realtime_stop_event.set()
                self.realtime_thread.join(timeout=5)
                logger.info("Realtime client stopped")
            except Exception as e:
                logger.warning(f"Error stopping realtime client: {e}")
        
        # Koapys 연결 종료
        self.koapys.close()
        
        logger.info("[OK] Shutdown complete")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Live Trading System")
    parser.add_argument(
        '--config',
        type=str,
        default='config/trading_config.json',
        help='Trading configuration file path'
    )
    parser.add_argument(
        '--stocks',
        type=str,
        nargs='+',
        help='Target stock codes (overrides config)'
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = TradingConfig(args.config if os.path.exists(args.config) else None)
    
    # 명령줄 인자로 오버라이드
    if args.stocks:
        config.target_stocks = args.stocks
    
    # 종목 코드 확인: target_stocks 미설정 시에도 조건검색 모니터가 활성화되어 있으면 진행
    if not config.target_stocks:
        if getattr(config, 'use_condition_monitor', False):
            logger.info("No static target stocks provided; proceeding with dynamic codes from condition monitor.")
        else:
            logger.error("No target stocks specified. Use --stocks or config file, or enable condition monitor.")
            return False
    
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    # 거래 시스템 실행
    trader = LiveTrader(config)
    trader.run()
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
