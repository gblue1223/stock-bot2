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
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, time as dt_time
from collections import deque

import numpy as np
import torch
from dotenv import load_dotenv

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from koapys.client import KoapyRestSimple
from koapys.types import OrderType, OrderBookType
from ai_trader.grpo.inference.infer_grpo import GRPOInference
from ai_trader.grpo.inference.enhanced_inference import EnhancedGRPOInference, Position, Action

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        self.model_path = "models/grpo_direct_features@20251016/direct_features_model.pt"
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
        self.seq_len = 30  # 시퀀스 길이
        self.num_features = 24  # 특징 수
        self.update_interval = 1.0  # 데이터 업데이트 간격 (초)
        
        # Koapys 설정
        self.koapys_base_url = os.getenv('KOAPYS_BASE_URL', 'http://localhost:5000')
        self.koapys_api_key = os.getenv('KOAPYS_API_KEY')
        self.simulation = os.getenv('SIMULATION', 'true').lower() == 'true'
        
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
            base_url=config.koapys_base_url,
            api_key=config.koapys_api_key,
            simulation=config.simulation
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
        base_inference = GRPOInference(
            policy_path=config.model_path,
            embedding_model_path=config.embedding_model_path,
            device=config.device,
            use_torchscript=False
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
            'max_drawdown': 0.0
        }
        
        # 실행 상태
        self.is_running = False
        self.last_update_time = {}  # code -> timestamp
    
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
    
    def extract_features(self, market_data: Dict) -> np.ndarray:
        """
        시장 데이터에서 특징 추출
        
        Args:
            market_data: Koapys에서 받은 시장 데이터
            
        Returns:
            특징 벡터 (num_features,)
        """
        # TODO: 실제 특징 추출 로직 구현
        # 현재는 더미 데이터 반환
        features = np.random.randn(self.config.num_features)
        return features
    
    def update_market_data(self, code: str):
        """
        시장 데이터 업데이트
        
        Args:
            code: 종목 코드
        """
        try:
            # 현재 가격 조회
            market_data = self.koapys.get_stock_basic_info(code)
            
            # 특징 추출
            features = self.extract_features(market_data)
            
            # 버퍼에 추가
            self.data_buffers[code].add(features)
            
            # 업데이트 시간 기록
            self.last_update_time[code] = time.time()
            
        except Exception as e:
            logger.error(f"Failed to update market data for {code}: {e}")
    
    def get_current_price(self, code: str) -> float:
        """
        현재 가격 조회
        
        Args:
            code: 종목 코드
            
        Returns:
            현재 가격
        """
        try:
            price = self.koapys.get_current_price(code)
            return float(price)
        except Exception as e:
            logger.error(f"Failed to get current price for {code}: {e}")
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
            # 현재 가격
            current_price = self.get_current_price(code)
            if current_price <= 0:
                logger.warning(f"Invalid price for {code}: {current_price}")
                return
            
            # 매수 수량 계산
            quantity = int(self.config.max_position_size / current_price)
            if quantity <= 0:
                logger.warning(f"Invalid quantity for {code}: {quantity}")
                return
            
            # 주문 실행
            logger.info(f"[BUY] {code}: price={current_price}, qty={quantity}, confidence={confidence:.3f}")
            
            order_result = self.koapys.send_order(
                rqname=f"BUY_{code}_{datetime.now().strftime('%H%M%S')}",
                account_no=self.account_no,
                order_type=OrderType.BUY,
                code=code,
                quantity=quantity,
                price=int(current_price),
                hoga=OrderBookType.MARKET_IOC
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
            
            # 주문 실행
            logger.info(
                f"[SELL] {code}: price={current_price}, qty={quantity}, "
                f"profit={profit_rate:.2f}%, confidence={confidence:.3f}, reason={reason}"
            )
            
            order_result = self.koapys.send_order(
                rqname=f"SELL_{code}_{datetime.now().strftime('%H%M%S')}",
                account_no=self.account_no,
                order_type=OrderType.SELL,
                code=code,
                quantity=quantity,
                price=int(current_price),
                hoga=OrderBookType.MARKET_IOC
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
        # 데이터 업데이트
        self.update_market_data(code)
        
        # 시퀀스가 준비되지 않았으면 대기
        if not self.data_buffers[code].is_ready():
            return
        
        # 시퀀스 가져오기
        sequence = self.data_buffers[code].get_sequence()
        
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
        logger.info(f"Target stocks: {self.config.target_stocks}")
        logger.info(f"Simulation mode: {self.config.simulation}")
        logger.info(f"Device: {self.config.device}")
        logger.info("=" * 80)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # 장 시간 확인
                if not self.is_market_open():
                    logger.info("Market is closed. Waiting...")
                    time.sleep(60)
                    continue
                
                # 각 종목 처리
                for code in self.config.target_stocks:
                    try:
                        self.process_stock(code)
                    except Exception as e:
                        logger.error(f"Error processing {code}: {e}", exc_info=True)
                
                # 통계 출력 (1분마다)
                if int(time.time()) % 60 == 0:
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
        
        # 추론 엔진 통계
        inference_stats = self.inference.get_stats()
        logger.info(f"\n[AI] Inference Stats:")
        logger.info(f"Total Predictions: {inference_stats['total_predictions']}")
        logger.info(f"Buy Signal Rate: {inference_stats.get('buy_signal_rate', 0):.2%}")
        logger.info(f"Buy Filter Rate: {inference_stats.get('buy_filter_rate', 0):.2%}")
        logger.info(f"Auto Exit Rate: {inference_stats.get('auto_exit_rate', 0):.2%}")
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
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Run in simulation mode'
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = TradingConfig(args.config if os.path.exists(args.config) else None)
    
    # 명령줄 인자로 오버라이드
    if args.stocks:
        config.target_stocks = args.stocks
    if args.simulation:
        config.simulation = True
    
    # 종목 코드 확인
    if not config.target_stocks:
        logger.error("No target stocks specified. Use --stocks or config file.")
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
