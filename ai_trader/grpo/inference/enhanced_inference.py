"""
개선된 GRPO 추론 엔진 (실전 거래용)

분석에서 발견한 주의점을 반영:
1. Buy 신호 과다 (91%) → 신뢰도 기반 필터링
2. Sell 신호 부족 (4.1%) → 자동 손절/익절 로직
3. Look-ahead bias 경고
"""

import logging
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .infer_grpo import GRPOInference

logger = logging.getLogger(__name__)


class Action(Enum):
    """행동 타입"""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    """포지션 정보"""
    entry_price: float
    entry_time: int
    current_price: float
    holding_period: int
    
    @property
    def profit_rate(self) -> float:
        """수익률 계산"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    @property
    def is_profit(self) -> bool:
        """수익 포지션 여부"""
        return self.profit_rate > 0


class EnhancedGRPOInference:
    """
    실전 거래를 위한 개선된 GRPO 추론 엔진
    
    주요 개선사항:
    1. 신뢰도 기반 Buy 필터링 (과다 진입 방지)
    2. 자동 손절/익절 (Sell 신호 보완)
    3. 포지션 관리 및 리스크 제어
    
    Args:
        base_inference: 기본 GRPOInference 인스턴스
        min_buy_confidence: Buy 신호 최소 신뢰도 (기본값: 0.0, 필터링 없음)
        min_sell_confidence: Sell 신호 최소 신뢰도 (기본값: 0.0, 필터링 없음)
        stop_loss_rate: 손절 비율 (기본값: -2.0%)
        take_profit_rate: 익절 비율 (기본값: 5.0%)
        max_holding_period: 최대 보유 기간 (기본값: 100)
        enable_auto_exit: 자동 손절/익절 활성화 (기본값: True)
        
    Note:
        검증 결과, 신뢰도 필터링 없이 자동 청산만 사용하는 것이 최고 성과.
        - None (0.0): Mean Reward 299.39, 75% 승률
        - Light (0.36): Mean Reward 4.70 (거래 거의 없음)
        - Moderate (0.358): Mean Reward 217.21 (27% 감소)
        
        포지션 관리(중복 진입 방지)로 이미 88.96% 자동 필터링됨.
    """
    
    def __init__(
        self,
        base_inference: GRPOInference,
        min_buy_confidence: float = 0.0,
        min_sell_confidence: float = 0.0,
        stop_loss_rate: float = -2.0,
        take_profit_rate: float = 5.0,
        max_holding_period: int = 100,
        enable_auto_exit: bool = True
    ):
        self.base_inference = base_inference
        
        # 신뢰도 임계값
        self.min_buy_confidence = min_buy_confidence
        self.min_sell_confidence = min_sell_confidence
        
        # 리스크 관리 파라미터
        self.stop_loss_rate = stop_loss_rate
        self.take_profit_rate = take_profit_rate
        self.max_holding_period = max_holding_period
        self.enable_auto_exit = enable_auto_exit
        
        # 통계
        self.stats = {
            'total_predictions': 0,
            'buy_signals': 0,
            'buy_filtered': 0,
            'sell_signals': 0,
            'auto_exits': 0,
            'stop_losses': 0,
            'take_profits': 0,
            'max_holding_exits': 0
        }
        
        logger.info("EnhancedGRPOInference initialized")
        logger.info(f"  Min Buy Confidence: {min_buy_confidence}")
        logger.info(f"  Min Sell Confidence: {min_sell_confidence}")
        logger.info(f"  Stop Loss: {stop_loss_rate}%")
        logger.info(f"  Take Profit: {take_profit_rate}%")
        logger.info(f"  Max Holding Period: {max_holding_period}")
        logger.info(f"  Auto Exit: {enable_auto_exit}")
    
    def predict(
        self,
        sequence: np.ndarray,
        current_position: Optional[Position] = None,
        current_price: float = 0.0,
        deterministic: bool = True
    ) -> Tuple[int, float, Dict]:
        """
        개선된 행동 예측
        
        Args:
            sequence: 입력 시퀀스
            current_position: 현재 포지션 (있으면)
            current_price: 현재 가격
            deterministic: 결정적 예측 여부
            
        Returns:
            (action, confidence, info) 튜플
            - action: 최종 행동 (0: Hold, 1: Buy, 2: Sell)
            - confidence: 신뢰도
            - info: 추가 정보 (원본 행동, 필터링 이유 등)
        """
        self.stats['total_predictions'] += 1
        
        # 기본 추론
        raw_action, raw_confidence = self.base_inference.predict(
            sequence, 
            deterministic=deterministic
        )
        
        info = {
            'raw_action': raw_action,
            'raw_confidence': raw_confidence,
            'filtered': False,
            'filter_reason': None,
            'auto_exit': False,
            'exit_reason': None
        }
        
        # 포지션이 있으면 자동 손절/익절 체크
        if current_position is not None and self.enable_auto_exit:
            # 포지션 업데이트
            current_position.current_price = current_price
            current_position.holding_period += 1
            
            exit_action, exit_reason = self._check_auto_exit(current_position)
            
            if exit_action == Action.SELL.value:
                self.stats['auto_exits'] += 1
                info['auto_exit'] = True
                info['exit_reason'] = exit_reason
                
                logger.debug(
                    f"Auto exit: {exit_reason}, "
                    f"profit={current_position.profit_rate:.2f}%, "
                    f"holding={current_position.holding_period}"
                )
                
                return Action.SELL.value, 1.0, info
        
        # Buy 신호 필터링
        if raw_action == Action.BUY.value:
            self.stats['buy_signals'] += 1
            
            if raw_confidence < self.min_buy_confidence:
                self.stats['buy_filtered'] += 1
                info['filtered'] = True
                info['filter_reason'] = f"Low confidence ({raw_confidence:.3f} < {self.min_buy_confidence})"
                
                logger.debug(
                    f"Buy signal filtered: confidence={raw_confidence:.3f} "
                    f"< threshold={self.min_buy_confidence}"
                )
                
                return Action.HOLD.value, raw_confidence, info
            
            # 이미 포지션이 있으면 추가 매수 방지
            if current_position is not None:
                self.stats['buy_filtered'] += 1
                info['filtered'] = True
                info['filter_reason'] = "Position already exists"
                
                logger.debug("Buy signal filtered: already in position")
                
                return Action.HOLD.value, raw_confidence, info
        
        # Sell 신호 검증
        elif raw_action == Action.SELL.value:
            self.stats['sell_signals'] += 1
            
            # 포지션이 없으면 Sell 무시
            if current_position is None:
                info['filtered'] = True
                info['filter_reason'] = "No position to sell"
                
                logger.debug("Sell signal filtered: no position")
                
                return Action.HOLD.value, raw_confidence, info
            
            # 신뢰도가 너무 낮으면 Hold로 변경
            if raw_confidence < self.min_sell_confidence:
                info['filtered'] = True
                info['filter_reason'] = f"Low confidence ({raw_confidence:.3f} < {self.min_sell_confidence})"
                
                logger.debug(
                    f"Sell signal filtered: confidence={raw_confidence:.3f} "
                    f"< threshold={self.min_sell_confidence}"
                )
                
                return Action.HOLD.value, raw_confidence, info
        
        # 필터링 통과
        return raw_action, raw_confidence, info
    
    def _check_auto_exit(self, position: Position) -> Tuple[int, str]:
        """
        자동 손절/익절 체크
        
        Args:
            position: 현재 포지션
            
        Returns:
            (action, reason) 튜플
        """
        # 손절 체크
        if position.profit_rate <= self.stop_loss_rate:
            self.stats['stop_losses'] += 1
            return Action.SELL.value, f"Stop loss (profit={position.profit_rate:.2f}%)"
        
        # 익절 체크
        if position.profit_rate >= self.take_profit_rate:
            self.stats['take_profits'] += 1
            return Action.SELL.value, f"Take profit (profit={position.profit_rate:.2f}%)"
        
        # 최대 보유 기간 체크
        if position.holding_period >= self.max_holding_period:
            self.stats['max_holding_exits'] += 1
            return Action.SELL.value, f"Max holding period (period={position.holding_period})"
        
        return Action.HOLD.value, None
    
    def get_stats(self) -> Dict:
        """
        통계 반환
        
        Returns:
            통계 딕셔너리
        """
        stats = self.stats.copy()
        
        if stats['total_predictions'] > 0:
            stats['buy_signal_rate'] = stats['buy_signals'] / stats['total_predictions']
            stats['buy_filter_rate'] = stats['buy_filtered'] / stats['buy_signals'] if stats['buy_signals'] > 0 else 0
            stats['sell_signal_rate'] = stats['sell_signals'] / stats['total_predictions']
            stats['auto_exit_rate'] = stats['auto_exits'] / stats['total_predictions']
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Statistics reset")
    
    def update_thresholds(
        self,
        min_buy_confidence: Optional[float] = None,
        min_sell_confidence: Optional[float] = None,
        stop_loss_rate: Optional[float] = None,
        take_profit_rate: Optional[float] = None,
        max_holding_period: Optional[int] = None
    ):
        """
        임계값 업데이트
        
        Args:
            min_buy_confidence: Buy 최소 신뢰도
            min_sell_confidence: Sell 최소 신뢰도
            stop_loss_rate: 손절 비율
            take_profit_rate: 익절 비율
            max_holding_period: 최대 보유 기간
        """
        if min_buy_confidence is not None:
            self.min_buy_confidence = min_buy_confidence
            logger.info(f"Updated min_buy_confidence: {min_buy_confidence}")
        
        if min_sell_confidence is not None:
            self.min_sell_confidence = min_sell_confidence
            logger.info(f"Updated min_sell_confidence: {min_sell_confidence}")
        
        if stop_loss_rate is not None:
            self.stop_loss_rate = stop_loss_rate
            logger.info(f"Updated stop_loss_rate: {stop_loss_rate}%")
        
        if take_profit_rate is not None:
            self.take_profit_rate = take_profit_rate
            logger.info(f"Updated take_profit_rate: {take_profit_rate}%")
        
        if max_holding_period is not None:
            self.max_holding_period = max_holding_period
            logger.info(f"Updated max_holding_period: {max_holding_period}")
