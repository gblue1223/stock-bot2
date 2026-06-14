"""
xLSTM 기반 E2E 추론 엔진 (실전 거래 및 백테스트용)

임베딩 오토인코더를 사용하지 않고 원본 시계열(seq_len, features)을 직접 입력받아 
리스크 관리 로직(자동 손절, 트레일링 스탑, 침체기 청산)을 수행합니다.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from enum import Enum
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM

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
    cumulative_return: float = 0.0  # 누적 수익률 (비율, 0.01 = 1%)
    max_price: float = 0.0  # 최고가 (트레일링 스탑용)
    
    @property
    def profit_rate(self) -> float:
        """수익률 계산 (백분율)"""
        return self.cumulative_return * 100
    
    @property
    def is_profit(self) -> bool:
        """수익 포지션 여부"""
        return self.cumulative_return > 0


class GRPOInferenceE2EXLSTM:
    """
    xLSTM E2E 모델 전용 추론 엔진
    """
    def __init__(
        self,
        policy_path: Union[str, Path],
        device: str = 'cuda',
        normalization_stats: Optional[Dict] = None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing GRPOInferenceE2EXLSTM on device: {self.device}")
        
        # 정책 로드
        self.policy = self._load_policy(policy_path)
        
        # 정규화 통계 세팅
        self.mean = None
        self.std = None
        if normalization_stats:
            self.set_normalization_stats(normalization_stats)

    def _load_policy(self, policy_path: Union[str, Path]) -> nn.Module:
        checkpoint = torch.load(policy_path, map_location=self.device)
        state_dict = checkpoint.get('policy_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # state_dict로부터 차원 감지
        # conv1.weight: (out_channels, in_channels, kernel_size)
        obs_dim = state_dict['conv1.weight'].shape[1] if 'conv1.weight' in state_dict else 31
        cnn_channels = state_dict['conv1.weight'].shape[0] if 'conv1.weight' in state_dict else 64
        
        # xlstm.cells.0.w_q.weight: (hidden_size, hidden_size)
        rnn_hidden_dim = state_dict['xlstm.cells.0.w_q.weight'].shape[0] if 'xlstm.cells.0.w_q.weight' in state_dict else 128
        fc_hidden_dim = state_dict['fc1.weight'].shape[0] if 'fc1.weight' in state_dict else 256
        action_dim = state_dict['policy_head.weight'].shape[0] if 'policy_head.weight' in state_dict else 3
        
        policy = GRPOPolicyE2EXLSTM(
            obs_dim=obs_dim,
            cnn_channels=cnn_channels,
            rnn_hidden_dim=rnn_hidden_dim,
            fc_hidden_dim=fc_hidden_dim,
            action_dim=action_dim
        )
        
        # 가중치 로드
        policy.load_state_dict(state_dict, strict=False)
        policy.to(self.device)
        policy.eval()
        return policy

    def set_normalization_stats(self, stats: Dict):
        if 'mean' in stats and 'std' in stats:
            self.mean = torch.tensor(stats['mean'], dtype=torch.float32, device=self.device)
            self.std = torch.tensor(stats['std'], dtype=torch.float32, device=self.device)
            logger.info("Normalization statistics set successfully")

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        # features에만 적용하거나, 전체 시퀀스 크기 맞춰 Broadcasting 적용
        # E2E 입력의 앞쪽 28개 피처가 정규화 대상임
        if x.dim() == 2:
            features = x[:, :len(self.mean)]
            normalized_features = (features - self.mean) / (self.std + 1e-8)
            return torch.cat([normalized_features, x[:, len(self.mean):]], dim=-1)
        elif x.dim() == 3:
            features = x[:, :, :len(self.mean)]
            mean = self.mean.unsqueeze(0).unsqueeze(0)
            std = self.std.unsqueeze(0).unsqueeze(0)
            normalized_features = (features - mean) / (std + 1e-8)
            return torch.cat([normalized_features, x[:, :, len(self.mean):]], dim=-1)
        return x

    def predict(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True
    ) -> Tuple[int, float]:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
        if obs.device != self.device:
            obs = obs.to(self.device)
            
        # 정규화
        normalized_obs = self._normalize_input(obs)
        if normalized_obs.dim() == 2:
            normalized_obs = normalized_obs.unsqueeze(0)
            
        with torch.no_grad():
            action_logits, _ = self.policy(normalized_obs)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                
            confidence = action_probs[0, action].item()
            
        return action.item(), confidence


class EnhancedGRPOInferenceXLSTM:
    """
    실전 거래를 위한 개선된 xLSTM 추론 엔진 (리스크 제어 포함)
    """
    def __init__(
        self,
        base_inference: GRPOInferenceE2EXLSTM,
        min_buy_confidence: float = 0.0,
        min_sell_confidence: float = 0.0,
        stop_loss_rate: float = -2.0,      # 손절 퍼센트 (-2.0%)
        take_profit_rate: float = 5.0,     # 익절 퍼센트 (5.0%)
        trailing_stop_activation_rate: float = 3.0,
        trailing_stop_callback_rate: float = 1.0,
        stagnation_exit_seconds: int = 2,
        stagnation_threshold: float = 0.5,
        enable_auto_exit: bool = True
    ):
        self.base_inference = base_inference
        
        self.min_buy_confidence = min_buy_confidence
        self.min_sell_confidence = min_sell_confidence
        self.stop_loss_rate = stop_loss_rate
        self.take_profit_rate = take_profit_rate
        self.trailing_stop_activation_rate = trailing_stop_activation_rate
        self.trailing_stop_callback_rate = trailing_stop_callback_rate
        self.stagnation_exit_seconds = stagnation_exit_seconds
        self.stagnation_threshold = stagnation_threshold
        self.enable_auto_exit = enable_auto_exit
        
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
        logger.info("EnhancedGRPOInferenceXLSTM initialized with risk rules")

    def predict(
        self,
        sequence: np.ndarray,
        current_position: Optional[Position] = None,
        current_price: float = 0.0,
        deterministic: bool = True
    ) -> Tuple[int, float, Dict]:
        self.stats['total_predictions'] += 1
        
        # 리스크 룰 우선 적용 (자동 익절/손절)
        if self.enable_auto_exit and current_position is not None:
            profit_pct = current_position.profit_rate
            
            # 1. 손절 검사
            if profit_pct <= self.stop_loss_rate:
                self.stats['auto_exits'] += 1
                self.stats['stop_losses'] += 1
                return Action.SELL.value, 1.0, {"reason": "Stop Loss Hit", "profit_rate": profit_pct}
                
            # 2. 익절 검사
            if profit_pct >= self.take_profit_rate:
                self.stats['auto_exits'] += 1
                self.stats['take_profits'] += 1
                return Action.SELL.value, 1.0, {"reason": "Take Profit Hit", "profit_rate": profit_pct}
                
            # 3. 트레일링 스탑 검사
            if profit_pct >= self.trailing_stop_activation_rate:
                current_position.max_price = max(current_position.max_price, current_price)
                peak_profit = ((current_position.max_price - current_position.entry_price) / current_position.entry_price) * 100
                if peak_profit - profit_pct >= self.trailing_stop_callback_rate:
                    self.stats['auto_exits'] += 1
                    return Action.SELL.value, 1.0, {"reason": "Trailing Stop Hit", "profit_rate": profit_pct}
                    
        # xLSTM 모델 예측 실행
        action, confidence = self.base_inference.predict(sequence, deterministic)
        
        info = {"raw_action": action, "confidence": confidence}
        
        # 신뢰도 기반 필터링
        if action == Action.BUY.value:
            self.stats['buy_signals'] += 1
            if confidence < self.min_buy_confidence:
                self.stats['buy_filtered'] += 1
                action = Action.HOLD.value
                info["filtered"] = True
                info["filter_reason"] = f"Buy confidence {confidence:.4f} below threshold {self.min_buy_confidence}"
        elif action == Action.SELL.value:
            self.stats['sell_signals'] += 1
            if confidence < self.min_sell_confidence:
                action = Action.HOLD.value
                info["filtered"] = True
                info["filter_reason"] = f"Sell confidence {confidence:.4f} below threshold {self.min_sell_confidence}"
                
        return action, confidence, info
