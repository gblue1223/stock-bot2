"""
GRPO 정책 네트워크 구현 (End-to-End 강화학습용)

이 모듈은 오토인코더를 거치지 않고 원본 시계열(seq_len, features)을 
직접 입력받아 행동을 결정하는 E2E 정책 네트워크를 정의합니다.
"""

import logging
from typing import Tuple, Optional, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class GRPOPolicyE2E(nn.Module):
    """
    GRPO를 위한 End-to-End 정책 네트워크
    
    입력: 원본 틱 시계열 데이터 (batch_size, seq_len, obs_dim)
    출력: 행동 확률 분포 (3,) - 보유/매수/매도
    
    아키텍처:
    - Feature Extraction: 1D CNN (로컬 패턴 및 노이즈 압축)
    - Sequence Modeling: LSTM 또는 GRU (장기 시계열 의존성)
    - Dense Layers: GRU의 마지막 은닉 상태 -> MLP
    - 정책 헤드: hidden_dim -> action_dim (행동 확률)
    - 가치 헤드: hidden_dim -> 1 (상태 가치)
    
    Args:
        obs_dim: 입력 특징 차원 (기본 31: 28 피처 + 3 상태)
        cnn_channels: CNN 필터 수 (기본 64)
        rnn_hidden_dim: RNN 은닉 상태 차원 (기본 128)
        fc_hidden_dim: MLP 은닉 레이어 차원 (기본 256)
        action_dim: 행동 공간 차원 (기본값: 3)
    """
    
    def __init__(
        self,
        obs_dim: int = 31,
        cnn_channels: int = 64,
        rnn_hidden_dim: int = 128,
        fc_hidden_dim: int = 256,
        action_dim: int = 3
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 1. 1D CNN for local feature extraction and downsampling
        # 입력 형태: (batch_size, channels=obs_dim, seq_len)
        self.conv1 = nn.Conv1d(in_channels=obs_dim, out_channels=cnn_channels, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        
        # 2. Sequence Modeling
        # CNN 출력 형태 (batch_size, cnn_channels, reduced_seq_len)를 
        # RNN 입력 형태 (batch_size, reduced_seq_len, cnn_channels)로 변경 후 입력
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 3. 공유 Dense 레이어
        self.fc1 = nn.Linear(rnn_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        
        # 4. 정책 및 가치 헤드
        self.policy_head = nn.Linear(fc_hidden_dim, action_dim)
        self.value_head = nn.Linear(fc_hidden_dim, 1)
        
        logger.info(f"GRPOPolicyE2E initialized: obs_dim={obs_dim}, "
                   f"cnn_channels={cnn_channels}, rnn_hidden_dim={rnn_hidden_dim}, "
                   f"action_dim={action_dim}")
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        정책 네트워크 forward pass
        
        Args:
            state: 원본 시계열 텐서 (batch_size, seq_len, obs_dim) 혹은 (seq_len, obs_dim)
            
        Returns:
            (action_logits, state_value) 튜플
            - action_logits: 행동 로짓 (batch_size, action_dim)
            - state_value: 상태 가치 (batch_size, 1)
        """
        # (seq_len, obs_dim) 입력인 경우 배치 차원 추가
        if state.dim() == 2:
            state = state.unsqueeze(0)
            
        batch_size = state.size(0)
        
        # CNN 입력 형태 맞추기: (batch_size, channels, seq_len)
        x = state.transpose(1, 2)
        
        # CNN Layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # RNN 입력 형태 맞추기: (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # GRU Layer
        _, hidden = self.gru(x)
        # hidden의 형태: (num_layers, batch_size, rnn_hidden_dim)
        # 마지막 레이어의 은닉 상태를 사용
        gru_out = hidden[-1]
        
        # 공유 특성 추출 (Dense)
        fc_out = F.relu(self.fc1(gru_out))
        fc_out = F.relu(self.fc2(fc_out))
        
        # 정책 및 가치 헤드
        action_logits = self.policy_head(fc_out)
        state_value = self.value_head(fc_out)
        
        return action_logits, state_value
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        정책에서 행동 샘플링 (GRPO 호환)
        
        Args:
            obs: 상태 텐서 (batch_size, seq_len, obs_dim)
            deterministic: True면 최대 확률 행동 선택, False면 확률적 샘플링
            
        Returns:
            (action, log_prob) 튜플
            - action: 선택된 행동 (batch_size,)
            - log_prob: 행동의 로그 확률 (batch_size,)
        """
        action_logits, _ = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        
        return action, log_prob
        
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 상태와 행동에 대한 로그 확률, 엔트로피, 가치 계산
        
        이 메서드는 정책 업데이트(PPO/GRPO loss computation) 시 사용됩니다.
        
        Args:
            states: 시계열 상태 텐서 (batch_size, seq_len, obs_dim)
            actions: 행동 텐서 (batch_size,)
            
        Returns:
            (log_probs, entropy, values) 튜플
            - log_probs: 행동의 로그 확률 (batch_size,)
            - entropy: 정책 엔트로피 (batch_size,)
            - values: 상태 가치 (batch_size,)
        """
        # Forward pass
        action_logits, state_values = self.forward(states)
        
        # 행동 확률 분포 생성
        dist = Categorical(logits=action_logits)
        
        # 로그 확률 계산
        log_probs = dist.log_prob(actions)
        
        # 엔트로피 계산
        entropy = dist.entropy()
        
        # 가치를 1D로 변환
        values = state_values.squeeze(-1)
        
        return log_probs, entropy, values
