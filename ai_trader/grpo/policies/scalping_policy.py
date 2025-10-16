"""
GRPO 정책 네트워크 구현

이 모듈은 GRPO를 위한 정책 네트워크를 정의합니다.
"""

import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class GRPOPolicy(nn.Module):
    """
    GRPO를 위한 정책 네트워크
    
    입력: 임베딩 벡터 (embedding_dim,)
    출력: 행동 확률 분포 (3,) - 보유/매수/매도
    
    아키텍처:
    - 입력 레이어: embedding_dim -> hidden_dim
    - 은닉 레이어: hidden_dim -> hidden_dim
    - 정책 헤드: hidden_dim -> action_dim (행동 확률)
    - 가치 헤드: hidden_dim -> 1 (상태 가치)
    
    Args:
        embedding_dim: 임베딩 차원
        hidden_dim: 은닉 레이어 차원 (기본값: 256)
        action_dim: 행동 공간 차원 (기본값: 3)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        action_dim: int = 3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 공유 특징 추출 레이어
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 정책 헤드 (행동 확률)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # 가치 헤드 (상태 가치)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        logger.info(f"GRPOPolicy initialized: embedding_dim={embedding_dim}, "
                   f"hidden_dim={hidden_dim}, action_dim={action_dim}")
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        정책 네트워크 forward pass
        
        Args:
            state: 상태 텐서 (batch_size, embedding_dim)
            
        Returns:
            (action_logits, state_value) 튜플
            - action_logits: 행동 로짓 (batch_size, action_dim)
            - state_value: 상태 가치 (batch_size, 1)
        """
        # 공유 특징 추출
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # 정책 헤드: 행동 로짓
        action_logits = self.policy_head(x)
        
        # 가치 헤드: 상태 가치
        state_value = self.value_head(x)
        
        return action_logits, state_value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        정책에서 행동 샘플링
        
        Args:
            state: 상태 텐서 (batch_size, embedding_dim)
            deterministic: True면 최대 확률 행동 선택, False면 확률적 샘플링
            
        Returns:
            (action, log_prob) 튜플
            - action: 선택된 행동 (batch_size,)
            - log_prob: 행동의 로그 확률 (batch_size,)
        """
        # Forward pass
        action_logits, _ = self.forward(state)
        
        # 행동 확률 분포 생성
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if deterministic:
            # 최대 확률 행동 선택
            action = torch.argmax(action_probs, dim=-1)
        else:
            # 확률적 샘플링
            action = dist.sample()
        
        # 로그 확률 계산
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 상태와 행동에 대한 로그 확률, 엔트로피, 가치 계산
        
        이 메서드는 정책 업데이트 시 사용됩니다.
        
        Args:
            states: 상태 텐서 (batch_size, embedding_dim)
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
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # 로그 확률 계산
        log_probs = dist.log_prob(actions)
        
        # 엔트로피 계산
        entropy = dist.entropy()
        
        # 가치를 1D로 변환
        values = state_values.squeeze(-1)
        
        return log_probs, entropy, values
