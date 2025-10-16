"""
직접 특징 사용 정책 네트워크
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class DirectFeaturePolicy(nn.Module):
    """
    직접 특징 사용 정책 네트워크
    
    임베딩 없이 평탄화된 시퀀스를 직접 입력으로 사용
    """
    
    def __init__(
        self,
        input_dim: int,  # seq_len * features
        hidden_dim: int = 128,
        action_dim: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # GRPO 훈련기 호환성을 위한 속성
        self.embedding_dim = input_dim
        
        # 입력 차원 축소
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 공유 특징 추출 레이어
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 정책 헤드 (행동 확률)
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # 가치 헤드 (상태 가치)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        logger.info(f"DirectFeaturePolicy initialized: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, action_dim={action_dim}")
    
    def forward(self, state):
        """정책 네트워크 forward pass"""
        # 입력 차원 축소
        x = F.relu(self.input_projection(state))
        
        # 공유 특징 추출
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 정책 헤드: 행동 로짓
        action_logits = self.policy_head(x)
        
        # 가치 헤드: 상태 가치
        state_value = self.value_head(x)
        
        return action_logits, state_value
    
    def get_action(self, state, deterministic=False):
        """정책에서 행동 샘플링"""
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
    
    def evaluate_actions(self, states, actions):
        """주어진 상태와 행동에 대한 로그 확률, 엔트로피, 가치 계산"""
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
