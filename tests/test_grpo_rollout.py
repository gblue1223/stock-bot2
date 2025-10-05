"""
GRPO 롤아웃 수집 테스트

Task 7.2: 롤아웃 수집 구현 검증
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.env import GRPOScalpingEnv


class MockEmbeddingModel(nn.Module):
    """테스트용 임베딩 모델"""
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(60, embedding_dim)
    
    def forward(self, x):
        # (batch, seq_len, features) -> (batch, embedding_dim)
        batch_size = x.shape[0]
        # 간단히 평균을 취한 후 선형 변환
        x_mean = x.mean(dim=1)  # (batch, features)
        return self.fc(x_mean)


class MockPolicy(nn.Module):
    """테스트용 정책 네트워크"""
    def __init__(self, embedding_dim=128, action_dim=3):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, action_dim)
    
    def forward(self, x):
        return self.fc(x)
    
    def get_action(self, state, deterministic=False):
        """
        행동 샘플링
        
        Args:
            state: 상태 텐서 (1, embedding_dim)
            deterministic: 결정론적 행동 선택 여부
            
        Returns:
            (action, log_prob) 튜플
        """
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = torch.log(probs[0, action])
        
        return action.item(), log_prob.item()


@pytest.fixture
def mock_env():
    """Mock 환경 생성"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    # 실제 환경 대신 Mock 사용
    env = Mock(spec=GRPOScalpingEnv)
    env.action_space = Mock()
    env.action_space.sample = Mock(return_value=0)
    env.action_space.n = 3
    
    # reset 메서드 Mock
    env.reset = Mock(return_value=(
        np.random.randn(128).astype(np.float32),
        {'stock_code': '005930', 'date': '2024-01-01', 'time': 0.0}
    ))
    
    # step 메서드 Mock
    def mock_step(action):
        next_state = np.random.randn(128).astype(np.float32)
        reward = np.random.randn()
        terminated = np.random.rand() < 0.1  # 10% 확률로 종료
        truncated = False
        info = {
            'position': 0,
            'current_price': 100.0,
            'current_time': 1.0,
            'quick_exit_violations': 0,
            'quick_exit_triggered': False
        }
        
        if terminated:
            info['episode'] = {
                'total_return': np.random.randn(),
                'num_trades': np.random.randint(0, 10),
                'avg_holding_time': np.random.rand() * 30,
                'sharpe_ratio': np.random.randn(),
                'quick_exit_violations': np.random.randint(0, 5),
                'win_rate': np.random.rand(),
                'avg_profit_per_trade': np.random.randn(),
                'episode_length': 100,
                'steps_taken': 50
            }
        
        return next_state, reward, terminated, truncated, info
    
    env.step = Mock(side_effect=mock_step)
    env.close = Mock()
    
    return env


@pytest.fixture
def mock_policy():
    """Mock 정책 생성"""
    return MockPolicy(embedding_dim=128, action_dim=3)


def test_collect_rollouts_basic(mock_env, mock_policy):
    """기본 롤아웃 수집 테스트"""
    trainer = GRPOTrainer(
        policy=mock_policy,
        env=mock_env,
        episodes_per_group=4,
        num_groups=2,
        device='cpu'
    )
    
    # 4개 에피소드 수집
    episodes = trainer.collect_rollouts(num_episodes=4)
    
    # 에피소드 수 확인
    assert len(episodes) == 4, f"Expected 4 episodes, got {len(episodes)}"
    
    # 각 에피소드 구조 확인
    for episode in episodes:
        assert 'states' in episode
        assert 'actions' in episode
        assert 'rewards' in episode
        assert 'next_states' in episode
        assert 'dones' in episode
        assert 'log_probs' in episode
        assert 'metadata' in episode
        
        # 데이터 타입 확인
        assert isinstance(episode['states'], np.ndarray)
        assert isinstance(episode['actions'], np.ndarray)
        assert isinstance(episode['rewards'], np.ndarray)
        assert isinstance(episode['next_states'], np.ndarray)
        assert isinstance(episode['dones'], np.ndarray)
        assert isinstance(episode['log_probs'], np.ndarray)
        assert isinstance(episode['metadata'], dict)
        
        # 길이 일관성 확인
        episode_length = len(episode['states'])
        assert len(episode['actions']) == episode_length
        assert len(episode['rewards']) == episode_length
        assert len(episode['next_states']) == episode_length
        assert len(episode['dones']) == episode_length
        assert len(episode['log_probs']) == episode_length


def test_collect_rollouts_metadata(mock_env, mock_policy):
    """에피소드 메타데이터 수집 테스트"""
    trainer = GRPOTrainer(
        policy=mock_policy,
        env=mock_env,
        episodes_per_group=2,
        num_groups=1,
        device='cpu'
    )
    
    episodes = trainer.collect_rollouts(num_episodes=2)
    
    # 메타데이터 필드 확인
    for episode in episodes:
        metadata = episode['metadata']
        
        # 필수 필드 확인
        assert 'episode_reward' in metadata
        assert 'episode_steps' in metadata
        
        # 타입 확인
        assert isinstance(metadata['episode_reward'], (int, float))
        assert isinstance(metadata['episode_steps'], int)


def test_collect_rollouts_timesteps(mock_env, mock_policy):
    """총 타임스텝 카운팅 테스트"""
    trainer = GRPOTrainer(
        policy=mock_policy,
        env=mock_env,
        episodes_per_group=3,
        num_groups=1,
        device='cpu'
    )
    
    initial_timesteps = trainer.total_timesteps
    
    episodes = trainer.collect_rollouts(num_episodes=3)
    
    # 타임스텝이 증가했는지 확인
    assert trainer.total_timesteps > initial_timesteps
    
    # 타임스텝이 에피소드 스텝 합과 일치하는지 확인
    total_steps = sum(ep['metadata']['episode_steps'] for ep in episodes)
    assert trainer.total_timesteps == initial_timesteps + total_steps


def test_sample_action_with_policy(mock_policy):
    """정책을 사용한 행동 샘플링 테스트"""
    mock_env = Mock()
    mock_env.action_space = Mock()
    mock_env.action_space.n = 3
    
    trainer = GRPOTrainer(
        policy=mock_policy,
        env=mock_env,
        device='cpu'
    )
    
    # 상태 텐서 생성
    state = torch.randn(1, 128)
    
    # 행동 샘플링
    action, log_prob = trainer._sample_action(state)
    
    # 행동이 유효한 범위인지 확인
    assert 0 <= action < 3
    
    # 로그 확률이 음수인지 확인 (확률은 0~1이므로 로그는 음수)
    assert log_prob <= 0


def test_sample_action_without_policy():
    """정책이 없을 때 행동 샘플링 테스트"""
    mock_env = Mock()
    mock_env.action_space = Mock()
    mock_env.action_space.sample = Mock(return_value=1)
    mock_env.action_space.n = 3
    
    # get_action 메서드가 없는 정책 (하지만 to 메서드와 parameters는 있어야 함)
    # 간단한 더미 네트워크 생성
    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.randn(1))
    
    mock_policy = DummyPolicy()
    
    trainer = GRPOTrainer(
        policy=mock_policy,
        env=mock_env,
        device='cpu'
    )
    
    state = torch.randn(1, 128)
    
    # 행동 샘플링 (랜덤 행동 사용)
    action, log_prob = trainer._sample_action(state)
    
    # 랜덤 행동이 반환되었는지 확인
    assert action == 1
    
    # 균등 분포의 로그 확률 확인: log(1/3) ≈ -1.0986
    assert abs(log_prob - (-np.log(3))) < 1e-6


def test_collect_rollouts_multiple_batches(mock_env, mock_policy):
    """여러 배치의 롤아웃 수집 테스트"""
    trainer = GRPOTrainer(
        policy=mock_policy,
        env=mock_env,
        episodes_per_group=8,
        num_groups=2,
        device='cpu'
    )
    
    # 첫 번째 배치
    episodes1 = trainer.collect_rollouts(num_episodes=8)
    assert len(episodes1) == 8
    
    timesteps_after_first = trainer.total_timesteps
    
    # 두 번째 배치
    episodes2 = trainer.collect_rollouts(num_episodes=8)
    assert len(episodes2) == 8
    
    # 타임스텝이 계속 증가하는지 확인
    assert trainer.total_timesteps > timesteps_after_first


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
