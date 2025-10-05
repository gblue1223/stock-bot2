"""
GRPO TensorBoard 로깅 테스트

Task 7.6: TensorBoard 로깅 구현 검증
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from ai_trader.grpo.grpo import GRPOTrainer


class MockPolicy(nn.Module):
    """테스트용 모의 정책"""
    
    def __init__(self, embedding_dim=128, hidden_dim=256, action_dim=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """행동 확률 분포 반환"""
        h = torch.relu(self.fc1(x))
        logits = self.fc2(h)
        return torch.softmax(logits, dim=-1)
    
    def evaluate_actions(self, states, actions):
        """행동 평가"""
        h = torch.relu(self.fc1(states))
        logits = self.fc2(h)
        probs = torch.softmax(logits, dim=-1)
        
        # 로그 확률
        log_probs = torch.log(probs + 1e-8)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # 엔트로피
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # 가치
        values = self.value_head(h).squeeze(-1)
        
        return action_log_probs, entropy, values


def test_log_metrics_group_level():
    """그룹 수준 메트릭 로깅 테스트"""
    
    # Mock 환경 및 정책 생성
    mock_env = Mock()
    mock_env.action_space = Mock()
    mock_env.action_space.n = 3
    
    policy = MockPolicy(embedding_dim=128, hidden_dim=256, action_dim=3)
    
    # GRPOTrainer 생성 (TensorBoard writer는 Mock)
    with patch('ai_trader.grpo.grpo.SummaryWriter') as mock_writer_class:
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer
        
        trainer = GRPOTrainer(
            policy=policy,
            env=mock_env,
            episodes_per_group=4,
            num_groups=2,
            device='cpu',
            tensorboard_log_dir='test_logs'
        )
        
        # 테스트 에피소드 데이터 생성
        episodes = []
        for i in range(8):
            episode = {
                'states': np.random.randn(10, 128).astype(np.float32),
                'actions': np.random.randint(0, 3, 10),
                'rewards': np.random.randn(10).astype(np.float32),
                'next_states': np.random.randn(10, 128).astype(np.float32),
                'dones': np.zeros(10, dtype=bool),
                'log_probs': np.random.randn(10).astype(np.float32),
                'metadata': {
                    'episode_reward': np.random.randn(),
                    'episode_steps': 10,
                    'win_rate': np.random.rand(),
                    'avg_holding_time': np.random.rand() * 30,
                    'quick_exit_violations': np.random.randint(0, 5),
                    'num_trades': np.random.randint(1, 10),
                    'sharpe_ratio': np.random.randn()
                }
            }
            episodes.append(episode)
        
        # 에피소드 그룹화
        grouped_episodes = {
            0: episodes[:4],
            1: episodes[4:]
        }
        
        # 각 에피소드에 그룹 정보 추가
        for i, ep in enumerate(episodes[:4]):
            ep['group_id'] = 0
        for i, ep in enumerate(episodes[4:]):
            ep['group_id'] = 1
        
        # 업데이트 메트릭
        update_metrics = {
            'policy_loss': 0.5,
            'value_loss': 0.3,
            'entropy': 1.0,
            'kl_divergence': 0.01,
            'clip_fraction': 0.2
        }
        
        # 참조 정책 설정
        trainer.reference_policy = MockPolicy(embedding_dim=128, hidden_dim=256, action_dim=3)
        
        # 메트릭 로깅 실행
        trainer._log_metrics(
            iteration=0,
            episodes=episodes,
            grouped_episodes=grouped_episodes,
            update_metrics=update_metrics
        )
        
        # TensorBoard writer의 add_scalar 호출 확인
        assert mock_writer.add_scalar.called
        
        # 호출된 메트릭 이름 수집
        called_metrics = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
        
        # 전체 메트릭 확인
        assert 'train/mean_reward' in called_metrics
        assert 'train/win_rate' in called_metrics
        assert 'train/avg_holding_time' in called_metrics
        assert 'train/quick_exit_violations_total' in called_metrics
        assert 'train/quick_exit_violations_mean' in called_metrics
        
        # 정책 업데이트 메트릭 확인
        assert 'train/policy_loss' in called_metrics
        assert 'train/entropy' in called_metrics
        assert 'train/kl_divergence' in called_metrics
        
        # 그룹 수준 메트릭 확인
        assert 'group_0/mean_return' in called_metrics
        assert 'group_0/policy_entropy' in called_metrics
        assert 'group_0/kl_divergence' in called_metrics
        
        assert 'group_1/mean_return' in called_metrics
        assert 'group_1/policy_entropy' in called_metrics
        assert 'group_1/kl_divergence' in called_metrics
        
        print("✓ 그룹 수준 메트릭 로깅 테스트 통과")


def test_log_metrics_overall():
    """전체 메트릭 로깅 테스트"""
    
    # Mock 환경 및 정책 생성
    mock_env = Mock()
    mock_env.action_space = Mock()
    mock_env.action_space.n = 3
    
    policy = MockPolicy(embedding_dim=128, hidden_dim=256, action_dim=3)
    
    # GRPOTrainer 생성 (TensorBoard writer는 Mock)
    with patch('ai_trader.grpo.grpo.SummaryWriter') as mock_writer_class:
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer
        
        trainer = GRPOTrainer(
            policy=policy,
            env=mock_env,
            device='cpu',
            tensorboard_log_dir='test_logs'
        )
        
        # 테스트 에피소드 데이터 생성
        episodes = []
        for i in range(4):
            episode = {
                'states': np.random.randn(10, 128).astype(np.float32),
                'actions': np.random.randint(0, 3, 10),
                'rewards': np.random.randn(10).astype(np.float32),
                'next_states': np.random.randn(10, 128).astype(np.float32),
                'dones': np.zeros(10, dtype=bool),
                'log_probs': np.random.randn(10).astype(np.float32),
                'metadata': {
                    'episode_reward': 0.5 + i * 0.1,
                    'episode_steps': 10,
                    'win_rate': 0.6,
                    'avg_holding_time': 20.0,
                    'quick_exit_violations': 2,
                    'num_trades': 5,
                    'sharpe_ratio': 1.5
                }
            }
            episodes.append(episode)
        
        grouped_episodes = {0: episodes}
        for ep in episodes:
            ep['group_id'] = 0
        
        update_metrics = {
            'policy_loss': 0.5,
            'value_loss': 0.3,
            'entropy': 1.0,
            'kl_divergence': 0.01
        }
        
        # 메트릭 로깅 실행
        trainer._log_metrics(
            iteration=0,
            episodes=episodes,
            grouped_episodes=grouped_episodes,
            update_metrics=update_metrics
        )
        
        # 호출된 메트릭 확인
        called_metrics = {call[0][0]: call[0][1] for call in mock_writer.add_scalar.call_args_list}
        
        # 평균 보상 확인
        assert 'train/mean_reward' in called_metrics
        expected_mean_reward = np.mean([0.5, 0.6, 0.7, 0.8])
        assert abs(called_metrics['train/mean_reward'] - expected_mean_reward) < 0.01
        
        # 승률 확인
        assert 'train/win_rate' in called_metrics
        assert called_metrics['train/win_rate'] == 0.6
        
        # 평균 보유 시간 확인
        assert 'train/avg_holding_time' in called_metrics
        assert called_metrics['train/avg_holding_time'] == 20.0
        
        # 빠른 손절 룰 위반 횟수 확인
        assert 'train/quick_exit_violations_total' in called_metrics
        assert called_metrics['train/quick_exit_violations_total'] == 8  # 2 * 4 episodes
        
        print("✓ 전체 메트릭 로깅 테스트 통과")


def test_log_metrics_empty_episodes():
    """빈 에피소드 처리 테스트"""
    
    mock_env = Mock()
    mock_env.action_space = Mock()
    mock_env.action_space.n = 3
    
    policy = MockPolicy(embedding_dim=128, hidden_dim=256, action_dim=3)
    
    with patch('ai_trader.grpo.grpo.SummaryWriter') as mock_writer_class:
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer
        
        trainer = GRPOTrainer(
            policy=policy,
            env=mock_env,
            device='cpu',
            tensorboard_log_dir='test_logs'
        )
        
        # 빈 에피소드 리스트
        episodes = []
        grouped_episodes = {}
        update_metrics = {}
        
        # 에러 없이 실행되어야 함
        try:
            trainer._log_metrics(
                iteration=0,
                episodes=episodes,
                grouped_episodes=grouped_episodes,
                update_metrics=update_metrics
            )
            print("✓ 빈 에피소드 처리 테스트 통과")
        except Exception as e:
            pytest.fail(f"빈 에피소드 처리 중 에러 발생: {e}")


if __name__ == '__main__':
    print("Task 7.6: TensorBoard 로깅 구현 테스트")
    print("=" * 60)
    
    test_log_metrics_group_level()
    test_log_metrics_overall()
    test_log_metrics_empty_episodes()
    
    print("=" * 60)
    print("모든 테스트 통과!")
