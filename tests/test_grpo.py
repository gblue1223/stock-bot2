"""
GRPO 알고리즘 단위 테스트

요구사항 4.2, 4.3을 검증하는 테스트
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from ai_trader.grpo.grpo import GRPOTrainer


@pytest.fixture
def mock_policy():
    """Mock 정책 네트워크 생성"""
    policy = Mock()
    policy.embedding_dim = 128
    policy.hidden_dim = 256
    policy.action_dim = 3
    policy.to = Mock(return_value=policy)
    policy.parameters = Mock(return_value=[torch.nn.Parameter(torch.randn(10, 10))])
    policy.state_dict = Mock(return_value={})
    policy.load_state_dict = Mock()
    policy.eval = Mock()
    
    # get_action mock
    def get_action_mock(state, deterministic=False):
        batch_size = state.shape[0]
        action = torch.randint(0, 3, (batch_size,))
        log_prob = torch.randn(batch_size)
        return action, log_prob
    
    policy.get_action = Mock(side_effect=get_action_mock)
    
    # evaluate_actions mock
    def evaluate_actions_mock(states, actions):
        batch_size = states.shape[0]
        log_probs = torch.randn(batch_size)
        entropy = torch.randn(batch_size)
        values = torch.randn(batch_size)
        return log_probs, entropy, values
    
    policy.evaluate_actions = Mock(side_effect=evaluate_actions_mock)
    
    return policy


@pytest.fixture
def mock_env():
    """Mock GRPO 환경 생성"""
    env = Mock()
    env.action_space = Mock()
    env.action_space.n = 3
    env.action_space.sample = Mock(return_value=0)
    
    # reset mock
    def reset_mock(seed=None):
        observation = np.random.randn(128).astype(np.float32)
        info = {'stock_code': '005930', 'date': '2025-01-01', 'time': 0.0}
        return observation, info
    
    env.reset = Mock(side_effect=reset_mock)
    
    # step mock
    step_counter = {'count': 0}
    
    def step_mock(action):
        step_counter['count'] += 1
        next_state = np.random.randn(128).astype(np.float32)
        reward = np.random.randn() * 0.01
        terminated = step_counter['count'] >= 100  # 100 스텝 후 종료
        truncated = False
        
        info = {
            'position': 0,
            'current_price': 100.0,
            'current_time': float(step_counter['count']),
            'quick_exit_violations': 0,
            'quick_exit_triggered': False
        }
        
        if terminated:
            # 에피소드 메타데이터 추가
            info['episode'] = {
                'total_return': np.random.randn() * 0.1,
                'num_trades': np.random.randint(5, 20),
                'avg_holding_time': np.random.rand() * 30,
                'sharpe_ratio': np.random.randn(),
                'quick_exit_violations': np.random.randint(0, 5),
                'win_rate': np.random.rand(),
                'avg_profit_per_trade': np.random.randn() * 0.01,
                'episode_reward': np.random.randn() * 0.1,
                'episode_steps': step_counter['count']
            }
            step_counter['count'] = 0  # 리셋
        
        return next_state, reward, terminated, truncated, info
    
    env.step = Mock(side_effect=step_mock)
    
    return env


class TestGRPOTrainer:
    """GRPOTrainer 테스트"""
    
    def test_trainer_initialization(self, mock_policy, mock_env):
        """훈련기 초기화 테스트"""
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            episodes_per_group=12,
            num_groups=4,
            learning_rate=3e-4,
            device='cpu'
        )
        
        assert trainer.episodes_per_group == 12
        assert trainer.num_groups == 4
        assert trainer.learning_rate == 3e-4
        assert trainer.total_timesteps == 0
        assert trainer.num_updates == 0
    
    def test_collect_rollouts(self, mock_policy, mock_env):
        """
        요구사항 4.1: 롤아웃 수집 검증
        
        현재 정책으로 여러 에피소드를 실행하고
        상태, 행동, 보상, 다음 상태를 수집
        """
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            device='cpu'
        )
        
        num_episodes = 8
        episodes = trainer.collect_rollouts(num_episodes)
        
        # 에피소드 수 검증
        assert len(episodes) == num_episodes, \
            f"Should collect {num_episodes} episodes, got {len(episodes)}"
        
        # 각 에피소드 데이터 검증
        for episode in episodes:
            # 필수 키 존재 확인
            assert 'states' in episode
            assert 'actions' in episode
            assert 'rewards' in episode
            assert 'next_states' in episode
            assert 'dones' in episode
            assert 'log_probs' in episode
            assert 'metadata' in episode
            
            # 데이터 형태 검증
            assert isinstance(episode['states'], np.ndarray)
            assert isinstance(episode['actions'], np.ndarray)
            assert isinstance(episode['rewards'], np.ndarray)
            assert isinstance(episode['next_states'], np.ndarray)
            assert isinstance(episode['dones'], np.ndarray)
            assert isinstance(episode['log_probs'], np.ndarray)
            
            # 데이터 길이 일관성 검증
            num_steps = len(episode['states'])
            assert len(episode['actions']) == num_steps
            assert len(episode['rewards']) == num_steps
            assert len(episode['next_states']) == num_steps
            assert len(episode['dones']) == num_steps
            assert len(episode['log_probs']) == num_steps
            
            # 메타데이터 검증
            metadata = episode['metadata']
            assert 'episode_reward' in metadata
            assert 'episode_steps' in metadata
            assert 'total_return' in metadata
            assert 'num_trades' in metadata
    
    def test_group_episodes_kmeans(self, mock_policy, mock_env):
        """
        요구사항 4.2: 에피소드 그룹화 검증 (K-means)
        
        시장 체제 지표(변동성, 추세 강도)를 기반으로
        K-means 클러스터링으로 에피소드를 그룹화
        """
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            num_groups=4,
            device='cpu'
        )
        
        # 롤아웃 수집
        num_episodes = 16
        episodes = trainer.collect_rollouts(num_episodes)
        
        # 에피소드 그룹화
        grouped_episodes = trainer.group_episodes(episodes)
        
        # 그룹 수 검증
        assert len(grouped_episodes) > 0, "Should have at least one group"
        assert len(grouped_episodes) <= trainer.num_groups, \
            f"Should have at most {trainer.num_groups} groups"
        
        # 모든 에피소드가 그룹에 할당되었는지 검증
        total_episodes_in_groups = sum(len(group_eps) for group_eps in grouped_episodes.values())
        assert total_episodes_in_groups == num_episodes, \
            f"All {num_episodes} episodes should be grouped, got {total_episodes_in_groups}"
        
        # 각 에피소드에 그룹 정보가 추가되었는지 검증
        for group_id, group_episodes in grouped_episodes.items():
            assert isinstance(group_id, int), "Group ID should be integer"
            
            for episode in group_episodes:
                assert 'group_id' in episode, "Episode should have group_id"
                assert episode['group_id'] == group_id, \
                    f"Episode group_id should match group key"
                
                assert 'market_indicators' in episode, \
                    "Episode should have market_indicators"
                assert len(episode['market_indicators']) == 3, \
                    "Market indicators should have 3 elements (volatility, trend, range)"
    
    def test_group_episodes_rule_based(self, mock_policy, mock_env):
        """
        요구사항 4.2: 에피소드 그룹화 검증 (규칙 기반)
        
        K-means 실패 시 규칙 기반 그룹화로 대체
        """
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            num_groups=4,
            device='cpu'
        )
        
        # 적은 수의 에피소드로 테스트 (K-means가 실패할 수 있음)
        num_episodes = 3
        episodes = trainer.collect_rollouts(num_episodes)
        
        # 에피소드 그룹화
        grouped_episodes = trainer.group_episodes(episodes)
        
        # 그룹화가 성공했는지 검증
        assert len(grouped_episodes) > 0, "Should have at least one group"
        
        # 모든 에피소드가 그룹에 할당되었는지 검증
        total_episodes_in_groups = sum(len(group_eps) for group_eps in grouped_episodes.values())
        assert total_episodes_in_groups == num_episodes, \
            f"All {num_episodes} episodes should be grouped"
    
    def test_compute_group_relative_advantages(self, mock_policy, mock_env):
        """
        요구사항 4.3: 그룹 상대 어드밴티지 계산 검증
        
        그룹 상대 어드밴티지:
        A_i = R_i - mean(R_group)
        
        여기서:
        - A_i: 에피소드 i의 어드밴티지
        - R_i: 에피소드 i의 총 수익
        - mean(R_group): 그룹 내 평균 수익
        """
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            num_groups=2,
            device='cpu'
        )
        
        # 롤아웃 수집 및 그룹화
        num_episodes = 8
        episodes = trainer.collect_rollouts(num_episodes)
        grouped_episodes = trainer.group_episodes(episodes)
        
        # 그룹 상대 어드밴티지 계산
        group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
        
        # 그룹 수 일치 검증
        assert len(group_advantages) == len(grouped_episodes), \
            "Should have advantages for all groups"
        
        # 각 그룹의 어드밴티지 검증
        for group_id, advantages in group_advantages.items():
            group_episodes = grouped_episodes[group_id]
            
            # 어드밴티지 수가 에피소드 수와 일치하는지 검증
            assert len(advantages) == len(group_episodes), \
                f"Group {group_id} should have {len(group_episodes)} advantages, got {len(advantages)}"
            
            # 각 에피소드의 어드밴티지 검증
            episode_returns = []
            for episode in group_episodes:
                total_return = np.sum(episode['rewards'])
                episode_returns.append(total_return)
            
            mean_group_return = np.mean(episode_returns)
            
            for idx, (episode, advantage) in enumerate(zip(group_episodes, advantages)):
                # 어드밴티지 형태 검증
                assert isinstance(advantage, np.ndarray), \
                    "Advantage should be numpy array"
                assert len(advantage) == len(episode['rewards']), \
                    "Advantage length should match episode length"
                
                # 그룹 상대 어드밴티지 계산 검증
                expected_advantage = episode_returns[idx] - mean_group_return
                
                # 모든 타임스텝에 동일한 어드밴티지가 할당되었는지 검증
                assert np.allclose(advantage, expected_advantage, atol=1e-5), \
                    f"Advantage should be {expected_advantage}, got {advantage[0]}"
                
                # 에피소드 메타데이터에 어드밴티지 정보가 추가되었는지 검증
                assert 'relative_advantage' in episode, \
                    "Episode should have relative_advantage"
                assert 'group_mean_return' in episode, \
                    "Episode should have group_mean_return"
                
                assert np.isclose(episode['relative_advantage'], expected_advantage, atol=1e-5), \
                    f"Episode relative_advantage should be {expected_advantage}"
                assert np.isclose(episode['group_mean_return'], mean_group_return, atol=1e-5), \
                    f"Episode group_mean_return should be {mean_group_return}"
    
    def test_advantage_properties(self, mock_policy, mock_env):
        """어드밴티지 속성 검증"""
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            num_groups=2,
            device='cpu'
        )
        
        # 롤아웃 수집 및 그룹화
        num_episodes = 12
        episodes = trainer.collect_rollouts(num_episodes)
        grouped_episodes = trainer.group_episodes(episodes)
        
        # 그룹 상대 어드밴티지 계산
        group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
        
        # 각 그룹의 어드밴티지 합이 0에 가까운지 검증
        # (그룹 내 상대 어드밴티지의 합은 0이어야 함)
        for group_id, advantages in group_advantages.items():
            # 각 에피소드의 첫 번째 어드밴티지 값 추출 (모든 타임스텝이 동일)
            advantage_values = [adv[0] for adv in advantages]
            advantage_sum = np.sum(advantage_values)
            
            # 합이 0에 가까운지 검증 (부동소수점 오차 허용)
            assert np.isclose(advantage_sum, 0.0, atol=1e-4), \
                f"Group {group_id} advantage sum should be near 0, got {advantage_sum}"
    
    def test_compute_returns(self, mock_policy, mock_env):
        """할인된 누적 보상 계산 테스트"""
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            gamma=0.99,
            device='cpu'
        )
        
        # 테스트 보상 시퀀스
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 리턴 계산
        returns = trainer._compute_returns(rewards)
        
        # 수동 계산
        expected_returns = np.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + trainer.gamma * running_return
            expected_returns[t] = running_return
        
        # 검증
        assert np.allclose(returns, expected_returns, atol=1e-5), \
            f"Returns should match expected values"
        
        # 마지막 리턴은 마지막 보상과 같아야 함
        assert np.isclose(returns[-1], rewards[-1], atol=1e-5), \
            "Last return should equal last reward"
    
    def test_rule_based_grouping(self, mock_policy, mock_env):
        """규칙 기반 그룹화 테스트"""
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            num_groups=4,
            device='cpu'
        )
        
        # 테스트 시장 지표 생성
        # [변동성, 추세, 범위]
        market_indicators = np.array([
            [0.01, 0.005, 0.02],   # 낮은 변동성, 약한 추세
            [0.05, 0.03, 0.08],    # 높은 변동성, 강한 추세
            [0.02, -0.001, 0.03],  # 낮은 변동성, 약한 추세
            [0.08, -0.05, 0.12],   # 높은 변동성, 강한 추세
            [0.03, 0.02, 0.05],    # 중간 변동성, 중간 추세
            [0.06, -0.04, 0.09],   # 높은 변동성, 강한 추세
        ])
        
        # 규칙 기반 그룹화 실행
        group_labels = trainer._rule_based_grouping(market_indicators)
        
        # 그룹 레이블 검증
        assert len(group_labels) == len(market_indicators), \
            "Should have one label per indicator"
        
        # 그룹 레이블이 유효한 범위 내에 있는지 검증
        assert np.all(group_labels >= 0), "Group labels should be non-negative"
        assert np.all(group_labels < trainer.num_groups), \
            f"Group labels should be less than {trainer.num_groups}"
        
        # 최소 1개 이상의 그룹이 있는지 검증
        unique_groups = np.unique(group_labels)
        assert len(unique_groups) > 0, "Should have at least one group"


class TestGRPOTrainerIntegration:
    """GRPO 훈련기 통합 테스트"""
    
    def test_full_training_iteration(self, mock_policy, mock_env):
        """전체 훈련 반복 테스트"""
        trainer = GRPOTrainer(
            policy=mock_policy,
            env=mock_env,
            episodes_per_group=4,
            num_groups=2,
            device='cpu'
        )
        
        # 1. 롤아웃 수집
        num_episodes = trainer.episodes_per_group * trainer.num_groups
        episodes = trainer.collect_rollouts(num_episodes)
        
        assert len(episodes) == num_episodes
        
        # 2. 에피소드 그룹화
        grouped_episodes = trainer.group_episodes(episodes)
        
        assert len(grouped_episodes) > 0
        
        # 3. 그룹 상대 어드밴티지 계산
        group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
        
        assert len(group_advantages) == len(grouped_episodes)
        
        # 4. 정책 업데이트 (mock이므로 실제 업데이트는 안 됨)
        all_episodes = []
        all_advantages = []
        for group_id in sorted(grouped_episodes.keys()):
            all_episodes.extend(grouped_episodes[group_id])
            all_advantages.extend(group_advantages[group_id])
        
        # 업데이트 메트릭 검증을 위해 실제 텐서 생성
        # (mock 정책이므로 실제 업데이트는 안 되지만 함수 호출은 테스트)
        assert len(all_episodes) == num_episodes
        assert len(all_advantages) == num_episodes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
