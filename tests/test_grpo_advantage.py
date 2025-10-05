"""
GRPO 그룹 상대 어드밴티지 계산 테스트

Task 7.4의 요구사항을 검증합니다:
- 각 그룹의 평균 수익 계산
- 그룹 내 상대 어드밴티지: A_i = R_i - mean(R_group)
- 요구사항: 4.3
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from ai_trader.grpo.grpo import GRPOTrainer


class MockPolicy(nn.Module):
    """테스트용 간단한 정책 네트워크"""
    
    def __init__(self, embedding_dim: int = 128, num_actions: int = 3):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc(x), dim=-1)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """행동 샘플링"""
        probs = self.forward(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
        
        return action, log_prob


class DummyEnv:
    """테스트용 더미 환경"""
    
    def __init__(self):
        self.action_space = type('obj', (object,), {
            'n': 3,
            'sample': lambda: np.random.randint(0, 3)
        })()
    
    def close(self):
        pass


def create_grouped_episodes_with_known_returns():
    """
    알려진 수익을 가진 그룹화된 에피소드 생성
    
    그룹 0: 수익 [10, 20, 30] -> 평균 20
    그룹 1: 수익 [-5, 0, 5] -> 평균 0
    그룹 2: 수익 [100, 200] -> 평균 150
    """
    grouped_episodes = {}
    
    # 그룹 0: 평균 수익 20
    group_0_returns = [10.0, 20.0, 30.0]
    grouped_episodes[0] = []
    for ret in group_0_returns:
        # 보상을 분산시켜 총합이 ret이 되도록
        num_steps = 10
        rewards = np.full(num_steps, ret / num_steps, dtype=np.float32)
        
        episode = {
            'states': np.random.randn(num_steps, 128),
            'actions': np.random.randint(0, 3, size=num_steps),
            'rewards': rewards,
            'next_states': np.random.randn(num_steps, 128),
            'dones': np.zeros(num_steps, dtype=bool),
            'log_probs': np.random.randn(num_steps),
            'metadata': {
                'episode_reward': ret,
                'episode_steps': num_steps
            }
        }
        grouped_episodes[0].append(episode)
    
    # 그룹 1: 평균 수익 0
    group_1_returns = [-5.0, 0.0, 5.0]
    grouped_episodes[1] = []
    for ret in group_1_returns:
        num_steps = 10
        rewards = np.full(num_steps, ret / num_steps, dtype=np.float32)
        
        episode = {
            'states': np.random.randn(num_steps, 128),
            'actions': np.random.randint(0, 3, size=num_steps),
            'rewards': rewards,
            'next_states': np.random.randn(num_steps, 128),
            'dones': np.zeros(num_steps, dtype=bool),
            'log_probs': np.random.randn(num_steps),
            'metadata': {
                'episode_reward': ret,
                'episode_steps': num_steps
            }
        }
        grouped_episodes[1].append(episode)
    
    # 그룹 2: 평균 수익 150
    group_2_returns = [100.0, 200.0]
    grouped_episodes[2] = []
    for ret in group_2_returns:
        num_steps = 10
        rewards = np.full(num_steps, ret / num_steps, dtype=np.float32)
        
        episode = {
            'states': np.random.randn(num_steps, 128),
            'actions': np.random.randint(0, 3, size=num_steps),
            'rewards': rewards,
            'next_states': np.random.randn(num_steps, 128),
            'dones': np.zeros(num_steps, dtype=bool),
            'log_probs': np.random.randn(num_steps),
            'metadata': {
                'episode_reward': ret,
                'episode_steps': num_steps
            }
        }
        grouped_episodes[2].append(episode)
    
    return grouped_episodes


def test_requirement_4_3_group_mean_return_calculation():
    """
    요구사항 4.3 검증: 각 그룹의 평균 수익 계산
    
    WHEN 어드밴티지가 계산되면 THEN 시스템은 각 에피소드의 수익을 
    그룹 평균 수익과 비교하여 그룹 상대 어드밴티지를 계산해야 합니다
    """
    # 트레이너 생성
    policy = MockPolicy()
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=3,
        num_groups=3,
        device='cpu'
    )
    
    # 알려진 수익을 가진 그룹화된 에피소드 생성
    grouped_episodes = create_grouped_episodes_with_known_returns()
    
    # 그룹 상대 어드밴티지 계산
    group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
    
    # 검증: 그룹 수가 일치하는지 확인
    assert len(group_advantages) == 3, f"Expected 3 groups, got {len(group_advantages)}"
    
    # 검증: 각 그룹의 평균 수익이 올바르게 계산되었는지 확인
    # 그룹 0: 평균 20
    group_0_episodes = grouped_episodes[0]
    expected_mean_0 = 20.0
    
    for episode in group_0_episodes:
        assert 'group_mean_return' in episode, "Episode missing group_mean_return"
        assert np.isclose(episode['group_mean_return'], expected_mean_0, atol=1e-5), \
            f"Group 0 mean return mismatch: expected {expected_mean_0}, got {episode['group_mean_return']}"
    
    # 그룹 1: 평균 0
    group_1_episodes = grouped_episodes[1]
    expected_mean_1 = 0.0
    
    for episode in group_1_episodes:
        assert np.isclose(episode['group_mean_return'], expected_mean_1, atol=1e-5), \
            f"Group 1 mean return mismatch: expected {expected_mean_1}, got {episode['group_mean_return']}"
    
    # 그룹 2: 평균 150
    group_2_episodes = grouped_episodes[2]
    expected_mean_2 = 150.0
    
    for episode in group_2_episodes:
        assert np.isclose(episode['group_mean_return'], expected_mean_2, atol=1e-5), \
            f"Group 2 mean return mismatch: expected {expected_mean_2}, got {episode['group_mean_return']}"
    
    print("✓ Group mean returns calculated correctly")


def test_requirement_4_3_relative_advantage_formula():
    """
    요구사항 4.3 검증: 그룹 내 상대 어드밴티지 A_i = R_i - mean(R_group)
    
    WHEN 어드밴티지가 계산되면 THEN 시스템은 A_i = R_i - mean(R_group) 공식을 
    사용하여 그룹 상대 어드밴티지를 계산해야 합니다
    """
    # 트레이너 생성
    policy = MockPolicy()
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=3,
        num_groups=3,
        device='cpu'
    )
    
    # 알려진 수익을 가진 그룹화된 에피소드 생성
    grouped_episodes = create_grouped_episodes_with_known_returns()
    
    # 그룹 상대 어드밴티지 계산
    group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
    
    # 검증: 그룹 0
    # 수익: [10, 20, 30], 평균: 20
    # 어드밴티지: [10-20=-10, 20-20=0, 30-20=10]
    expected_advantages_0 = [-10.0, 0.0, 10.0]
    
    for i, episode in enumerate(grouped_episodes[0]):
        assert 'relative_advantage' in episode, "Episode missing relative_advantage"
        expected_adv = expected_advantages_0[i]
        actual_adv = episode['relative_advantage']
        
        assert np.isclose(actual_adv, expected_adv, atol=1e-5), \
            f"Group 0 episode {i} advantage mismatch: expected {expected_adv}, got {actual_adv}"
        
        # 어드밴티지 배열도 확인
        advantage_array = group_advantages[0][i]
        assert np.all(np.isclose(advantage_array, expected_adv, atol=1e-5)), \
            f"Group 0 episode {i} advantage array mismatch"
    
    # 검증: 그룹 1
    # 수익: [-5, 0, 5], 평균: 0
    # 어드밴티지: [-5-0=-5, 0-0=0, 5-0=5]
    expected_advantages_1 = [-5.0, 0.0, 5.0]
    
    for i, episode in enumerate(grouped_episodes[1]):
        expected_adv = expected_advantages_1[i]
        actual_adv = episode['relative_advantage']
        
        assert np.isclose(actual_adv, expected_adv, atol=1e-5), \
            f"Group 1 episode {i} advantage mismatch: expected {expected_adv}, got {actual_adv}"
    
    # 검증: 그룹 2
    # 수익: [100, 200], 평균: 150
    # 어드밴티지: [100-150=-50, 200-150=50]
    expected_advantages_2 = [-50.0, 50.0]
    
    for i, episode in enumerate(grouped_episodes[2]):
        expected_adv = expected_advantages_2[i]
        actual_adv = episode['relative_advantage']
        
        assert np.isclose(actual_adv, expected_adv, atol=1e-5), \
            f"Group 2 episode {i} advantage mismatch: expected {expected_adv}, got {actual_adv}"
    
    print("✓ Relative advantages calculated correctly using A_i = R_i - mean(R_group)")


def test_advantage_array_shape():
    """
    어드밴티지 배열의 형태가 올바른지 검증
    
    각 에피소드의 어드밴티지 배열은 에피소드의 타임스텝 수와 일치해야 합니다
    """
    policy = MockPolicy()
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=3,
        num_groups=3,
        device='cpu'
    )
    
    grouped_episodes = create_grouped_episodes_with_known_returns()
    group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
    
    # 각 그룹의 어드밴티지 배열 검증
    for group_id, advantages in group_advantages.items():
        episodes = grouped_episodes[group_id]
        
        assert len(advantages) == len(episodes), \
            f"Group {group_id}: number of advantage arrays mismatch"
        
        for i, (episode, advantage_array) in enumerate(zip(episodes, advantages)):
            num_steps = len(episode['rewards'])
            
            assert len(advantage_array) == num_steps, \
                f"Group {group_id} episode {i}: advantage array length mismatch. " \
                f"Expected {num_steps}, got {len(advantage_array)}"
            
            # 모든 타임스텝에 동일한 어드밴티지가 할당되었는지 확인
            assert np.all(advantage_array == advantage_array[0]), \
                f"Group {group_id} episode {i}: advantage values should be uniform"
    
    print("✓ Advantage array shapes are correct")


def test_advantage_sum_is_zero_per_group():
    """
    각 그룹 내에서 어드밴티지의 합이 0에 가까운지 검증
    
    그룹 상대 어드밴티지는 그룹 평균을 기준으로 하므로,
    그룹 내 모든 에피소드의 어드밴티지 합은 0이어야 합니다
    """
    policy = MockPolicy()
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=3,
        num_groups=3,
        device='cpu'
    )
    
    grouped_episodes = create_grouped_episodes_with_known_returns()
    group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
    
    # 각 그룹의 어드밴티지 합 검증
    for group_id, episodes in grouped_episodes.items():
        # 그룹 내 모든 에피소드의 상대 어드밴티지 합
        advantage_sum = sum(ep['relative_advantage'] for ep in episodes)
        
        # 부동소수점 오차를 고려하여 0에 가까운지 확인
        assert np.isclose(advantage_sum, 0.0, atol=1e-4), \
            f"Group {group_id}: advantage sum should be close to 0, got {advantage_sum}"
    
    print("✓ Advantage sums are zero within each group")


def test_edge_case_single_episode_per_group():
    """
    에지 케이스: 그룹당 단일 에피소드
    
    그룹에 에피소드가 하나만 있는 경우, 어드밴티지는 0이어야 합니다
    """
    policy = MockPolicy()
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=1,
        num_groups=2,
        device='cpu'
    )
    
    # 각 그룹에 에피소드 하나씩
    grouped_episodes = {
        0: [{
            'states': np.random.randn(10, 128),
            'actions': np.random.randint(0, 3, size=10),
            'rewards': np.full(10, 1.0, dtype=np.float32),
            'next_states': np.random.randn(10, 128),
            'dones': np.zeros(10, dtype=bool),
            'log_probs': np.random.randn(10),
            'metadata': {'episode_reward': 10.0, 'episode_steps': 10}
        }],
        1: [{
            'states': np.random.randn(10, 128),
            'actions': np.random.randint(0, 3, size=10),
            'rewards': np.full(10, -0.5, dtype=np.float32),
            'next_states': np.random.randn(10, 128),
            'dones': np.zeros(10, dtype=bool),
            'log_probs': np.random.randn(10),
            'metadata': {'episode_reward': -5.0, 'episode_steps': 10}
        }]
    }
    
    group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
    
    # 각 그룹의 단일 에피소드 어드밴티지는 0이어야 함
    for group_id, episodes in grouped_episodes.items():
        assert len(episodes) == 1, f"Group {group_id} should have 1 episode"
        
        episode = episodes[0]
        assert 'relative_advantage' in episode
        assert np.isclose(episode['relative_advantage'], 0.0, atol=1e-5), \
            f"Group {group_id}: single episode advantage should be 0, got {episode['relative_advantage']}"
    
    print("✓ Single episode per group has zero advantage")


def test_advantage_metadata_added():
    """
    에피소드 메타데이터에 어드밴티지 정보가 추가되는지 검증
    """
    policy = MockPolicy()
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=3,
        num_groups=3,
        device='cpu'
    )
    
    grouped_episodes = create_grouped_episodes_with_known_returns()
    group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
    
    # 모든 에피소드에 메타데이터가 추가되었는지 확인
    for group_id, episodes in grouped_episodes.items():
        for episode in episodes:
            # relative_advantage 필드 확인
            assert 'relative_advantage' in episode, \
                f"Group {group_id}: episode missing 'relative_advantage'"
            
            # group_mean_return 필드 확인
            assert 'group_mean_return' in episode, \
                f"Group {group_id}: episode missing 'group_mean_return'"
            
            # 값이 숫자인지 확인
            assert isinstance(episode['relative_advantage'], (int, float, np.number)), \
                f"Group {group_id}: relative_advantage should be numeric"
            
            assert isinstance(episode['group_mean_return'], (int, float, np.number)), \
                f"Group {group_id}: group_mean_return should be numeric"
    
    print("✓ Advantage metadata added to episodes")


def test_different_episode_lengths():
    """
    다양한 길이의 에피소드에 대한 어드밴티지 계산 검증
    """
    policy = MockPolicy()
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=3,
        num_groups=1,
        device='cpu'
    )
    
    # 다양한 길이의 에피소드 생성
    grouped_episodes = {
        0: [
            {
                'states': np.random.randn(5, 128),
                'actions': np.random.randint(0, 3, size=5),
                'rewards': np.full(5, 2.0, dtype=np.float32),  # 총 10
                'next_states': np.random.randn(5, 128),
                'dones': np.zeros(5, dtype=bool),
                'log_probs': np.random.randn(5),
                'metadata': {'episode_reward': 10.0, 'episode_steps': 5}
            },
            {
                'states': np.random.randn(10, 128),
                'actions': np.random.randint(0, 3, size=10),
                'rewards': np.full(10, 2.0, dtype=np.float32),  # 총 20
                'next_states': np.random.randn(10, 128),
                'dones': np.zeros(10, dtype=bool),
                'log_probs': np.random.randn(10),
                'metadata': {'episode_reward': 20.0, 'episode_steps': 10}
            },
            {
                'states': np.random.randn(15, 128),
                'actions': np.random.randint(0, 3, size=15),
                'rewards': np.full(15, 2.0, dtype=np.float32),  # 총 30
                'next_states': np.random.randn(15, 128),
                'dones': np.zeros(15, dtype=bool),
                'log_probs': np.random.randn(15),
                'metadata': {'episode_reward': 30.0, 'episode_steps': 15}
            }
        ]
    }
    
    group_advantages = trainer.compute_group_relative_advantages(grouped_episodes)
    
    # 평균 수익: (10 + 20 + 30) / 3 = 20
    # 어드밴티지: [-10, 0, 10]
    expected_advantages = [-10.0, 0.0, 10.0]
    
    for i, episode in enumerate(grouped_episodes[0]):
        # 어드밴티지 값 확인
        assert np.isclose(episode['relative_advantage'], expected_advantages[i], atol=1e-5)
        
        # 어드밴티지 배열 길이 확인
        advantage_array = group_advantages[0][i]
        expected_length = episode['metadata']['episode_steps']
        assert len(advantage_array) == expected_length, \
            f"Episode {i}: expected length {expected_length}, got {len(advantage_array)}"
    
    print("✓ Advantages calculated correctly for different episode lengths")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
