"""
GRPO 에피소드 그룹화 테스트

Task 7.3의 요구사항을 검증합니다:
- 시장 체제 지표 추출 (변동성, 추세 강도)
- K-means 클러스터링 또는 규칙 기반 그룹화
- 그룹별 에피소드 분류
- 요구사항: 4.2
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.env import GRPOScalpingEnv


class MockEmbeddingModel(nn.Module):
    """테스트용 간단한 임베딩 모델"""
    
    def __init__(self, input_dim: int = 60, embedding_dim: int = 128, seq_len: int = 60):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.fc = nn.Linear(input_dim * seq_len, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, embedding_dim)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)


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


def create_mock_episodes(num_episodes: int = 20) -> list:
    """
    테스트용 모의 에피소드 생성
    
    다양한 시장 상황을 시뮬레이션:
    - 낮은 변동성, 횡보
    - 낮은 변동성, 추세
    - 높은 변동성, 횡보
    - 높은 변동성, 추세
    """
    episodes = []
    
    for i in range(num_episodes):
        # 에피소드 타입 결정
        episode_type = i % 4
        
        if episode_type == 0:
            # 낮은 변동성, 횡보
            rewards = np.random.normal(0.0, 0.001, size=50)
        elif episode_type == 1:
            # 낮은 변동성, 상승 추세
            rewards = np.random.normal(0.002, 0.001, size=50)
        elif episode_type == 2:
            # 높은 변동성, 횡보
            rewards = np.random.normal(0.0, 0.01, size=50)
        else:
            # 높은 변동성, 하락 추세
            rewards = np.random.normal(-0.002, 0.01, size=50)
        
        episode = {
            'states': np.random.randn(50, 128),
            'actions': np.random.randint(0, 3, size=50),
            'rewards': rewards,
            'next_states': np.random.randn(50, 128),
            'dones': np.zeros(50, dtype=bool),
            'log_probs': np.random.randn(50),
            'metadata': {
                'episode_reward': np.sum(rewards),
                'episode_steps': 50,
                'num_trades': np.random.randint(1, 10),
                'avg_holding_time': np.random.uniform(5.0, 30.0),
                'win_rate': np.random.uniform(0.3, 0.7),
                'quick_exit_violations': np.random.randint(0, 5)
            }
        }
        
        episodes.append(episode)
    
    return episodes


def test_requirement_4_2_market_regime_extraction():
    """
    요구사항 4.2 검증: 시장 체제 지표 추출
    
    WHEN 에피소드가 그룹화되면 THEN 시스템은 시장 체제 지표(변동성 분위수, 추세 강도)를 
    기반으로 클러스터링 또는 규칙 기반 그룹화를 사용해야 합니다
    """
    # 모의 환경 및 정책 생성
    embedding_model = MockEmbeddingModel()
    policy = MockPolicy()
    
    # 더미 환경 (실제 DB 연결 없이)
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'n': 3, 'sample': lambda: np.random.randint(0, 3)})()
        
        def close(self):
            pass
    
    env = DummyEnv()
    
    # GRPOTrainer 생성
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=5,
        num_groups=4,
        device='cpu'
    )
    
    # 모의 에피소드 생성
    episodes = create_mock_episodes(num_episodes=20)
    
    # 에피소드 그룹화
    grouped_episodes = trainer.group_episodes(episodes)
    
    # 그룹화 결과 검증
    assert len(grouped_episodes) > 0, "No groups created"
    assert len(grouped_episodes) <= 4, "Too many groups created"
    
    # 모든 에피소드가 그룹에 할당되었는지 확인
    total_episodes = sum(len(group) for group in grouped_episodes.values())
    assert total_episodes == 20, f"Expected 20 episodes, got {total_episodes}"
    
    # 각 에피소드에 시장 지표가 추가되었는지 확인
    for group_id, group_episodes in grouped_episodes.items():
        for episode in group_episodes:
            assert 'group_id' in episode, "Episode missing group_id"
            assert 'market_indicators' in episode, "Episode missing market_indicators"
            
            # 시장 지표가 3개 (변동성, 추세 강도, 범위)인지 확인
            indicators = episode['market_indicators']
            assert len(indicators) == 3, f"Expected 3 indicators, got {len(indicators)}"
            
            # 변동성, 추세 강도, 범위가 유효한 값인지 확인
            volatility, trend_strength, reward_range = indicators
            assert volatility >= 0, "Volatility should be non-negative"
            assert isinstance(trend_strength, (int, float)), "Trend strength should be numeric"
            assert reward_range >= 0, "Reward range should be non-negative"
    
    print(f"Successfully grouped {total_episodes} episodes into {len(grouped_episodes)} groups")
    for group_id, group_episodes in grouped_episodes.items():
        print(f"  Group {group_id}: {len(group_episodes)} episodes")


def test_kmeans_clustering():
    """
    K-means 클러스터링 동작 검증
    """
    embedding_model = MockEmbeddingModel()
    policy = MockPolicy()
    
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'n': 3, 'sample': lambda: np.random.randint(0, 3)})()
        
        def close(self):
            pass
    
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=5,
        num_groups=4,
        device='cpu'
    )
    
    # 충분한 수의 에피소드 생성 (K-means가 작동하도록)
    episodes = create_mock_episodes(num_episodes=40)
    
    # 그룹화
    grouped_episodes = trainer.group_episodes(episodes)
    
    # K-means가 작동했는지 확인 (4개 그룹 생성)
    assert len(grouped_episodes) == 4, f"Expected 4 groups, got {len(grouped_episodes)}"
    
    # 각 그룹의 시장 지표 평균 계산
    group_stats = {}
    for group_id, group_episodes in grouped_episodes.items():
        indicators = np.array([ep['market_indicators'] for ep in group_episodes])
        mean_indicators = np.mean(indicators, axis=0)
        
        group_stats[group_id] = {
            'size': len(group_episodes),
            'mean_volatility': mean_indicators[0],
            'mean_trend': mean_indicators[1],
            'mean_range': mean_indicators[2]
        }
    
    print("\nGroup statistics:")
    for group_id, stats in group_stats.items():
        print(f"  Group {group_id}: size={stats['size']}, "
              f"volatility={stats['mean_volatility']:.6f}, "
              f"trend={stats['mean_trend']:.6f}, "
              f"range={stats['mean_range']:.6f}")
    
    # 그룹 간 차이가 있는지 확인 (완벽하게 다를 필요는 없지만 어느 정도 차이가 있어야 함)
    volatilities = [stats['mean_volatility'] for stats in group_stats.values()]
    assert np.std(volatilities) > 0, "Groups should have different volatility characteristics"


def test_rule_based_grouping_fallback():
    """
    규칙 기반 그룹화 대체 메커니즘 검증
    """
    embedding_model = MockEmbeddingModel()
    policy = MockPolicy()
    
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'n': 3, 'sample': lambda: np.random.randint(0, 3)})()
        
        def close(self):
            pass
    
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=5,
        num_groups=4,
        device='cpu'
    )
    
    # 규칙 기반 그룹화 직접 테스트
    market_indicators = np.array([
        [0.001, 0.002, 0.003],   # 낮은 변동성, 양의 추세
        [0.001, -0.002, 0.003],  # 낮은 변동성, 음의 추세
        [0.01, 0.002, 0.03],     # 높은 변동성, 양의 추세
        [0.01, -0.002, 0.03],    # 높은 변동성, 음의 추세
        [0.001, 0.0, 0.002],     # 낮은 변동성, 횡보
        [0.01, 0.0, 0.02],       # 높은 변동성, 횡보
    ])
    
    group_labels = trainer._rule_based_grouping(market_indicators)
    
    # 그룹 레이블이 올바른 범위인지 확인
    assert len(group_labels) == len(market_indicators)
    assert all(0 <= label < 4 for label in group_labels), "Group labels out of range"
    
    # 최소 2개 이상의 그룹이 생성되었는지 확인
    unique_groups = len(set(group_labels))
    assert unique_groups >= 2, f"Expected at least 2 groups, got {unique_groups}"
    
    print(f"\nRule-based grouping results: {group_labels}")
    print(f"Number of unique groups: {unique_groups}")


def test_edge_case_single_episode():
    """
    에지 케이스: 단일 에피소드 그룹화
    """
    embedding_model = MockEmbeddingModel()
    policy = MockPolicy()
    
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'n': 3, 'sample': lambda: np.random.randint(0, 3)})()
        
        def close(self):
            pass
    
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=5,
        num_groups=4,
        device='cpu'
    )
    
    # 단일 에피소드
    episodes = create_mock_episodes(num_episodes=1)
    
    # 그룹화
    grouped_episodes = trainer.group_episodes(episodes)
    
    # 단일 그룹에 할당되었는지 확인
    assert len(grouped_episodes) == 1, "Single episode should be in one group"
    assert 0 in grouped_episodes, "Group 0 should exist"
    assert len(grouped_episodes[0]) == 1, "Group 0 should have 1 episode"


def test_edge_case_empty_episodes():
    """
    에지 케이스: 빈 에피소드 리스트
    """
    embedding_model = MockEmbeddingModel()
    policy = MockPolicy()
    
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'n': 3, 'sample': lambda: np.random.randint(0, 3)})()
        
        def close(self):
            pass
    
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=5,
        num_groups=4,
        device='cpu'
    )
    
    # 빈 에피소드 리스트
    episodes = []
    
    # 그룹화
    grouped_episodes = trainer.group_episodes(episodes)
    
    # 빈 딕셔너리 반환 확인
    assert len(grouped_episodes) == 0, "Empty episodes should return empty groups"


def test_group_statistics_logging():
    """
    그룹 통계 로깅 검증
    """
    embedding_model = MockEmbeddingModel()
    policy = MockPolicy()
    
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'n': 3, 'sample': lambda: np.random.randint(0, 3)})()
        
        def close(self):
            pass
    
    env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=5,
        num_groups=4,
        device='cpu'
    )
    
    # 에피소드 생성
    episodes = create_mock_episodes(num_episodes=20)
    
    # 그룹화
    grouped_episodes = trainer.group_episodes(episodes)
    
    # 각 그룹의 통계 확인
    for group_id, group_episodes in grouped_episodes.items():
        # 그룹 크기
        assert len(group_episodes) > 0, f"Group {group_id} is empty"
        
        # 그룹 평균 보상
        group_rewards = [ep['metadata']['episode_reward'] for ep in group_episodes]
        mean_reward = np.mean(group_rewards)
        assert isinstance(mean_reward, (int, float)), "Mean reward should be numeric"
        
        # 그룹 평균 지표
        group_indicators = [ep['market_indicators'] for ep in group_episodes]
        mean_indicators = np.mean(group_indicators, axis=0)
        assert len(mean_indicators) == 3, "Should have 3 mean indicators"
        
        print(f"Group {group_id}: size={len(group_episodes)}, "
              f"mean_reward={mean_reward:.6f}, "
              f"mean_volatility={mean_indicators[0]:.6f}, "
              f"mean_trend={mean_indicators[1]:.6f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
