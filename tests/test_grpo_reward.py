"""
GRPO 환경 보상 계산 테스트

보상 계산 로직이 요구사항을 충족하는지 검증합니다.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn


class MockEmbeddingModel(nn.Module):
    """테스트용 임베딩 모델"""
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(60, embedding_dim)
    
    def forward(self, x):
        # (batch, seq_len, features) -> (batch, embedding_dim)
        return self.fc(x.mean(dim=1))


def test_reward_calculation_basic():
    """기본 보상 계산 테스트"""
    from ai_trader.grpo.env import GRPOScalpingEnv
    
    # 임베딩 모델 생성
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    # 환경 생성 (실제 DB 없이 테스트하기 위해 mock 필요)
    # 여기서는 _calculate_reward 메서드만 테스트
    env = GRPOScalpingEnv.__new__(GRPOScalpingEnv)
    env.transaction_cost_rate = 0.00215  # 0.215%
    env.round_trip_cost = 0.0043  # 0.43%
    env.max_holding_time = 60.0
    env.holding_penalty_rate = 0.001
    
    # 테스트 케이스 1: 수익률 1%, 보유 시간 30초
    # 예상 보상: 0.01 - 0.0043 = 0.0057
    entry_price = 100.0
    exit_price = 101.0
    holding_time = 30.0
    
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    expected_profit_rate = (exit_price - entry_price) / entry_price  # 0.01
    expected_reward = expected_profit_rate - env.round_trip_cost  # 0.01 - 0.0043 = 0.0057
    
    assert abs(components['profit_rate'] - expected_profit_rate) < 1e-6
    assert abs(components['transaction_cost'] - (-env.round_trip_cost)) < 1e-6
    assert components['holding_penalty'] == 0.0  # 60초 이하이므로 페널티 없음
    assert abs(reward - expected_reward) < 1e-6
    
    print(f"✓ Test 1 passed: profit_rate={components['profit_rate']:.4f}, reward={reward:.4f}")


def test_reward_positive_condition():
    """양의 보상 조건 테스트: 수익률 > 0.43%"""
    from ai_trader.grpo.env import GRPOScalpingEnv
    
    env = GRPOScalpingEnv.__new__(GRPOScalpingEnv)
    env.transaction_cost_rate = 0.00215
    env.round_trip_cost = 0.0043
    env.max_holding_time = 60.0
    env.holding_penalty_rate = 0.001
    
    # 테스트 케이스 2: 수익률 0.43% (경계값)
    # 예상 보상: 0.0043 - 0.0043 = 0.0
    entry_price = 100.0
    exit_price = 100.43
    holding_time = 30.0
    
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    assert abs(reward) < 1e-6  # 0에 가까워야 함
    
    print(f"✓ Test 2 passed: profit_rate={components['profit_rate']:.4f}, reward={reward:.4f}")
    
    # 테스트 케이스 3: 수익률 0.5% (양의 보상)
    # 예상 보상: 0.005 - 0.0043 = 0.0007
    exit_price = 100.5
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    expected_reward = 0.005 - 0.0043
    assert reward > 0  # 양의 보상
    assert abs(reward - expected_reward) < 1e-6
    
    print(f"✓ Test 3 passed: profit_rate={components['profit_rate']:.4f}, reward={reward:.4f}")
    
    # 테스트 케이스 4: 수익률 0.3% (음의 보상)
    # 예상 보상: 0.003 - 0.0043 = -0.0013
    exit_price = 100.3
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    expected_reward = 0.003 - 0.0043
    assert reward < 0  # 음의 보상
    assert abs(reward - expected_reward) < 1e-6
    
    print(f"✓ Test 4 passed: profit_rate={components['profit_rate']:.4f}, reward={reward:.4f}")


def test_holding_penalty():
    """장기 보유 페널티 테스트: -0.001 * (보유시간 - 60초)"""
    from ai_trader.grpo.env import GRPOScalpingEnv
    
    env = GRPOScalpingEnv.__new__(GRPOScalpingEnv)
    env.transaction_cost_rate = 0.00215
    env.round_trip_cost = 0.0043
    env.max_holding_time = 60.0
    env.holding_penalty_rate = 0.001
    
    # 테스트 케이스 5: 보유 시간 60초 (페널티 없음)
    entry_price = 100.0
    exit_price = 101.0
    holding_time = 60.0
    
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    assert components['holding_penalty'] == 0.0
    expected_reward = 0.01 - 0.0043
    assert abs(reward - expected_reward) < 1e-6
    
    print(f"✓ Test 5 passed: holding_time={holding_time}s, penalty={components['holding_penalty']:.4f}")
    
    # 테스트 케이스 6: 보유 시간 70초 (페널티 적용)
    # 예상 페널티: -0.001 * (70 - 60) = -0.01
    holding_time = 70.0
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    expected_penalty = 0.001 * (70.0 - 60.0)
    assert abs(components['holding_penalty'] - (-expected_penalty)) < 1e-6  # 음수로 저장됨
    expected_reward = 0.01 - 0.0043 - expected_penalty
    assert abs(reward - expected_reward) < 1e-6
    
    print(f"✓ Test 6 passed: holding_time={holding_time}s, penalty={components['holding_penalty']:.4f}")
    
    # 테스트 케이스 7: 보유 시간 100초 (큰 페널티)
    # 예상 페널티: -0.001 * (100 - 60) = -0.04
    holding_time = 100.0
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    expected_penalty = 0.001 * (100.0 - 60.0)
    assert abs(components['holding_penalty'] - (-expected_penalty)) < 1e-6  # 음수로 저장됨
    expected_reward = 0.01 - 0.0043 - expected_penalty
    assert abs(reward - expected_reward) < 1e-6
    
    print(f"✓ Test 7 passed: holding_time={holding_time}s, penalty={components['holding_penalty']:.4f}")


def test_transaction_cost():
    """거래 비용 계산 테스트: 수수료 0.015% + 세금 0.2% = 0.215%, 왕복 0.43%"""
    from ai_trader.grpo.env import GRPOScalpingEnv
    
    env = GRPOScalpingEnv.__new__(GRPOScalpingEnv)
    env.transaction_cost_rate = 0.00215
    env.round_trip_cost = 0.0043
    env.max_holding_time = 60.0
    env.holding_penalty_rate = 0.001
    
    # 거래 비용 검증
    assert abs(env.transaction_cost_rate - 0.00215) < 1e-6  # 0.215%
    assert abs(env.round_trip_cost - 0.0043) < 1e-6  # 0.43%
    
    # 테스트 케이스 8: 손실 거래
    # 수익률 -1%, 보유 시간 30초
    # 예상 보상: -0.01 - 0.0043 = -0.0143
    entry_price = 100.0
    exit_price = 99.0
    holding_time = 30.0
    
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    expected_profit_rate = -0.01
    expected_reward = expected_profit_rate - env.round_trip_cost
    
    assert abs(components['profit_rate'] - expected_profit_rate) < 1e-6
    assert abs(reward - expected_reward) < 1e-6
    assert reward < 0  # 손실 거래는 항상 음의 보상
    
    print(f"✓ Test 8 passed: profit_rate={components['profit_rate']:.4f}, reward={reward:.4f}")


def test_reward_components():
    """보상 구성 요소 검증"""
    from ai_trader.grpo.env import GRPOScalpingEnv
    
    env = GRPOScalpingEnv.__new__(GRPOScalpingEnv)
    env.transaction_cost_rate = 0.00215
    env.round_trip_cost = 0.0043
    env.max_holding_time = 60.0
    env.holding_penalty_rate = 0.001
    
    # 테스트 케이스 9: 모든 구성 요소 포함
    entry_price = 100.0
    exit_price = 102.0  # 2% 수익
    holding_time = 80.0  # 80초 보유 (20초 초과)
    
    reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
    
    # 구성 요소 검증
    assert 'profit_rate' in components
    assert 'transaction_cost' in components
    assert 'holding_penalty' in components
    assert 'net_reward' in components
    
    # 값 검증
    expected_profit_rate = 0.02
    expected_transaction_cost = -0.0043
    expected_holding_penalty = 0.001 * (80.0 - 60.0)
    expected_net_reward = expected_profit_rate + expected_transaction_cost - expected_holding_penalty
    
    assert abs(components['profit_rate'] - expected_profit_rate) < 1e-6
    assert abs(components['transaction_cost'] - expected_transaction_cost) < 1e-6
    assert abs(components['holding_penalty'] - (-expected_holding_penalty)) < 1e-6  # 음수로 저장됨
    assert abs(components['net_reward'] - expected_net_reward) < 1e-6
    assert abs(reward - expected_net_reward) < 1e-6
    
    print(f"✓ Test 9 passed: all components verified")
    print(f"  profit_rate={components['profit_rate']:.4f}")
    print(f"  transaction_cost={components['transaction_cost']:.4f}")
    print(f"  holding_penalty={components['holding_penalty']:.4f}")
    print(f"  net_reward={components['net_reward']:.4f}")


if __name__ == '__main__':
    print("Running GRPO reward calculation tests...\n")
    
    test_reward_calculation_basic()
    print()
    
    test_reward_positive_condition()
    print()
    
    test_holding_penalty()
    print()
    
    test_transaction_cost()
    print()
    
    test_reward_components()
    print()
    
    print("All tests passed! ✓")
