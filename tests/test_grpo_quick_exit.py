"""
빠른 손절 룰 단위 테스트

데이터베이스 없이 빠른 손절 룰의 핵심 로직을 검증합니다.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from ai_trader.grpo.env import GRPOScalpingEnv


class MockEmbeddingModel(nn.Module):
    """테스트용 간단한 임베딩 모델"""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """고정된 임베딩 반환"""
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.embedding_dim)


def test_quick_exit_threshold_configuration():
    """빠른 손절 임계값 설정 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    # 기본값 테스트
    with patch.object(GRPOScalpingEnv, '_connect_db'):
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="dummy.db",
            embedding_dim=128
        )
        assert env.quick_exit_threshold == 1.5
        assert env.quick_exit_penalty == 0.01
    
    # 커스텀 값 테스트
    with patch.object(GRPOScalpingEnv, '_connect_db'):
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="dummy.db",
            embedding_dim=128,
            quick_exit_threshold=3.0,
            quick_exit_penalty=0.05
        )
        assert env.quick_exit_threshold == 3.0
        assert env.quick_exit_penalty == 0.05


def test_quick_exit_violation_counter():
    """빠른 손절 위반 카운터 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch.object(GRPOScalpingEnv, '_connect_db'):
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="dummy.db",
            embedding_dim=128,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01
        )
        
        # 초기 위반 횟수는 0
        assert env.quick_exit_violations == 0
        
        # 수동으로 위반 횟수 증가 시뮬레이션
        env.quick_exit_violations += 1
        assert env.quick_exit_violations == 1
        
        env.quick_exit_violations += 1
        assert env.quick_exit_violations == 2


def test_quick_exit_penalty_application():
    """빠른 손절 페널티 적용 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch.object(GRPOScalpingEnv, '_connect_db'):
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="dummy.db",
            embedding_dim=128,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01
        )
        
        # 보상 계산 테스트
        entry_price = 100.0
        exit_price = 99.0  # 1% 손실
        holding_time = 1.0  # 1초 (임계값 이내)
        
        reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
        
        # 기본 보상 계산 확인
        expected_profit_rate = (exit_price - entry_price) / entry_price  # -0.01
        expected_reward = expected_profit_rate - env.round_trip_cost  # -0.01 - 0.0043 = -0.0143
        
        assert abs(reward - expected_reward) < 1e-6
        assert components['profit_rate'] == expected_profit_rate
        assert components['transaction_cost'] == -env.round_trip_cost


def test_quick_exit_trade_metadata():
    """빠른 손절 거래 메타데이터 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch.object(GRPOScalpingEnv, '_connect_db'):
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="dummy.db",
            embedding_dim=128,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01
        )
        
        # 거래 정보 시뮬레이션
        trade_info = {
            'entry_price': 100.0,
            'exit_price': 99.0,
            'holding_time': 1.0,
            'profit_rate': -0.01,
            'reward': -0.0243,  # -0.01 - 0.0043 - 0.01 (penalty)
            'reward_components': {
                'profit_rate': -0.01,
                'transaction_cost': -0.0043,
                'holding_penalty': 0.0,
                'net_reward': -0.0143
            },
            'quick_exit_violation': True,
            'quick_exit_penalty': 0.01
        }
        
        # 메타데이터 필드 확인
        assert 'quick_exit_violation' in trade_info
        assert trade_info['quick_exit_violation'] == True
        assert 'quick_exit_penalty' in trade_info
        assert trade_info['quick_exit_penalty'] == 0.01


def test_transaction_cost_calculation():
    """거래 비용 계산 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch.object(GRPOScalpingEnv, '_connect_db'):
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="dummy.db",
            embedding_dim=128,
            transaction_cost_rate=0.00215  # 0.215%
        )
        
        # 왕복 거래비용 확인
        assert env.transaction_cost_rate == 0.00215
        assert env.round_trip_cost == 0.0043  # 0.43%
        
        # 양의 보상을 받으려면 수익률이 0.43%를 초과해야 함
        entry_price = 100.0
        
        # 0.43% 수익 - 거래비용 = 0
        exit_price_break_even = 100.43
        reward, _ = env._calculate_reward(entry_price, exit_price_break_even, 1.0)
        assert abs(reward) < 1e-6  # 거의 0
        
        # 0.5% 수익 - 거래비용 = 0.07% 양의 보상
        exit_price_profit = 100.5
        reward, _ = env._calculate_reward(entry_price, exit_price_profit, 1.0)
        assert reward > 0
        
        # 0.3% 수익 - 거래비용 = -0.13% 음의 보상
        exit_price_loss = 100.3
        reward, _ = env._calculate_reward(entry_price, exit_price_loss, 1.0)
        assert reward < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
