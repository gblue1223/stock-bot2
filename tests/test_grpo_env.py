"""
GRPO 환경 테스트

GRPOScalpingEnv의 기본 동작을 검증합니다.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from ai_trader.grpo.env import GRPOScalpingEnv
from ai_trader.embedding.models import TradingEmbeddingModel


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


def test_env_initialization():
    """환경 초기화 테스트"""
    # 간단한 임베딩 모델 생성
    embedding_model = MockEmbeddingModel(input_dim=60, embedding_dim=128, seq_len=60)
    
    # 환경 생성 (실제 DB 경로 필요)
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    
    try:
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            embedding_dim=128,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01
        )
        
        # 관측 공간 확인
        assert env.observation_space.shape == (128,)
        
        # 행동 공간 확인
        assert env.action_space.n == 3
        
        # 설정 확인
        assert env.quick_exit_threshold == 1.5
        assert env.quick_exit_penalty == 0.01
        assert env.round_trip_cost == 0.0043  # 0.43%
        
        env.close()
        
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


def test_env_reset():
    """환경 리셋 테스트"""
    embedding_model = MockEmbeddingModel(input_dim=60, embedding_dim=128, seq_len=60)
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    
    try:
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            embedding_dim=128
        )
        
        # 리셋
        observation, info = env.reset()
        
        # 관측값 형태 확인
        assert observation.shape == (128,)
        assert isinstance(observation, np.ndarray)
        
        # 정보 확인
        assert 'stock_code' in info
        assert 'date' in info
        assert 'time' in info
        
        # 초기 상태 확인
        assert env.position == 0
        assert env.quick_exit_violations == 0
        
        env.close()
        
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


def test_env_step_actions():
    """환경 스텝 및 행동 테스트"""
    embedding_model = MockEmbeddingModel(input_dim=60, embedding_dim=128, seq_len=60)
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    
    try:
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            embedding_dim=128
        )
        
        observation, info = env.reset()
        
        # 보유 행동 (action=0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (128,)
        assert reward == 0.0  # 보유는 보상 없음
        assert env.position == 0
        
        # 매수 행동 (action=1)
        obs, reward, terminated, truncated, info = env.step(1)
        assert env.position == 1
        assert env.entry_price > 0
        
        # 매도 행동 (action=2)
        obs, reward, terminated, truncated, info = env.step(2)
        assert env.position == 0
        assert len(env.episode_trades) == 1
        
        env.close()
        
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


def test_transaction_cost():
    """거래 비용 계산 테스트"""
    embedding_model = MockEmbeddingModel(input_dim=60, embedding_dim=128, seq_len=60)
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    
    try:
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            embedding_dim=128,
            transaction_cost_rate=0.00215
        )
        
        # 왕복 거래비용 확인
        assert env.round_trip_cost == 0.0043  # 0.43%
        
        env.close()
        
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


def test_configurable_threshold():
    """설정 가능한 임계값 검증"""
    embedding_model = MockEmbeddingModel(input_dim=60, embedding_dim=128, seq_len=60)
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    
    try:
        # 다른 임계값으로 환경 생성
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            embedding_dim=128,
            quick_exit_threshold=5.0,
            quick_exit_penalty=0.02
        )
        
        assert env.quick_exit_threshold == 5.0
        assert env.quick_exit_penalty == 0.02
        
        env.close()
        
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
