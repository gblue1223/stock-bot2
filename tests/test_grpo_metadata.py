"""
GRPO 환경 메타데이터 계산 테스트

에피소드 메타데이터 계산 로직을 검증합니다.
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


def test_calculate_episode_metadata_basic():
    """기본 메타데이터 계산 테스트"""
    # Mock 임베딩 모델
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    # Mock 데이터베이스 연결
    with patch('duckdb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # 환경 생성
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="mock.db",
            embedding_dim=128
        )
        
        # 에피소드 데이터 시뮬레이션
        env.episode_rewards = [0.01, -0.005, 0.02, 0.015, -0.01]
        env.episode_trades = [
            {'holding_time': 5.0, 'reward': 0.01},
            {'holding_time': 3.0, 'reward': -0.005},
            {'holding_time': 7.0, 'reward': 0.02},
            {'holding_time': 4.0, 'reward': 0.015},
            {'holding_time': 2.0, 'reward': -0.01}
        ]
        env.quick_exit_violations = 2
        env.episode_length = 100
        env.current_step = 50
        
        # 메타데이터 계산
        metadata = env._calculate_episode_metadata()
        
        # 검증
        assert 'total_return' in metadata
        assert 'num_trades' in metadata
        assert 'avg_holding_time' in metadata
        assert 'sharpe_ratio' in metadata
        assert 'quick_exit_violations' in metadata
        
        # 값 검증
        assert metadata['total_return'] == pytest.approx(0.03, abs=1e-6)
        assert metadata['num_trades'] == 5
        assert metadata['avg_holding_time'] == pytest.approx(4.2, abs=1e-6)
        assert metadata['quick_exit_violations'] == 2
        
        # 샤프 비율 계산 검증
        mean_reward = np.mean(env.episode_rewards)
        std_reward = np.std(env.episode_rewards)
        expected_sharpe = mean_reward / std_reward if std_reward > 0 else 0.0
        assert metadata['sharpe_ratio'] == pytest.approx(expected_sharpe, abs=1e-6)
        
        env.close()


def test_calculate_episode_metadata_no_trades():
    """거래가 없는 경우 메타데이터 계산 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch('duckdb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="mock.db",
            embedding_dim=128
        )
        
        # 거래 없음
        env.episode_rewards = []
        env.episode_trades = []
        env.quick_exit_violations = 0
        env.episode_length = 100
        env.current_step = 50
        
        # 메타데이터 계산
        metadata = env._calculate_episode_metadata()
        
        # 검증
        assert metadata['total_return'] == 0.0
        assert metadata['num_trades'] == 0
        assert metadata['avg_holding_time'] == 0.0
        assert metadata['sharpe_ratio'] == 0.0
        assert metadata['quick_exit_violations'] == 0
        assert metadata['win_rate'] == 0.0
        assert metadata['avg_profit_per_trade'] == 0.0
        
        env.close()


def test_calculate_episode_metadata_single_reward():
    """단일 보상인 경우 샤프 비율 계산 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch('duckdb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="mock.db",
            embedding_dim=128
        )
        
        # 단일 보상
        env.episode_rewards = [0.01]
        env.episode_trades = [{'holding_time': 5.0, 'reward': 0.01}]
        env.quick_exit_violations = 0
        env.episode_length = 100
        env.current_step = 50
        
        # 메타데이터 계산
        metadata = env._calculate_episode_metadata()
        
        # 단일 보상인 경우 샤프 비율은 0
        assert metadata['sharpe_ratio'] == 0.0
        assert metadata['num_trades'] == 1
        
        env.close()


def test_calculate_episode_metadata_zero_std():
    """표준편차가 0인 경우 샤프 비율 계산 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch('duckdb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="mock.db",
            embedding_dim=128
        )
        
        # 모든 보상이 동일 (표준편차 0)
        env.episode_rewards = [0.01, 0.01, 0.01, 0.01]
        env.episode_trades = [
            {'holding_time': 5.0, 'reward': 0.01},
            {'holding_time': 5.0, 'reward': 0.01},
            {'holding_time': 5.0, 'reward': 0.01},
            {'holding_time': 5.0, 'reward': 0.01}
        ]
        env.quick_exit_violations = 0
        env.episode_length = 100
        env.current_step = 50
        
        # 메타데이터 계산
        metadata = env._calculate_episode_metadata()
        
        # 표준편차가 0이면 샤프 비율은 0
        assert metadata['sharpe_ratio'] == 0.0
        
        env.close()


def test_calculate_episode_metadata_win_rate():
    """승률 계산 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch('duckdb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="mock.db",
            embedding_dim=128
        )
        
        # 5개 거래 중 3개 승리
        env.episode_rewards = [0.01, -0.005, 0.02, 0.015, -0.01]
        env.episode_trades = [
            {'holding_time': 5.0, 'reward': 0.01},   # 승
            {'holding_time': 3.0, 'reward': -0.005}, # 패
            {'holding_time': 7.0, 'reward': 0.02},   # 승
            {'holding_time': 4.0, 'reward': 0.015},  # 승
            {'holding_time': 2.0, 'reward': -0.01}   # 패
        ]
        env.quick_exit_violations = 0
        env.episode_length = 100
        env.current_step = 50
        
        # 메타데이터 계산
        metadata = env._calculate_episode_metadata()
        
        # 승률 검증: 3/5 = 0.6
        assert metadata['win_rate'] == pytest.approx(0.6, abs=1e-6)
        
        env.close()


def test_calculate_episode_metadata_avg_profit():
    """평균 거래당 수익 계산 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch('duckdb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="mock.db",
            embedding_dim=128
        )
        
        # 거래 데이터
        env.episode_rewards = [0.01, -0.005, 0.02, 0.015, -0.01]
        env.episode_trades = [
            {'holding_time': 5.0, 'reward': 0.01},
            {'holding_time': 3.0, 'reward': -0.005},
            {'holding_time': 7.0, 'reward': 0.02},
            {'holding_time': 4.0, 'reward': 0.015},
            {'holding_time': 2.0, 'reward': -0.01}
        ]
        env.quick_exit_violations = 0
        env.episode_length = 100
        env.current_step = 50
        
        # 메타데이터 계산
        metadata = env._calculate_episode_metadata()
        
        # 평균 거래당 수익 검증
        expected_avg = (0.01 - 0.005 + 0.02 + 0.015 - 0.01) / 5
        assert metadata['avg_profit_per_trade'] == pytest.approx(expected_avg, abs=1e-6)
        
        env.close()


def test_calculate_episode_metadata_types():
    """메타데이터 타입 검증 테스트"""
    embedding_model = MockEmbeddingModel(embedding_dim=128)
    
    with patch('duckdb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path="mock.db",
            embedding_dim=128
        )
        
        # 거래 데이터
        env.episode_rewards = [0.01, -0.005]
        env.episode_trades = [
            {'holding_time': 5.0, 'reward': 0.01},
            {'holding_time': 3.0, 'reward': -0.005}
        ]
        env.quick_exit_violations = 1
        env.episode_length = 100
        env.current_step = 50
        
        # 메타데이터 계산
        metadata = env._calculate_episode_metadata()
        
        # 타입 검증
        assert isinstance(metadata['total_return'], float)
        assert isinstance(metadata['num_trades'], int)
        assert isinstance(metadata['avg_holding_time'], float)
        assert isinstance(metadata['sharpe_ratio'], float)
        assert isinstance(metadata['quick_exit_violations'], int)
        assert isinstance(metadata['win_rate'], float)
        assert isinstance(metadata['avg_profit_per_trade'], float)
        assert isinstance(metadata['episode_length'], int)
        assert isinstance(metadata['steps_taken'], int)
        
        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
