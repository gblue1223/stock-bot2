"""
GRPO 환경 단위 테스트

요구사항 3.2, 3.3, 3.5, 3.6을 검증하는 테스트
"""

import pytest
import numpy as np
import torch
import tempfile
import duckdb
from unittest.mock import Mock, MagicMock, patch

from ai_trader.grpo.env import GRPOScalpingEnv
from ai_trader.embedding import AutoEncoderEmbedding

@pytest.fixture
def mock_embedding_model():
    """Mock 임베딩 모델 생성 - 실제 모델 인스턴스 사용"""
    # 실제 임베딩 모델 생성 (테스트용 작은 크기)
    # 테스트 DB 특징 수: 28 (등락률, 누적거래대금, 거래회전율, 체결강도 + 매도/매수대기금액1-10 + 종목명_scalar, 시간_sin, 시간_cos, 시간_scalar2)
    model = AutoEncoderEmbedding(
        input_dim=28,  # 테스트 데이터의 특징 수에 맞춤
        embedding_dim=128,
        seq_len=60
    )
    model.eval()
    return model

@pytest.fixture
def mock_db_path():
    """Mock 데이터베이스 경로 생성"""
    # 임시 DuckDB 데이터베이스 경로 생성 (파일은 생성하지 않음)
    import os
    db_path = os.path.join(tempfile.gettempdir(), f'test_grpo_{np.random.randint(10000, 99999)}.duckdb')
    
    # 테스트 데이터 생성
    conn = duckdb.connect(db_path)
    
    # 테스트 테이블 생성 (28개 특징)
    conn.execute("""
        CREATE TABLE datasets (
            날짜 DATE,
            종목코드 VARCHAR,
            번호 INTEGER,
            등락률 REAL,
            누적거래대금 REAL,
            거래회전율 REAL,
            체결강도 REAL,
            매도대기금액1 REAL,
            매도대기금액2 REAL,
            매도대기금액3 REAL,
            매도대기금액4 REAL,
            매도대기금액5 REAL,
            매도대기금액6 REAL,
            매도대기금액7 REAL,
            매도대기금액8 REAL,
            매도대기금액9 REAL,
            매도대기금액10 REAL,
            매수대기금액1 REAL,
            매수대기금액2 REAL,
            매수대기금액3 REAL,
            매수대기금액4 REAL,
            매수대기금액5 REAL,
            매수대기금액6 REAL,
            매수대기금액7 REAL,
            매수대기금액8 REAL,
            매수대기금액9 REAL,
            매수대기금액10 REAL,
            종목명_scalar REAL,
            시간_sin REAL,
            시간_cos REAL,
            시간_scalar2 REAL
        )
    """)
    
    # 테스트 데이터 삽입 (200개 행)
    for i in range(200):
        # 가격 변화 시뮬레이션 (등락률)
        price_change = np.random.randn() * 0.01  # 평균 0, 표준편차 1%
        
        conn.execute(f"""
            INSERT INTO datasets VALUES (
                '2025-01-01',
                '005930',
                {i},
                {price_change},
                {np.random.rand()},
                {np.random.rand()},
                {np.random.rand()},
                {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()},
                {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()},
                {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()},
                {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()}, {np.random.rand()},
                {np.random.rand()},
                {np.sin(i * 0.1)},
                {np.cos(i * 0.1)},
                {i / 200.0}
            )
        """)
    
    conn.close()
    
    yield db_path
    
    # 정리
    import os
    try:
        os.unlink(db_path)
    except:
        pass


class TestGRPOScalpingEnv:
    """GRPOScalpingEnv 테스트"""
    
    def test_env_initialization(self, mock_embedding_model, mock_db_path):
        """환경 초기화 테스트"""
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path,
            embedding_dim=128,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01
        )
        
        # 관측 공간 검증
        assert env.observation_space.shape == (128,), \
            f"Expected observation space shape (128,), got {env.observation_space.shape}"
        
        # 행동 공간 검증
        assert env.action_space.n == 3, \
            f"Expected 3 actions, got {env.action_space.n}"
        
        # 거래 비용 검증
        assert env.transaction_cost_rate == 0.00215, \
            "Transaction cost rate should be 0.215%"
        assert env.round_trip_cost == 0.0043, \
            "Round trip cost should be 0.43%"
        
        # 빠른 손절 룰 설정 검증
        assert env.quick_exit_threshold == 1.5, \
            "Quick exit threshold should be 1.5"
        assert env.quick_exit_penalty == 0.01, \
            "Quick exit penalty should be 0.01"
        
        env.close()
    
    def test_configurable_thresholds(self, mock_embedding_model, mock_db_path):
        """
        요구사항 3.3: 설정 가능한 임계값 검증
        
        빠른 손절 시간 임계값과 페널티가 설정 가능해야 함
        """
        # 다양한 임계값으로 환경 생성
        threshold_configs = [
            (1.5, 0.01),
            (3.0, 0.02),
            (5.0, 0.05),
        ]
        
        for threshold, penalty in threshold_configs:
            env = GRPOScalpingEnv(
                embedding_model=mock_embedding_model,
                db_path=mock_db_path,
                quick_exit_threshold=threshold,
                quick_exit_penalty=penalty
            )
            
            assert env.quick_exit_threshold == threshold, \
                f"Quick exit threshold should be {threshold}, got {env.quick_exit_threshold}"
            assert env.quick_exit_penalty == penalty, \
                f"Quick exit penalty should be {penalty}, got {env.quick_exit_penalty}"
            
            env.close()
    
    def test_reset(self, mock_embedding_model, mock_db_path):
        """환경 리셋 테스트"""
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path,
            embedding_dim=128
        )
        
        observation, info = env.reset()
        
        # 관측값 형태 검증
        assert observation.shape == (128,), \
            f"Expected observation shape (128,), got {observation.shape}"
        
        # 초기 상태 검증
        assert env.position == 0, "Initial position should be 0"
        assert env.entry_price == 0.0, "Initial entry price should be 0"
        assert env.entry_time == 0.0, "Initial entry time should be 0"
        
        # 메타데이터 초기화 검증
        assert len(env.episode_trades) == 0, "Episode trades should be empty"
        assert len(env.episode_rewards) == 0, "Episode rewards should be empty"
        assert env.quick_exit_violations == 0, "Quick exit violations should be 0"
        
        # 정보 검증
        assert 'stock_code' in info
        assert 'date' in info
        assert 'time' in info
        
        env.close()
    
    def test_reward_calculation(self, mock_embedding_model, mock_db_path):
        """
        요구사항 3.2: 보상 계산 검증
        
        보상 구조:
        - 수익률 기반 보상: (청산가 - 진입가) / 진입가 - 거래비용
        - 거래비용: 0.43% (왕복)
        - 양의 보상 조건: 수익률 > 0.43%
        """
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path,
            transaction_cost_rate=0.00215
        )
        
        # 테스트 케이스 1: 수익률 1% (거래비용 차감 후 0.57%)
        entry_price = 100.0
        exit_price = 101.0  # 1% 상승
        holding_time = 30.0
        
        reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
        
        expected_profit_rate = (exit_price - entry_price) / entry_price  # 0.01
        expected_reward = expected_profit_rate - env.round_trip_cost  # 0.01 - 0.0043 = 0.0057
        
        assert np.isclose(components['profit_rate'], expected_profit_rate, atol=1e-6), \
            f"Profit rate should be {expected_profit_rate}, got {components['profit_rate']}"
        assert np.isclose(components['transaction_cost'], -env.round_trip_cost, atol=1e-6), \
            f"Transaction cost should be {-env.round_trip_cost}, got {components['transaction_cost']}"
        assert np.isclose(reward, expected_reward, atol=1e-6), \
            f"Reward should be {expected_reward}, got {reward}"
        
        # 테스트 케이스 2: 수익률 0.3% (거래비용 차감 후 -0.13%, 음의 보상)
        exit_price = 100.3  # 0.3% 상승
        reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
        
        expected_profit_rate = 0.003
        expected_reward = expected_profit_rate - env.round_trip_cost  # 0.003 - 0.0043 = -0.0013
        
        assert reward < 0, "Reward should be negative when profit < transaction cost"
        assert np.isclose(reward, expected_reward, atol=1e-6), \
            f"Reward should be {expected_reward}, got {reward}"
        
        # 테스트 케이스 3: 장기 보유 페널티
        exit_price = 101.0
        holding_time = 70.0  # 60초 초과
        
        reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
        
        expected_holding_penalty = env.holding_penalty_rate * (holding_time - env.max_holding_time)
        assert components['holding_penalty'] == -expected_holding_penalty, \
            f"Holding penalty should be {-expected_holding_penalty}, got {components['holding_penalty']}"
        
        env.close()
    
    def test_quick_exit_rule_violation(self, mock_embedding_model, mock_db_path):
        """
        요구사항 3.3: 빠른 손절 룰 동작 검증
        
        매수 후 설정 가능한 시간 임계값 이내에 가격이 상승하지 않으면
        자동 매도 및 페널티 적용
        """
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01
        )
        
        # 환경 리셋
        observation, info = env.reset()
        
        # 매수 행동
        observation, reward, terminated, truncated, info = env.step(1)  # 매수
        
        assert env.position == 1, "Position should be 1 after buy"
        entry_price = env.entry_price
        entry_time = env.entry_time
        
        # 가격이 하락하도록 시뮬레이션
        # 진입 가격이 양수인 경우 낮게, 음수인 경우 더 음수로 설정
        if entry_price > 0:
            env.current_price = entry_price * 0.99  # 1% 하락
        else:
            env.current_price = entry_price * 1.01  # 더 음수로 (하락)
        
        # 보유 행동 (빠른 손절 룰 체크)
        # 시간이 임계값 이내이고 가격이 상승하지 않았으므로 위반
        observation, reward, terminated, truncated, info = env.step(0)  # 보유
        
        # 빠른 손절 룰 위반 검증
        assert env.quick_exit_violations == 1, \
            f"Quick exit violations should be 1, got {env.quick_exit_violations}"
        
        # 자동 매도 검증
        assert env.position == 0, "Position should be 0 after quick exit"
        
        # 정보에 빠른 손절 트리거 플래그 확인
        assert info['quick_exit_triggered'] == True, \
            "quick_exit_triggered should be True"
        
        # 거래 기록 확인
        assert len(env.episode_trades) == 1, "Should have 1 trade recorded"
        trade = env.episode_trades[0]
        assert trade.get('quick_exit_violation') == True, \
            "Trade should be marked as quick exit violation"
        assert trade.get('quick_exit_penalty') == env.quick_exit_penalty, \
            f"Trade should have penalty {env.quick_exit_penalty}"
        
        # 페널티가 적용되었는지 확인 (거래 보상에서 페널티가 차감됨)
        # 실제 보상은 수익률 - 거래비용 - 빠른 손절 페널티
        assert 'quick_exit_penalty' in trade, \
            "Trade should record quick exit penalty"
        
        env.close()
    
    def test_quick_exit_rule_no_violation(self, mock_embedding_model, mock_db_path):
        """빠른 손절 룰 위반하지 않는 경우 테스트"""
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01
        )
        
        observation, info = env.reset()
        
        # 매수
        observation, reward, terminated, truncated, info = env.step(1)
        entry_price = env.entry_price
        entry_time = env.entry_time
        
        # 가격이 상승하도록 시뮬레이션
        # 중요: 현재 가격을 진입 가격보다 높게 설정
        original_get_price = env._get_current_price
        
        def mock_get_price():
            return entry_price + 0.01  # 가격 상승
        
        env._get_current_price = mock_get_price
        
        # 다음 스텝으로 이동하기 전에 가격 업데이트
        env.current_step += 1
        env.current_price = env._get_current_price()
        env.current_time = float(env.episode_metadata[env.current_step, 2])
        env.current_step -= 1  # step 메서드에서 다시 증가시킬 것이므로 되돌림
        
        # 보유 (가격이 상승했으므로 위반 없음)
        observation, reward, terminated, truncated, info = env.step(0)
        
        # 위반 없음 검증
        assert env.quick_exit_violations == 0, \
            "Should have no quick exit violations when price increases"
        assert env.position == 1, \
            "Position should still be 1 (not auto-sold)"
        assert info['quick_exit_triggered'] == False, \
            "quick_exit_triggered should be False"
        
        # 원래 함수 복원
        env._get_current_price = original_get_price
        
        env.close()
    
    def test_episode_metadata(self, mock_embedding_model, mock_db_path):
        """
        요구사항 3.5, 3.6, 3.7: 에피소드 메타데이터 검증
        
        에피소드 종료 시 다음 메타데이터를 기록해야 함:
        - 총 수익 (total_return)
        - 거래 횟수 (num_trades)
        - 평균 보유 시간 (avg_holding_time)
        - 샤프 비율 (sharpe_ratio)
        - 빠른 손절 룰 위반 횟수 (quick_exit_violations)
        """
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path,
            quick_exit_threshold=1.5
        )
        
        observation, info = env.reset()
        
        # 여러 거래 시뮬레이션
        # 거래 1: 매수 -> 매도
        env.step(1)  # 매수
        env.step(2)  # 매도
        
        # 거래 2: 매수 -> 매도
        env.step(1)  # 매수
        env.step(2)  # 매도
        
        # 에피소드 종료까지 진행 (최대 200 스텝)
        max_steps = 200
        for i in range(max_steps):
            observation, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        
        # 에피소드가 종료되었는지 확인
        assert terminated or truncated, \
            f"Episode should terminate within {max_steps} steps"
        
        # 에피소드 메타데이터 검증
        assert 'episode' in info, "Info should contain episode metadata"
        
        metadata = info['episode']
        
        # 필수 메트릭 존재 확인
        assert 'total_return' in metadata, "Should have total_return"
        assert 'num_trades' in metadata, "Should have num_trades"
        assert 'avg_holding_time' in metadata, "Should have avg_holding_time"
        assert 'sharpe_ratio' in metadata, "Should have sharpe_ratio"
        assert 'quick_exit_violations' in metadata, "Should have quick_exit_violations"
        
        # 추가 메트릭 확인
        assert 'win_rate' in metadata, "Should have win_rate"
        assert 'avg_profit_per_trade' in metadata, "Should have avg_profit_per_trade"
        assert 'episode_length' in metadata, "Should have episode_length"
        assert 'steps_taken' in metadata, "Should have steps_taken"
        
        # 거래 횟수 검증
        assert metadata['num_trades'] >= 2, \
            f"Should have at least 2 trades, got {metadata['num_trades']}"
        
        # 빠른 손절 룰 위반 횟수 검증
        assert metadata['quick_exit_violations'] == env.quick_exit_violations, \
            "Metadata should match env quick_exit_violations"
        
        env.close()
    
    def test_action_space(self, mock_embedding_model, mock_db_path):
        """행동 공간 테스트"""
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path
        )
        
        observation, info = env.reset()
        
        # 행동 0: 보유
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.position == 0, "Position should remain 0 after hold"
        
        # 행동 1: 매수
        observation, reward, terminated, truncated, info = env.step(1)
        assert env.position == 1, "Position should be 1 after buy"
        assert env.entry_price != 0, "Entry price should be set"
        assert env.entry_time != 0, "Entry time should be set"
        
        # 행동 2: 매도
        observation, reward, terminated, truncated, info = env.step(2)
        assert env.position == 0, "Position should be 0 after sell"
        assert len(env.episode_trades) == 1, "Should have 1 trade recorded"
        
        env.close()
    
    def test_transaction_cost_rates(self, mock_embedding_model, mock_db_path):
        """
        요구사항 3.2.1, 3.2.2: 거래 비용 검증
        
        - 수수료(0.015%) + 세금(0.2%) = 0.215% (매수/매도 각각)
        - 왕복 거래비용: 0.43%
        - 양의 보상 조건: 수익률 > 0.43%
        """
        env = GRPOScalpingEnv(
            embedding_model=mock_embedding_model,
            db_path=mock_db_path,
            transaction_cost_rate=0.00215
        )
        
        # 거래 비용 비율 검증
        assert env.transaction_cost_rate == 0.00215, \
            "Transaction cost rate should be 0.215%"
        
        # 왕복 거래비용 검증
        assert env.round_trip_cost == 0.0043, \
            "Round trip cost should be 0.43%"
        
        # 양의 보상 조건 테스트
        # 수익률 0.43%: 보상 = 0
        entry_price = 100.0
        exit_price = 100.43
        holding_time = 30.0
        
        reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
        
        # 수익률이 정확히 거래비용과 같으면 보상은 0에 가까움
        assert np.isclose(reward, 0.0, atol=1e-6), \
            f"Reward should be near 0 when profit equals transaction cost, got {reward}"
        
        # 수익률 0.44%: 양의 보상
        exit_price = 100.44
        reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
        assert reward > 0, \
            "Reward should be positive when profit > transaction cost"
        
        # 수익률 0.42%: 음의 보상
        exit_price = 100.42
        reward, components = env._calculate_reward(entry_price, exit_price, holding_time)
        assert reward < 0, \
            "Reward should be negative when profit < transaction cost"
        
        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
