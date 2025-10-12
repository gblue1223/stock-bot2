"""
GRPO 백테스팅 시뮬레이터 테스트

이 모듈은 백테스팅 시뮬레이터의 기능을 테스트합니다.
"""

import pytest
import numpy as np
import torch
import tempfile
import duckdb
from pathlib import Path

from ai_trader.grpo.backtest import BacktestSimulator
from ai_trader.inference.infer_grpo import GRPOInference
from ai_trader.embedding import AutoEncoderEmbedding
from ai_trader.grpo.policy import GRPOPolicy


@pytest.fixture
def temp_db():
    """임시 테스트 데이터베이스 생성"""
    # 임시 디렉토리에 데이터베이스 파일 생성
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / 'test.duckdb'
    
    # 테스트 데이터 생성
    conn = duckdb.connect(str(db_path))
    
    # 테이블 생성
    conn.execute("""
        CREATE TABLE datasets (
            날짜 DATE,
            종목코드 VARCHAR,
            번호 INTEGER,
            등락률 REAL,
            누적거래대금 REAL,
            거래회전율 REAL,
            체결강도 REAL
        )
    """)
    
    # 샘플 데이터 삽입 (2개 종목, 각 100개 샘플)
    np.random.seed(42)
    for stock_code in ['000001', '000002']:
        for i in range(100):
            conn.execute("""
                INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                '2024-01-01',
                stock_code,
                i,
                float(np.random.randn() * 0.01),  # 등락률
                float(np.random.rand() * 1000),   # 누적거래대금
                float(np.random.rand()),          # 거래회전율
                float(np.random.rand() * 100)     # 체결강도
            ])
    
    conn.close()
    
    yield str(db_path)
    
    # 정리
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_inference_engine(tmp_path):
    """모의 추론 엔진 생성"""
    # 임베딩 모델 생성
    embedding_model = AutoEncoderEmbedding(
        input_dim=4,
        embedding_dim=16,
        seq_len=10,
        num_heads=2
    )
    
    # 정책 생성
    policy = GRPOPolicy(
        embedding_dim=16,
        hidden_dim=32,
        action_dim=3
    )
    
    # 체크포인트 저장
    embedding_path = tmp_path / 'embedding.pt'
    policy_path = tmp_path / 'policy.pt'
    
    torch.save({
        'state_dict': embedding_model.state_dict(),
        'config': {
            'input_dim': 4,
            'embedding_dim': 16,
            'seq_len': 10,
            'num_heads': 2
        },
        'normalization_stats': {
            'mean': np.zeros(4),
            'std': np.ones(4)
        }
    }, embedding_path)
    
    torch.save({
        'state_dict': policy.state_dict(),
        'config': {
            'embedding_dim': 16,
            'hidden_dim': 32,
            'action_dim': 3
        }
    }, policy_path)
    
    # 추론 엔진 생성
    inference_engine = GRPOInference(
        embedding_model_path=str(embedding_path),
        policy_path=str(policy_path),
        device='cpu',
        use_torchscript=False
    )
    
    return inference_engine


def test_backtest_simulator_initialization(mock_inference_engine, temp_db):
    """백테스팅 시뮬레이터 초기화 테스트"""
    simulator = BacktestSimulator(
        inference_engine=mock_inference_engine,
        db_path=temp_db,
        table_name='datasets',
        seq_len=10
    )
    
    assert simulator.seq_len == 10
    assert simulator.transaction_cost_rate == 0.00215
    assert simulator.round_trip_cost == 0.0043
    assert simulator.slippage_rate == 0.0001
    
    simulator.close()


def test_slippage_application(mock_inference_engine, temp_db):
    """슬리피지 적용 테스트"""
    simulator = BacktestSimulator(
        inference_engine=mock_inference_engine,
        db_path=temp_db,
        slippage_rate=0.001
    )
    
    # 매수 시 가격 상승
    buy_price = simulator._apply_slippage(100.0, action=1)
    assert buy_price == 100.1  # 100 * (1 + 0.001)
    
    # 매도 시 가격 하락
    sell_price = simulator._apply_slippage(100.0, action=2)
    assert sell_price == 99.9  # 100 * (1 - 0.001)
    
    # 보유 시 가격 변화 없음
    hold_price = simulator._apply_slippage(100.0, action=0)
    assert hold_price == 100.0
    
    simulator.close()


def test_reward_calculation(mock_inference_engine, temp_db):
    """보상 계산 테스트"""
    simulator = BacktestSimulator(
        inference_engine=mock_inference_engine,
        db_path=temp_db,
        transaction_cost_rate=0.00215,
        max_holding_time=60.0,
        holding_penalty_rate=0.001
    )
    
    # 수익 거래 (거래 비용 포함)
    entry_price = 100.0
    exit_price = 101.0
    holding_time = 30.0
    
    reward, components = simulator._calculate_reward(entry_price, exit_price, holding_time)
    
    # 수익률 = (101 - 100) / 100 = 0.01 = 1%
    # 거래 비용 = 0.43%
    # 순수익 = 1% - 0.43% = 0.57%
    assert components['profit_rate'] == pytest.approx(0.01, abs=1e-6)
    assert components['transaction_cost'] == pytest.approx(-0.0043, abs=1e-6)
    assert components['holding_penalty'] == 0.0  # 60초 이내
    assert reward == pytest.approx(0.0057, abs=1e-6)
    
    # 손실 거래
    exit_price = 99.0
    reward, components = simulator._calculate_reward(entry_price, exit_price, holding_time)
    
    # 수익률 = (99 - 100) / 100 = -0.01 = -1%
    # 거래 비용 = 0.43%
    # 순손실 = -1% - 0.43% = -1.43%
    assert components['profit_rate'] == pytest.approx(-0.01, abs=1e-6)
    assert reward == pytest.approx(-0.0143, abs=1e-6)
    
    # 장기 보유 페널티
    holding_time = 90.0  # 60초 초과
    reward, components = simulator._calculate_reward(entry_price, exit_price, holding_time)
    
    # 보유 페널티 = 0.001 * (90 - 60) = 0.03
    assert components['holding_penalty'] == pytest.approx(-0.03, abs=1e-6)
    assert reward == pytest.approx(-0.0443, abs=1e-6)  # -0.01 - 0.0043 - 0.03
    
    simulator.close()


def test_load_test_data(mock_inference_engine, temp_db):
    """테스트 데이터 로드 테스트"""
    simulator = BacktestSimulator(
        inference_engine=mock_inference_engine,
        db_path=temp_db,
        table_name='datasets',
        seq_len=10
    )
    
    # 모든 데이터 로드
    data, metadata = simulator._load_test_data()
    
    assert len(data) == 200  # 2개 종목 * 100개 샘플
    assert data.shape[1] == 4  # 4개 특징
    assert metadata.shape == (200, 3)  # [종목코드, 날짜, 번호]
    
    # 특정 종목만 로드
    data, metadata = simulator._load_test_data(stock_codes=['000001'])
    
    assert len(data) == 100  # 1개 종목 * 100개 샘플
    
    simulator.close()


def test_run_backtest(mock_inference_engine, temp_db):
    """백테스팅 실행 테스트"""
    simulator = BacktestSimulator(
        inference_engine=mock_inference_engine,
        db_path=temp_db,
        table_name='datasets',
        seq_len=10,
        transaction_cost_rate=0.00215,
        slippage_rate=0.0001
    )
    
    # 백테스팅 실행
    results = simulator.run_backtest(
        test_start_date='2024-01-01',
        test_end_date='2024-01-01',
        deterministic=True,
        verbose=False
    )
    
    # 결과 검증
    assert 'episodes' in results
    assert 'metrics' in results
    assert 'trades' in results
    assert 'summary' in results
    
    # 에피소드 수 확인 (2개 종목)
    assert len(results['episodes']) == 2
    
    # 메트릭 확인
    metrics = results['metrics']
    assert 'win_rate' in metrics
    assert 'avg_profit_per_trade' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'avg_holding_time' in metrics
    
    # 요약 통계 확인
    summary = results['summary']
    assert summary['num_stocks'] == 2
    assert summary['num_days'] == 1
    assert 'total_slippage_cost' in summary
    assert 'total_transaction_cost' in summary
    
    simulator.close()


def test_save_results(mock_inference_engine, temp_db, tmp_path):
    """결과 저장 테스트"""
    simulator = BacktestSimulator(
        inference_engine=mock_inference_engine,
        db_path=temp_db,
        table_name='datasets',
        seq_len=10
    )
    
    # 백테스팅 실행
    results = simulator.run_backtest(
        deterministic=True,
        verbose=False
    )
    
    # 결과 저장
    output_path = tmp_path / 'backtest_results.json'
    simulator.save_results(results, str(output_path))
    
    # 파일 존재 확인
    assert output_path.exists()
    
    # JSON 파일 읽기
    import json
    with open(output_path, 'r', encoding='utf-8') as f:
        loaded_results = json.load(f)
    
    # 데이터 검증
    assert 'episodes' in loaded_results
    assert 'metrics' in loaded_results
    assert 'trades' in loaded_results
    assert 'summary' in loaded_results
    
    simulator.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
