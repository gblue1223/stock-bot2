#!/usr/bin/env python3
"""
AutoEncoder + GRPO 통합 테스트 (합성 데이터 사용)

실제 데이터베이스가 없는 경우를 위한 합성 데이터를 사용한 통합 테스트입니다.
"""

import tempfile
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integrate_autoencoder_grpo import AutoEncoderGRPOIntegration

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_database(db_path: str, n_samples: int = 10000):
    """합성 주식 데이터로 DuckDB 생성"""
    logger.info(f"Creating synthetic database with {n_samples:,} samples...")
    
    # 합성 데이터 생성
    np.random.seed(42)  # 재현 가능한 결과를 위해
    
    # 종목과 날짜 조합 생성 (순차적 데이터 보장)
    stocks = ['005930', '000660', '035420', '051910', '068270']
    stock_names = ['삼성전자', 'SK하이닉스', 'NAVER', 'LG화학', '셀트리온']
    dates = [20240901, 20240902, 20240903, 20240904, 20240905]
    
    # 각 종목/날짜 조합당 충분한 데이터 생성
    samples_per_combination = max(200, n_samples // (len(stocks) * len(dates)))
    
    all_data = []
    
    for stock_idx, (stock_code, stock_name) in enumerate(zip(stocks, stock_names)):
        for date in dates:
            # 각 종목/날짜 조합에 대해 순차적 데이터 생성
            # 정확히 28개의 숫자형 특성을 생성 (모델이 기대하는 차원에 맞춤)
            combination_data = {
                '날짜': np.full(samples_per_combination, date, dtype=np.int64),
                '종목코드': np.full(samples_per_combination, stock_code),
                '종목명': np.full(samples_per_combination, stock_name),
                '시간': np.linspace(90000, 153000, samples_per_combination).astype(np.int64),  # 순차적 시간
                '등락률': np.random.normal(0, 2.0, samples_per_combination).astype(np.float64),
                '누적거래대금': np.random.exponential(1e9, samples_per_combination).astype(np.float64),
                '거래회전율': np.random.exponential(0.1, samples_per_combination).astype(np.float64),
                '체결강도': np.random.normal(100, 30, samples_per_combination).astype(np.float64)
            }
            
            # 호가 데이터 (매도/매수 대기금액 1~10) - 20개 특성
            for i in range(1, 11):
                combination_data[f'매도대기금액{i}'] = np.random.exponential(1e6, samples_per_combination).astype(np.float64)
                combination_data[f'매수대기금액{i}'] = np.random.exponential(1e6, samples_per_combination).astype(np.float64)
            
            # 정규화된 특성들 - 4개 특성 (총 28개가 되도록)
            # 기본 특성 5개 + 호가 특성 20개 + 정규화 특성 3개 = 28개
            combination_data['종목명_scalar'] = np.full(samples_per_combination, stock_idx / len(stocks), dtype=np.float64)
            time_normalized = np.linspace(0, 1, samples_per_combination)
            combination_data['시간_sin'] = np.sin(2 * np.pi * time_normalized).astype(np.float64)
            combination_data['시간_cos'] = np.cos(2 * np.pi * time_normalized).astype(np.float64)
            # 시간_scalar 제거하여 정확히 28개 특성 유지
            
            # DataFrame 생성 및 추가
            combination_df = pd.DataFrame(combination_data)
            all_data.append(combination_df)
    
    # 모든 데이터 결합
    df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Generated {len(df):,} samples across {len(stocks)} stocks and {len(dates)} dates")
    logger.info(f"Samples per stock/date combination: {samples_per_combination}")
    
    # 데이터 검증
    for stock_code in stocks:
        for date in dates:
            count = len(df[(df['종목코드'] == stock_code) & (df['날짜'] == date)])
            logger.info(f"Stock {stock_code}, Date {date}: {count} samples")
    
    # DuckDB에 저장
    import duckdb
    
    conn = duckdb.connect(db_path)
    
    # 테이블 생성 및 데이터 삽입
    conn.execute("CREATE TABLE datasets AS SELECT * FROM df")
    
    # 인덱스 생성 (성능 향상)
    conn.execute("CREATE INDEX idx_date_time ON datasets(날짜, 시간)")
    conn.execute("CREATE INDEX idx_stock ON datasets(종목코드)")
    
    # 통계 확인
    result = conn.execute("SELECT COUNT(*) as count, MIN(날짜) as min_date, MAX(날짜) as max_date FROM datasets").fetchone()
    logger.info(f"Database created: {result[0]:,} records from {result[1]} to {result[2]}")
    
    conn.close()
    logger.info(f"Synthetic database saved to: {db_path}")


def main():
    """메인 함수"""
    logger.info("AutoEncoder + GRPO Integration Test with Synthetic Data")
    logger.info("=" * 80)
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. 합성 데이터베이스 생성
        db_path = temp_path / "synthetic_data.duckdb"
        create_synthetic_database(str(db_path), n_samples=5000)  # 작은 데이터셋으로 테스트
        
        # 2. AutoEncoder 모델 경로 확인
        autoencoder_path = "models/autoencoder/20251012_013148"
        if not Path(autoencoder_path).exists():
            logger.error(f"AutoEncoder model not found: {autoencoder_path}")
            logger.error("Please ensure the model exists before running this test.")
            return 1
        
        try:
            # 3. 통합 객체 생성
            logger.info("Creating integration object...")
            integration = AutoEncoderGRPOIntegration(
                autoencoder_path=autoencoder_path,
                device='auto'
            )
            
            # 4. 통합 테스트 실행
            logger.info("Running integration test...")
            test_results = integration.test_integration(
                db_path=str(db_path),
                table_name='datasets',
                n_samples=50  # 빠른 테스트를 위해 적은 샘플
            )
            
            # 5. 결과 출력
            logger.info("=" * 80)
            logger.info("INTEGRATION TEST RESULTS")
            logger.info("=" * 80)
            
            for key, value in test_results.items():
                logger.info(f"{key}: {value}")
            
            # 6. 간단한 GRPO 훈련 테스트 (매우 짧은 훈련)
            logger.info("\nTesting short GRPO training...")
            
            output_dir = temp_path / "grpo_test_output"
            
            trainer, history = integration.train_grpo_agent(
                db_path=str(db_path),
                output_dir=str(output_dir),
                table_name='datasets',
                episodes_per_group=2,  # 매우 작은 값
                num_groups=2,          # 매우 작은 값
                total_timesteps=1000,  # 매우 짧은 훈련
                learning_rate=3e-4
            )
            
            logger.info("=" * 80)
            logger.info("🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info("✓ AutoEncoder model loaded correctly")
            logger.info("✓ GRPO environment created successfully")
            logger.info("✓ Policy network initialized correctly")
            logger.info("✓ Integration works end-to-end")
            logger.info("✓ Short training completed without errors")
            
            if test_results['avg_inference_time_ms'] < 10:
                logger.info("✓ Meets real-time trading requirements")
            else:
                logger.warning("⚠ May not meet real-time trading requirements")
            
            return 0
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(main())