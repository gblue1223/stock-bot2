#!/usr/bin/env python3
"""
DirectFeaturePolicy 추론 테스트

지정된 모델로 202509 월 데이터에 대해 추론을 테스트합니다.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import duckdb
import numpy as np
import torch

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.inference.infer_grpo import GRPOInference

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(db_path, year_month, seq_len=30, limit=1000):
    """
    테스트 데이터 로드 - 다양한 종목/날짜에서 균등 샘플링
    
    Args:
        db_path: DuckDB 경로
        year_month: 테스트 기간 (예: 202509)
        seq_len: 시퀀스 길이
        limit: 최대 샘플 수
        
    Returns:
        (sequences, dates, codes) 튜플
    """
    logger.info(f"Loading test data from {year_month}...")
    
    conn = duckdb.connect(db_path, read_only=True)
    
    try:
        # 202509 월 데이터 조회
        query = f"""
        SELECT *
        FROM datasets_raw
        WHERE 날짜 >= '{year_month}01' AND 날짜 < '{year_month}32'
        ORDER BY 종목코드, 날짜, 시간
        """
        
        df = conn.execute(query).fetchdf()
        
        if len(df) == 0:
            logger.error(f"No data found for {year_month}")
            return None, None, None
        
        logger.info(f"Loaded {len(df)} rows")
        
        # 모델 학습 시 사용한 특징 컬럼만 선택 (24개)
        feature_cols = [
            '등락률',
            '누적거래대금',
            '거래회전율',
            '체결강도',
            '매도대기금액1', '매도대기금액2', '매도대기금액3', '매도대기금액4', '매도대기금액5',
            '매도대기금액6', '매도대기금액7', '매도대기금액8', '매도대기금액9', '매도대기금액10',
            '매수대기금액1', '매수대기금액2', '매수대기금액3', '매수대기금액4', '매수대기금액5',
            '매수대기금액6', '매수대기금액7', '매수대기금액8', '매수대기금액9', '매수대기금액10'
        ]
        
        # 컬럼 존재 확인
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) != len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Missing columns: {missing}, using {len(available_cols)} columns")
            feature_cols = available_cols
        
        # 종목-날짜 조합별로 그룹화
        sequences = []
        dates = []
        codes = []
        times = []
        
        # 각 종목-날짜 조합에서 균등하게 샘플링
        grouped = df.groupby(['종목코드', '날짜'])
        
        # 샘플링할 그룹 수 계산
        total_groups = len(grouped)
        samples_per_group = max(1, limit // total_groups)
        
        logger.info(f"Found {total_groups} stock-date combinations")
        logger.info(f"Sampling up to {samples_per_group} sequences per combination")
        
        for (code, date), group in grouped:
            if len(sequences) >= limit:
                break
            
            # 시간순 정렬
            group = group.sort_values('시간')
            
            # NaN을 0으로 채우기
            features = group[feature_cols].fillna(0.0).values.astype(np.float32)
            
            # 해당 종목-날짜 조합에서 가능한 시퀀스 수
            max_sequences = len(features) - seq_len
            if max_sequences <= 0:
                continue
            
            # 균등한 간격으로 샘플링
            if max_sequences <= samples_per_group:
                # 가능한 모든 시퀀스 사용
                sample_indices = list(range(seq_len, len(features)))
            else:
                # 균등한 간격으로 샘플링
                step = max_sequences // samples_per_group
                sample_indices = list(range(seq_len, len(features), step))[:samples_per_group]
            
            for i in sample_indices:
                if len(sequences) >= limit:
                    break
                    
                seq = features[i-seq_len:i]
                sequences.append(seq)
                dates.append(date)
                codes.append(code)
                times.append(group.iloc[i]['시간'] if '시간' in group.columns else 0)
        
        if len(sequences) == 0:
            logger.error("No valid sequences found")
            return None, None, None
        
        sequences = np.array(sequences, dtype=np.float32)
        logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")
        logger.info(f"Unique stocks: {len(set(codes))}, Unique dates: {len(set(dates))}")
        
        return sequences, dates, codes
        
    finally:
        conn.close()


def test_inference(model_path, db_path, year_month='202509', limit=1000):
    """
    추론 테스트 실행
    
    Args:
        model_path: 모델 체크포인트 경로
        db_path: 데이터베이스 경로
        year_month: 테스트 기간
        limit: 최대 샘플 수
    """
    logger.info("=" * 80)
    logger.info("DirectFeaturePolicy Inference Test")
    logger.info("=" * 80)
    
    # 파일 확인
    logger.info("Checking files...")
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return False
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return False
    
    logger.info("✅ All files found")
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        # 1. 추론 엔진 초기화
        logger.info("Initializing inference engine...")
        
        inference = GRPOInference(
            policy_path=model_path,
            embedding_model_path=None,  # DirectFeaturePolicy는 불필요
            device=device,
            use_torchscript=False,  # 테스트에서는 비활성화
            cache_size=100
        )
        
        logger.info(f"✅ Inference engine initialized")
        logger.info(f"   Policy type: {inference.policy_type}")
        
        # 2. 테스트 데이터 로드
        sequences, dates, codes = load_test_data(db_path, year_month, seq_len=30, limit=limit)
        
        if sequences is None:
            return False
        
        # 3. 단일 샘플 추론 테스트
        logger.info("=" * 80)
        logger.info("Testing single sample inference...")
        
        sample_seq = sequences[0]
        start_time = time.time()
        action, confidence = inference.predict(sample_seq, deterministic=True)
        inference_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Single inference completed")
        logger.info(f"   Action: {action} (0=Hold, 1=Buy, 2=Sell)")
        logger.info(f"   Confidence: {confidence:.4f}")
        logger.info(f"   Inference time: {inference_time:.2f}ms")
        
        # 4. 배치 추론 테스트
        logger.info("=" * 80)
        logger.info("Testing batch inference...")
        
        batch_size = min(100, len(sequences))
        batch_sequences = sequences[:batch_size]
        
        start_time = time.time()
        actions, confidences = inference.predict_batch(batch_sequences, deterministic=True)
        batch_time = time.time() - start_time
        
        logger.info(f"✅ Batch inference completed")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Total time: {batch_time*1000:.2f}ms")
        logger.info(f"   Per sample: {(batch_time/batch_size)*1000:.2f}ms")
        
        # 5. 행동 분포 분석
        logger.info("=" * 80)
        logger.info("Analyzing action distribution...")
        
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in actions:
            action_counts[action] += 1
        
        logger.info(f"Action distribution:")
        logger.info(f"   Hold (0): {action_counts[0]} ({action_counts[0]/batch_size*100:.1f}%)")
        logger.info(f"   Buy  (1): {action_counts[1]} ({action_counts[1]/batch_size*100:.1f}%)")
        logger.info(f"   Sell (2): {action_counts[2]} ({action_counts[2]/batch_size*100:.1f}%)")
        
        # 6. 신뢰도 분석
        logger.info("=" * 80)
        logger.info("Analyzing confidence scores...")
        
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)
        
        logger.info(f"Confidence statistics:")
        logger.info(f"   Mean: {mean_conf:.4f}")
        logger.info(f"   Std:  {std_conf:.4f}")
        logger.info(f"   Min:  {min_conf:.4f}")
        logger.info(f"   Max:  {max_conf:.4f}")
        
        # 7. 전체 데이터셋 추론 (더 많은 샘플)
        if len(sequences) > batch_size:
            logger.info("=" * 80)
            logger.info(f"Testing on full dataset ({len(sequences)} samples)...")
            
            start_time = time.time()
            all_actions, all_confidences = inference.predict_batch(sequences, deterministic=True)
            full_time = time.time() - start_time
            
            logger.info(f"✅ Full dataset inference completed")
            logger.info(f"   Total samples: {len(sequences)}")
            logger.info(f"   Total time: {full_time:.2f}s")
            logger.info(f"   Per sample: {(full_time/len(sequences))*1000:.2f}ms")
            logger.info(f"   Throughput: {len(sequences)/full_time:.1f} samples/sec")
            
            # 전체 데이터셋 행동 분포
            full_action_counts = {0: 0, 1: 0, 2: 0}
            for action in all_actions:
                full_action_counts[action] += 1
            
            logger.info(f"Full dataset action distribution:")
            logger.info(f"   Hold (0): {full_action_counts[0]} ({full_action_counts[0]/len(sequences)*100:.1f}%)")
            logger.info(f"   Buy  (1): {full_action_counts[1]} ({full_action_counts[1]/len(sequences)*100:.1f}%)")
            logger.info(f"   Sell (2): {full_action_counts[2]} ({full_action_counts[2]/len(sequences)*100:.1f}%)")
        
        # 8. 성능 통계
        logger.info("=" * 80)
        logger.info("Performance statistics...")
        
        stats = inference.get_performance_stats()
        logger.info(f"   Mean inference time: {stats['mean_inference_time_ms']:.2f}ms")
        logger.info(f"   Median inference time: {stats['median_inference_time_ms']:.2f}ms")
        logger.info(f"   P95 inference time: {stats['p95_inference_time_ms']:.2f}ms")
        logger.info(f"   P99 inference time: {stats['p99_inference_time_ms']:.2f}ms")
        
        # 9. 샘플 예측 출력
        logger.info("=" * 80)
        logger.info("Sample predictions (first 10):")
        logger.info(f"{'Index':<8} {'Date':<12} {'Code':<10} {'Action':<8} {'Confidence':<12}")
        logger.info("-" * 60)
        
        for i in range(min(10, len(sequences))):
            action = all_actions[i] if len(sequences) > batch_size else actions[i]
            conf = all_confidences[i] if len(sequences) > batch_size else confidences[i]
            action_name = ['Hold', 'Buy', 'Sell'][action]
            logger.info(f"{i:<8} {dates[i]:<12} {codes[i]:<10} {action_name:<8} {conf:.6f}")
        
        logger.info("=" * 80)
        logger.info("✅ Inference test completed successfully!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}", exc_info=True)
        return False


def main():
    """메인 함수"""
    
    # 경로 설정
    model_path = r"d:\Workspace\Project\stock-bot\stock-bot2\models\grpo_direct_features@20251016\direct_features_model.pt"
    db_path = r"C:\Users\user\Workspace\datasets@20251016\datasets_raw_all.duckdb"
    year_month = "202509"
    limit = 1000
    
    logger.info("DirectFeaturePolicy Inference Test")
    logger.info(f"Model: {model_path}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Test period: {year_month}")
    logger.info(f"Sample limit: {limit}")
    
    success = test_inference(model_path, db_path, year_month, limit)
    
    if success:
        print("\n✅ Inference test completed successfully!")
    else:
        print("\n❌ Inference test failed.")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
