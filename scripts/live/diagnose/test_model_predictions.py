#!/usr/bin/env python3
"""
모델 예측 분포 테스트

실제 데이터로 모델의 예측 분포를 확인합니다.
- BUY/SELL/HOLD 비율
- 신뢰도 분포
- 정규화 전후 비교
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import json
from collections import Counter

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.inference.infer_grpo import GRPOInference
from scripts.live.online_normalizer import OnlineNormalizer
import duckdb

def load_sample_data(db_path: str, num_samples: int = 1000):
    """DuckDB에서 샘플 데이터 로드"""
    print(f"Loading {num_samples} samples from {db_path}...")
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # 테이블 확인
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Available tables: {tables}")
    
    # 첫 번째 테이블 사용
    table_name = tables[0][0] if tables else None
    if not table_name:
        raise ValueError("No tables found in database")
    
    print(f"Using table: {table_name}")
    
    # 랜덤 샘플 추출
    query = f"""
    SELECT * FROM {table_name}
    ORDER BY RANDOM()
    LIMIT {num_samples}
    """
    
    df = conn.execute(query).fetchdf()
    conn.close()
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def create_sequences(df, seq_len: int = 60, num_features: int = 28):
    """데이터프레임에서 시퀀스 생성"""
    print(f"Creating sequences (seq_len={seq_len}, num_features={num_features})...")
    
    sequences = []
    
    # 종목별로 그룹화
    if 'code' in df.columns:
        grouped = df.groupby('code')
    else:
        # code 컬럼이 없으면 전체를 하나의 그룹으로
        grouped = [(None, df)]
    
    for code, group in grouped:
        if len(group) < seq_len:
            continue
        
        # 특징 컬럼 추출 (숫자 컬럼만)
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['code', 'timestamp', 'date', 'time']]
        
        if len(feature_cols) < num_features:
            print(f"Warning: Only {len(feature_cols)} features found, expected {num_features}")
            continue
        
        # 첫 num_features 컬럼 사용
        feature_cols = feature_cols[:num_features]
        
        # 시퀀스 생성
        for i in range(len(group) - seq_len + 1):
            seq = group.iloc[i:i+seq_len][feature_cols].values
            if seq.shape == (seq_len, num_features):
                sequences.append(seq)
    
    print(f"Created {len(sequences)} sequences")
    return sequences

def test_model_predictions(
    model_path: str,
    embedding_model_path: str,
    normalization_stats_path: str,
    db_path: str,
    num_samples: int = 1000,
    device: str = 'cuda'
):
    """모델 예측 분포 테스트"""
    
    print("="*80)
    print("모델 예측 분포 테스트")
    print("="*80)
    
    # 1. 정규화 통계 로드
    print("\n[1] Loading normalization statistics...")
    with open(normalization_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    training_mean = np.array(norm_stats['mean'], dtype=np.float32)
    training_std = np.array(norm_stats['std'], dtype=np.float32)
    
    print(f"   Mean range: [{training_mean.min():.4f}, {training_mean.max():.4f}]")
    print(f"   Std range: [{training_std.min():.4f}, {training_std.max():.4f}]")
    
    # 2. 정규화기 초기화
    print("\n[2] Initializing normalizer...")
    normalizer = OnlineNormalizer(
        num_features=28,
        use_training_stats=True,
        training_mean=training_mean,
        training_std=training_std
    )
    
    # 3. 모델 로드
    print("\n[3] Loading model...")
    inference = GRPOInference(
        policy_path=model_path,
        embedding_model_path=embedding_model_path,
        device=device,
        use_torchscript=False,
        normalization_stats=None  # 정규화는 외부에서 처리
    )
    print(f"   Model loaded on {device}")
    
    # 4. 샘플 데이터 로드
    print("\n[4] Loading sample data...")
    df = load_sample_data(db_path, num_samples)
    
    # 5. 시퀀스 생성
    print("\n[5] Creating sequences...")
    sequences = create_sequences(df, seq_len=60, num_features=28)
    
    if len(sequences) == 0:
        print("ERROR: No sequences created!")
        return
    
    # 6. 예측 수행
    print(f"\n[6] Running predictions on {len(sequences)} sequences...")
    
    predictions = []
    confidences = []
    
    for i, seq in enumerate(sequences):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(sequences)}")
        
        # 정규화
        normalized_seq = np.zeros_like(seq)
        for j in range(seq.shape[0]):
            normalized_seq[j] = normalizer.normalize(seq[j], update=False)
        
        # 예측
        action, confidence = inference.predict(normalized_seq, deterministic=True)
        predictions.append(action)
        confidences.append(confidence)
    
    # 7. 결과 분석
    print("\n" + "="*80)
    print("예측 결과 분석")
    print("="*80)
    
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    action_counts = Counter(predictions)
    
    print("\n[행동 분포]")
    for action in [0, 1, 2]:
        count = action_counts[action]
        percentage = (count / len(predictions)) * 100
        print(f"  {action_names[action]:5s} ({action}): {count:5d} ({percentage:5.2f}%)")
    
    print("\n[신뢰도 통계]")
    confidences_array = np.array(confidences)
    print(f"  평균: {confidences_array.mean():.4f}")
    print(f"  표준편차: {confidences_array.std():.4f}")
    print(f"  최소: {confidences_array.min():.4f}")
    print(f"  최대: {confidences_array.max():.4f}")
    
    # 행동별 신뢰도
    print("\n[행동별 신뢰도]")
    for action in [0, 1, 2]:
        action_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == action]
        if action_confidences:
            print(f"  {action_names[action]:5s}: {np.mean(action_confidences):.4f} ± {np.std(action_confidences):.4f}")
        else:
            print(f"  {action_names[action]:5s}: N/A (no predictions)")
    
    # 8. 샘플 예측 출력
    print("\n[샘플 예측 (처음 20개)]")
    for i in range(min(20, len(predictions))):
        action = predictions[i]
        confidence = confidences[i]
        print(f"  #{i+1:3d}: {action_names[action]:5s} (confidence={confidence:.4f})")
    
    # 9. 정규화 전후 비교
    print("\n[정규화 전후 비교 (첫 번째 시퀀스)]")
    if len(sequences) > 0:
        original = sequences[0]
        normalized = np.zeros_like(original)
        for j in range(original.shape[0]):
            normalized[j] = normalizer.normalize(original[j], update=False)
        
        print(f"  원본 데이터:")
        print(f"    Mean: {original.mean():.2f}, Std: {original.std():.2f}")
        print(f"    Range: [{original.min():.2f}, {original.max():.2f}]")
        print(f"  정규화 후:")
        print(f"    Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        print(f"    Range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    print("\n" + "="*80)
    print("테스트 완료")
    print("="*80)
    
    return {
        'predictions': predictions,
        'confidences': confidences,
        'action_counts': action_counts
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model predictions')
    parser.add_argument('--model', default='models/grpo_scalping@2025120/model.pt',
                        help='Model path')
    parser.add_argument('--embedding', default='models/autoencoder/best_model.pt',
                        help='Embedding model path')
    parser.add_argument('--norm-stats', default='models/grpo_scalping@2025120/normalization_stats.json',
                        help='Normalization stats path')
    parser.add_argument('--db', default='datasets_norm_all.duckdb',
                        help='DuckDB database path')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to test')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    for path, name in [
        (args.model, 'Model'),
        (args.embedding, 'Embedding model'),
        (args.norm_stats, 'Normalization stats'),
        (args.db, 'Database')
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)
    
    # 테스트 실행
    results = test_model_predictions(
        model_path=args.model,
        embedding_model_path=args.embedding,
        normalization_stats_path=args.norm_stats,
        db_path=args.db,
        num_samples=args.samples,
        device=args.device
    )
