#!/usr/bin/env python3
"""
합성 데이터로 모델 테스트

실제 시장 데이터와 유사한 합성 데이터를 생성하여 모델의 예측 분포를 확인합니다.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.inference.infer_grpo import GRPOInference
from scripts.live.online_normalizer import OnlineNormalizer

def generate_synthetic_sequences(num_sequences: int = 100, seq_len: int = 60, num_features: int = 28):
    """실제 시장 데이터와 유사한 합성 시퀀스 생성"""
    
    print(f"Generating {num_sequences} synthetic sequences...")
    
    sequences = []
    
    for i in range(num_sequences):
        # 시나리오별 데이터 생성
        scenario = i % 5
        
        if scenario == 0:
            # 상승 추세
            base_price = 10000
            trend = np.linspace(0, 1000, seq_len)
            noise = np.random.randn(seq_len) * 100
            prices = base_price + trend + noise
            
        elif scenario == 1:
            # 하락 추세
            base_price = 10000
            trend = np.linspace(0, -1000, seq_len)
            noise = np.random.randn(seq_len) * 100
            prices = base_price + trend + noise
            
        elif scenario == 2:
            # 횡보
            base_price = 10000
            noise = np.random.randn(seq_len) * 50
            prices = base_price + noise
            
        elif scenario == 3:
            # 급등
            base_price = 10000
            trend = np.exp(np.linspace(0, 1, seq_len)) - 1
            trend = trend * 500
            noise = np.random.randn(seq_len) * 50
            prices = base_price + trend + noise
            
        else:
            # 급락
            base_price = 10000
            trend = -(np.exp(np.linspace(0, 1, seq_len)) - 1)
            trend = trend * 500
            noise = np.random.randn(seq_len) * 50
            prices = base_price + trend + noise
        
        # 28개 특징 생성
        seq = np.zeros((seq_len, num_features), dtype=np.float32)
        
        for t in range(seq_len):
            # 파생 피처 (0-3): 정규화된 값
            seq[t, 0] = np.random.randn() * 0.5  # 가격 변화율
            seq[t, 1] = np.random.randn() * 0.3  # 거래량 변화율
            seq[t, 2] = np.random.randn() * 0.2  # 모멘텀
            seq[t, 3] = np.random.randn() * 0.1  # 변동성
            
            # 기본 지표 (4-7): 원본 스케일
            seq[t, 4] = (prices[t] - prices[max(0, t-1)]) / prices[max(0, t-1)] * 100  # 등락률
            seq[t, 5] = np.random.uniform(1e9, 1e11)  # 누적거래대금
            seq[t, 6] = np.random.uniform(0.1, 10.0)  # 거래회전율
            seq[t, 7] = np.random.uniform(50, 150)  # 체결강도
            
            # 대기금액 (8-27): 매도/매수 각 10개
            for j in range(20):
                seq[t, 8+j] = np.random.uniform(1e6, 1e9)
        
        sequences.append(seq)
    
    return sequences

def test_model_with_synthetic_data(
    model_path: str,
    embedding_model_path: str,
    normalization_stats_path: str,
    num_sequences: int = 100,
    device: str = 'cuda'
):
    """합성 데이터로 모델 테스트"""
    
    print("="*80)
    print("합성 데이터로 모델 테스트")
    print("="*80)
    
    # 1. 정규화 통계 로드
    print("\n[1] Loading normalization statistics...")
    with open(normalization_stats_path, 'r', encoding='utf-8') as f:
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
        normalization_stats=None
    )
    print(f"   Model loaded on {device}")
    
    # 4. 합성 데이터 생성
    print(f"\n[4] Generating {num_sequences} synthetic sequences...")
    sequences = generate_synthetic_sequences(num_sequences, seq_len=60, num_features=28)
    
    # 5. 예측 수행
    print(f"\n[5] Running predictions...")
    
    predictions = []
    confidences = []
    scenario_predictions = {i: [] for i in range(5)}
    
    for i, seq in enumerate(sequences):
        scenario = i % 5
        
        # 정규화
        normalized_seq = np.zeros_like(seq)
        for j in range(seq.shape[0]):
            normalized_seq[j] = normalizer.normalize(seq[j], update=False)
        
        # 예측
        action, confidence = inference.predict(normalized_seq, deterministic=True)
        predictions.append(action)
        confidences.append(confidence)
        scenario_predictions[scenario].append((action, confidence))
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{num_sequences}")
    
    # 6. 결과 분석
    print("\n" + "="*80)
    print("예측 결과 분석")
    print("="*80)
    
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    # 전체 분포
    print("\n[전체 행동 분포]")
    from collections import Counter
    action_counts = Counter(predictions)
    
    for action in [0, 1, 2]:
        count = action_counts[action]
        percentage = (count / len(predictions)) * 100
        print(f"  {action_names[action]:5s} ({action}): {count:5d} ({percentage:5.2f}%)")
    
    # 시나리오별 분포
    print("\n[시나리오별 행동 분포]")
    scenario_names = {
        0: '상승 추세',
        1: '하락 추세',
        2: '횡보',
        3: '급등',
        4: '급락'
    }
    
    for scenario in range(5):
        preds = scenario_predictions[scenario]
        actions = [p[0] for p in preds]
        action_counts_scenario = Counter(actions)
        
        print(f"\n  {scenario_names[scenario]}:")
        for action in [0, 1, 2]:
            count = action_counts_scenario[action]
            percentage = (count / len(actions)) * 100 if actions else 0
            avg_conf = np.mean([p[1] for p in preds if p[0] == action]) if count > 0 else 0
            print(f"    {action_names[action]:5s}: {count:2d} ({percentage:5.1f}%) - 평균 신뢰도: {avg_conf:.4f}")
    
    # 신뢰도 통계
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
    
    # 샘플 예측
    print("\n[샘플 예측 (각 시나리오별 첫 3개)]")
    for scenario in range(5):
        print(f"\n  {scenario_names[scenario]}:")
        preds = scenario_predictions[scenario][:3]
        for i, (action, confidence) in enumerate(preds):
            print(f"    #{i+1}: {action_names[action]:5s} (confidence={confidence:.4f})")
    
    print("\n" + "="*80)
    print("테스트 완료")
    print("="*80)
    
    return {
        'predictions': predictions,
        'confidences': confidences,
        'scenario_predictions': scenario_predictions
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model with synthetic data')
    parser.add_argument('--model', default='models/grpo_scalping@2025120/model.pt',
                        help='Model path')
    parser.add_argument('--embedding', default='models/autoencoder@20251013/model.pt',
                        help='Embedding model path')
    parser.add_argument('--norm-stats', default='models/grpo_scalping@2025120/normalization_stats.json',
                        help='Normalization stats path')
    parser.add_argument('--sequences', type=int, default=100,
                        help='Number of sequences to test')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    for path, name in [
        (args.model, 'Model'),
        (args.embedding, 'Embedding model'),
        (args.norm_stats, 'Normalization stats')
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)
    
    # 테스트 실행
    results = test_model_with_synthetic_data(
        model_path=args.model,
        embedding_model_path=args.embedding,
        normalization_stats_path=args.norm_stats,
        num_sequences=args.sequences,
        device=args.device
    )
