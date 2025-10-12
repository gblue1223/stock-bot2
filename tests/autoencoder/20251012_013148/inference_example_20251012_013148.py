#!/usr/bin/env python3
"""
AutoEncoder 모델 추론 예제
모델 경로: models/autoencoder/20251012_013148

이 스크립트는 훈련된 autoencoder 모델을 사용하여 실제 추론을 수행하는 예제입니다.
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder


def load_autoencoder_model(model_path: str, device: str = 'auto'):
    """AutoEncoder 모델 로드"""
    model_path = Path(model_path)
    
    # 디바이스 설정
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 정보 로드
    with open(model_path / 'model_info.json', 'r') as f:
        model_info = json.load(f)
    
    # 모델 생성
    config = model_info['config']
    data_shape = model_info['data_shape']
    
    model = MaskedAutoEncoder(
        input_dim=data_shape['num_features'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        seq_len=data_shape['seq_len'],
        dropout=config['dropout'],
        mask_ratio=config['mask_ratio']
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path / 'model.pt', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, model_info, device


def generate_sample_data(seq_len: int, num_features: int, batch_size: int = 1):
    """샘플 시계열 데이터 생성 (주식 데이터와 유사한 패턴)"""
    
    # 기본 가격 데이터 (정규화된 형태)
    data = torch.zeros(batch_size, seq_len, num_features)
    
    # 각 배치에 대해 다른 패턴 생성
    for b in range(batch_size):
        # 초기값 설정
        data[b, 0, :] = torch.randn(num_features) * 0.1
        
        # 시계열 패턴 생성
        for t in range(1, seq_len):
            # 이전 값에 작은 변화 추가 (주식 가격 변화와 유사)
            change = torch.randn(num_features) * 0.02
            
            # 일부 특성에 트렌드 추가
            if t > seq_len // 2:
                trend = torch.randn(num_features) * 0.001 * (t - seq_len // 2)
                change += trend
            
            data[b, t, :] = data[b, t-1, :] + change
    
    return data


def main():
    """메인 함수"""
    model_path = "models/autoencoder/20251012_013148"
    
    print("AutoEncoder 모델 추론 예제")
    print("=" * 50)
    
    # 1. 모델 로드
    print("1. 모델 로딩...")
    model, model_info, device = load_autoencoder_model(model_path)
    
    config = model_info['config']
    data_shape = model_info['data_shape']
    
    print(f"   - 디바이스: {device}")
    print(f"   - 입력 차원: {data_shape['num_features']} features, {data_shape['seq_len']} sequence length")
    print(f"   - 임베딩 차원: {config['embedding_dim']}")
    print(f"   - 모델 파라미터: {model_info['total_params']:,}")
    
    # 2. 샘플 데이터 생성
    print("\n2. 샘플 데이터 생성...")
    batch_size = 5
    sample_data = generate_sample_data(
        data_shape['seq_len'], 
        data_shape['num_features'], 
        batch_size
    ).to(device)
    
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 데이터 형태: {sample_data.shape}")
    print(f"   - 데이터 범위: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
    
    # 3. 추론 실행
    print("\n3. 추론 실행...")
    
    with torch.no_grad():
        start_time = time.time()
        reconstruction, embeddings, mask = model(sample_data)
        inference_time = time.time() - start_time
    
    print(f"   - 추론 시간: {inference_time:.4f}초")
    print(f"   - 샘플당 시간: {(inference_time/batch_size)*1000:.2f}ms")
    
    # 4. 결과 분석
    print("\n4. 결과 분석...")
    
    # 마스킹 정보
    mask_ratio = mask.float().mean().item()
    print(f"   - 마스킹 비율: {mask_ratio:.3f} (설정값: {config['mask_ratio']:.3f})")
    
    # 임베딩 통계
    emb_mean = embeddings.mean().item()
    emb_std = embeddings.std().item()
    emb_min = embeddings.min().item()
    emb_max = embeddings.max().item()
    
    print(f"   - 임베딩 통계:")
    print(f"     * 평균: {emb_mean:.4f}")
    print(f"     * 표준편차: {emb_std:.4f}")
    print(f"     * 범위: [{emb_min:.4f}, {emb_max:.4f}]")
    
    # 재구성 품질
    mask_expanded = mask.unsqueeze(-1).expand_as(sample_data)
    unmasked_input = sample_data[~mask_expanded]
    unmasked_recon = reconstruction[~mask_expanded]
    
    mse_loss = torch.nn.functional.mse_loss(unmasked_recon, unmasked_input)
    mae_loss = torch.nn.functional.l1_loss(unmasked_recon, unmasked_input)
    
    print(f"   - 재구성 품질:")
    print(f"     * MSE: {mse_loss:.6f}")
    print(f"     * MAE: {mae_loss:.6f}")
    
    # 5. 개별 샘플 분석
    print("\n5. 개별 샘플 분석...")
    
    for i in range(min(3, batch_size)):  # 처음 3개 샘플만 분석
        sample_embedding = embeddings[i]
        sample_mask = mask[i]
        
        # 임베딩 벡터의 주요 성분
        top_indices = torch.topk(torch.abs(sample_embedding), 5).indices
        top_values = sample_embedding[top_indices]
        
        print(f"   샘플 {i+1}:")
        print(f"     - 마스킹된 시점: {sample_mask.sum().item()}/{data_shape['seq_len']}")
        print(f"     - 주요 임베딩 성분: {top_values.cpu().numpy()}")
    
    # 6. 사용 예제
    print("\n6. 실제 사용 예제...")
    print("   다음과 같이 모델을 사용할 수 있습니다:")
    print()
    print("   ```python")
    print("   # 새로운 데이터에 대한 임베딩 생성")
    print("   with torch.no_grad():")
    print("       _, embeddings, _ = model(new_data)")
    print("   ")
    print("   # 임베딩을 다른 모델의 입력으로 사용")
    print("   # 예: GRPO 에이전트, 분류기 등")
    print("   trading_decision = trading_model(embeddings)")
    print("   ```")
    
    print("\n✓ 추론 예제 완료!")


if __name__ == '__main__':
    main()