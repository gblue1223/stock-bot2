#!/usr/bin/env python3
"""
모델 체크포인트 검사

모델 파일을 로드하여 구조와 파라미터를 확인합니다.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.policies import GRPOPolicy
from ai_trader.embedding import AutoEncoderEmbedding

def inspect_model(model_path: str, embedding_model_path: str):
    """모델 체크포인트 검사"""
    
    print("="*80)
    print("모델 체크포인트 검사")
    print("="*80)
    
    # 1. 정책 모델 로드
    print(f"\n[1] Loading policy model: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\n체크포인트 키:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # 정책 상태 로드
    if 'policy_state_dict' in checkpoint:
        policy_state = checkpoint['policy_state_dict']
    elif 'model_state_dict' in checkpoint:
        policy_state = checkpoint['model_state_dict']
    else:
        policy_state = checkpoint
    
    print(f"\n정책 파라미터:")
    total_params = 0
    for name, param in policy_state.items():
        num_params = param.numel()
        total_params += num_params
        print(f"  {name:40s}: {str(param.shape):20s} ({num_params:,} params)")
    
    print(f"\n총 파라미터 수: {total_params:,}")
    
    # 2. 정책 초기화 및 로드
    print(f"\n[2] Initializing policy...")
    
    # GRPOPolicy 초기화 (기본 파라미터)
    policy = GRPOPolicy(
        embedding_dim=128,
        hidden_dim=128,
        action_dim=3
    )
    
    try:
        policy.load_state_dict(policy_state)
        print("✅ Policy loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        return
    
    # 3. 임베딩 모델 로드
    print(f"\n[3] Loading embedding model: {embedding_model_path}")
    
    emb_checkpoint = torch.load(embedding_model_path, map_location='cpu')
    
    print(f"\n임베딩 체크포인트 키:")
    for key in emb_checkpoint.keys():
        print(f"  - {key}")
    
    if 'model_state_dict' in emb_checkpoint:
        emb_state = emb_checkpoint['model_state_dict']
    else:
        emb_state = emb_checkpoint
    
    print(f"\n임베딩 파라미터:")
    total_emb_params = 0
    for name, param in emb_state.items():
        num_params = param.numel()
        total_emb_params += num_params
        print(f"  {name:40s}: {str(param.shape):20s} ({num_params:,} params)")
    
    print(f"\n총 임베딩 파라미터 수: {total_emb_params:,}")
    
    # 4. 테스트 추론
    print(f"\n[4] Testing inference...")
    
    policy.eval()
    
    # 랜덤 임베딩 생성 (배치 크기 10)
    batch_size = 10
    embedding_dim = 128
    test_embeddings = torch.randn(batch_size, embedding_dim)
    
    print(f"  입력 shape: {test_embeddings.shape}")
    
    with torch.no_grad():
        action_logits, values = policy(test_embeddings)
    
    print(f"  출력 action_logits shape: {action_logits.shape}")
    print(f"  출력 values shape: {values.shape}")
    
    # 행동 확률 계산
    action_probs = torch.softmax(action_logits, dim=-1)
    
    print(f"\n  행동 확률 (첫 5개 샘플):")
    for i in range(min(5, batch_size)):
        probs = action_probs[i].numpy()
        action = torch.argmax(action_probs[i]).item()
        print(f"    샘플 {i+1}: HOLD={probs[0]:.4f}, BUY={probs[1]:.4f}, SELL={probs[2]:.4f} -> {['HOLD', 'BUY', 'SELL'][action]}")
    
    # 5. 행동 분포 통계
    print(f"\n[5] Action distribution statistics (100 random samples)...")
    
    num_test_samples = 100
    test_embeddings = torch.randn(num_test_samples, embedding_dim)
    
    with torch.no_grad():
        action_logits, _ = policy(test_embeddings)
        action_probs = torch.softmax(action_logits, dim=-1)
        actions = torch.argmax(action_probs, dim=-1)
    
    action_counts = {
        0: (actions == 0).sum().item(),
        1: (actions == 1).sum().item(),
        2: (actions == 2).sum().item()
    }
    
    print(f"\n  행동 분포 (랜덤 임베딩):")
    for action, name in [(0, 'HOLD'), (1, 'BUY'), (2, 'SELL')]:
        count = action_counts[action]
        percentage = (count / num_test_samples) * 100
        print(f"    {name:5s}: {count:3d} / {num_test_samples} ({percentage:5.2f}%)")
    
    # 평균 확률
    mean_probs = action_probs.mean(dim=0).numpy()
    print(f"\n  평균 행동 확률:")
    print(f"    HOLD: {mean_probs[0]:.4f}")
    print(f"    BUY:  {mean_probs[1]:.4f}")
    print(f"    SELL: {mean_probs[2]:.4f}")
    
    # 6. 정책 네트워크 구조
    print(f"\n[6] Policy network structure:")
    print(policy)
    
    print("\n" + "="*80)
    print("검사 완료")
    print("="*80)
    
    # 요약
    print("\n[요약]")
    print(f"  정책 파라미터: {total_params:,}")
    print(f"  임베딩 파라미터: {total_emb_params:,}")
    print(f"  총 파라미터: {total_params + total_emb_params:,}")
    print(f"\n  랜덤 임베딩 테스트 결과:")
    print(f"    HOLD: {action_counts[0]:3d} ({action_counts[0]/num_test_samples*100:.1f}%)")
    print(f"    BUY:  {action_counts[1]:3d} ({action_counts[1]/num_test_samples*100:.1f}%)")
    print(f"    SELL: {action_counts[2]:3d} ({action_counts[2]/num_test_samples*100:.1f}%)")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect model checkpoint')
    parser.add_argument('--model', default='models/grpo_scalping@2025120/model.pt',
                        help='Model path')
    parser.add_argument('--embedding', default='models/autoencoder@20251013/model.pt',
                        help='Embedding model path')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    for path, name in [
        (args.model, 'Model'),
        (args.embedding, 'Embedding model')
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)
    
    inspect_model(args.model, args.embedding)
