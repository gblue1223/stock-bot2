#!/usr/bin/env python3
"""
임베딩 공간 시각화

랜덤 임베딩 vs 실제 임베딩의 분포를 비교합니다.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.inference.infer_grpo import GRPOInference
from ai_trader.embedding import AutoEncoderEmbedding
from scripts.live.online_normalizer import OnlineNormalizer

def generate_synthetic_sequences(num_sequences: int = 100, seq_len: int = 60, num_features: int = 28):
    """합성 시퀀스 생성"""
    sequences = []
    labels = []  # 시나리오 레이블
    
    for i in range(num_sequences):
        scenario = i % 5
        
        if scenario == 0:  # 상승
            base_price = 10000
            trend = np.linspace(0, 1000, seq_len)
            noise = np.random.randn(seq_len) * 100
            prices = base_price + trend + noise
        elif scenario == 1:  # 하락
            base_price = 10000
            trend = np.linspace(0, -1000, seq_len)
            noise = np.random.randn(seq_len) * 100
            prices = base_price + trend + noise
        elif scenario == 2:  # 횡보
            base_price = 10000
            noise = np.random.randn(seq_len) * 50
            prices = base_price + noise
        elif scenario == 3:  # 급등
            base_price = 10000
            trend = np.exp(np.linspace(0, 1, seq_len)) - 1
            trend = trend * 500
            noise = np.random.randn(seq_len) * 50
            prices = base_price + trend + noise
        else:  # 급락
            base_price = 10000
            trend = -(np.exp(np.linspace(0, 1, seq_len)) - 1)
            trend = trend * 500
            noise = np.random.randn(seq_len) * 50
            prices = base_price + trend + noise
        
        # 28개 특징 생성
        seq = np.zeros((seq_len, num_features), dtype=np.float32)
        
        for t in range(seq_len):
            seq[t, 0] = np.random.randn() * 0.5
            seq[t, 1] = np.random.randn() * 0.3
            seq[t, 2] = np.random.randn() * 0.2
            seq[t, 3] = np.random.randn() * 0.1
            seq[t, 4] = (prices[t] - prices[max(0, t-1)]) / prices[max(0, t-1)] * 100
            seq[t, 5] = np.random.uniform(1e9, 1e11)
            seq[t, 6] = np.random.uniform(0.1, 10.0)
            seq[t, 7] = np.random.uniform(50, 150)
            for j in range(20):
                seq[t, 8+j] = np.random.uniform(1e6, 1e9)
        
        sequences.append(seq)
        labels.append(scenario)
    
    return sequences, labels

def visualize_embedding_space(
    model_path: str,
    embedding_model_path: str,
    normalization_stats_path: str,
    num_sequences: int = 100,
    device: str = 'cuda'
):
    """임베딩 공간 시각화"""
    
    print("="*80)
    print("임베딩 공간 시각화")
    print("="*80)
    
    # 1. 모델 로드
    print("\n[1] Loading models...")
    
    # 정규화 통계
    with open(normalization_stats_path, 'r', encoding='utf-8') as f:
        norm_stats = json.load(f)
    training_mean = np.array(norm_stats['mean'], dtype=np.float32)
    training_std = np.array(norm_stats['std'], dtype=np.float32)
    
    normalizer = OnlineNormalizer(
        num_features=28,
        use_training_stats=True,
        training_mean=training_mean,
        training_std=training_std
    )
    
    # 임베딩 모델
    emb_checkpoint = torch.load(embedding_model_path, map_location=device)
    embedding_model = AutoEncoderEmbedding(
        input_dim=28,
        embedding_dim=128,
        hidden_dim=256,
        seq_len=60
    ).to(device)
    embedding_model.load_state_dict(emb_checkpoint['model_state_dict'])
    embedding_model.eval()
    
    # 정책 모델
    inference = GRPOInference(
        policy_path=model_path,
        embedding_model_path=embedding_model_path,
        device=device,
        use_torchscript=False,
        normalization_stats=None
    )
    
    print("   Models loaded")
    
    # 2. 랜덤 임베딩 생성
    print("\n[2] Generating random embeddings...")
    num_random = 100
    random_embeddings = torch.randn(num_random, 128).to(device)
    
    with torch.no_grad():
        random_logits, _ = inference.policy(random_embeddings)
        random_probs = torch.softmax(random_logits, dim=-1)
        random_actions = torch.argmax(random_probs, dim=-1).cpu().numpy()
    
    random_embeddings = random_embeddings.cpu().numpy()
    
    print(f"   Random embeddings: {random_embeddings.shape}")
    print(f"   Action distribution: HOLD={np.sum(random_actions==0)}, BUY={np.sum(random_actions==1)}, SELL={np.sum(random_actions==2)}")
    
    # 3. 실제 임베딩 생성
    print("\n[3] Generating real embeddings from synthetic data...")
    sequences, scenario_labels = generate_synthetic_sequences(num_sequences)
    
    real_embeddings = []
    real_actions = []
    
    for seq in sequences:
        # 정규화
        normalized_seq = np.zeros_like(seq)
        for j in range(seq.shape[0]):
            normalized_seq[j] = normalizer.normalize(seq[j], update=False)
        
        # 임베딩 생성
        seq_tensor = torch.tensor(normalized_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = embedding_model.encode(seq_tensor).squeeze(0)
            real_embeddings.append(embedding.cpu().numpy())
            
            # 예측
            logits, _ = inference.policy(embedding.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            real_actions.append(action)
    
    real_embeddings = np.array(real_embeddings)
    real_actions = np.array(real_actions)
    
    print(f"   Real embeddings: {real_embeddings.shape}")
    print(f"   Action distribution: HOLD={np.sum(real_actions==0)}, BUY={np.sum(real_actions==1)}, SELL={np.sum(real_actions==2)}")
    
    # 4. 통계 비교
    print("\n[4] Comparing statistics...")
    
    print("\n  랜덤 임베딩:")
    print(f"    Mean: {random_embeddings.mean():.4f}")
    print(f"    Std:  {random_embeddings.std():.4f}")
    print(f"    Range: [{random_embeddings.min():.4f}, {random_embeddings.max():.4f}]")
    
    print("\n  실제 임베딩:")
    print(f"    Mean: {real_embeddings.mean():.4f}")
    print(f"    Std:  {real_embeddings.std():.4f}")
    print(f"    Range: [{real_embeddings.min():.4f}, {real_embeddings.max():.4f}]")
    
    # 차원별 통계
    print("\n  차원별 분산 비교 (처음 10개 차원):")
    for i in range(10):
        random_var = random_embeddings[:, i].var()
        real_var = real_embeddings[:, i].var()
        print(f"    Dim {i:2d}: Random={random_var:.4f}, Real={real_var:.4f}, Ratio={real_var/random_var:.4f}")
    
    # 5. PCA 시각화
    print("\n[5] Creating PCA visualization...")
    
    # 모든 임베딩 결합
    all_embeddings = np.vstack([random_embeddings, real_embeddings])
    all_labels = np.array(['Random']*len(random_embeddings) + ['Real']*len(real_embeddings))
    all_actions = np.concatenate([random_actions, real_actions])
    
    # PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(all_embeddings)
    
    # 플롯 1: 랜덤 vs 실제
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    random_mask = all_labels == 'Random'
    real_mask = all_labels == 'Real'
    plt.scatter(emb_2d[random_mask, 0], emb_2d[random_mask, 1], 
                c='blue', alpha=0.5, label='Random', s=50)
    plt.scatter(emb_2d[real_mask, 0], emb_2d[real_mask, 1], 
                c='red', alpha=0.5, label='Real', s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Random vs Real Embeddings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 플롯 2: 행동별 (랜덤)
    plt.subplot(1, 3, 2)
    for action, color, name in [(0, 'blue', 'HOLD'), (1, 'green', 'BUY'), (2, 'red', 'SELL')]:
        mask = (all_labels == 'Random') & (all_actions == action)
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                    c=color, alpha=0.6, label=name, s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Random Embeddings by Action')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 플롯 3: 행동별 (실제)
    plt.subplot(1, 3, 3)
    for action, color, name in [(0, 'blue', 'HOLD'), (1, 'green', 'BUY'), (2, 'red', 'SELL')]:
        mask = (all_labels == 'Real') & (all_actions == action)
        if np.sum(mask) > 0:
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                        c=color, alpha=0.6, label=name, s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Real Embeddings by Action')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_space_pca.png', dpi=150, bbox_inches='tight')
    print("   Saved: embedding_space_pca.png")
    
    # 6. t-SNE 시각화 (시간이 오래 걸림)
    print("\n[6] Creating t-SNE visualization (this may take a while)...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_tsne = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(emb_tsne[random_mask, 0], emb_tsne[random_mask, 1], 
                c='blue', alpha=0.5, label='Random', s=50)
    plt.scatter(emb_tsne[real_mask, 0], emb_tsne[real_mask, 1], 
                c='red', alpha=0.5, label='Real', s=50)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Random vs Real Embeddings (t-SNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    for action, color, name in [(0, 'blue', 'HOLD'), (1, 'green', 'BUY'), (2, 'red', 'SELL')]:
        mask = (all_labels == 'Random') & (all_actions == action)
        plt.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1], 
                    c=color, alpha=0.6, label=name, s=50)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Random Embeddings by Action (t-SNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    for action, color, name in [(0, 'blue', 'HOLD'), (1, 'green', 'BUY'), (2, 'red', 'SELL')]:
        mask = (all_labels == 'Real') & (all_actions == action)
        if np.sum(mask) > 0:
            plt.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1], 
                        c=color, alpha=0.6, label=name, s=50)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Real Embeddings by Action (t-SNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_space_tsne.png', dpi=150, bbox_inches='tight')
    print("   Saved: embedding_space_tsne.png")
    
    print("\n" + "="*80)
    print("시각화 완료")
    print("="*80)
    print("\n생성된 파일:")
    print("  - embedding_space_pca.png")
    print("  - embedding_space_tsne.png")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize embedding space')
    parser.add_argument('--model', default='models/grpo_scalping@2025120/model.pt')
    parser.add_argument('--embedding', default='models/autoencoder@20251013/model.pt')
    parser.add_argument('--norm-stats', default='models/grpo_scalping@2025120/normalization_stats.json')
    parser.add_argument('--sequences', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    visualize_embedding_space(
        model_path=args.model,
        embedding_model_path=args.embedding,
        normalization_stats_path=args.norm_stats,
        num_sequences=args.sequences,
        device=args.device
    )
