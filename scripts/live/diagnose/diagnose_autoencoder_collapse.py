#!/usr/bin/env python3
"""
AutoEncoder 임베딩 공간 붕괴 진단 (3491 모델)

AutoEncoderEmbedding 모델의 임베딩 공간 붕괴 여부를 확인합니다.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import entropy

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.embedding import AutoEncoderEmbedding

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_autoencoder(model_path: str, device: str = 'cuda') -> AutoEncoderEmbedding:
    """AutoEncoder 모델 로드"""
    logger.info(f"Loading AutoEncoder from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 설정 추출
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # model_info.json에서 로드
        model_dir = os.path.dirname(model_path)
        info_path = os.path.join(model_dir, 'model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                config = info.get('config', {})
        else:
            config = {}
    
    # state_dict에서 실제 차원 추론
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # input_dim 추론
    if 'encoder.input_projection.weight' in state_dict:
        input_dim = state_dict['encoder.input_projection.weight'].shape[1]
    else:
        input_dim = config.get('input_dim', 60)
    
    # 모델 생성
    model = AutoEncoderEmbedding(
        input_dim=input_dim,
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        seq_len=config.get('seq_len', 60),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1)
    )
    
    # 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded:")
    logger.info(f"  Input Dim: {input_dim}")
    logger.info(f"  Embedding Dim: {model.embedding_dim}")
    logger.info(f"  Hidden Dim: {config.get('hidden_dim', 256)}")
    logger.info(f"  Seq Len: {config.get('seq_len', 60)}")
    
    return model


def generate_diverse_data(
    num_samples: int,
    seq_len: int,
    input_dim: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """다양한 패턴의 테스트 데이터 생성"""
    logger.info(f"Generating {num_samples} diverse test samples")
    
    data = []
    samples_per_pattern = num_samples // 6
    
    # 1. 랜덤 노이즈
    data.append(torch.randn(samples_per_pattern, seq_len, input_dim))
    
    # 2. 상승 트렌드
    trend_up = torch.randn(samples_per_pattern, seq_len, input_dim) * 0.5
    trend_up += torch.linspace(0, 3, seq_len).view(1, -1, 1)
    data.append(trend_up)
    
    # 3. 하락 트렌드
    trend_down = torch.randn(samples_per_pattern, seq_len, input_dim) * 0.5
    trend_down += torch.linspace(3, 0, seq_len).view(1, -1, 1)
    data.append(trend_down)
    
    # 4. 주기적 패턴 (sin)
    t = torch.linspace(0, 4 * np.pi, seq_len).view(1, -1, 1)
    periodic_sin = torch.randn(samples_per_pattern, seq_len, input_dim) * 0.3
    periodic_sin += torch.sin(t) * 2
    data.append(periodic_sin)
    
    # 5. 주기적 패턴 (cos)
    periodic_cos = torch.randn(samples_per_pattern, seq_len, input_dim) * 0.3
    periodic_cos += torch.cos(t) * 2
    data.append(periodic_cos)
    
    # 6. 변동성 높은 패턴
    volatile = torch.randn(samples_per_pattern, seq_len, input_dim) * 3
    data.append(volatile)
    
    data = torch.cat(data, dim=0).to(device)
    logger.info(f"Test data shape: {data.shape}")
    
    return data


def extract_embeddings(
    model: AutoEncoderEmbedding,
    data: torch.Tensor,
    batch_size: int = 128
) -> np.ndarray:
    """임베딩 추출"""
    logger.info("Extracting embeddings...")
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            emb = model.encode(batch)
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings


def analyze_variance(embeddings: np.ndarray) -> Dict:
    """차원별 분산 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("1. Variance Analysis")
    logger.info("=" * 80)
    
    variances = np.var(embeddings, axis=0)
    
    stats = {
        'mean_variance': float(np.mean(variances)),
        'std_variance': float(np.std(variances)),
        'min_variance': float(np.min(variances)),
        'max_variance': float(np.max(variances)),
        'zero_variance_dims': int(np.sum(variances < 1e-6)),
        'low_variance_dims': int(np.sum(variances < 0.01)),
        'very_low_variance_dims': int(np.sum(variances < 0.001)),
        'variances': variances
    }
    
    logger.info(f"Mean Variance: {stats['mean_variance']:.6f}")
    logger.info(f"Std Variance: {stats['std_variance']:.6f}")
    logger.info(f"Min Variance: {stats['min_variance']:.6f}")
    logger.info(f"Max Variance: {stats['max_variance']:.6f}")
    logger.info(f"Zero Variance Dims (<1e-6): {stats['zero_variance_dims']}")
    logger.info(f"Very Low Variance Dims (<0.001): {stats['very_low_variance_dims']}")
    logger.info(f"Low Variance Dims (<0.01): {stats['low_variance_dims']}")
    
    # 경고
    if stats['zero_variance_dims'] > 0:
        logger.error(f"[CRITICAL] {stats['zero_variance_dims']} dimensions have ZERO variance!")
        logger.error("This indicates SEVERE embedding collapse!")
    
    if stats['very_low_variance_dims'] > len(variances) * 0.5:
        logger.error(f"[CRITICAL] {stats['very_low_variance_dims']} dimensions have very low variance (>50%)!")
    elif stats['low_variance_dims'] > len(variances) * 0.5:
        logger.warning(f"[WARNING] {stats['low_variance_dims']} dimensions have low variance (>50%)!")
    
    return stats


def analyze_effective_rank(embeddings: np.ndarray) -> Dict:
    """유효 차원 수 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("2. Effective Rank Analysis")
    logger.info("=" * 80)
    
    # SVD
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    
    # 정규화된 특이값
    S_normalized = S / np.sum(S)
    
    # Effective Rank
    effective_rank = np.exp(entropy(S_normalized))
    
    # 누적 분산
    cumsum_variance = np.cumsum(S_normalized)
    dims_50 = np.searchsorted(cumsum_variance, 0.50) + 1
    dims_90 = np.searchsorted(cumsum_variance, 0.90) + 1
    dims_95 = np.searchsorted(cumsum_variance, 0.95) + 1
    dims_99 = np.searchsorted(cumsum_variance, 0.99) + 1
    
    stats = {
        'effective_rank': float(effective_rank),
        'total_dims': len(S),
        'effective_ratio': float(effective_rank / len(S)),
        'dims_50': int(dims_50),
        'dims_90': int(dims_90),
        'dims_95': int(dims_95),
        'dims_99': int(dims_99),
        'singular_values': S,
        'cumsum_variance': cumsum_variance
    }
    
    logger.info(f"Total Dimensions: {stats['total_dims']}")
    logger.info(f"Effective Rank: {stats['effective_rank']:.2f}")
    logger.info(f"Effective Ratio: {stats['effective_ratio']:.2%}")
    logger.info(f"Dims for 50% variance: {stats['dims_50']}")
    logger.info(f"Dims for 90% variance: {stats['dims_90']}")
    logger.info(f"Dims for 95% variance: {stats['dims_95']}")
    logger.info(f"Dims for 99% variance: {stats['dims_99']}")
    
    # 경고
    if stats['effective_ratio'] < 0.05:
        logger.error(f"[CRITICAL] Effective rank is only {stats['effective_ratio']:.1%}!")
        logger.error("SEVERE EMBEDDING COLLAPSE DETECTED!")
    elif stats['effective_ratio'] < 0.1:
        logger.error(f"[CRITICAL] Effective rank is {stats['effective_ratio']:.1%}!")
        logger.error("Significant embedding collapse detected!")
    elif stats['effective_ratio'] < 0.3:
        logger.warning(f"[WARNING] Effective rank is {stats['effective_ratio']:.1%}.")
        logger.warning("Embedding space may be underutilized.")
    
    return stats


def analyze_distances(embeddings: np.ndarray, sample_size: int = 1000) -> Dict:
    """거리 분포 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("3. Distance Distribution Analysis")
    logger.info("=" * 80)
    
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample = embeddings[indices]
    else:
        sample = embeddings
    
    distances = pdist(sample, metric='euclidean')
    
    stats = {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'median_distance': float(np.median(distances)),
        'cv': float(np.std(distances) / np.mean(distances)),  # Coefficient of Variation
        'distances': distances
    }
    
    logger.info(f"Mean Distance: {stats['mean_distance']:.4f}")
    logger.info(f"Std Distance: {stats['std_distance']:.4f}")
    logger.info(f"Min Distance: {stats['min_distance']:.4f}")
    logger.info(f"Max Distance: {stats['max_distance']:.4f}")
    logger.info(f"Median Distance: {stats['median_distance']:.4f}")
    logger.info(f"Coefficient of Variation: {stats['cv']:.4f}")
    
    # 경고
    if stats['cv'] < 0.05:
        logger.error("[CRITICAL] Very low distance variance - severe collapse!")
    elif stats['cv'] < 0.1:
        logger.warning("[WARNING] Low distance variance - embeddings too similar!")
    
    if stats['min_distance'] < 0.001:
        logger.warning(f"[WARNING] Some embeddings are extremely close (min={stats['min_distance']:.6f})")
    
    return stats


def analyze_reconstruction(
    model: AutoEncoderEmbedding,
    data: torch.Tensor,
    num_samples: int = 100
) -> Dict:
    """재구성 품질 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("4. Reconstruction Quality Analysis")
    logger.info("=" * 80)
    
    # 샘플링
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    sample = data[indices]
    
    with torch.no_grad():
        reconstruction, embeddings = model(sample)
    
    # MSE 계산
    mse = torch.mean((sample - reconstruction) ** 2, dim=(1, 2))
    mse_np = mse.cpu().numpy()
    
    stats = {
        'mean_mse': float(np.mean(mse_np)),
        'std_mse': float(np.std(mse_np)),
        'min_mse': float(np.min(mse_np)),
        'max_mse': float(np.max(mse_np))
    }
    
    logger.info(f"Mean MSE: {stats['mean_mse']:.6f}")
    logger.info(f"Std MSE: {stats['std_mse']:.6f}")
    logger.info(f"Min MSE: {stats['min_mse']:.6f}")
    logger.info(f"Max MSE: {stats['max_mse']:.6f}")
    
    # 경고
    if stats['mean_mse'] > 1.0:
        logger.warning(f"[WARNING] High reconstruction error (MSE={stats['mean_mse']:.4f})")
    
    return stats


def visualize_results(
    variance_stats: Dict,
    rank_stats: Dict,
    distance_stats: Dict,
    embeddings: np.ndarray,
    output_dir: str = 'logs/autoencoder_diagnosis'
):
    """결과 시각화"""
    logger.info("\n" + "=" * 80)
    logger.info("5. Visualization")
    logger.info("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 차원별 분산
    ax = axes[0, 0]
    variances = variance_stats['variances']
    ax.bar(range(len(variances)), variances)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.set_title('Variance per Dimension')
    ax.axhline(y=0.001, color='r', linestyle='--', label='Very low (0.001)')
    ax.axhline(y=0.01, color='orange', linestyle='--', label='Low (0.01)')
    ax.legend()
    ax.set_yscale('log')
    
    # 2. 특이값 분포
    ax = axes[0, 1]
    S = rank_stats['singular_values']
    ax.plot(S, 'b-', linewidth=2)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. 누적 분산
    ax = axes[0, 2]
    cumsum = rank_stats['cumsum_variance']
    ax.plot(cumsum, 'b-', linewidth=2)
    ax.axhline(y=0.50, color='purple', linestyle='--', label='50%')
    ax.axhline(y=0.90, color='r', linestyle='--', label='90%')
    ax.axhline(y=0.95, color='g', linestyle='--', label='95%')
    ax.axhline(y=0.99, color='b', linestyle='--', label='99%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 거리 분포
    ax = axes[1, 0]
    distances = distance_stats['distances']
    ax.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('Pairwise Distance Distribution')
    ax.axvline(x=distance_stats['mean_distance'], color='r', linestyle='--', 
               linewidth=2, label=f"Mean: {distance_stats['mean_distance']:.2f}")
    ax.legend()
    
    # 5. PCA 2D
    ax = axes[1, 1]
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings[:1000])
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        alpha=0.5, s=10, c=range(len(embeddings_2d)), cmap='viridis')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA 2D Projection')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    
    # 6. 요약
    ax = axes[1, 2]
    ax.axis('off')
    
    # 상태 판정
    if rank_stats['effective_ratio'] < 0.05:
        status = '[CRITICAL] SEVERE COLLAPSE!'
        color = 'red'
    elif rank_stats['effective_ratio'] < 0.1:
        status = '[CRITICAL] Significant Collapse'
        color = 'red'
    elif rank_stats['effective_ratio'] < 0.3:
        status = '[WARNING] Underutilized'
        color = 'orange'
    else:
        status = '[OK] Normal'
        color = 'green'
    
    summary_text = f"""
AutoEncoder Embedding Space Diagnosis

1. Variance Analysis:
   Mean Variance: {variance_stats['mean_variance']:.6f}
   Zero Variance: {variance_stats['zero_variance_dims']}
   Very Low (<0.001): {variance_stats['very_low_variance_dims']}
   Low (<0.01): {variance_stats['low_variance_dims']}

2. Effective Rank:
   Total Dims: {rank_stats['total_dims']}
   Effective Rank: {rank_stats['effective_rank']:.2f}
   Effective Ratio: {rank_stats['effective_ratio']:.2%}
   Dims for 50%: {rank_stats['dims_50']}
   Dims for 90%: {rank_stats['dims_90']}
   Dims for 95%: {rank_stats['dims_95']}

3. Distance Distribution:
   Mean: {distance_stats['mean_distance']:.4f}
   Std: {distance_stats['std_distance']:.4f}
   CV: {distance_stats['cv']:.4f}

Status: {status}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', color=color if status != '[OK] Normal' else 'black')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'autoencoder_collapse_diagnosis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    
    plt.close()


def main():
    """메인 함수"""
    logger.info("=" * 80)
    logger.info("AutoEncoder Embedding Space Collapse Diagnosis")
    logger.info("=" * 80)
    
    # 모델 경로
    model_path = "models/autoencoder@20251013/model.pt"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    try:
        # 1. 모델 로드
        model = load_autoencoder(model_path, device)
        
        # 2. 테스트 데이터 생성
        # state_dict에서 차원 추론
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint.get('state_dict', checkpoint)
        
        input_dim = state_dict['encoder.input_projection.weight'].shape[1]
        seq_len = 60  # 기본값
        
        if 'config' in checkpoint:
            seq_len = checkpoint['config'].get('seq_len', 60)
        
        logger.info(f"Detected: input_dim={input_dim}, seq_len={seq_len}")
        
        test_data = generate_diverse_data(
            num_samples=1200,
            seq_len=seq_len,
            input_dim=input_dim,
            device=device
        )
        
        # 3. 임베딩 추출
        embeddings = extract_embeddings(model, test_data)
        
        # 4. 분석
        variance_stats = analyze_variance(embeddings)
        rank_stats = analyze_effective_rank(embeddings)
        distance_stats = analyze_distances(embeddings)
        recon_stats = analyze_reconstruction(model, test_data)
        
        # 5. 시각화
        visualize_results(
            variance_stats,
            rank_stats,
            distance_stats,
            embeddings
        )
        
        # 6. 최종 판정
        logger.info("\n" + "=" * 80)
        logger.info("Final Diagnosis")
        logger.info("=" * 80)
        
        issues = []
        critical_issues = []
        
        # 심각한 문제
        if variance_stats['zero_variance_dims'] > 0:
            critical_issues.append(f"ZERO variance in {variance_stats['zero_variance_dims']} dimensions")
        
        if rank_stats['effective_ratio'] < 0.05:
            critical_issues.append(f"SEVERE collapse: effective rank only {rank_stats['effective_ratio']:.1%}")
        elif rank_stats['effective_ratio'] < 0.1:
            critical_issues.append(f"Significant collapse: effective rank {rank_stats['effective_ratio']:.1%}")
        
        if distance_stats['cv'] < 0.05:
            critical_issues.append("SEVERE: Very low distance variance")
        
        # 경미한 문제
        if variance_stats['very_low_variance_dims'] > len(variance_stats['variances']) * 0.5:
            issues.append(f"Very low variance in {variance_stats['very_low_variance_dims']} dimensions (>50%)")
        
        if rank_stats['effective_ratio'] < 0.3 and rank_stats['effective_ratio'] >= 0.1:
            issues.append(f"Underutilized: effective rank {rank_stats['effective_ratio']:.1%}")
        
        if distance_stats['cv'] < 0.1 and distance_stats['cv'] >= 0.05:
            issues.append("Low distance variance")
        
        # 결과 출력
        if critical_issues:
            logger.error("\n[CRITICAL ISSUES DETECTED]")
            for issue in critical_issues:
                logger.error(f"  - {issue}")
            
            logger.error("\nThis model has SEVERE embedding collapse!")
            logger.error("DO NOT use for production!")
            
            logger.info("\nUrgent Actions Required:")
            logger.info("1. Retrain with different architecture")
            logger.info("2. Increase embedding dimension significantly")
            logger.info("3. Add strong regularization")
            logger.info("4. Use contrastive loss")
            logger.info("5. Check data preprocessing")
            
            return False
            
        elif issues:
            logger.warning("\n[ISSUES DETECTED]")
            for issue in issues:
                logger.warning(f"  - {issue}")
            
            logger.info("\nRecommendations:")
            logger.info("1. Consider increasing embedding dimension")
            logger.info("2. Add regularization (dropout, weight decay)")
            logger.info("3. Use contrastive loss")
            logger.info("4. Monitor during training")
        else:
            logger.info("\n[OK] No critical issues detected")
            logger.info("Embedding space appears healthy")
        
        return len(critical_issues) == 0
        
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
