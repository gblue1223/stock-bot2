#!/usr/bin/env python3
"""
임베딩 공간 붕괴 진단 스크립트

임베딩 공간의 극단적 붕괴(collapse) 여부를 확인합니다.
- 차원별 분산 분석
- 유효 차원 수 (Effective Rank)
- 임베딩 간 거리 분포
- 클러스터링 품질
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
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


def load_model(model_path: str, device: str = 'cuda') -> AutoEncoderEmbedding:
    """모델 로드"""
    logger.info(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = AutoEncoderEmbedding(
        input_dim=config.get('input_dim', 60),
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        seq_len=config.get('seq_len', 60),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1)
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: embedding_dim={config.get('embedding_dim', 128)}")
    return model


def generate_test_data(
    num_samples: int = 1000,
    seq_len: int = 60,
    input_dim: int = 60,
    device: str = 'cuda'
) -> torch.Tensor:
    """테스트 데이터 생성"""
    logger.info(f"Generating {num_samples} test samples")
    
    # 다양한 패턴의 데이터 생성
    data = []
    
    # 1. 랜덤 노이즈
    data.append(torch.randn(num_samples // 4, seq_len, input_dim))
    
    # 2. 상승 트렌드
    trend_up = torch.randn(num_samples // 4, seq_len, input_dim)
    trend_up += torch.linspace(0, 2, seq_len).view(1, -1, 1)
    data.append(trend_up)
    
    # 3. 하락 트렌드
    trend_down = torch.randn(num_samples // 4, seq_len, input_dim)
    trend_down += torch.linspace(2, 0, seq_len).view(1, -1, 1)
    data.append(trend_down)
    
    # 4. 주기적 패턴
    t = torch.linspace(0, 4 * np.pi, seq_len).view(1, -1, 1)
    periodic = torch.randn(num_samples // 4, seq_len, input_dim)
    periodic += torch.sin(t)
    data.append(periodic)
    
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
    
    # 차원별 분산
    variances = np.var(embeddings, axis=0)
    
    # 통계
    stats = {
        'mean_variance': float(np.mean(variances)),
        'std_variance': float(np.std(variances)),
        'min_variance': float(np.min(variances)),
        'max_variance': float(np.max(variances)),
        'zero_variance_dims': int(np.sum(variances < 1e-6)),
        'low_variance_dims': int(np.sum(variances < 0.01)),
        'variances': variances
    }
    
    logger.info(f"Mean Variance: {stats['mean_variance']:.6f}")
    logger.info(f"Std Variance: {stats['std_variance']:.6f}")
    logger.info(f"Min Variance: {stats['min_variance']:.6f}")
    logger.info(f"Max Variance: {stats['max_variance']:.6f}")
    logger.info(f"Zero Variance Dims: {stats['zero_variance_dims']}")
    logger.info(f"Low Variance Dims (<0.01): {stats['low_variance_dims']}")
    
    # 경고
    if stats['zero_variance_dims'] > 0:
        logger.warning(f"[WARNING] {stats['zero_variance_dims']} dimensions have zero variance!")
    
    if stats['low_variance_dims'] > len(variances) * 0.5:
        logger.warning(f"[WARNING] {stats['low_variance_dims']} dimensions have low variance (>50%)!")
    
    return stats


def analyze_effective_rank(embeddings: np.ndarray) -> Dict:
    """유효 차원 수 (Effective Rank) 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("2. Effective Rank Analysis")
    logger.info("=" * 80)
    
    # SVD
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    
    # 정규화된 특이값
    S_normalized = S / np.sum(S)
    
    # Effective Rank (Shannon entropy 기반)
    effective_rank = np.exp(entropy(S_normalized))
    
    # 누적 분산 설명률
    cumsum_variance = np.cumsum(S_normalized)
    dims_90 = np.searchsorted(cumsum_variance, 0.90) + 1
    dims_95 = np.searchsorted(cumsum_variance, 0.95) + 1
    dims_99 = np.searchsorted(cumsum_variance, 0.99) + 1
    
    stats = {
        'effective_rank': float(effective_rank),
        'total_dims': len(S),
        'effective_ratio': float(effective_rank / len(S)),
        'dims_90': int(dims_90),
        'dims_95': int(dims_95),
        'dims_99': int(dims_99),
        'singular_values': S,
        'cumsum_variance': cumsum_variance
    }
    
    logger.info(f"Total Dimensions: {stats['total_dims']}")
    logger.info(f"Effective Rank: {stats['effective_rank']:.2f}")
    logger.info(f"Effective Ratio: {stats['effective_ratio']:.2%}")
    logger.info(f"Dims for 90% variance: {stats['dims_90']}")
    logger.info(f"Dims for 95% variance: {stats['dims_95']}")
    logger.info(f"Dims for 99% variance: {stats['dims_99']}")
    
    # 경고
    if stats['effective_ratio'] < 0.1:
        logger.error(f"[CRITICAL] Effective rank is only {stats['effective_ratio']:.1%} of total dimensions!")
        logger.error("This indicates severe embedding collapse!")
    elif stats['effective_ratio'] < 0.3:
        logger.warning(f"[WARNING] Effective rank is {stats['effective_ratio']:.1%} of total dimensions.")
        logger.warning("Embedding space may be underutilized.")
    
    return stats


def analyze_distances(embeddings: np.ndarray, sample_size: int = 1000) -> Dict:
    """임베딩 간 거리 분포 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("3. Distance Distribution Analysis")
    logger.info("=" * 80)
    
    # 샘플링 (계산 효율)
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample = embeddings[indices]
    else:
        sample = embeddings
    
    # 쌍별 거리 계산
    distances = pdist(sample, metric='euclidean')
    
    stats = {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'median_distance': float(np.median(distances)),
        'distances': distances
    }
    
    logger.info(f"Mean Distance: {stats['mean_distance']:.4f}")
    logger.info(f"Std Distance: {stats['std_distance']:.4f}")
    logger.info(f"Min Distance: {stats['min_distance']:.4f}")
    logger.info(f"Max Distance: {stats['max_distance']:.4f}")
    logger.info(f"Median Distance: {stats['median_distance']:.4f}")
    
    # 경고
    if stats['std_distance'] < stats['mean_distance'] * 0.1:
        logger.warning("[WARNING] Very low distance variance - embeddings may be too similar!")
    
    if stats['min_distance'] < 0.01:
        logger.warning(f"[WARNING] Some embeddings are very close (min={stats['min_distance']:.6f})")
    
    return stats


def analyze_clustering(embeddings: np.ndarray) -> Dict:
    """클러스터링 품질 분석"""
    logger.info("\n" + "=" * 80)
    logger.info("4. Clustering Quality Analysis")
    logger.info("=" * 80)
    
    # 중심으로부터의 거리
    center = np.mean(embeddings, axis=0)
    distances_from_center = np.linalg.norm(embeddings - center, axis=1)
    
    # 통계
    stats = {
        'mean_dist_from_center': float(np.mean(distances_from_center)),
        'std_dist_from_center': float(np.std(distances_from_center)),
        'max_dist_from_center': float(np.max(distances_from_center)),
        'center_norm': float(np.linalg.norm(center))
    }
    
    logger.info(f"Mean Distance from Center: {stats['mean_dist_from_center']:.4f}")
    logger.info(f"Std Distance from Center: {stats['std_dist_from_center']:.4f}")
    logger.info(f"Max Distance from Center: {stats['max_dist_from_center']:.4f}")
    logger.info(f"Center Norm: {stats['center_norm']:.4f}")
    
    # 경고
    if stats['std_dist_from_center'] < stats['mean_dist_from_center'] * 0.2:
        logger.warning("[WARNING] Embeddings are tightly clustered around center!")
    
    return stats


def visualize_results(
    variance_stats: Dict,
    rank_stats: Dict,
    distance_stats: Dict,
    embeddings: np.ndarray,
    output_dir: str = 'logs/embedding_diagnosis'
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
    ax.axhline(y=0.01, color='r', linestyle='--', label='Low variance threshold')
    ax.legend()
    
    # 2. 특이값 분포
    ax = axes[0, 1]
    S = rank_stats['singular_values']
    ax.plot(S)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values')
    ax.set_yscale('log')
    
    # 3. 누적 분산 설명률
    ax = axes[0, 2]
    cumsum = rank_stats['cumsum_variance']
    ax.plot(cumsum)
    ax.axhline(y=0.90, color='r', linestyle='--', label='90%')
    ax.axhline(y=0.95, color='g', linestyle='--', label='95%')
    ax.axhline(y=0.99, color='b', linestyle='--', label='99%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()
    ax.grid(True)
    
    # 4. 거리 분포
    ax = axes[1, 0]
    distances = distance_stats['distances']
    ax.hist(distances, bins=50, edgecolor='black')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('Pairwise Distance Distribution')
    ax.axvline(x=distance_stats['mean_distance'], color='r', linestyle='--', label='Mean')
    ax.legend()
    
    # 5. PCA 2D
    ax = axes[1, 1]
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings[:1000])  # 샘플링
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA 2D Projection')
    ax.grid(True)
    
    # 6. 요약 텍스트
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
Embedding Space Diagnosis Summary

1. Variance Analysis:
   - Mean Variance: {variance_stats['mean_variance']:.6f}
   - Zero Variance Dims: {variance_stats['zero_variance_dims']}
   - Low Variance Dims: {variance_stats['low_variance_dims']}

2. Effective Rank:
   - Total Dims: {rank_stats['total_dims']}
   - Effective Rank: {rank_stats['effective_rank']:.2f}
   - Effective Ratio: {rank_stats['effective_ratio']:.2%}
   - Dims for 95%: {rank_stats['dims_95']}

3. Distance Distribution:
   - Mean Distance: {distance_stats['mean_distance']:.4f}
   - Std Distance: {distance_stats['std_distance']:.4f}

Status: {'[CRITICAL] Collapse Detected!' if rank_stats['effective_ratio'] < 0.1 
         else '[WARNING] Underutilized' if rank_stats['effective_ratio'] < 0.3
         else '[OK] Normal'}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'embedding_collapse_diagnosis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    
    plt.close()


def load_direct_feature_policy(model_path: str, device: str = 'cuda'):
    """DirectFeaturePolicy 로드"""
    from ai_trader.grpo.policies.direct_feature_policy import DirectFeaturePolicy
    
    logger.info(f"Loading DirectFeaturePolicy from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # state_dict에서 차원 추론
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    else:
        state_dict = checkpoint['state_dict']
    
    input_dim = state_dict['input_projection.weight'].shape[1]
    hidden_dim = state_dict['input_projection.weight'].shape[0]
    
    policy = DirectFeaturePolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=config.get('action_dim', 3)
    )
    
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    
    logger.info(f"Policy loaded: input_dim={input_dim}, hidden_dim={hidden_dim}")
    return policy, input_dim, hidden_dim


def extract_hidden_activations(
    policy,
    data: torch.Tensor,
    batch_size: int = 128
) -> np.ndarray:
    """Hidden layer 활성화 추출"""
    logger.info("Extracting hidden activations...")
    
    activations = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            # input_projection 후 활성화
            hidden = policy.input_projection(batch)
            hidden = torch.relu(hidden)
            activations.append(hidden.cpu().numpy())
    
    activations = np.vstack(activations)
    logger.info(f"Activations shape: {activations.shape}")
    
    return activations


def main():
    """메인 함수"""
    logger.info("=" * 80)
    logger.info("Embedding Space Collapse Diagnosis")
    logger.info("=" * 80)
    
    # 모델 경로 - DirectFeaturePolicy 사용
    model_path = "models/grpo_direct_features@20251016/direct_features_model.pt"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Trying alternative path...")
        model_path = "models/grpo_scalping@2025120/model.pt"
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    try:
        # 1. 모델 로드 (DirectFeaturePolicy)
        policy, input_dim, hidden_dim = load_direct_feature_policy(model_path, device)
        
        # 2. 테스트 데이터 생성 (평탄화된 형태)
        logger.info(f"Generating test data: input_dim={input_dim}")
        num_samples = 1000
        
        # 다양한 패턴의 데이터
        data = []
        data.append(torch.randn(num_samples // 4, input_dim))  # 랜덤
        data.append(torch.randn(num_samples // 4, input_dim) + 1.0)  # 상승
        data.append(torch.randn(num_samples // 4, input_dim) - 1.0)  # 하락
        data.append(torch.randn(num_samples // 4, input_dim) * 2.0)  # 변동성
        
        test_data = torch.cat(data, dim=0).to(device)
        logger.info(f"Test data shape: {test_data.shape}")
        
        # 3. Hidden 활성화 추출
        embeddings = extract_hidden_activations(policy, test_data)
        
        # 4. 분석
        variance_stats = analyze_variance(embeddings)
        rank_stats = analyze_effective_rank(embeddings)
        distance_stats = analyze_distances(embeddings)
        clustering_stats = analyze_clustering(embeddings)
        
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
        
        if variance_stats['zero_variance_dims'] > 0:
            issues.append(f"Zero variance in {variance_stats['zero_variance_dims']} dimensions")
        
        if rank_stats['effective_ratio'] < 0.1:
            issues.append(f"Severe collapse: effective rank only {rank_stats['effective_ratio']:.1%}")
        elif rank_stats['effective_ratio'] < 0.3:
            issues.append(f"Underutilized: effective rank {rank_stats['effective_ratio']:.1%}")
        
        if distance_stats['std_distance'] < distance_stats['mean_distance'] * 0.1:
            issues.append("Very low distance variance")
        
        if issues:
            logger.error("[ISSUES DETECTED]")
            for issue in issues:
                logger.error(f"  - {issue}")
            
            logger.info("\nRecommendations:")
            logger.info("1. Increase embedding dimension")
            logger.info("2. Add regularization (dropout, weight decay)")
            logger.info("3. Use contrastive loss")
            logger.info("4. Check data normalization")
            logger.info("5. Reduce model capacity if overfitting")
        else:
            logger.info("[OK] No critical issues detected")
            logger.info("Embedding space appears healthy")
        
        return len(issues) == 0
        
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
