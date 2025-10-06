"""
임베딩 모델 평가 메트릭

임베딩 품질을 평가하기 위한 메트릭을 제공합니다:
- Silhouette score: 종목별 클러스터링 품질 평가
- Temporal coherence: 시간적 일관성 평가
"""

import logging
from typing import List, Union, Optional
import numpy as np
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def compute_silhouette_score(
    embeddings: np.ndarray,
    stock_codes: List[Union[str, int]]
) -> float:
    """
    Silhouette score 계산 - 종목별 클러스터링 품질 평가
    
    Silhouette score는 클러스터링 품질을 측정하는 지표로,
    같은 종목의 임베딩이 얼마나 가까이 모여있고 다른 종목과는
    얼마나 멀리 떨어져 있는지를 평가합니다.
    
    점수 해석:
    - 1에 가까울수록: 클러스터가 잘 분리되어 있음 (좋음)
    - 0에 가까울수록: 클러스터가 겹쳐있음 (보통)
    - -1에 가까울수록: 잘못된 클러스터링 (나쁨)
    
    Args:
        embeddings: 임베딩 벡터 배열 (n_samples, embedding_dim)
        stock_codes: 종목 코드 리스트 (n_samples,)
        
    Returns:
        Silhouette score (-1 ~ 1, 높을수록 좋음)
        
    Raises:
        ValueError: 입력 데이터가 유효하지 않은 경우
        
    Example:
        >>> embeddings = np.random.randn(100, 128)
        >>> stock_codes = ['005930'] * 50 + ['000660'] * 50
        >>> score = compute_silhouette_score(embeddings, stock_codes)
        >>> print(f"Silhouette score: {score:.4f}")
    """
    # 입력 검증
    if len(embeddings) == 0:
        raise ValueError("임베딩 배열이 비어있습니다")
    
    if len(stock_codes) == 0:
        raise ValueError("종목 코드 리스트가 비어있습니다")
    
    if len(embeddings) != len(stock_codes):
        raise ValueError(
            f"임베딩 개수({len(embeddings)})와 종목 코드 개수({len(stock_codes)})가 일치하지 않습니다"
        )
    
    # 종목 코드를 숫자 레이블로 변환
    unique_stocks = list(set(stock_codes))
    
    if len(unique_stocks) < 2:
        # 클러스터가 2개 미만이면 silhouette score 계산 불가
        logger.warning(
            f"종목이 {len(unique_stocks)}개뿐입니다. "
            "Silhouette score 계산을 위해서는 최소 2개의 종목이 필요합니다."
        )
        return 0.0
    
    stock_to_label = {stock: idx for idx, stock in enumerate(unique_stocks)}
    labels = np.array([stock_to_label[stock] for stock in stock_codes])
    
    try:
        # Silhouette score 계산 (코사인 유사도 사용)
        # 코사인 유사도는 임베딩 공간에서 방향의 유사성을 측정하므로
        # 대조 학습으로 훈련된 임베딩 평가에 적합합니다
        score = silhouette_score(embeddings, labels, metric='cosine')
        return float(score)
    except Exception as e:
        logger.error(f"Silhouette score 계산 중 오류 발생: {e}")
        raise


def compute_temporal_coherence(
    embeddings: np.ndarray,
    timestamps: List[Union[str, int, float]],
    window_size: Optional[int] = 1
) -> float:
    """
    Temporal coherence 계산 - 시간적으로 가까운 샘플의 유사도 평가
    
    시간적으로 가까운 샘플들의 임베딩이 유사한지 측정합니다.
    시장 상황은 시간에 따라 연속적으로 변하므로, 좋은 임베딩은
    시간적으로 가까운 샘플들이 유사한 벡터를 가져야 합니다.
    
    점수 해석:
    - 1에 가까울수록: 시간적으로 가까운 샘플이 매우 유사 (좋음)
    - 0.5 근처: 보통 수준의 시간적 일관성
    - 0에 가까울수록: 시간적 일관성이 없음 (나쁨)
    
    Args:
        embeddings: 임베딩 벡터 배열 (n_samples, embedding_dim)
        timestamps: 타임스탬프 리스트 (n_samples,)
        window_size: 비교할 이웃 샘플 개수 (기본값: 1, 즉 바로 다음 샘플)
        
    Returns:
        Temporal coherence score (0 ~ 1, 높을수록 좋음)
        
    Raises:
        ValueError: 입력 데이터가 유효하지 않은 경우
        
    Example:
        >>> embeddings = np.random.randn(100, 128)
        >>> timestamps = list(range(100))  # 순차적 타임스탬프
        >>> score = compute_temporal_coherence(embeddings, timestamps)
        >>> print(f"Temporal coherence: {score:.4f}")
    """
    # 입력 검증
    if len(embeddings) == 0:
        raise ValueError("임베딩 배열이 비어있습니다")
    
    if len(timestamps) == 0:
        raise ValueError("타임스탬프 리스트가 비어있습니다")
    
    if len(embeddings) != len(timestamps):
        raise ValueError(
            f"임베딩 개수({len(embeddings)})와 타임스탬프 개수({len(timestamps)})가 일치하지 않습니다"
        )
    
    if len(embeddings) < 2:
        logger.warning("샘플이 2개 미만입니다. Temporal coherence를 계산할 수 없습니다.")
        return 0.0
    
    if window_size < 1:
        raise ValueError(f"window_size는 1 이상이어야 합니다: {window_size}")
    
    try:
        # 타임스탬프를 숫자로 변환 (정렬을 위해)
        numeric_timestamps = []
        for t in timestamps:
            if isinstance(t, (int, float)):
                numeric_timestamps.append(float(t))
            elif isinstance(t, str):
                # 문자열 타임스탬프를 해시값으로 변환
                numeric_timestamps.append(float(hash(t)))
            elif hasattr(t, 'item'):
                # PyTorch Tensor 또는 NumPy scalar
                numeric_timestamps.append(float(t.item()))
            else:
                raise ValueError(f"지원하지 않는 타임스탬프 타입: {type(t)}")
        
        # 타임스탬프 순서로 정렬
        sorted_indices = np.argsort(numeric_timestamps)
        sorted_embeddings = embeddings[sorted_indices]
        
        # 연속된 샘플 간의 코사인 유사도 계산
        similarities = []
        
        for i in range(len(sorted_embeddings) - window_size):
            emb1 = sorted_embeddings[i]
            emb2 = sorted_embeddings[i + window_size]
            
            # 코사인 유사도 계산
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 > 1e-8 and norm2 > 1e-8:
                # 코사인 유사도: dot(a, b) / (||a|| * ||b||)
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                similarities.append(similarity)
        
        if len(similarities) == 0:
            logger.warning("유효한 유사도를 계산할 수 없습니다.")
            return 0.0
        
        # 평균 유사도 계산
        avg_similarity = np.mean(similarities)
        
        # 코사인 유사도는 -1 ~ 1 범위이므로 0 ~ 1로 정규화
        temporal_coherence = (avg_similarity + 1) / 2
        
        return float(temporal_coherence)
        
    except Exception as e:
        logger.error(f"Temporal coherence 계산 중 오류 발생: {e}")
        raise


def evaluate_embedding_quality(
    embeddings: np.ndarray,
    stock_codes: List[Union[str, int]],
    timestamps: List[Union[str, int, float]],
    window_size: int = 1
) -> dict:
    """
    임베딩 품질 종합 평가
    
    Silhouette score와 Temporal coherence를 모두 계산하여
    임베딩 품질을 종합적으로 평가합니다.
    
    Args:
        embeddings: 임베딩 벡터 배열 (n_samples, embedding_dim)
        stock_codes: 종목 코드 리스트 (n_samples,)
        timestamps: 타임스탬프 리스트 (n_samples,)
        window_size: Temporal coherence 계산 시 윈도우 크기
        
    Returns:
        평가 메트릭 딕셔너리:
        {
            'silhouette_score': float,
            'temporal_coherence': float,
            'num_samples': int,
            'num_stocks': int,
            'embedding_dim': int
        }
        
    Example:
        >>> embeddings = np.random.randn(100, 128)
        >>> stock_codes = ['005930'] * 50 + ['000660'] * 50
        >>> timestamps = list(range(100))
        >>> metrics = evaluate_embedding_quality(embeddings, stock_codes, timestamps)
        >>> print(f"Silhouette: {metrics['silhouette_score']:.4f}")
        >>> print(f"Temporal coherence: {metrics['temporal_coherence']:.4f}")
    """
    # 기본 통계
    num_samples = len(embeddings)
    num_stocks = len(set(stock_codes)) if stock_codes else 0
    embedding_dim = embeddings.shape[1] if len(embeddings) > 0 else 0
    
    # Silhouette score 계산
    silhouette = 0.0
    try:
        if len(stock_codes) > 0 and num_stocks >= 2:
            silhouette = compute_silhouette_score(embeddings, stock_codes)
    except Exception as e:
        logger.warning(f"Silhouette score 계산 실패: {e}")
    
    # Temporal coherence 계산
    temporal_coherence = 0.0
    try:
        if len(timestamps) > 0:
            temporal_coherence = compute_temporal_coherence(
                embeddings, timestamps, window_size
            )
    except Exception as e:
        logger.warning(f"Temporal coherence 계산 실패: {e}")
    
    return {
        'silhouette_score': silhouette,
        'temporal_coherence': temporal_coherence,
        'num_samples': num_samples,
        'num_stocks': num_stocks,
        'embedding_dim': embedding_dim
    }
