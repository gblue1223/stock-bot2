"""
임베딩 모듈 - 매매 데이터를 밀집 벡터로 변환

이 모듈은 대조 학습을 통해 60개 이상의 매매 특징을 저차원 임베딩으로 변환하는
모델을 제공합니다.

주요 컴포넌트:
- TradingEmbeddingModel: 임베딩 모델 아키텍처
- InfoNCELoss: 대조 학습 손실 함수
- EmbeddingDataLoader: 데이터 로더
- compute_silhouette_score: 종목 클러스터링 품질 평가
- compute_temporal_coherence: 시간적 일관성 평가
- evaluate_embedding_quality: 임베딩 품질 종합 평가
"""

from .data import EmbeddingDataLoader, ContrastiveDataset, DataLoadError
from .models import TradingEmbeddingModel
from .evaluation import (
    compute_silhouette_score,
    compute_temporal_coherence,
    evaluate_embedding_quality
)

__all__ = [
    'EmbeddingDataLoader',
    'ContrastiveDataset',
    'DataLoadError',
    'TradingEmbeddingModel',
    'compute_silhouette_score',
    'compute_temporal_coherence',
    'evaluate_embedding_quality'
]
