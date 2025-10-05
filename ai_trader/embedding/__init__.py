"""
임베딩 모듈 - 매매 데이터를 밀집 벡터로 변환

이 모듈은 대조 학습을 통해 60개 이상의 매매 특징을 저차원 임베딩으로 변환하는
모델을 제공합니다.

주요 컴포넌트:
- TradingEmbeddingModel: 임베딩 모델 아키텍처
- InfoNCELoss: 대조 학습 손실 함수
- EmbeddingDataLoader: 데이터 로더
"""

from .data import EmbeddingDataLoader, ContrastiveDataset, DataLoadError
from .models import TradingEmbeddingModel

__all__ = [
    'EmbeddingDataLoader',
    'ContrastiveDataset',
    'DataLoadError',
    'TradingEmbeddingModel'
]
