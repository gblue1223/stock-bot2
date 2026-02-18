"""
임베딩 모듈 - 매매 데이터를 밀집 벡터로 변환

이 모듈은 AutoEncoder 기반 자기지도 학습을 통해 60개 이상의 매매 특징을 
저차원 임베딩으로 변환하는 모델을 제공합니다.

주요 컴포넌트:
- AutoEncoderEmbedding: 기본 AutoEncoder 모델
- MaskedAutoEncoder: 마스킹 기법을 사용한 AutoEncoder
- FineTunedEmbedding: 트레이딩 특화 Fine-tuned 모델
- AutoEncoderTrainer: 훈련 파이프라인
- TimeSeriesDataset: 시계열 데이터셋
"""

from .autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder, create_autoencoder_model
from .autoencoder_trainer import AutoEncoderTrainer, TimeSeriesDataset, train_autoencoder_embedding

__all__ = [
    'AutoEncoderEmbedding',
    'MaskedAutoEncoder', 
    'create_autoencoder_model',
    'AutoEncoderTrainer',
    'TimeSeriesDataset',
    'train_autoencoder_embedding',
    'FineTuner',
    '',
    'fine_tune_model'
]
