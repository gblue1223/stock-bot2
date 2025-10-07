"""
임베딩 모델 아키텍처

이 모듈은 매매 데이터를 밀집 벡터로 변환하는 임베딩 모델을 정의합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TradingEmbeddingModel(nn.Module):
    """
    매매 데이터를 밀집 벡터로 인코딩하는 임베딩 모델
    
    아키텍처:
    - Input: (batch, seq_len, input_dim) 특징
    - Conv1D layers: 지역 패턴 추출
    - Multi-head Attention: 중요 특징 포착
    - Projection layers: 최종 임베딩 생성
    - Output: (batch, embedding_dim) 벡터
    
    Args:
        input_dim: 입력 특징 차원 (기본값: 60+)
        embedding_dim: 출력 임베딩 차원 (기본값: 128)
        seq_len: 시퀀스 길이 (기본값: 60)
        num_heads: Multi-head Attention의 헤드 수 (기본값: 4)
        conv_channels: Conv1D 레이어의 채널 수 (기본값: [64, 128, 256])
        dropout: 드롭아웃 비율 (기본값: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 60,
        embedding_dim: int = 128,
        seq_len: int = 60,
        num_heads: int = 4,
        conv_channels: Optional[list] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        self.conv_channels = conv_channels
        
        # Conv1D 레이어로 지역 패턴 추출
        # Input: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len) for Conv1d
        conv_layers = []
        in_channels = input_dim
        
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Multi-head Attention으로 중요 특징 포착
        # Attention expects (seq_len, batch, feature_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=conv_channels[-1],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (batch, seq_len, feature_dim)
        )
        
        self.attention_norm = nn.LayerNorm(conv_channels[-1])
        
        # Projection 레이어로 최종 임베딩 생성
        self.projection = nn.Sequential(
            nn.Linear(conv_channels[-1], embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # 임베딩 정규화를 위한 레이어
        self.output_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 (batch, seq_len, input_dim)
            
        Returns:
            임베딩 벡터 (batch, embedding_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Conv1D는 (batch, channels, length) 형태를 기대
        # (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Conv1D 레이어로 지역 패턴 추출
        x = self.conv_layers(x)  # (batch, conv_channels[-1], seq_len)
        
        # (batch, conv_channels[-1], seq_len) -> (batch, seq_len, conv_channels[-1])
        x = x.transpose(1, 2)
        
        # Multi-head Attention으로 중요 특징 포착
        # Self-attention: query, key, value 모두 동일
        attn_output, _ = self.attention(x, x, x)  # (batch, seq_len, conv_channels[-1])
        
        # Residual connection + Layer Normalization
        x = self.attention_norm(x + attn_output)
        
        # 시퀀스를 평균 풀링하여 단일 벡터로 변환
        x = x.mean(dim=1)  # (batch, conv_channels[-1])
        
        # Projection 레이어로 최종 임베딩 생성
        embedding = self.projection(x)  # (batch, embedding_dim)
        
        # 출력 정규화
        embedding = self.output_norm(embedding)
        
        # L2 정규화로 단위 구에 투영 (대조 학습에 유용)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        단일 스텝 인코딩 (시퀀스가 아닌 단일 타임스텝)
        
        Args:
            x: 입력 텐서 (batch, input_dim)
            
        Returns:
            임베딩 벡터 (batch, embedding_dim)
        """
        # (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        return self.forward(x)
    
    def get_config(self) -> dict:
        """모델 설정 반환"""
        return {
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'seq_len': self.seq_len,
            'num_heads': self.num_heads,
            'conv_channels': self.conv_channels
        }
