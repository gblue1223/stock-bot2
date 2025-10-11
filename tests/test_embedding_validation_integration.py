"""
임베딩 모델 검증 통합 테스트
"""

import pytest
import numpy as np
import torch
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_trader.embedding.models import TradingEmbeddingModel
from ai_trader.embedding.losses import InfoNCELoss
from ai_trader.embedding.data import TimeSeriesSequenceDataset
from ai_trader.embedding.train_embedding import (
    compute_silhouette_score,
    compute_temporal_coherence
)


class TestValidationIntegration:
    """검증 통합 테스트"""
    
    def test_validation_with_metadata(self):
        """메타데이터를 포함한 검증 테스트"""
        # 테스트 데이터 생성
        n_samples = 100
        n_features = 60
        seq_len = 10
        
        # 특징 데이터 생성
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # 메타데이터 생성 (종목코드, 날짜, 번호)
        stock_codes = ['A'] * 50 + ['B'] * 50
        dates = ['2024-01-01'] * n_samples
        numbers = list(range(n_samples))
        metadata = np.array(list(zip(stock_codes, dates, numbers)))
        
        # Dataset 생성 (메타데이터 반환 활성화)
        dataset = TimeSeriesSequenceDataset(
            data=data,
            metadata=metadata,
            seq_len=seq_len,
            return_metadata=True
        )
        
        # 데이터 로드 테스트
        if len(dataset) > 0:
            item = dataset[0]
            
            # 메타데이터 포함 여부 확인
            assert len(item) == 5, f"Expected 5 items (anchor, positive, negative, stock_code, timestamp), got {len(item)}"
            
            anchor, positive, negative, stock_code, timestamp = item
            
            # 텐서 형태 확인
            assert anchor.shape == (seq_len, n_features)
            assert positive.shape == (seq_len, n_features)
            assert negative.shape == (seq_len, n_features)
            
            # 메타데이터 타입 확인
            assert isinstance(stock_code, str)
            assert isinstance(timestamp, float)
    
    def test_validation_metrics_with_model(self):
        """모델과 함께 검증 메트릭 테스트"""
        # 모델 생성
        input_dim = 60
        embedding_dim = 32
        seq_len = 10
        
        model = TradingEmbeddingModel(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            num_heads=2
        )
        model.eval()
        
        # 테스트 데이터 생성
        n_samples = 50
        data = np.random.randn(n_samples, input_dim).astype(np.float32)
        stock_codes = ['A'] * 25 + ['B'] * 25
        dates = ['2024-01-01'] * n_samples
        numbers = list(range(n_samples))
        metadata = np.array(list(zip(stock_codes, dates, numbers)))
        
        # Dataset 생성
        dataset = TimeSeriesSequenceDataset(
            data=data,
            metadata=metadata,
            seq_len=seq_len,
            return_metadata=True
        )
        
        # 임베딩 생성
        embeddings = []
        stock_code_list = []
        timestamp_list = []
        
        with torch.no_grad():
            for i in range(min(len(dataset), 10)):  # 처음 10개만 테스트
                anchor, positive, negative, stock_code, timestamp = dataset[i]
                
                # 배치 차원 추가
                anchor = anchor.unsqueeze(0)
                
                # 임베딩 생성
                emb = model(anchor)
                embeddings.append(emb.squeeze(0).numpy())
                stock_code_list.append(stock_code)
                timestamp_list.append(timestamp)
        
        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            
            # Silhouette score 계산
            silhouette = compute_silhouette_score(embeddings, stock_code_list)
            assert -1 <= silhouette <= 1, f"Silhouette score should be in [-1, 1], got {silhouette}"
            
            # Temporal coherence 계산
            coherence = compute_temporal_coherence(embeddings, timestamp_list)
            assert 0 <= coherence <= 1, f"Temporal coherence should be in [0, 1], got {coherence}"
            
            print(f"Silhouette score: {silhouette:.4f}")
            print(f"Temporal coherence: {coherence:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
