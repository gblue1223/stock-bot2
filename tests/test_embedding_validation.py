"""
임베딩 모델 검증 메트릭 테스트
"""

import pytest
import numpy as np
import torch
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_trader.embedding.evaluation import (
    compute_silhouette_score,
    compute_temporal_coherence
)


class TestValidationMetrics:
    """검증 메트릭 테스트"""
    
    def test_silhouette_score_basic(self):
        """기본 Silhouette score 계산 테스트"""
        # 두 개의 명확한 클러스터 생성
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [0.9, -0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 1.1, 0.0],
            [-0.1, 0.9, 0.0]
        ])
        
        stock_codes = ['A', 'A', 'A', 'B', 'B', 'B']
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        # 명확한 클러스터이므로 양수 점수 기대
        assert score > 0, f"Expected positive score, got {score}"
        assert -1 <= score <= 1, f"Score should be in [-1, 1], got {score}"
    
    def test_silhouette_score_single_cluster(self):
        """단일 클러스터 Silhouette score 테스트"""
        embeddings = np.array([
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, -0.1]
        ])
        
        stock_codes = ['A', 'A', 'A']
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        # 단일 클러스터는 계산 불가하므로 0 반환
        assert score == 0.0
    
    def test_silhouette_score_empty(self):
        """빈 데이터 Silhouette score 테스트"""
        embeddings = np.array([])
        stock_codes = []
        
        # 빈 데이터는 예외를 발생시켜야 함
        with pytest.raises(ValueError, match="임베딩 배열이 비어있습니다"):
            compute_silhouette_score(embeddings, stock_codes)
    
    def test_temporal_coherence_basic(self):
        """기본 Temporal coherence 계산 테스트"""
        # 시간 순서로 유사한 임베딩 생성
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.2, 0.0],
            [1.0, 0.3, 0.0]
        ])
        
        timestamps = [1.0, 2.0, 3.0, 4.0]
        
        coherence = compute_temporal_coherence(embeddings, timestamps)
        
        # 유사한 임베딩이므로 높은 coherence 기대
        assert coherence > 0.5, f"Expected high coherence, got {coherence}"
        assert 0 <= coherence <= 1, f"Coherence should be in [0, 1], got {coherence}"
    
    def test_temporal_coherence_random_order(self):
        """무작위 순서 Temporal coherence 테스트"""
        # 무작위 임베딩
        np.random.seed(42)
        embeddings = np.random.randn(10, 5)
        
        # 무작위 타임스탬프
        timestamps = np.random.permutation(10).tolist()
        
        coherence = compute_temporal_coherence(embeddings, timestamps)
        
        # 무작위이므로 중간 정도의 coherence 기대
        assert 0 <= coherence <= 1, f"Coherence should be in [0, 1], got {coherence}"
    
    def test_temporal_coherence_empty(self):
        """빈 데이터 Temporal coherence 테스트"""
        embeddings = np.array([])
        timestamps = []
        
        # 빈 데이터는 예외를 발생시켜야 함
        with pytest.raises(ValueError, match="임베딩 배열이 비어있습니다"):
            compute_temporal_coherence(embeddings, timestamps)
    
    def test_temporal_coherence_single_sample(self):
        """단일 샘플 Temporal coherence 테스트"""
        embeddings = np.array([[1.0, 0.0]])
        timestamps = [1.0]
        
        coherence = compute_temporal_coherence(embeddings, timestamps)
        
        # 단일 샘플은 계산 불가하므로 0 반환
        assert coherence == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
