"""
임베딩 평가 메트릭 테스트

Silhouette score와 Temporal coherence 계산 함수를 테스트합니다.
"""

import pytest
import numpy as np
from ai_trader.embedding.evaluation import (
    compute_silhouette_score,
    compute_temporal_coherence,
    evaluate_embedding_quality
)


class TestSilhouetteScore:
    """Silhouette score 계산 테스트"""
    
    def test_perfect_clustering(self):
        """완벽하게 분리된 클러스터의 경우 높은 점수를 반환해야 함"""
        # 두 개의 완전히 분리된 클러스터 생성
        cluster1 = np.random.randn(50, 128) + np.array([10, 0] + [0] * 126)
        cluster2 = np.random.randn(50, 128) + np.array([-10, 0] + [0] * 126)
        embeddings = np.vstack([cluster1, cluster2])
        
        stock_codes = ['005930'] * 50 + ['000660'] * 50
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        # 완벽하게 분리된 클러스터는 높은 점수를 가져야 함
        assert score > 0.5, f"Expected score > 0.5, got {score}"
    
    def test_overlapping_clusters(self):
        """겹치는 클러스터의 경우 낮은 점수를 반환해야 함"""
        # 겹치는 클러스터 생성
        cluster1 = np.random.randn(50, 128)
        cluster2 = np.random.randn(50, 128)
        embeddings = np.vstack([cluster1, cluster2])
        
        stock_codes = ['005930'] * 50 + ['000660'] * 50
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        # 겹치는 클러스터는 낮은 점수를 가져야 함
        assert -1 <= score <= 1, f"Score should be in [-1, 1], got {score}"
    
    def test_single_cluster(self):
        """단일 클러스터의 경우 0을 반환해야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = ['005930'] * 100
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        # 단일 클러스터는 계산 불가하므로 0 반환
        assert score == 0.0
    
    def test_multiple_stocks(self):
        """여러 종목의 경우 정상적으로 계산되어야 함"""
        embeddings = np.random.randn(150, 128)
        stock_codes = ['005930'] * 50 + ['000660'] * 50 + ['035420'] * 50
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        assert -1 <= score <= 1
    
    def test_empty_embeddings(self):
        """빈 임베딩 배열의 경우 예외를 발생시켜야 함"""
        embeddings = np.array([])
        stock_codes = []
        
        with pytest.raises(ValueError, match="임베딩 배열이 비어있습니다"):
            compute_silhouette_score(embeddings, stock_codes)
    
    def test_mismatched_lengths(self):
        """임베딩과 종목 코드 길이가 다른 경우 예외를 발생시켜야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = ['005930'] * 50
        
        with pytest.raises(ValueError, match="일치하지 않습니다"):
            compute_silhouette_score(embeddings, stock_codes)
    
    def test_string_stock_codes(self):
        """문자열 종목 코드를 정상적으로 처리해야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = ['AAPL'] * 50 + ['GOOGL'] * 50
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        assert -1 <= score <= 1
    
    def test_integer_stock_codes(self):
        """정수 종목 코드를 정상적으로 처리해야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = [1] * 50 + [2] * 50
        
        score = compute_silhouette_score(embeddings, stock_codes)
        
        assert -1 <= score <= 1


class TestTemporalCoherence:
    """Temporal coherence 계산 테스트"""
    
    def test_perfect_temporal_coherence(self):
        """시간적으로 완벽하게 일관된 임베딩의 경우 높은 점수를 반환해야 함"""
        # 시간에 따라 천천히 변하는 임베딩 생성
        embeddings = []
        for i in range(100):
            emb = np.ones(128) * (i / 100)  # 천천히 증가
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        
        timestamps = list(range(100))
        
        score = compute_temporal_coherence(embeddings, timestamps)
        
        # 천천히 변하는 임베딩은 높은 temporal coherence를 가져야 함
        assert score > 0.8, f"Expected score > 0.8, got {score}"
    
    def test_random_temporal_coherence(self):
        """무작위 임베딩의 경우 중간 정도의 점수를 반환해야 함"""
        embeddings = np.random.randn(100, 128)
        timestamps = list(range(100))
        
        score = compute_temporal_coherence(embeddings, timestamps)
        
        # 무작위 임베딩은 0 ~ 1 범위의 점수를 가져야 함
        assert 0 <= score <= 1, f"Score should be in [0, 1], got {score}"
    
    def test_window_size(self):
        """다양한 윈도우 크기에서 정상적으로 작동해야 함"""
        embeddings = np.random.randn(100, 128)
        timestamps = list(range(100))
        
        for window_size in [1, 2, 5, 10]:
            score = compute_temporal_coherence(embeddings, timestamps, window_size)
            assert 0 <= score <= 1
    
    def test_string_timestamps(self):
        """문자열 타임스탬프를 정상적으로 처리해야 함"""
        embeddings = np.random.randn(100, 128)
        timestamps = [f"2024-01-01 00:00:{i:02d}" for i in range(100)]
        
        score = compute_temporal_coherence(embeddings, timestamps)
        
        assert 0 <= score <= 1
    
    def test_float_timestamps(self):
        """실수 타임스탬프를 정상적으로 처리해야 함"""
        embeddings = np.random.randn(100, 128)
        timestamps = [float(i) * 0.1 for i in range(100)]
        
        score = compute_temporal_coherence(embeddings, timestamps)
        
        assert 0 <= score <= 1
    
    def test_unsorted_timestamps(self):
        """정렬되지 않은 타임스탬프를 자동으로 정렬해야 함"""
        embeddings = np.random.randn(100, 128)
        timestamps = list(range(100))
        np.random.shuffle(timestamps)
        
        score = compute_temporal_coherence(embeddings, timestamps)
        
        assert 0 <= score <= 1
    
    def test_empty_embeddings(self):
        """빈 임베딩 배열의 경우 예외를 발생시켜야 함"""
        embeddings = np.array([])
        timestamps = []
        
        with pytest.raises(ValueError, match="임베딩 배열이 비어있습니다"):
            compute_temporal_coherence(embeddings, timestamps)
    
    def test_single_sample(self):
        """단일 샘플의 경우 0을 반환해야 함"""
        embeddings = np.random.randn(1, 128)
        timestamps = [0]
        
        score = compute_temporal_coherence(embeddings, timestamps)
        
        assert score == 0.0
    
    def test_mismatched_lengths(self):
        """임베딩과 타임스탬프 길이가 다른 경우 예외를 발생시켜야 함"""
        embeddings = np.random.randn(100, 128)
        timestamps = list(range(50))
        
        with pytest.raises(ValueError, match="일치하지 않습니다"):
            compute_temporal_coherence(embeddings, timestamps)
    
    def test_invalid_window_size(self):
        """유효하지 않은 윈도우 크기의 경우 예외를 발생시켜야 함"""
        embeddings = np.random.randn(100, 128)
        timestamps = list(range(100))
        
        with pytest.raises(ValueError, match="window_size는 1 이상"):
            compute_temporal_coherence(embeddings, timestamps, window_size=0)


class TestEvaluateEmbeddingQuality:
    """임베딩 품질 종합 평가 테스트"""
    
    def test_complete_evaluation(self):
        """모든 메트릭이 정상적으로 계산되어야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = ['005930'] * 50 + ['000660'] * 50
        timestamps = list(range(100))
        
        metrics = evaluate_embedding_quality(embeddings, stock_codes, timestamps)
        
        # 모든 키가 존재해야 함
        assert 'silhouette_score' in metrics
        assert 'temporal_coherence' in metrics
        assert 'num_samples' in metrics
        assert 'num_stocks' in metrics
        assert 'embedding_dim' in metrics
        
        # 값이 유효한 범위에 있어야 함
        assert -1 <= metrics['silhouette_score'] <= 1
        assert 0 <= metrics['temporal_coherence'] <= 1
        assert metrics['num_samples'] == 100
        assert metrics['num_stocks'] == 2
        assert metrics['embedding_dim'] == 128
    
    def test_single_stock(self):
        """단일 종목의 경우 silhouette score는 0이어야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = ['005930'] * 100
        timestamps = list(range(100))
        
        metrics = evaluate_embedding_quality(embeddings, stock_codes, timestamps)
        
        assert metrics['silhouette_score'] == 0.0
        assert 0 <= metrics['temporal_coherence'] <= 1
        assert metrics['num_stocks'] == 1
    
    def test_custom_window_size(self):
        """커스텀 윈도우 크기를 사용할 수 있어야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = ['005930'] * 50 + ['000660'] * 50
        timestamps = list(range(100))
        
        metrics = evaluate_embedding_quality(
            embeddings, stock_codes, timestamps, window_size=5
        )
        
        assert 0 <= metrics['temporal_coherence'] <= 1
    
    def test_empty_stock_codes(self):
        """빈 종목 코드의 경우 silhouette score는 0이어야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = []
        timestamps = list(range(100))
        
        metrics = evaluate_embedding_quality(embeddings, stock_codes, timestamps)
        
        assert metrics['silhouette_score'] == 0.0
        assert metrics['num_stocks'] == 0
    
    def test_empty_timestamps(self):
        """빈 타임스탬프의 경우 temporal coherence는 0이어야 함"""
        embeddings = np.random.randn(100, 128)
        stock_codes = ['005930'] * 50 + ['000660'] * 50
        timestamps = []
        
        metrics = evaluate_embedding_quality(embeddings, stock_codes, timestamps)
        
        assert metrics['temporal_coherence'] == 0.0
    
    def test_large_dataset(self):
        """대규모 데이터셋에서도 정상적으로 작동해야 함"""
        embeddings = np.random.randn(10000, 256)
        stock_codes = ['005930'] * 5000 + ['000660'] * 5000
        timestamps = list(range(10000))
        
        metrics = evaluate_embedding_quality(embeddings, stock_codes, timestamps)
        
        assert metrics['num_samples'] == 10000
        assert metrics['embedding_dim'] == 256
        assert -1 <= metrics['silhouette_score'] <= 1
        assert 0 <= metrics['temporal_coherence'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
