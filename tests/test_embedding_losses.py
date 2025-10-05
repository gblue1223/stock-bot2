"""
임베딩 손실 함수 테스트
"""

import torch
import pytest
from ai_trader.embedding.losses import InfoNCELoss, TripletLoss


def test_infonce_loss_output_shape():
    """InfoNCE 손실 함수 출력 형태 검증"""
    loss_fn = InfoNCELoss(temperature=0.07)
    
    batch_size = 32
    embedding_dim = 128
    num_negatives = 10
    
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negatives = torch.randn(batch_size, num_negatives, embedding_dim)
    
    loss = loss_fn(anchor, positive, negatives)
    
    # 손실은 스칼라여야 함
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0  # 손실은 항상 양수


def test_infonce_loss_reduction():
    """InfoNCE 손실 함수 reduction 모드 검증"""
    batch_size = 16
    embedding_dim = 64
    num_negatives = 5
    
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negatives = torch.randn(batch_size, num_negatives, embedding_dim)
    
    # Mean reduction
    loss_fn_mean = InfoNCELoss(temperature=0.1, reduction='mean')
    loss_mean = loss_fn_mean(anchor, positive, negatives)
    assert loss_mean.shape == torch.Size([])
    
    # Sum reduction
    loss_fn_sum = InfoNCELoss(temperature=0.1, reduction='sum')
    loss_sum = loss_fn_sum(anchor, positive, negatives)
    assert loss_sum.shape == torch.Size([])
    
    # None reduction
    loss_fn_none = InfoNCELoss(temperature=0.1, reduction='none')
    loss_none = loss_fn_none(anchor, positive, negatives)
    assert loss_none.shape == torch.Size([batch_size])


def test_infonce_loss_temperature_effect():
    """Temperature 파라미터 효과 검증"""
    batch_size = 8
    embedding_dim = 32
    num_negatives = 4
    
    # 동일한 데이터로 다른 temperature 테스트
    torch.manual_seed(42)
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negatives = torch.randn(batch_size, num_negatives, embedding_dim)
    
    # 낮은 temperature
    loss_fn_low = InfoNCELoss(temperature=0.01)
    loss_low = loss_fn_low(anchor, positive, negatives)
    
    # 높은 temperature
    loss_fn_high = InfoNCELoss(temperature=1.0)
    loss_high = loss_fn_high(anchor, positive, negatives)
    
    # Temperature가 다르면 손실 값도 달라야 함
    assert loss_low.item() != loss_high.item()


def test_infonce_loss_perfect_positive():
    """완벽한 긍정 쌍에 대한 손실 검증"""
    batch_size = 4
    embedding_dim = 16
    num_negatives = 3
    
    # 앵커와 긍정이 동일한 경우
    anchor = torch.randn(batch_size, embedding_dim)
    positive = anchor.clone()  # 완벽한 긍정 쌍
    negatives = torch.randn(batch_size, num_negatives, embedding_dim)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(anchor, positive, negatives)
    
    # 완벽한 긍정 쌍이므로 손실이 낮아야 함
    assert loss.item() < 2.0  # 경험적 임계값


def test_infonce_loss_symmetric():
    """대칭적 InfoNCE 손실 검증"""
    batch_size = 8
    embedding_dim = 32
    num_negatives = 5
    
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negatives = torch.randn(batch_size, num_negatives, embedding_dim)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # 대칭적 손실 계산
    loss_symmetric = loss_fn.forward_symmetric(anchor, positive, negatives)
    
    # 일반 손실 계산
    loss_forward = loss_fn(anchor, positive, negatives)
    loss_backward = loss_fn(positive, anchor, negatives)
    expected_symmetric = (loss_forward + loss_backward) / 2.0
    
    # 대칭적 손실이 올바르게 계산되었는지 확인
    assert torch.allclose(loss_symmetric, expected_symmetric, atol=1e-6)


def test_triplet_loss_output_shape():
    """Triplet Loss 출력 형태 검증"""
    loss_fn = TripletLoss(margin=1.0, distance_metric='euclidean')
    
    batch_size = 16
    embedding_dim = 64
    
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)
    
    loss = loss_fn(anchor, positive, negative)
    
    # 손실은 스칼라여야 함
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_triplet_loss_distance_metrics():
    """Triplet Loss 거리 메트릭 검증"""
    batch_size = 8
    embedding_dim = 32
    
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)
    
    # Euclidean distance
    loss_fn_euclidean = TripletLoss(margin=1.0, distance_metric='euclidean')
    loss_euclidean = loss_fn_euclidean(anchor, positive, negative)
    
    # Cosine distance
    loss_fn_cosine = TripletLoss(margin=1.0, distance_metric='cosine')
    loss_cosine = loss_fn_cosine(anchor, positive, negative)
    
    # 두 메트릭 모두 유효한 손실을 반환해야 함
    assert loss_euclidean.item() >= 0
    assert loss_cosine.item() >= 0


def test_triplet_loss_margin_effect():
    """Triplet Loss 마진 효과 검증"""
    batch_size = 4
    embedding_dim = 16
    
    # 긍정 샘플이 앵커에 가깝고, 부정 샘플이 먼 경우
    anchor = torch.zeros(batch_size, embedding_dim)
    positive = torch.ones(batch_size, embedding_dim) * 0.1  # 가까움
    negative = torch.ones(batch_size, embedding_dim) * 10.0  # 멀음
    
    # 작은 마진
    loss_fn_small = TripletLoss(margin=0.1)
    loss_small = loss_fn_small(anchor, positive, negative)
    
    # 큰 마진
    loss_fn_large = TripletLoss(margin=5.0)
    loss_large = loss_fn_large(anchor, positive, negative)
    
    # 큰 마진일수록 손실이 커야 함 (같은 데이터에 대해)
    assert loss_large.item() >= loss_small.item()


def test_infonce_loss_gradient_flow():
    """InfoNCE 손실의 그래디언트 흐름 검증"""
    batch_size = 8
    embedding_dim = 32
    num_negatives = 5
    
    anchor = torch.randn(batch_size, embedding_dim, requires_grad=True)
    positive = torch.randn(batch_size, embedding_dim, requires_grad=True)
    negatives = torch.randn(batch_size, num_negatives, embedding_dim, requires_grad=True)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(anchor, positive, negatives)
    
    # 역전파
    loss.backward()
    
    # 그래디언트가 계산되었는지 확인
    assert anchor.grad is not None
    assert positive.grad is not None
    assert negatives.grad is not None
    
    # 그래디언트가 0이 아닌지 확인
    assert torch.any(anchor.grad != 0)
    assert torch.any(positive.grad != 0)
    assert torch.any(negatives.grad != 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
