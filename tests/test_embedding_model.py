"""
임베딩 모델 단위 테스트

요구사항 1.1, 1.2, 1.3, 1.4를 검증하는 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from ai_trader.embedding.models import TradingEmbeddingModel
from ai_trader.embedding.losses import InfoNCELoss, TripletLoss


class TestTradingEmbeddingModel:
    """TradingEmbeddingModel 테스트"""
    
    def test_output_shape_sequence(self):
        """
        요구사항 1.1: 임베딩 모델 출력 형태 검증 (시퀀스 입력)
        
        입력: (batch, seq_len, input_dim)
        출력: (batch, embedding_dim)
        """
        batch_size = 32
        seq_len = 60
        input_dim = 60
        embedding_dim = 128
        
        model = TradingEmbeddingModel(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len
        )
        
        # 랜덤 입력 생성
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = model(x)
        
        # 출력 형태 검증
        assert output.shape == (batch_size, embedding_dim), \
            f"Expected shape ({batch_size}, {embedding_dim}), got {output.shape}"
        
        # 출력이 L2 정규화되었는지 검증 (단위 구에 투영)
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), \
            "Output embeddings should be L2 normalized"
    
    def test_output_shape_single_step(self):
        """
        요구사항 1.2: 단일 스텝 인코딩 모드 검증
        
        입력: (batch, input_dim)
        출력: (batch, embedding_dim)
        """
        batch_size = 16
        input_dim = 60
        embedding_dim = 128
        
        model = TradingEmbeddingModel(
            input_dim=input_dim,
            embedding_dim=embedding_dim
        )
        
        # 단일 스텝 입력 생성
        x = torch.randn(batch_size, input_dim)
        
        # 단일 스텝 인코딩
        output = model.encode_single(x)
        
        # 출력 형태 검증
        assert output.shape == (batch_size, embedding_dim), \
            f"Expected shape ({batch_size}, {embedding_dim}), got {output.shape}"
    
    def test_model_architecture(self):
        """
        요구사항 1.4: 모델 아키텍처 검증
        
        - Conv1D 레이어 존재
        - Multi-head Attention 존재
        - Projection 레이어 존재
        """
        model = TradingEmbeddingModel(
            input_dim=60,
            embedding_dim=128,
            num_heads=4
        )
        
        # Conv1D 레이어 검증
        assert hasattr(model, 'conv_layers'), "Model should have conv_layers"
        assert isinstance(model.conv_layers, nn.Sequential), \
            "conv_layers should be nn.Sequential"
        
        # Multi-head Attention 검증
        assert hasattr(model, 'attention'), "Model should have attention layer"
        assert isinstance(model.attention, nn.MultiheadAttention), \
            "attention should be nn.MultiheadAttention"
        assert model.attention.num_heads == 4, "Number of attention heads should be 4"
        
        # Projection 레이어 검증
        assert hasattr(model, 'projection'), "Model should have projection layer"
        assert isinstance(model.projection, nn.Sequential), \
            "projection should be nn.Sequential"
    
    def test_different_embedding_dims(self):
        """다양한 임베딩 차원 테스트"""
        batch_size = 8
        seq_len = 60
        input_dim = 60
        
        for embedding_dim in [64, 128, 256]:
            model = TradingEmbeddingModel(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                seq_len=seq_len
            )
            
            x = torch.randn(batch_size, seq_len, input_dim)
            output = model(x)
            
            assert output.shape == (batch_size, embedding_dim), \
                f"Failed for embedding_dim={embedding_dim}"
    
    def test_model_config(self):
        """모델 설정 반환 테스트"""
        input_dim = 60
        embedding_dim = 128
        seq_len = 60
        num_heads = 4
        conv_channels = [64, 128, 256]
        
        model = TradingEmbeddingModel(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            conv_channels=conv_channels
        )
        
        config = model.get_config()
        
        assert config['input_dim'] == input_dim
        assert config['embedding_dim'] == embedding_dim
        assert config['seq_len'] == seq_len
        assert config['num_heads'] == num_heads
        assert config['conv_channels'] == conv_channels
    
    def test_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        model = TradingEmbeddingModel(
            input_dim=60,
            embedding_dim=128
        )
        
        x = torch.randn(4, 60, 60, requires_grad=True)
        output = model(x)
        
        # 더미 손실 계산
        loss = output.sum()
        loss.backward()
        
        # 그래디언트가 계산되었는지 확인
        assert x.grad is not None, "Gradients should flow to input"
        
        # 모델 파라미터에 그래디언트가 있는지 확인
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient not computed for {name}"


class TestInfoNCELoss:
    """InfoNCELoss 테스트"""
    
    def test_loss_computation(self):
        """
        요구사항 1.3, 5.4: InfoNCE 손실 계산 검증
        
        긍정 쌍의 유사도는 최대화, 부정 쌍의 유사도는 최소화
        """
        batch_size = 32
        embedding_dim = 128
        num_negatives = 10
        
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # 랜덤 임베딩 생성
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negatives = torch.randn(batch_size, num_negatives, embedding_dim)
        
        # 손실 계산
        loss = loss_fn(anchor, positive, negatives)
        
        # 손실이 스칼라인지 확인
        assert loss.dim() == 0, "Loss should be a scalar"
        
        # 손실이 양수인지 확인
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_perfect_positive_pair(self):
        """완벽한 긍정 쌍 테스트 (손실이 낮아야 함)"""
        batch_size = 16
        embedding_dim = 64
        num_negatives = 5
        
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # 앵커와 긍정 샘플을 동일하게 설정
        anchor = torch.randn(batch_size, embedding_dim)
        positive = anchor.clone()  # 완벽한 긍정 쌍
        negatives = torch.randn(batch_size, num_negatives, embedding_dim)
        
        # 손실 계산
        loss = loss_fn(anchor, positive, negatives)
        
        # 완벽한 긍정 쌍의 경우 손실이 낮아야 함
        assert loss.item() < 1.0, \
            f"Loss for perfect positive pair should be low, got {loss.item()}"
    
    def test_symmetric_loss(self):
        """대칭적 InfoNCE 손실 테스트"""
        batch_size = 16
        embedding_dim = 64
        num_negatives = 5
        
        loss_fn = InfoNCELoss(temperature=0.07)
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negatives = torch.randn(batch_size, num_negatives, embedding_dim)
        
        # 대칭적 손실 계산
        symmetric_loss = loss_fn.forward_symmetric(anchor, positive, negatives)
        
        # 손실이 스칼라인지 확인
        assert symmetric_loss.dim() == 0, "Symmetric loss should be a scalar"
        
        # 손실이 양수인지 확인
        assert symmetric_loss.item() >= 0, "Symmetric loss should be non-negative"
    
    def test_temperature_effect(self):
        """Temperature 파라미터 효과 테스트"""
        batch_size = 16
        embedding_dim = 64
        num_negatives = 5
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negatives = torch.randn(batch_size, num_negatives, embedding_dim)
        
        # 낮은 temperature
        loss_fn_low = InfoNCELoss(temperature=0.01)
        loss_low = loss_fn_low(anchor, positive, negatives)
        
        # 높은 temperature
        loss_fn_high = InfoNCELoss(temperature=1.0)
        loss_high = loss_fn_high(anchor, positive, negatives)
        
        # Temperature가 다르면 손실도 달라야 함
        assert not torch.isclose(loss_low, loss_high, atol=1e-3), \
            "Different temperatures should produce different losses"
    
    def test_reduction_modes(self):
        """손실 축소 모드 테스트"""
        batch_size = 16
        embedding_dim = 64
        num_negatives = 5
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negatives = torch.randn(batch_size, num_negatives, embedding_dim)
        
        # Mean reduction
        loss_fn_mean = InfoNCELoss(reduction='mean')
        loss_mean = loss_fn_mean(anchor, positive, negatives)
        assert loss_mean.dim() == 0, "Mean reduction should produce scalar"
        
        # Sum reduction
        loss_fn_sum = InfoNCELoss(reduction='sum')
        loss_sum = loss_fn_sum(anchor, positive, negatives)
        assert loss_sum.dim() == 0, "Sum reduction should produce scalar"
        
        # None reduction
        loss_fn_none = InfoNCELoss(reduction='none')
        loss_none = loss_fn_none(anchor, positive, negatives)
        assert loss_none.shape == (batch_size,), \
            f"None reduction should produce per-sample losses, got shape {loss_none.shape}"
    
    def test_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        batch_size = 8
        embedding_dim = 32
        num_negatives = 5
        
        loss_fn = InfoNCELoss()
        
        anchor = torch.randn(batch_size, embedding_dim, requires_grad=True)
        positive = torch.randn(batch_size, embedding_dim, requires_grad=True)
        negatives = torch.randn(batch_size, num_negatives, embedding_dim, requires_grad=True)
        
        loss = loss_fn(anchor, positive, negatives)
        loss.backward()
        
        # 그래디언트가 계산되었는지 확인
        assert anchor.grad is not None, "Gradients should flow to anchor"
        assert positive.grad is not None, "Gradients should flow to positive"
        assert negatives.grad is not None, "Gradients should flow to negatives"


class TestTripletLoss:
    """TripletLoss 테스트 (대안적 대조 학습 손실)"""
    
    def test_loss_computation_euclidean(self):
        """Euclidean 거리 기반 Triplet Loss 테스트"""
        batch_size = 16
        embedding_dim = 64
        
        loss_fn = TripletLoss(margin=1.0, distance_metric='euclidean')
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negative = torch.randn(batch_size, embedding_dim)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_loss_computation_cosine(self):
        """Cosine 거리 기반 Triplet Loss 테스트"""
        batch_size = 16
        embedding_dim = 64
        
        loss_fn = TripletLoss(margin=0.5, distance_metric='cosine')
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negative = torch.randn(batch_size, embedding_dim)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_perfect_triplet(self):
        """완벽한 triplet 테스트 (손실이 0이어야 함)"""
        batch_size = 8
        embedding_dim = 32
        
        loss_fn = TripletLoss(margin=1.0, distance_metric='euclidean')
        
        # 앵커와 긍정 샘플을 동일하게, 부정 샘플을 멀리 설정
        anchor = torch.randn(batch_size, embedding_dim)
        positive = anchor.clone()
        negative = anchor + 10.0  # 매우 먼 부정 샘플
        
        loss = loss_fn(anchor, positive, negative)
        
        # 완벽한 triplet의 경우 손실이 0이어야 함
        assert loss.item() < 0.1, \
            f"Loss for perfect triplet should be near zero, got {loss.item()}"
    
    def test_invalid_distance_metric(self):
        """잘못된 거리 메트릭 테스트"""
        with pytest.raises(ValueError):
            TripletLoss(distance_metric='invalid')
    
    def test_invalid_reduction(self):
        """잘못된 reduction 모드 테스트"""
        with pytest.raises(ValueError):
            TripletLoss(reduction='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
