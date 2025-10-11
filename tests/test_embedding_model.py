"""
AutoEncoder 임베딩 모델 단위 테스트

요구사항 1.1, 1.2, 1.3, 1.4를 검증하는 테스트
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder, create_autoencoder_model


class TestAutoEncoderEmbedding:
    """AutoEncoderEmbedding 테스트"""
    
    def test_output_shape_sequence(self):
        """
        요구사항 1.1: AutoEncoder 모델 출력 형태 검증 (시퀀스 입력)
        
        입력: (batch, seq_len, input_dim)
        출력: reconstruction (batch, seq_len, input_dim), embedding (batch, embedding_dim)
        """
        batch_size = 32
        seq_len = 60
        input_dim = 60
        embedding_dim = 128
        
        model = AutoEncoderEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len
        )
        
        # 랜덤 입력 생성
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        reconstruction, embedding = model(x)
        
        # 출력 형태 검증
        assert reconstruction.shape == (batch_size, seq_len, input_dim), \
            f"Expected reconstruction shape ({batch_size}, {seq_len}, {input_dim}), got {reconstruction.shape}"
        assert embedding.shape == (batch_size, embedding_dim), \
            f"Expected embedding shape ({batch_size}, {embedding_dim}), got {embedding.shape}"
    
    def test_encode_only(self):
        """
        요구사항 1.2: 인코딩만 수행하는 모드 검증
        
        입력: (batch, seq_len, input_dim)
        출력: (batch, embedding_dim)
        """
        batch_size = 16
        seq_len = 60
        input_dim = 60
        embedding_dim = 128
        
        model = AutoEncoderEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len
        )
        
        # 시퀀스 입력 생성
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # 인코딩만 수행
        embedding = model.encode(x)
        
        # 출력 형태 검증
        assert embedding.shape == (batch_size, embedding_dim), \
            f"Expected shape ({batch_size}, {embedding_dim}), got {embedding.shape}"
    
    def test_model_architecture(self):
        """
        요구사항 1.4: AutoEncoder 모델 아키텍처 검증
        
        - Encoder 존재
        - Decoder 존재
        - Transformer 기반 구조
        """
        model = AutoEncoderEmbedding(
            input_dim=60,
            embedding_dim=128,
            seq_len=60
        )
        
        # Encoder 검증
        assert hasattr(model, 'encoder'), "Model should have encoder"
        assert hasattr(model.encoder, 'transformer'), "Encoder should have transformer"
        assert hasattr(model.encoder, 'bottleneck'), "Encoder should have bottleneck"
        
        # Decoder 검증
        assert hasattr(model, 'decoder'), "Model should have decoder"
        assert hasattr(model.decoder, 'transformer'), "Decoder should have transformer"
        assert hasattr(model.decoder, 'output_projection'), "Decoder should have output projection"
    
    def test_different_embedding_dims(self):
        """다양한 임베딩 차원 테스트"""
        batch_size = 8
        seq_len = 60
        input_dim = 60
        
        for embedding_dim in [64, 128, 256]:
            model = AutoEncoderEmbedding(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                seq_len=seq_len
            )
            
            x = torch.randn(batch_size, seq_len, input_dim)
            reconstruction, embedding = model(x)
            
            assert reconstruction.shape == (batch_size, seq_len, input_dim), \
                f"Failed reconstruction shape for embedding_dim={embedding_dim}"
            assert embedding.shape == (batch_size, embedding_dim), \
                f"Failed embedding shape for embedding_dim={embedding_dim}"
    
    def test_reconstruction_loss(self):
        """재구성 손실 계산 테스트"""
        batch_size = 16
        seq_len = 60
        input_dim = 60
        embedding_dim = 128
        
        model = AutoEncoderEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        reconstruction, embedding = model(x)
        
        # MSE 손실 계산
        loss = F.mse_loss(reconstruction, x)
        
        # 손실이 양수인지 확인
        assert loss.item() >= 0, "Reconstruction loss should be non-negative"
        
        # 손실이 유한한지 확인
        assert torch.isfinite(loss), "Reconstruction loss should be finite"
    
    def test_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        model = AutoEncoderEmbedding(
            input_dim=60,
            embedding_dim=128,
            seq_len=60
        )
        
        x = torch.randn(4, 60, 60, requires_grad=True)
        reconstruction, embedding = model(x)
        
        # 재구성 손실 계산
        loss = F.mse_loss(reconstruction, x)
        loss.backward()
        
        # 그래디언트가 계산되었는지 확인
        assert x.grad is not None, "Gradients should flow to input"
        
        # 모델 파라미터에 그래디언트가 있는지 확인
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient not computed for {name}"


class TestMaskedAutoEncoder:
    """MaskedAutoEncoder 테스트"""
    
    def test_masking_functionality(self):
        """
        요구사항 5.3, 5.4: 마스킹 기법 동작 검증
        """
        batch_size = 16
        seq_len = 60
        input_dim = 60
        embedding_dim = 128
        mask_ratio = 0.15
        
        model = MaskedAutoEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            mask_ratio=mask_ratio
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        reconstruction, embedding, mask = model(x)
        
        # 출력 형태 검증
        assert reconstruction.shape == (batch_size, seq_len, input_dim)
        assert embedding.shape == (batch_size, embedding_dim)
        assert mask.shape == (batch_size, seq_len)
        
        # 마스킹 비율 검증 (대략적으로)
        actual_mask_ratio = mask.float().mean().item()
        assert 0.1 <= actual_mask_ratio <= 0.2, \
            f"Mask ratio should be around {mask_ratio}, got {actual_mask_ratio}"
    
    def test_mask_creation(self):
        """마스크 생성 테스트"""
        batch_size = 8
        seq_len = 60
        input_dim = 60
        
        model = MaskedAutoEncoder(
            input_dim=input_dim,
            embedding_dim=128,
            seq_len=seq_len,
            mask_ratio=0.2
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = model.create_mask(x, mask_ratio=0.2)
        
        # 마스크 형태 검증
        assert mask.shape == (batch_size, seq_len)
        assert mask.dtype == torch.bool
        
        # 각 샘플에 최소 하나의 비마스킹 위치가 있는지 확인
        for i in range(batch_size):
            assert not mask[i].all(), f"Sample {i} should have at least one unmasked position"
    
    def test_mask_application(self):
        """마스크 적용 테스트"""
        batch_size = 4
        seq_len = 10
        input_dim = 5
        
        model = MaskedAutoEncoder(
            input_dim=input_dim,
            embedding_dim=32,
            seq_len=seq_len
        )
        
        x = torch.ones(batch_size, seq_len, input_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :3] = True  # 처음 3개 위치 마스킹
        
        masked_x = model.apply_mask(x, mask)
        
        # 마스킹된 위치는 0이어야 함
        assert torch.all(masked_x[:, :3] == 0), "Masked positions should be zero"
        
        # 마스킹되지 않은 위치는 원본 값이어야 함
        assert torch.all(masked_x[:, 3:] == 1), "Unmasked positions should retain original values"
    
    def test_reconstruction_loss_with_mask(self):
        """마스킹된 위치에서만 손실 계산 테스트"""
        batch_size = 8
        seq_len = 20
        input_dim = 10
        
        model = MaskedAutoEncoder(
            input_dim=input_dim,
            embedding_dim=64,
            seq_len=seq_len,
            mask_ratio=0.3
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        reconstruction, embedding, mask = model(x)
        
        # 마스킹된 위치에서만 손실 계산
        masked_loss = F.mse_loss(reconstruction[mask], x[mask])
        
        # 전체 손실 계산 (비교용)
        full_loss = F.mse_loss(reconstruction, x)
        
        # 마스킹된 손실이 유효한지 확인
        assert torch.isfinite(masked_loss), "Masked loss should be finite"
        assert masked_loss.item() >= 0, "Masked loss should be non-negative"


class TestCreateAutoEncoderModel:
    """create_autoencoder_model 팩토리 함수 테스트"""
    
    def test_standard_model_creation(self):
        """표준 AutoEncoder 모델 생성 테스트"""
        config = {
            'model_type': 'standard',
            'embedding_dim': 128,
            'hidden_dim': 256,
            'seq_len': 60,
            'num_layers': 3,
            'dropout': 0.1
        }
        
        model = create_autoencoder_model(input_dim=60, config=config)
        
        assert isinstance(model, AutoEncoderEmbedding)
        assert not isinstance(model, MaskedAutoEncoder)
        assert model.embedding_dim == 128
    
    def test_masked_model_creation(self):
        """Masked AutoEncoder 모델 생성 테스트"""
        config = {
            'model_type': 'masked',
            'embedding_dim': 128,
            'hidden_dim': 256,
            'seq_len': 60,
            'num_layers': 3,
            'dropout': 0.1,
            'mask_ratio': 0.15
        }
        
        model = create_autoencoder_model(input_dim=60, config=config)
        
        assert isinstance(model, MaskedAutoEncoder)
        assert model.embedding_dim == 128
        assert model.mask_ratio == 0.15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
