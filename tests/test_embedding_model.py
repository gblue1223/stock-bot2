"""
임베딩 모델 단위 테스트
"""

import pytest
import torch
from ai_trader.embedding.models import TradingEmbeddingModel


class TestTradingEmbeddingModel:
    """TradingEmbeddingModel 테스트"""
    
    def test_output_shape(self):
        """임베딩 모델 출력 형태 검증 (요구사항 1.1)"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128, seq_len=60)
        x = torch.randn(32, 60, 60)  # (batch, seq_len, features)
        output = model(x)
        
        assert output.shape == (32, 128), f"Expected shape (32, 128), got {output.shape}"
        
    def test_different_embedding_dims(self):
        """다양한 임베딩 차원 테스트 (요구사항 1.1)"""
        for embedding_dim in [64, 128, 256]:
            model = TradingEmbeddingModel(input_dim=60, embedding_dim=embedding_dim)
            x = torch.randn(16, 60, 60)
            output = model(x)
            
            assert output.shape == (16, embedding_dim)
    
    def test_single_step_encoding(self):
        """단일 스텝 인코딩 모드 테스트 (요구사항 1.2)"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128)
        x = torch.randn(32, 60)  # (batch, features) - 단일 타임스텝
        output = model.encode_single(x)
        
        assert output.shape == (32, 128)
    
    def test_sequence_encoding(self):
        """시퀀스 인코딩 모드 테스트 (요구사항 1.2)"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128, seq_len=60)
        x = torch.randn(32, 60, 60)  # (batch, seq_len, features)
        output = model(x)
        
        assert output.shape == (32, 128)
    
    def test_output_normalized(self):
        """출력이 L2 정규화되었는지 검증 (요구사항 1.3)"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128)
        x = torch.randn(32, 60, 60)
        output = model(x)
        
        # L2 norm이 1에 가까운지 확인
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_architecture_components(self):
        """모델 아키텍처 컴포넌트 검증 (요구사항 1.4)"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128)
        
        # Conv1D 레이어 존재 확인
        assert hasattr(model, 'conv_layers')
        
        # Multi-head Attention 존재 확인
        assert hasattr(model, 'attention')
        
        # Projection 레이어 존재 확인
        assert hasattr(model, 'projection')
    
    def test_batch_size_flexibility(self):
        """다양한 배치 크기 처리 테스트"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128)
        
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 60, 60)
            output = model(x)
            assert output.shape == (batch_size, 128)
    
    def test_different_input_dims(self):
        """다양한 입력 차원 테스트 (60+ 특징)"""
        for input_dim in [60, 65, 70]:
            model = TradingEmbeddingModel(input_dim=input_dim, embedding_dim=128)
            x = torch.randn(16, 60, input_dim)
            output = model(x)
            
            assert output.shape == (16, 128)
    
    def test_get_config(self):
        """모델 설정 반환 테스트"""
        config = {
            'input_dim': 65,
            'embedding_dim': 256,
            'seq_len': 60,
            'num_heads': 8,
            'conv_channels': [64, 128, 256]
        }
        
        model = TradingEmbeddingModel(**config)
        returned_config = model.get_config()
        
        assert returned_config['input_dim'] == config['input_dim']
        assert returned_config['embedding_dim'] == config['embedding_dim']
        assert returned_config['seq_len'] == config['seq_len']
        assert returned_config['num_heads'] == config['num_heads']
    
    def test_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128)
        x = torch.randn(16, 60, 60, requires_grad=True)
        output = model(x)
        
        # 간단한 손실 계산
        loss = output.sum()
        loss.backward()
        
        # 그래디언트가 계산되었는지 확인
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_inference_mode(self):
        """추론 모드 테스트"""
        model = TradingEmbeddingModel(input_dim=60, embedding_dim=128)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(32, 60, 60)
            output = model(x)
            
            assert output.shape == (32, 128)
            assert not output.requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
