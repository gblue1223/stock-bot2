"""
AutoEncoder 훈련 및 Fine-tuning 테스트
"""

import torch
import torch.nn.functional as F
import pytest
import numpy as np
from ai_trader.embedding.autoencoder_trainer import AutoEncoderTrainer, TimeSeriesDataset, train_autoencoder_embedding
from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder
from ai_trader.embedding.fine_tuning import FineTunedEmbedding, TradingTaskHead, FineTuner


def test_time_series_dataset():
    """TimeSeriesDataset 테스트"""
    # 더미 데이터 생성
    data = np.random.randn(1000, 50).astype(np.float32)
    seq_len = 60
    stride = 1
    
    dataset = TimeSeriesDataset(data, seq_len, stride)
    
    # 데이터셋 길이 검증
    expected_length = max(0, (len(data) - seq_len) // stride + 1)
    assert len(dataset) == expected_length
    
    # 샘플 형태 검증
    if len(dataset) > 0:
        sample = dataset[0]
        assert sample.shape == (seq_len, 50)
        assert sample.dtype == torch.float32


def test_time_series_dataset_stride():
    """TimeSeriesDataset stride 파라미터 테스트"""
    data = np.random.randn(100, 10).astype(np.float32)
    seq_len = 20
    
    # stride=1
    dataset_stride1 = TimeSeriesDataset(data, seq_len, stride=1)
    
    # stride=5
    dataset_stride5 = TimeSeriesDataset(data, seq_len, stride=5)
    
    # stride가 클수록 데이터셋 크기가 작아야 함
    assert len(dataset_stride5) <= len(dataset_stride1)


def test_autoencoder_trainer_initialization():
    """AutoEncoderTrainer 초기화 테스트"""
    model = AutoEncoderEmbedding(input_dim=50, embedding_dim=128, seq_len=60)
    trainer = AutoEncoderTrainer(model, device='cpu')
    
    assert trainer.model == model
    assert trainer.device == 'cpu'
    assert trainer.current_epoch == 0
    assert trainer.best_loss == float('inf')


def test_autoencoder_trainer_loss_computation():
    """AutoEncoderTrainer 손실 계산 테스트"""
    # 표준 AutoEncoder
    model = AutoEncoderEmbedding(input_dim=20, embedding_dim=64, seq_len=30)
    trainer = AutoEncoderTrainer(model, device='cpu')
    
    batch = torch.randn(8, 30, 20)
    loss, metrics = trainer.compute_loss(batch)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # 스칼라
    assert loss.item() >= 0
    assert 'reconstruction_loss' in metrics
    
    # Masked AutoEncoder
    masked_model = MaskedAutoEncoder(input_dim=20, embedding_dim=64, seq_len=30, mask_ratio=0.15)
    masked_trainer = AutoEncoderTrainer(masked_model, device='cpu')
    
    masked_loss, masked_metrics = masked_trainer.compute_loss(batch)
    
    assert isinstance(masked_loss, torch.Tensor)
    assert masked_loss.dim() == 0
    assert masked_loss.item() >= 0
    assert 'reconstruction_loss' in masked_metrics
    assert 'mask_ratio' in masked_metrics


def test_autoencoder_trainer_optimizer_creation():
    """AutoEncoderTrainer optimizer 생성 테스트"""
    model = AutoEncoderEmbedding(input_dim=30, embedding_dim=64, seq_len=40)
    trainer = AutoEncoderTrainer(model, device='cpu')
    
    config = {
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'max_epochs': 50
    }
    
    optimizer, scheduler = trainer.create_optimizer(config)
    
    assert optimizer is not None
    assert scheduler is not None
    assert optimizer.param_groups[0]['lr'] == 1e-3
    assert optimizer.param_groups[0]['weight_decay'] == 1e-4


def test_trading_task_head():
    """TradingTaskHead 테스트"""
    embedding_dim = 128
    
    # 분류 헤드
    classification_head = TradingTaskHead(
        embedding_dim=embedding_dim,
        task_type='classification',
        num_classes=3,
        hidden_dim=64
    )
    
    embeddings = torch.randn(16, embedding_dim)
    output = classification_head(embeddings)
    assert output.shape == (16, 3)
    
    # 회귀 헤드
    regression_head = TradingTaskHead(
        embedding_dim=embedding_dim,
        task_type='regression',
        hidden_dim=64
    )
    
    output = regression_head(embeddings)
    assert output.shape == (16, 1)
    
    # 랭킹 헤드
    ranking_head = TradingTaskHead(
        embedding_dim=embedding_dim,
        task_type='ranking',
        hidden_dim=64
    )
    
    output = ranking_head(embeddings)
    assert output.shape == (16, 1)


def test_finetuned_embedding():
    """FineTunedEmbedding 테스트"""
    # 기본 AutoEncoder 모델
    base_model = AutoEncoderEmbedding(input_dim=50, embedding_dim=128, seq_len=60)
    
    # 태스크 헤드
    task_head = TradingTaskHead(
        embedding_dim=128,
        task_type='classification',
        num_classes=3
    )
    
    # Fine-tuned 모델
    finetuned_model = FineTunedEmbedding(
        base_model=base_model,
        task_head=task_head,
        freeze_encoder=False
    )
    
    # 테스트 입력
    x = torch.randn(8, 60, 50)
    task_output, embeddings = finetuned_model(x)
    
    assert task_output.shape == (8, 3)  # 분류 출력
    assert embeddings.shape == (8, 128)  # 임베딩


def test_finetuned_embedding_frozen_encoder():
    """인코더가 고정된 FineTunedEmbedding 테스트"""
    base_model = AutoEncoderEmbedding(input_dim=30, embedding_dim=64, seq_len=40)
    task_head = TradingTaskHead(embedding_dim=64, task_type='regression')
    
    finetuned_model = FineTunedEmbedding(
        base_model=base_model,
        task_head=task_head,
        freeze_encoder=True
    )
    
    # 인코더 파라미터가 고정되었는지 확인
    for param in finetuned_model.encoder.parameters():
        assert not param.requires_grad, "Encoder parameters should be frozen"
    
    # 태스크 헤드 파라미터는 학습 가능해야 함
    for param in finetuned_model.task_head.parameters():
        assert param.requires_grad, "Task head parameters should be trainable"


def test_fine_tuner_loss_computation():
    """FineTuner 손실 계산 테스트"""
    base_model = AutoEncoderEmbedding(input_dim=20, embedding_dim=32, seq_len=30)
    task_head = TradingTaskHead(embedding_dim=32, task_type='classification', num_classes=3)
    finetuned_model = FineTunedEmbedding(base_model, task_head)
    
    trainer = FineTuner(finetuned_model, device='cpu')
    
    batch_x = torch.randn(8, 30, 20)
    batch_y = torch.randint(0, 3, (8,))
    
    loss, metrics = trainer.compute_loss_and_metrics(batch_x, batch_y, 'classification')
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_fine_tuner_regression_loss():
    """FineTuner 회귀 손실 테스트"""
    base_model = AutoEncoderEmbedding(input_dim=15, embedding_dim=32, seq_len=25)
    task_head = TradingTaskHead(embedding_dim=32, task_type='regression')
    finetuned_model = FineTunedEmbedding(base_model, task_head)
    
    trainer = FineTuner(finetuned_model, device='cpu')
    
    batch_x = torch.randn(8, 25, 15)
    batch_y = torch.randn(8)
    
    loss, metrics = trainer.compute_loss_and_metrics(batch_x, batch_y, 'regression')
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0
    assert 'r2' in metrics
    assert 'mse' in metrics


def test_train_autoencoder_embedding_function():
    """train_autoencoder_embedding 함수 테스트"""
    # 더미 데이터 생성
    train_data = np.random.randn(1000, 30).astype(np.float32)
    val_data = np.random.randn(200, 30).astype(np.float32)
    
    config = {
        'model_type': 'standard',
        'embedding_dim': 64,
        'hidden_dim': 128,
        'seq_len': 40,
        'batch_size': 32,
        'max_epochs': 2,  # 빠른 테스트를 위해 적은 에포크
        'learning_rate': 1e-3
    }
    
    model, trainer, history = train_autoencoder_embedding(
        train_data, val_data, config, output_dir='test_models'
    )
    
    assert isinstance(model, AutoEncoderEmbedding)
    assert isinstance(trainer, AutoEncoderTrainer)
    assert len(history) == config['max_epochs']
    assert all('train_loss' in entry for entry in history)
    assert all('val_loss' in entry for entry in history)


def test_reconstruction_loss_gradient_flow():
    """재구성 손실의 그래디언트 흐름 테스트"""
    model = AutoEncoderEmbedding(input_dim=20, embedding_dim=32, seq_len=25)
    
    x = torch.randn(4, 25, 20, requires_grad=True)
    reconstruction, embedding = model(x)
    
    # 재구성 손실 계산
    loss = F.mse_loss(reconstruction, x)
    loss.backward()
    
    # 그래디언트가 계산되었는지 확인
    assert x.grad is not None
    
    # 모델 파라미터에 그래디언트가 있는지 확인
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient not computed for {name}"
        assert torch.any(param.grad != 0), f"Gradient is zero for {name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
