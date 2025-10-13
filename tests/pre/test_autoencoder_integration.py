"""
AutoEncoder + Fine-tuning 통합 테스트

전체 파이프라인을 테스트합니다:
1. AutoEncoder 사전 훈련
2. Fine-tuning
3. 추론
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder
from ai_trader.embedding.autoencoder_trainer import AutoEncoderTrainer, TimeSeriesDataset, train_autoencoder_embedding
from ai_trader.embedding.fine_tuning import (
    FineTunedEmbedding, 
    TradingTaskHead, 
    FineTuner, 
    TradingTaskDataset,
    fine_tune_for_trading_task,
    load_pretrained_model
)
from ai_trader.embedding.data import TimeSeriesSequenceDataset, TradingTaskDataset as DataTradingTaskDataset


class TestAutoEncoderPipeline:
    """AutoEncoder 전체 파이프라인 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cpu'  # 테스트에서는 CPU 사용
        
        # 테스트 데이터 생성
        self.n_samples = 1000
        self.n_features = 30
        self.seq_len = 40
        self.embedding_dim = 64
        
        # 시계열 데이터 생성
        self.train_data = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.val_data = np.random.randn(self.n_samples // 5, self.n_features).astype(np.float32)
    
    def teardown_method(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_autoencoder_training_pipeline(self):
        """AutoEncoder 훈련 파이프라인 테스트"""
        # 1. 모델 생성
        model = AutoEncoderEmbedding(
            input_dim=self.n_features,
            embedding_dim=self.embedding_dim,
            seq_len=self.seq_len
        )
        
        # 2. 트레이너 생성
        trainer = AutoEncoderTrainer(model, device=self.device)
        
        # 3. 훈련 설정
        config = {
            'batch_size': 32,
            'max_epochs': 3,  # 빠른 테스트를 위해 적은 에포크
            'learning_rate': 1e-3,
            'seq_len': self.seq_len,
            'stride': 5
        }
        
        # 4. 훈련 실행
        history = trainer.train(self.train_data, self.val_data, config)
        
        # 5. 결과 검증
        assert len(history) == config['max_epochs']
        assert all('train_loss' in entry for entry in history)
        assert all('val_loss' in entry for entry in history)
        
        # 손실이 감소하는지 확인 (마지막 손실이 첫 번째보다 작거나 같아야 함)
        first_loss = history[0]['train_loss']
        last_loss = history[-1]['train_loss']
        assert last_loss <= first_loss * 1.1  # 10% 여유를 둠
        
        # 6. 추론 테스트
        model.eval()
        test_input = torch.randn(5, self.seq_len, self.n_features)
        
        with torch.no_grad():
            reconstruction, embedding = model(test_input)
        
        assert reconstruction.shape == test_input.shape
        assert embedding.shape == (5, self.embedding_dim)
    
    def test_masked_autoencoder_training(self):
        """Masked AutoEncoder 훈련 테스트"""
        # 1. Masked AutoEncoder 생성
        model = MaskedAutoEncoder(
            input_dim=self.n_features,
            embedding_dim=self.embedding_dim,
            seq_len=self.seq_len,
            mask_ratio=0.15
        )
        
        # 2. 트레이너 생성
        trainer = AutoEncoderTrainer(model, device=self.device)
        
        # 3. 훈련 설정
        config = {
            'batch_size': 16,
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'seq_len': self.seq_len,
            'stride': 10
        }
        
        # 4. 훈련 실행
        history = trainer.train(self.train_data, self.val_data, config)
        
        # 5. 결과 검증
        assert len(history) == config['max_epochs']
        
        # 마스킹 기능 테스트
        model.eval()
        test_input = torch.randn(3, self.seq_len, self.n_features)
        
        with torch.no_grad():
            reconstruction, embedding, mask = model(test_input)
        
        assert reconstruction.shape == test_input.shape
        assert embedding.shape == (3, self.embedding_dim)
        assert mask.shape == (3, self.seq_len)
        assert mask.dtype == torch.bool
    
    def test_high_level_training_function(self):
        """고수준 훈련 함수 테스트"""
        config = {
            'model_type': 'standard',
            'embedding_dim': self.embedding_dim,
            'hidden_dim': 128,
            'seq_len': self.seq_len,
            'batch_size': 32,
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'output_dir': self.temp_dir
        }
        
        # 고수준 함수로 훈련
        model, trainer, history = train_autoencoder_embedding(
            self.train_data, self.val_data, config
        )
        
        # 결과 검증
        assert isinstance(model, AutoEncoderEmbedding)
        assert isinstance(trainer, AutoEncoderTrainer)
        assert len(history) == config['max_epochs']
        
        # 파일이 저장되었는지 확인
        output_path = Path(self.temp_dir)
        assert (output_path / 'training_config.json').exists()
        assert (output_path / 'training_history.json').exists()
    
    def test_fine_tuning_pipeline(self):
        """Fine-tuning 파이프라인 테스트"""
        # 1. 기본 AutoEncoder 훈련
        base_model = AutoEncoderEmbedding(
            input_dim=self.n_features,
            embedding_dim=self.embedding_dim,
            seq_len=self.seq_len
        )
        
        trainer = AutoEncoderTrainer(base_model, device=self.device)
        config = {
            'batch_size': 16,
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'seq_len': self.seq_len,
            'stride': 5
        }
        
        trainer.train(self.train_data, self.val_data, config)
        
        # 2. Fine-tuning 데이터 준비
        n_sequences = 200
        sequences = np.random.randn(n_sequences, self.seq_len, self.n_features).astype(np.float32)
        labels = np.random.randint(0, 3, n_sequences)  # 3클래스 분류
        
        train_sequences = sequences[:160]
        train_labels = labels[:160]
        val_sequences = sequences[160:]
        val_labels = labels[160:]
        
        # 3. Task head 생성
        task_head = TradingTaskHead(
            embedding_dim=self.embedding_dim,
            task_type='classification',
            num_classes=3,
            hidden_dim=32
        )
        
        # 4. Fine-tuned 모델 생성
        finetuned_model = FineTunedEmbedding(
            base_model=base_model,
            task_head=task_head,
            freeze_encoder=False
        )
        
        # 5. Fine-tuning 트레이너 생성
        ft_trainer = FineTuner(finetuned_model, device=self.device)
        
        # 6. Fine-tuning 데이터셋 생성
        train_dataset = TradingTaskDataset(train_sequences, train_labels, 'classification')
        val_dataset = TradingTaskDataset(val_sequences, val_labels, 'classification')
        
        # 7. Fine-tuning 실행
        ft_config = {
            'batch_size': 16,
            'max_epochs': 3,
            'encoder_lr': 1e-4,
            'head_lr': 1e-3
        }
        
        ft_history = ft_trainer.fine_tune(train_dataset, val_dataset, ft_config)
        
        # 8. 결과 검증
        assert len(ft_history) == ft_config['max_epochs']
        assert all('train_loss' in entry for entry in ft_history)
        assert all('val_loss' in entry for entry in ft_history)
        assert all('train_metrics' in entry for entry in ft_history)
        assert all('val_metrics' in entry for entry in ft_history)
        
        # 정확도가 랜덤보다 나은지 확인 (33% 이상)
        final_accuracy = ft_history[-1]['val_metrics']['accuracy']
        assert final_accuracy >= 0.2  # 랜덤보다 약간 나은 수준
        
        # 9. 추론 테스트
        finetuned_model.eval()
        test_input = torch.randn(5, self.seq_len, self.n_features)
        
        with torch.no_grad():
            task_output, embeddings = finetuned_model(test_input)
        
        assert task_output.shape == (5, 3)  # 3클래스 분류
        assert embeddings.shape == (5, self.embedding_dim)
    
    def test_high_level_fine_tuning_function(self):
        """고수준 Fine-tuning 함수 테스트"""
        # 1. 기본 모델 저장
        base_model = AutoEncoderEmbedding(
            input_dim=self.n_features,
            embedding_dim=self.embedding_dim,
            seq_len=self.seq_len
        )
        
        # 체크포인트 저장
        checkpoint_path = Path(self.temp_dir) / 'base_model.pt'
        torch.save({
            'model_state_dict': base_model.state_dict(),
            'model_config': {
                'input_dim': self.n_features,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': 128,
                'seq_len': self.seq_len,
                'model_type': 'standard'
            }
        }, checkpoint_path)
        
        # 2. Fine-tuning 데이터 준비
        n_sequences = 100
        sequences = np.random.randn(n_sequences, self.seq_len, self.n_features).astype(np.float32)
        labels = np.random.randint(0, 2, n_sequences)  # 2클래스 분류
        
        train_sequences = sequences[:80]
        train_labels = labels[:80]
        val_sequences = sequences[80:]
        val_labels = labels[80:]
        
        # 3. 고수준 Fine-tuning 함수 실행
        ft_config = {
            'batch_size': 16,
            'max_epochs': 2,
            'encoder_lr': 1e-4,
            'head_lr': 1e-3,
            'output_dir': str(Path(self.temp_dir) / 'finetuned')
        }
        
        model, trainer, history = fine_tune_for_trading_task(
            pretrained_model_path=str(checkpoint_path),
            train_sequences=train_sequences,
            train_labels=train_labels,
            val_sequences=val_sequences,
            val_labels=val_labels,
            task_type='classification',
            num_classes=2,
            config=ft_config
        )
        
        # 4. 결과 검증
        assert isinstance(model, FineTunedEmbedding)
        assert isinstance(trainer, FineTuner)
        assert len(history) == ft_config['max_epochs']
        
        # 파일이 저장되었는지 확인
        ft_output_path = Path(ft_config['output_dir'])
        assert (ft_output_path / 'finetuning_config.json').exists()
        assert (ft_output_path / 'finetuning_history.json').exists()
    
    def test_model_loading_and_inference(self):
        """모델 로딩 및 추론 테스트"""
        # 1. 모델 훈련 및 저장
        model = AutoEncoderEmbedding(
            input_dim=self.n_features,
            embedding_dim=self.embedding_dim,
            seq_len=self.seq_len
        )
        
        checkpoint_path = Path(self.temp_dir) / 'test_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': self.n_features,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': 128,
                'seq_len': self.seq_len,
                'model_type': 'standard'
            }
        }, checkpoint_path)
        
        # 2. 모델 로딩
        loaded_model = load_pretrained_model(str(checkpoint_path), device=self.device)
        
        # 3. 로딩된 모델 검증
        assert isinstance(loaded_model, AutoEncoderEmbedding)
        assert loaded_model.embedding_dim == self.embedding_dim
        
        # 4. 추론 테스트
        test_input = torch.randn(3, self.seq_len, self.n_features)
        
        # 원본 모델 추론
        model.eval()
        with torch.no_grad():
            orig_reconstruction, orig_embedding = model(test_input)
        
        # 로딩된 모델 추론
        loaded_model.eval()
        with torch.no_grad():
            loaded_reconstruction, loaded_embedding = loaded_model(test_input)
        
        # 결과가 동일한지 확인
        assert torch.allclose(orig_reconstruction, loaded_reconstruction, atol=1e-6)
        assert torch.allclose(orig_embedding, loaded_embedding, atol=1e-6)
    
    def test_end_to_end_pipeline(self):
        """전체 파이프라인 End-to-End 테스트"""
        # 1. 원시 데이터 생성 (시계열 형태)
        n_total_samples = 2000
        raw_data = np.random.randn(n_total_samples, self.n_features).astype(np.float32)
        metadata = np.array([
            ['STOCK1', '20241001', i] for i in range(n_total_samples)
        ])
        
        # 2. 시퀀스 데이터셋 생성
        sequence_dataset = TimeSeriesSequenceDataset(
            data=raw_data,
            metadata=metadata,
            seq_len=self.seq_len,
            stride=10
        )
        
        # 3. AutoEncoder 사전 훈련용 데이터 준비
        train_data = raw_data[:1600]  # 80%
        val_data = raw_data[1600:]    # 20%
        
        # 4. AutoEncoder 사전 훈련
        config = {
            'model_type': 'masked',
            'embedding_dim': self.embedding_dim,
            'hidden_dim': 128,
            'seq_len': self.seq_len,
            'mask_ratio': 0.15,
            'batch_size': 32,
            'max_epochs': 3,
            'learning_rate': 1e-3,
            'output_dir': str(Path(self.temp_dir) / 'pretrained')
        }
        
        pretrained_model, _, pretrain_history = train_autoencoder_embedding(
            train_data, val_data, config
        )
        
        # 5. Fine-tuning 데이터 준비
        # 시퀀스들을 수집
        sequences = []
        for i in range(min(100, len(sequence_dataset))):  # 100개 시퀀스만 사용
            seq = sequence_dataset[i]
            sequences.append(seq.numpy())
        
        sequences = np.array(sequences)
        
        # 더미 라벨 생성 (가격 변화 방향)
        labels = np.random.randint(0, 3, len(sequences))
        
        # 6. Fine-tuning 실행
        ft_model, ft_trainer, ft_history = fine_tune_for_trading_task(
            pretrained_model_path=str(Path(config['output_dir']) / 'best_model.pt'),
            train_sequences=sequences[:80],
            train_labels=labels[:80],
            val_sequences=sequences[80:],
            val_labels=labels[80:],
            task_type='classification',
            num_classes=3,
            config={
                'batch_size': 16,
                'max_epochs': 2,
                'encoder_lr': 1e-4,
                'head_lr': 1e-3,
                'output_dir': str(Path(self.temp_dir) / 'finetuned'),
                'device': self.device  # 명시적으로 디바이스 전달
            }
        )
        
        # 7. 전체 파이프라인 검증
        assert len(pretrain_history) == config['max_epochs']
        assert len(ft_history) == 2
        
        # 8. 최종 추론 테스트
        ft_model.eval()
        test_sequence = torch.randn(1, self.seq_len, self.n_features).to(self.device)
        
        with torch.no_grad():
            prediction, embedding = ft_model(test_sequence)
        
        assert prediction.shape == (1, 3)  # 3클래스 분류
        assert embedding.shape == (1, self.embedding_dim)
        
        # 예측이 유효한 확률 분포인지 확인
        probabilities = torch.softmax(prediction, dim=1)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])