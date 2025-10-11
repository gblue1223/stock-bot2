#!/usr/bin/env python3
"""
train_autoencoder_preprocessed.py 테스트
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import json
import h5py
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.pre.train_autoencoder_preprocessed import (
    create_data_loaders,
    PreprocessedAutoEncoderTrainer,
    main
)
from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder


class TestCreateDataLoaders:
    """데이터 로더 생성 테스트"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """임시 데이터 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        
        # Create test data structure
        for month in ['2024_09', '2024_10', '2024_12']:
            month_dir = Path(temp_dir) / month
            month_dir.mkdir(parents=True, exist_ok=True)
            
            # Create batch_info.json
            batch_info = {
                'num_batches': 3,
                'sequences_per_batch': 1000,
                'seq_len': 60,
                'num_features': 28
            }
            with open(month_dir / 'batch_info.json', 'w') as f:
                json.dump(batch_info, f)
            
            # Create test batch files
            for i in range(3):
                batch_file = month_dir / f'batch_{i:06d}.h5'
                with h5py.File(batch_file, 'w') as f:
                    # Create random sequences data
                    sequences = np.random.randn(1000, 60, 28).astype(np.float32)
                    f.create_dataset('sequences', data=sequences)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_create_data_loaders_basic(self, temp_data_dir):
        """기본 데이터 로더 생성 테스트"""
        train_months = ['2024_09', '2024_10']
        val_months = ['2024_12']
        
        train_loader, val_loader, num_features = create_data_loaders(
            temp_data_dir,
            train_months,
            val_months,
            batch_size=2,
            max_batches_per_month=2,
            max_sequences_per_batch=500
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert num_features == 28
        assert len(train_loader) > 0
        assert len(val_loader) > 0
    
    def test_create_data_loaders_no_validation(self, temp_data_dir):
        """검증 데이터 없이 데이터 로더 생성 테스트"""
        train_months = ['2024_09']
        val_months = []
        
        train_loader, val_loader, num_features = create_data_loaders(
            temp_data_dir,
            train_months,
            val_months,
            batch_size=1
        )
        
        assert train_loader is not None
        assert val_loader is None
        assert num_features == 28
    
    def test_data_loader_iteration(self, temp_data_dir):
        """데이터 로더 반복 테스트"""
        train_months = ['2024_09']
        val_months = []
        
        train_loader, _, _ = create_data_loaders(
            temp_data_dir,
            train_months,
            val_months,
            batch_size=1,
            max_sequences_per_batch=100
        )
        
        # Test iteration
        for batch in train_loader:
            assert isinstance(batch, torch.Tensor)
            assert batch.dim() == 3  # (batch_size, seq_len, features)
            assert batch.shape[1] == 60  # seq_len
            assert batch.shape[2] == 28  # num_features
            break  # Just test first batch


class TestPreprocessedAutoEncoderTrainer:
    """전처리된 오토인코더 트레이너 테스트"""
    
    @pytest.fixture
    def model(self):
        """테스트용 모델"""
        return MaskedAutoEncoder(
            input_dim=28,
            embedding_dim=64,
            hidden_dim=128,
            seq_len=60,
            num_layers=2,
            dropout=0.1,
            mask_ratio=0.15
        )
    
    @pytest.fixture
    def trainer(self, model):
        """테스트용 트레이너"""
        device = 'cpu'  # Use CPU for testing
        return PreprocessedAutoEncoderTrainer(model, device)
    
    @pytest.fixture
    def sample_dataloader(self):
        """샘플 데이터 로더"""
        # Create sample data
        data = torch.randn(10, 60, 28)  # 10 sequences
        dataset = torch.utils.data.TensorDataset(data)
        return torch.utils.data.DataLoader(dataset, batch_size=2)
    
    def test_trainer_initialization(self, trainer):
        """트레이너 초기화 테스트"""
        assert trainer.model is not None
        assert trainer.device == 'cpu'
        assert trainer.current_epoch == 0
        assert trainer.best_loss == float('inf')
        assert trainer.training_history == []
    
    def test_compute_loss(self, trainer, sample_dataloader):
        """손실 계산 테스트"""
        trainer.model.eval()
        
        for batch in sample_dataloader:
            batch = batch[0]  # TensorDataset returns tuple
            loss, metrics = trainer.compute_loss(batch)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.item() > 0
            assert isinstance(metrics, dict)
            break
    
    def test_train_epoch_preprocessed(self, trainer, sample_dataloader):
        """전처리된 데이터 에포크 훈련 테스트"""
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
        
        # Create proper dataloader that returns tensors directly
        data = torch.randn(4, 60, 28)  # 4 sequences
        dataset = torch.utils.data.TensorDataset(data)
        proper_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Mock tqdm to return a mock object with set_postfix method
        class MockTqdm:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable
            def __iter__(self):
                return iter(self.iterable)
            def set_postfix(self, *args, **kwargs):
                pass
        
        with patch('scripts.pre.train_autoencoder_preprocessed.tqdm', MockTqdm):
            avg_loss = trainer.train_epoch_preprocessed(proper_dataloader, optimizer)
            
            assert isinstance(avg_loss, float)
            assert avg_loss > 0
    
    def test_validate_preprocessed(self, trainer, sample_dataloader):
        """전처리된 데이터 검증 테스트"""
        # Create proper dataloader that returns tensors directly
        data = torch.randn(4, 60, 28)  # 4 sequences
        dataset = torch.utils.data.TensorDataset(data)
        proper_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Mock tqdm to return a simple iterable
        class MockTqdm:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable
            def __iter__(self):
                return iter(self.iterable)
        
        with patch('scripts.pre.train_autoencoder_preprocessed.tqdm', MockTqdm):
            avg_loss = trainer.validate_preprocessed(proper_dataloader)
            
            assert isinstance(avg_loss, float)
            assert avg_loss > 0
    
    def test_train_preprocessed_short(self, trainer, sample_dataloader):
        """짧은 훈련 테스트"""
        config = {
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'save_every': 1,
            'output_dir': tempfile.mkdtemp()
        }
        
        # Create proper dataloader that returns tensors directly
        data = torch.randn(4, 60, 28)  # 4 sequences
        dataset = torch.utils.data.TensorDataset(data)
        proper_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Mock tqdm to return a simple iterable
        class MockTqdm:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable
            def __iter__(self):
                return iter(self.iterable)
            def set_postfix(self, *args, **kwargs):
                pass
        
        with patch('scripts.pre.train_autoencoder_preprocessed.tqdm', MockTqdm):
            history = trainer.train_preprocessed(proper_dataloader, proper_dataloader, config)
            
            assert len(history) == 2  # 2 epochs
            assert all('epoch' in entry for entry in history)
            assert all('train_loss' in entry for entry in history)
            assert all('val_loss' in entry for entry in history)
        
        # Cleanup
        shutil.rmtree(config['output_dir'])


class TestMainFunction:
    """메인 함수 테스트"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """임시 데이터 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        
        # Create minimal test data
        month_dir = Path(temp_dir) / '2024_09'
        month_dir.mkdir(parents=True, exist_ok=True)
        
        batch_info = {
            'num_batches': 1,
            'sequences_per_batch': 100,
            'seq_len': 60,
            'num_features': 28
        }
        with open(month_dir / 'batch_info.json', 'w') as f:
            json.dump(batch_info, f)
        
        batch_file = month_dir / 'batch_000000.h5'
        with h5py.File(batch_file, 'w') as f:
            sequences = np.random.randn(100, 60, 28).astype(np.float32)
            f.create_dataset('sequences', data=sequences)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_main_with_minimal_args(self, temp_data_dir):
        """최소 인자로 메인 함수 테스트"""
        output_dir = tempfile.mkdtemp()
        
        test_args = [
            'train_autoencoder_preprocessed.py',
            '--data-dir', temp_data_dir,
            '--output-dir', output_dir,
            '--train-months', '2024_09',
            '--val-months', '2024_09',  # Same as train for testing
            '--max-epochs', '1',
            '--batch-size', '1',
            '--max-sequences', '50',
            '--max-batches-per-month', '1'
        ]
        
        with patch('sys.argv', test_args):
            with patch('scripts.pre.train_autoencoder_preprocessed.tqdm') as mock_tqdm:
                # Mock tqdm to return the actual dataloader
                mock_tqdm.side_effect = lambda x, **kwargs: x
                
                try:
                    main()
                    
                    # Check if output files were created
                    output_path = Path(output_dir)
                    assert (output_path / 'training_config.json').exists()
                    assert (output_path / 'training_history.json').exists()
                    assert (output_path / 'training.log').exists()
                    
                except Exception as e:
                    pytest.fail(f"Main function failed: {e}")
        
        # Cleanup
        shutil.rmtree(output_dir)


def test_imports():
    """Import 테스트"""
    try:
        from scripts.pre.train_autoencoder_preprocessed import (
            create_data_loaders,
            PreprocessedAutoEncoderTrainer,
            main
        )
        from ai_trader.embedding.autoencoder_model import create_autoencoder_model
        from ai_trader.embedding.autoencoder_trainer import AutoEncoderTrainer
        
        print("✅ All imports successful")
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_model_creation():
    """모델 생성 테스트"""
    try:
        from ai_trader.embedding.autoencoder_model import create_autoencoder_model
        
        config = {
            'model_type': 'masked',
            'embedding_dim': 64,
            'hidden_dim': 128,
            'seq_len': 60,
            'num_layers': 2,
            'dropout': 0.1,
            'mask_ratio': 0.15
        }
        
        model = create_autoencoder_model(28, config)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(2, 60, 28)
        with torch.no_grad():
            output = model(x)
            assert len(output) == 3  # reconstruction, embedding, mask
        
        print("✅ Model creation and forward pass successful")
        
    except Exception as e:
        pytest.fail(f"Model creation failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running basic import and model tests...")
    
    test_imports()
    test_model_creation()
    
    print("\n🚀 Running pytest...")
    pytest.main([__file__, "-v"])