#!/usr/bin/env python3
"""
Tests for AutoEncoder fine-tuning functionality.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder
from ai_trader.embedding.fine_tuning.fine_tuning import (
    FineTuner, 
    create_fine_tuning_config, 
    fine_tune_model
)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model_config():
    """Standard model configuration."""
    return {
        'num_features': 28,
        'seq_len': 60,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'mask_ratio': 0.15
    }


@pytest.fixture
def model(model_config, device):
    """Create test model."""
    model = MaskedAutoEncoder(
        input_dim=model_config['num_features'],
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        seq_len=model_config['seq_len'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        mask_ratio=model_config['mask_ratio']
    ).to(device)
    return model


@pytest.fixture
def sample_data(device):
    """Create sample data for testing."""
    batch_size = 2
    seq_len = 60
    num_features = 28
    
    # Create random data
    data = torch.randn(batch_size, seq_len, num_features).to(device)
    return data


@pytest.fixture
def mock_data_loader(sample_data):
    """Create mock data loader."""
    class MockDataLoader:
        def __init__(self, data, num_batches=3):
            self.data = data
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield self.data
        
        def __len__(self):
            return self.num_batches
    
    return MockDataLoader(sample_data)


class TestFineTuningConfig:
    """Test fine-tuning configuration creation."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = create_fine_tuning_config()
        
        assert config['learning_rate'] == 1e-4
        assert config['max_epochs'] == 5
        assert config['weight_decay'] == 1e-5
        assert config['batch_size'] == 2
        assert config['freeze_encoder'] == False
        assert config['freeze_layers'] == 0
        assert config['warmup_epochs'] == 1
        assert config['lr_schedule'] == 'cosine'
        assert config['dropout_increase'] == 0.05
        assert config['gradient_clip'] == 0.5
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = create_fine_tuning_config(
            learning_rate=2e-4,
            max_epochs=10,
            freeze_encoder=True,
            custom_param='test'
        )
        
        assert config['learning_rate'] == 2e-4
        assert config['max_epochs'] == 10
        assert config['freeze_encoder'] == True
        assert config['custom_param'] == 'test'


class TestFineTuner:
    """Test FineTuner class."""
    
    def test_initialization(self, model, device):
        """Test FineTuner initialization."""
        config = create_fine_tuning_config()
        fine_tuner = FineTuner(model, device, config)
        
        assert fine_tuner.model == model
        assert fine_tuner.device == device
        assert fine_tuner.config == config
        assert fine_tuner.optimizer is not None
        assert fine_tuner.scheduler is not None
        assert fine_tuner.best_loss == float('inf')
    
    def test_scheduler_creation(self, model, device):
        """Test different scheduler types."""
        # Cosine scheduler
        config = create_fine_tuning_config(lr_schedule='cosine')
        fine_tuner = FineTuner(model, device, config)
        assert isinstance(fine_tuner.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Step scheduler
        config = create_fine_tuning_config(lr_schedule='step')
        fine_tuner = FineTuner(model, device, config)
        assert isinstance(fine_tuner.scheduler, torch.optim.lr_scheduler.StepLR)
        
        # No scheduler
        config = create_fine_tuning_config(lr_schedule=None)
        fine_tuner = FineTuner(model, device, config)
        assert fine_tuner.scheduler is None
    
    def test_apply_fine_tuning_strategy(self, model, device):
        """Test fine-tuning strategy application."""
        config = create_fine_tuning_config(
            freeze_layers=2,
            dropout_increase=0.1
        )
        fine_tuner = FineTuner(model, device, config)
        
        # Count trainable parameters before
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Apply strategy
        fine_tuner.apply_fine_tuning_strategy()
        
        # Count trainable parameters after
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have fewer trainable parameters due to freezing
        assert trainable_after < trainable_before
    
    def test_freeze_encoder(self, model, device):
        """Test encoder freezing."""
        config = create_fine_tuning_config(freeze_encoder=True)
        fine_tuner = FineTuner(model, device, config)
        
        fine_tuner.apply_fine_tuning_strategy()
        
        # Check that encoder parameters are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad
    
    def test_train_epoch(self, model, device, mock_data_loader):
        """Test training epoch."""
        config = create_fine_tuning_config()
        fine_tuner = FineTuner(model, device, config)
        
        # Run training epoch
        loss = fine_tuner.train_epoch(mock_data_loader)
        
        assert isinstance(loss, float)
        assert loss >= 0.0
    
    def test_validate(self, model, device, mock_data_loader):
        """Test validation."""
        config = create_fine_tuning_config()
        fine_tuner = FineTuner(model, device, config)
        
        # Run validation
        loss = fine_tuner.validate(mock_data_loader)
        
        assert isinstance(loss, float)
        assert loss >= 0.0
    
    def test_fine_tune(self, model, device, mock_data_loader):
        """Test complete fine-tuning process."""
        config = create_fine_tuning_config(max_epochs=2)
        fine_tuner = FineTuner(model, device, config)
        
        # Run fine-tuning
        results = fine_tuner.fine_tune(mock_data_loader, mock_data_loader)
        
        # Check results
        assert 'initial_val_loss' in results
        assert 'best_val_loss' in results
        assert 'improvement' in results
        assert 'improvement_percent' in results
        assert 'training_time_minutes' in results
        assert 'epochs' in results
        assert 'history' in results
        
        assert results['epochs'] == 2
        assert len(results['history']['train_loss']) == 2
        assert len(results['history']['val_loss']) == 2
    
    def test_save_checkpoint(self, model, device):
        """Test checkpoint saving."""
        config = create_fine_tuning_config()
        fine_tuner = FineTuner(model, device, config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'checkpoint.pt'
            
            fine_tuner.save_checkpoint(
                checkpoint_path,
                epoch=1,
                initial_val_loss=0.5,
                additional_info={'test': 'value'}
            )
            
            assert checkpoint_path.exists()
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            assert 'model_state_dict' in checkpoint
            assert 'config' in checkpoint
            assert 'epoch' in checkpoint
            assert 'test' in checkpoint
            assert checkpoint['test'] == 'value'
    
    def test_load_checkpoint(self, model, device):
        """Test checkpoint loading."""
        config = create_fine_tuning_config()
        fine_tuner = FineTuner(model, device, config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'checkpoint.pt'
            
            # Save checkpoint
            fine_tuner.best_loss = 0.123
            fine_tuner.history = {'train_loss': [0.5, 0.4], 'val_loss': [0.6, 0.5], 'learning_rate': [1e-4, 1e-4]}
            fine_tuner.save_checkpoint(checkpoint_path, epoch=1, initial_val_loss=0.6)
            
            # Create new model and load checkpoint
            new_model = MaskedAutoEncoder(
                input_dim=28,
                embedding_dim=128,
                hidden_dim=256,
                seq_len=60,
                num_layers=4,
                dropout=0.1,
                mask_ratio=0.15
            ).to(device)
            
            loaded_fine_tuner = FineTuner.load_checkpoint(new_model, checkpoint_path, device)
            
            assert loaded_fine_tuner.best_loss == 0.123
            assert len(loaded_fine_tuner.history['train_loss']) == 2


class TestFineTuningIntegration:
    """Test fine-tuning integration functions."""
    
    def test_fine_tune_model_function(self, model, device, mock_data_loader):
        """Test fine_tune_model convenience function."""
        results = fine_tune_model(
            model,
            mock_data_loader,
            mock_data_loader,
            device,
            config=create_fine_tuning_config(max_epochs=1)
        )
        
        assert 'best_val_loss' in results
        assert 'improvement' in results
        assert results['epochs'] == 1
    
    def test_fine_tune_model_with_default_config(self, model, device, mock_data_loader):
        """Test fine_tune_model with default configuration."""
        results = fine_tune_model(
            model,
            mock_data_loader,
            mock_data_loader,
            device
        )
        
        assert 'best_val_loss' in results
        assert results['epochs'] == 5  # Default max_epochs


class TestFineTuningEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data_loader(self, model, device):
        """Test with empty data loader."""
        class EmptyDataLoader:
            def __iter__(self):
                return iter([])
            def __len__(self):
                return 0
        
        config = create_fine_tuning_config(max_epochs=1)
        fine_tuner = FineTuner(model, device, config)
        
        empty_loader = EmptyDataLoader()
        
        # Should handle empty data gracefully
        train_loss = fine_tuner.train_epoch(empty_loader)
        val_loss = fine_tuner.validate(empty_loader)
        
        assert train_loss == 0.0
        assert val_loss == 0.0
    
    def test_gradient_clipping(self, model, device, mock_data_loader):
        """Test gradient clipping functionality."""
        config = create_fine_tuning_config(gradient_clip=0.1)  # Very small clip
        fine_tuner = FineTuner(model, device, config)
        
        # This should not raise an error
        loss = fine_tuner.train_epoch(mock_data_loader)
        assert isinstance(loss, float)
    
    def test_warmup_learning_rate(self, model, device, mock_data_loader):
        """Test warmup learning rate scheduling."""
        config = create_fine_tuning_config(
            max_epochs=3,
            warmup_epochs=2,
            learning_rate=1e-3
        )
        fine_tuner = FineTuner(model, device, config)
        
        results = fine_tuner.fine_tune(mock_data_loader, mock_data_loader)
        
        # Check that learning rate changed during warmup
        lr_history = results['history']['learning_rate']
        assert len(lr_history) == 3
        assert lr_history[0] < lr_history[1]  # Warmup increase


if __name__ == '__main__':
    pytest.main([__file__])