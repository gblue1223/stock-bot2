"""
Fast training pipeline for AutoEncoder-based embeddings.
Optimized for large datasets with efficient data loading and training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from tqdm import tqdm
import time

from .autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder, create_autoencoder_model


class TimeSeriesDataset(Dataset):
    """Efficient dataset for time series data."""
    
    def __init__(self, data: np.ndarray, seq_len: int = 60, stride: int = 1):
        """
        Args:
            data: Shape (n_samples, n_features)
            seq_len: Length of each sequence
            stride: Step size between sequences
        """
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.stride = stride
        
        # Calculate number of valid sequences
        self.n_sequences = max(0, (len(data) - seq_len) // stride + 1)
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        return self.data[start_idx:end_idx]


class AutoEncoderTrainer:
    """Fast trainer for AutoEncoder embedding models."""
    
    def __init__(self, model: AutoEncoderEmbedding, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
    def create_optimizer(self, config: dict):
        """Create optimizer with learning rate scheduling."""
        optimizer_type = config.get('optimizer', 'adamw')
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95)  # Better for transformers
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.get('max_epochs', 100),
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        else:
            scheduler = None
            
        return optimizer, scheduler
    
    def compute_loss(self, batch, mask_ratio: Optional[float] = None):
        """Compute reconstruction loss."""
        if isinstance(self.model, MaskedAutoEncoder):
            reconstruction, embedding, mask = self.model(batch, mask_ratio)
            
            # Only compute loss on masked positions
            loss = F.mse_loss(reconstruction[mask], batch[mask])
            
            # Add embedding regularization
            embedding_reg = 0.01 * torch.norm(embedding, dim=1).mean()
            loss = loss + embedding_reg
            
            return loss, {'reconstruction_loss': loss.item(), 'mask_ratio': mask.float().mean().item()}
        else:
            reconstruction, embedding = self.model(batch)
            loss = F.mse_loss(reconstruction, batch)
            
            # Add embedding regularization
            embedding_reg = 0.01 * torch.norm(embedding, dim=1).mean()
            loss = loss + embedding_reg
            
            return loss, {'reconstruction_loss': loss.item()}
    
    def train_epoch(self, dataloader: DataLoader, optimizer, mask_ratio: Optional[float] = None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            loss, metrics = self.compute_loss(batch, mask_ratio)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader, mask_ratio: Optional[float] = None):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                loss, metrics = self.compute_loss(batch, mask_ratio)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              config: dict = None):
        """
        Main training loop.
        
        Args:
            train_data: Training data (n_samples, n_features)
            val_data: Validation data (optional)
            config: Training configuration
        """
        if config is None:
            config = {}
        
        # Training parameters
        batch_size = config.get('batch_size', 256)
        max_epochs = config.get('max_epochs', 100)
        seq_len = config.get('seq_len', 60)
        stride = config.get('stride', 1)
        mask_ratio = config.get('mask_ratio', 0.15) if isinstance(self.model, MaskedAutoEncoder) else None
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, seq_len, stride)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_data is not None:
            val_dataset = TimeSeriesDataset(val_data, seq_len, stride)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Create optimizer
        optimizer, scheduler = self.create_optimizer(config)
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        if val_loader:
            self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, mask_ratio)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader, mask_ratio)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Log progress
            epoch_time = time.time() - start_time
            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
            if val_loss:
                log_msg += f", val_loss={val_loss:.4f}"
            log_msg += f", time={epoch_time:.1f}s"
            
            self.logger.info(log_msg)
            
            # Save training history
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'time': epoch_time
            }
            self.training_history.append(history_entry)
            
            # Save best model
            current_loss = val_loss if val_loss else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(config.get('output_dir', 'models'), is_best=True)
            
            # Regular checkpoint
            if epoch % config.get('save_every', 10) == 0:
                self.save_checkpoint(config.get('output_dir', 'models'))
        
        self.logger.info("Training completed!")
        return self.training_history
    
    def save_checkpoint(self, output_dir: str, is_best: bool = False, save_optimizer: bool = True):
        """Save model checkpoint with optimization options."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'model_config': {
                'input_dim': self.model.encoder.input_projection.in_features,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.encoder.bottleneck[0].in_features,
                'seq_len': self.model.decoder.seq_len,
                'model_type': 'masked' if isinstance(self.model, MaskedAutoEncoder) else 'standard'
            }
        }
        
        # 옵티마이저 상태 저장 (점진적 학습용)
        if save_optimizer and hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if hasattr(self, 'scheduler') and self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            torch.save(checkpoint, output_path / 'best_model.pt')
        else:
            torch.save(checkpoint, output_path / f'checkpoint_epoch_{self.current_epoch}.pt')
        
        # 경량화된 추론 전용 모델 저장
        if is_best:
            inference_checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_config': checkpoint['model_config']
            }
            torch.save(inference_checkpoint, output_path / 'inference_model.pt')
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint with optimizer state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        # 옵티마이저 상태 복원 (점진적 학습용)
        if load_optimizer and hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Loaded optimizer state")
        
        if load_optimizer and hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Loaded scheduler state")
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def resume_training(self, checkpoint_path: str, new_data: np.ndarray, 
                       val_data: Optional[np.ndarray] = None, config: dict = None):
        """
        점진적 학습: 기존 모델에서 새로운 데이터로 계속 훈련
        
        Args:
            checkpoint_path: 재개할 체크포인트 경로
            new_data: 새로운 훈련 데이터
            val_data: 새로운 검증 데이터
            config: 훈련 설정
        """
        if config is None:
            config = {}
        
        # 체크포인트 로드
        self.load_checkpoint(checkpoint_path, load_optimizer=True)
        
        # 새로운 데이터로 훈련 계속
        self.logger.info(f"Resuming training from epoch {self.current_epoch}")
        self.logger.info(f"New training data shape: {new_data.shape}")
        
        # 기존 훈련 함수 호출
        return self.train(new_data, val_data, config)


def train_autoencoder_embedding(train_data: np.ndarray, 
                              val_data: Optional[np.ndarray] = None,
                              config: dict = None,
                              output_dir: str = 'models/autoencoder'):
    """
    High-level function to train autoencoder embedding.
    
    Args:
        train_data: Training data (n_samples, n_features)
        val_data: Validation data (optional)
        config: Training configuration
        output_dir: Output directory for models
    """
    if config is None:
        config = {
            'model_type': 'masked',  # or 'standard'
            'embedding_dim': 128,
            'hidden_dim': 256,
            'seq_len': 120,
            'num_layers': 3,
            'dropout': 0.1,
            'mask_ratio': 0.15,
            'batch_size': 256,
            'max_epochs': 100,
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'output_dir': output_dir
        }
    
    # Create model
    input_dim = train_data.shape[1]
    model = create_autoencoder_model(input_dim, config)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = AutoEncoderTrainer(model, device)
    
    # Train
    history = trainer.train(train_data, val_data, config)
    
    # Save final model and config
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, trainer, history


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate dummy data
    n_samples, n_features = 100000, 50
    train_data = np.random.randn(n_samples, n_features).astype(np.float32)
    val_data = np.random.randn(n_samples // 10, n_features).astype(np.float32)
    
    # Train model
    model, trainer, history = train_autoencoder_embedding(
        train_data, val_data,
        config={'max_epochs': 5, 'batch_size': 128}
    )
    
    print("Training completed!")
    print(f"Final train loss: {history[-1]['train_loss']:.4f}")
    print(f"Final val loss: {history[-1]['val_loss']:.4f}")