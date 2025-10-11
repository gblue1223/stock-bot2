"""
Fine-tuning utilities for AutoEncoder models.

This module provides utilities for fine-tuning pre-trained AutoEncoder models
on new data while preserving existing knowledge.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FineTuner:
    """Fine-tuning manager for AutoEncoder models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model: Pre-trained AutoEncoder model
            device: Device to run training on
            config: Fine-tuning configuration
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Setup optimizer with lower learning rate
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.95)
        )
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        self.best_loss = float('inf')
        self.start_time = None
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        schedule_type = self.config.get('lr_schedule', 'cosine')
        max_epochs = self.config.get('max_epochs', 5)
        
        if schedule_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=self.config.get('learning_rate', 1e-4) * 0.01
            )
        elif schedule_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=2,
                gamma=0.5
            )
        else:
            return None
    
    def apply_fine_tuning_strategy(self) -> None:
        """Apply fine-tuning specific modifications to the model."""
        # Freeze layers if specified
        freeze_layers = self.config.get('freeze_layers', 0)
        if freeze_layers > 0:
            logger.info(f"Freezing first {freeze_layers} layers")
            for i, layer in enumerate(self.model.encoder.transformer.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Freeze entire encoder if specified
        if self.config.get('freeze_encoder', False):
            logger.info("Freezing entire encoder")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        
        # Increase dropout for regularization
        dropout_increase = self.config.get('dropout_increase', 0.0)
        if dropout_increase > 0:
            logger.info(f"Increasing dropout by {dropout_increase}")
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = min(0.5, module.p + dropout_increase)
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstruction, embedding, mask = self.model(batch)
            
            # Compute loss
            recon_loss = F.mse_loss(reconstruction[mask], batch[mask])
            reg_loss = 0.005 * torch.norm(embedding, dim=1).mean()
            total_loss_batch = recon_loss + reg_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            grad_clip = self.config.get('gradient_clip', 0.5)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                reconstruction, embedding, mask = self.model(batch)
                loss = F.mse_loss(reconstruction[mask], batch[mask])
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def fine_tune(
        self,
        train_loader,
        val_loader,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Perform fine-tuning.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting fine-tuning...")
        self.start_time = time.time()
        
        # Apply fine-tuning strategy
        self.apply_fine_tuning_strategy()
        
        # Get initial validation loss
        initial_val_loss = self.validate(val_loader)
        logger.info(f"Initial validation loss: {initial_val_loss:.6f}")
        
        max_epochs = self.config.get('max_epochs', 5)
        warmup_epochs = self.config.get('warmup_epochs', 1)
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = self.config.get('learning_rate', 1e-4) * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"Warmup epoch {epoch+1}, LR: {warmup_lr:.6f}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler (after warmup)
            if epoch >= warmup_epochs and self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch+1}/{max_epochs}: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if save_path:
                    self.save_checkpoint(save_path, epoch, initial_val_loss)
                logger.info(f"New best model saved! Loss: {val_loss:.6f}")
        
        total_time = time.time() - self.start_time
        
        results = {
            'initial_val_loss': initial_val_loss,
            'best_val_loss': self.best_loss,
            'improvement': initial_val_loss - self.best_loss,
            'improvement_percent': 100 * (initial_val_loss - self.best_loss) / initial_val_loss,
            'training_time_minutes': total_time / 60,
            'epochs': len(self.history['train_loss']),
            'history': self.history
        }
        
        logger.info(f"Fine-tuning completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_loss:.6f}")
        logger.info(f"Improvement: {results['improvement']:.6f} ({results['improvement_percent']:.2f}%)")
        
        return results
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        initial_val_loss: float,
        additional_info: Optional[Dict] = None
    ) -> None:
        """Save model checkpoint."""
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'epoch': epoch,
            'best_val_loss': self.best_loss,
            'initial_val_loss': initial_val_loss,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: Path,
        device: torch.device
    ) -> 'FineTuner':
        """Load fine-tuner from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create fine-tuner
        fine_tuner = cls(model, device, checkpoint['config'])
        
        # Load optimizer state
        fine_tuner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if checkpoint.get('scheduler_state_dict') and fine_tuner.scheduler:
            fine_tuner.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        fine_tuner.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'learning_rate': []})
        fine_tuner.best_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return fine_tuner


def create_fine_tuning_config(
    learning_rate: float = 1e-4,
    max_epochs: int = 5,
    weight_decay: float = 1e-5,
    batch_size: int = 2,
    freeze_encoder: bool = False,
    freeze_layers: int = 0,
    warmup_epochs: int = 1,
    lr_schedule: str = 'cosine',
    dropout_increase: float = 0.05,
    gradient_clip: float = 0.5,
    **kwargs
) -> Dict[str, Any]:
    """
    Create fine-tuning configuration.
    
    Args:
        learning_rate: Learning rate (lower than original training)
        max_epochs: Maximum number of epochs
        weight_decay: Weight decay for regularization
        batch_size: Batch size
        freeze_encoder: Whether to freeze encoder layers
        freeze_layers: Number of layers to freeze
        warmup_epochs: Number of warmup epochs
        lr_schedule: Learning rate schedule ('cosine', 'step', or None)
        dropout_increase: Amount to increase dropout
        gradient_clip: Gradient clipping threshold
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration dictionary
    """
    config = {
        'learning_rate': learning_rate,
        'max_epochs': max_epochs,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'freeze_encoder': freeze_encoder,
        'freeze_layers': freeze_layers,
        'warmup_epochs': warmup_epochs,
        'lr_schedule': lr_schedule,
        'dropout_increase': dropout_increase,
        'gradient_clip': gradient_clip
    }
    
    config.update(kwargs)
    return config


def fine_tune_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function for fine-tuning a model.
    
    Args:
        model: Pre-trained model to fine-tune
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run training on
        config: Fine-tuning configuration (uses defaults if None)
        save_path: Path to save best model
        
    Returns:
        Training results dictionary
    """
    if config is None:
        config = create_fine_tuning_config()
    
    fine_tuner = FineTuner(model, device, config)
    return fine_tuner.fine_tune(train_loader, val_loader, save_path)