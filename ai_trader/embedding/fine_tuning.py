"""
Fine-tuning pipeline for pre-trained AutoEncoder embeddings.
Adapts general embeddings to specific trading tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from tqdm import tqdm

from .autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder, create_autoencoder_model
from .autoencoder_trainer import AutoEncoderTrainer


class TradingTaskDataset(Dataset):
    """Dataset for trading-specific fine-tuning tasks."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, task_type: str = 'classification'):
        """
        Args:
            sequences: Input sequences (n_samples, seq_len, n_features)
            labels: Task labels (n_samples,) or (n_samples, n_classes)
            task_type: 'classification', 'regression', or 'ranking'
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).float()
        self.task_type = task_type
        
        if task_type == 'classification' and labels.ndim == 1:
            self.labels = self.labels.long()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TradingTaskHead(nn.Module):
    """Task-specific head for fine-tuning."""
    
    def __init__(self, embedding_dim: int, task_type: str, num_classes: int = None,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.task_type = task_type
        
        if task_type == 'classification':
            assert num_classes is not None, "num_classes required for classification"
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif task_type == 'regression':
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        elif task_type == 'ranking':
            # For pairwise ranking tasks
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def forward(self, embeddings):
        return self.head(embeddings)


class FineTunedEmbedding(nn.Module):
    """Fine-tuned embedding model for specific trading tasks."""
    
    def __init__(self, base_model: AutoEncoderEmbedding, task_head: TradingTaskHead,
                 freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = base_model.encoder
        self.task_head = task_head
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        embeddings = self.encoder(x)
        task_output = self.task_head(embeddings)
        return task_output, embeddings


class FineTuner:
    """Fine-tuning trainer for trading tasks."""
    
    def __init__(self, model: FineTunedEmbedding, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf')  # Assuming higher is better
        self.training_history = []
    
    def create_optimizer(self, config: dict):
        """Create optimizer with different learning rates for encoder and head."""
        encoder_lr = config.get('encoder_lr', 1e-4)  # Lower LR for pre-trained encoder
        head_lr = config.get('head_lr', 1e-3)  # Higher LR for new head
        weight_decay = config.get('weight_decay', 1e-4)
        
        # Different learning rates for different parts
        param_groups = [
            {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.model.task_head.parameters(), 'lr': head_lr}
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.get('max_epochs', 50),
            eta_min=encoder_lr * 0.01
        )
        
        return optimizer, scheduler
    
    def compute_loss_and_metrics(self, batch_x, batch_y, task_type: str):
        """Compute task-specific loss and metrics."""
        task_output, embeddings = self.model(batch_x)
        
        if task_type == 'classification':
            loss = nn.CrossEntropyLoss()(task_output, batch_y)
            
            # Accuracy
            pred = torch.argmax(task_output, dim=1)
            accuracy = (pred == batch_y).float().mean()
            
            return loss, {'accuracy': accuracy.item()}
            
        elif task_type == 'regression':
            loss = nn.MSELoss()(task_output.squeeze(), batch_y)
            
            # R-squared
            ss_res = torch.sum((batch_y - task_output.squeeze()) ** 2)
            ss_tot = torch.sum((batch_y - torch.mean(batch_y)) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            return loss, {'r2': r2.item(), 'mse': loss.item()}
            
        elif task_type == 'ranking':
            # Pairwise ranking loss
            # Assumes batch_y contains pairwise preferences
            loss = nn.MarginRankingLoss()(task_output.squeeze(), task_output.squeeze(), batch_y)
            
            return loss, {'ranking_loss': loss.item()}
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def train_epoch(self, dataloader: DataLoader, optimizer, task_type: str):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Fine-tune Epoch {self.current_epoch}")
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            loss, metrics = self.compute_loss_and_metrics(batch_x, batch_y, task_type)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                **{k: f"{v:.4f}" for k, v in metrics.items()}
            })
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return total_loss / num_batches, avg_metrics
    
    def validate(self, dataloader: DataLoader, task_type: str):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                loss, metrics = self.compute_loss_and_metrics(batch_x, batch_y, task_type)
                total_loss += loss.item()
                num_batches += 1
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return total_loss / num_batches, avg_metrics
    
    def fine_tune(self, train_dataset: TradingTaskDataset, 
                  val_dataset: Optional[TradingTaskDataset] = None,
                  config: dict = None):
        """
        Main fine-tuning loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            config: Fine-tuning configuration
        """
        if config is None:
            config = {}
        
        # Training parameters
        batch_size = config.get('batch_size', 128)
        max_epochs = config.get('max_epochs', 50)
        task_type = train_dataset.task_type
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        # Create optimizer
        optimizer, scheduler = self.create_optimizer(config)
        
        self.logger.info(f"Starting fine-tuning for {max_epochs} epochs")
        self.logger.info(f"Task type: {task_type}")
        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        if val_loader:
            self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, optimizer, task_type)
            
            # Validate
            val_loss, val_metrics = None, {}
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader, task_type)
            
            # Update scheduler
            scheduler.step()
            
            # Log progress
            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
            for key, value in train_metrics.items():
                log_msg += f", train_{key}={value:.4f}"
            
            if val_loss:
                log_msg += f", val_loss={val_loss:.4f}"
                for key, value in val_metrics.items():
                    log_msg += f", val_{key}={value:.4f}"
            
            self.logger.info(log_msg)
            
            # Save training history
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'lr': optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
            
            # Save best model (based on validation accuracy or R2)
            current_metric = self._get_primary_metric(val_metrics if val_metrics else train_metrics, task_type)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(config.get('output_dir', 'models'), is_best=True)
        
        self.logger.info("Fine-tuning completed!")
        return self.training_history
    
    def _get_primary_metric(self, metrics: dict, task_type: str) -> float:
        """Get primary metric for model selection."""
        if task_type == 'classification':
            return metrics.get('accuracy', 0.0)
        elif task_type == 'regression':
            return metrics.get('r2', -float('inf'))
        else:
            return -metrics.get('ranking_loss', float('inf'))
    
    def save_checkpoint(self, output_dir: str, is_best: bool = False):
        """Save fine-tuned model checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'training_history': self.training_history
        }
        
        if is_best:
            torch.save(checkpoint, output_path / 'best_finetuned_model.pt')
        else:
            torch.save(checkpoint, output_path / f'finetuned_checkpoint_epoch_{self.current_epoch}.pt')


def load_pretrained_model(checkpoint_path: str, device: str = 'cuda') -> AutoEncoderEmbedding:
    """Load pre-trained autoencoder model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = create_autoencoder_model(model_config['input_dim'], model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


def fine_tune_for_trading_task(pretrained_model_path: str,
                             train_sequences: np.ndarray,
                             train_labels: np.ndarray,
                             val_sequences: Optional[np.ndarray] = None,
                             val_labels: Optional[np.ndarray] = None,
                             task_type: str = 'classification',
                             num_classes: Optional[int] = None,
                             config: dict = None,
                             output_dir: str = 'models/finetuned'):
    """
    High-level function to fine-tune pre-trained embedding for trading tasks.
    
    Args:
        pretrained_model_path: Path to pre-trained autoencoder model
        train_sequences: Training sequences (n_samples, seq_len, n_features)
        train_labels: Training labels
        val_sequences: Validation sequences (optional)
        val_labels: Validation labels (optional)
        task_type: 'classification', 'regression', or 'ranking'
        num_classes: Number of classes for classification
        config: Fine-tuning configuration
        output_dir: Output directory
    """
    if config is None:
        config = {
            'batch_size': 128,
            'max_epochs': 50,
            'encoder_lr': 1e-4,
            'head_lr': 1e-3,
            'freeze_encoder': False,
            'output_dir': output_dir
        }
    
    # Load pre-trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model = load_pretrained_model(pretrained_model_path, device)
    
    # Create task head
    task_head = TradingTaskHead(
        embedding_dim=base_model.embedding_dim,
        task_type=task_type,
        num_classes=num_classes,
        hidden_dim=config.get('head_hidden_dim', 64),
        dropout=config.get('head_dropout', 0.1)
    )
    
    # Create fine-tuned model
    model = FineTunedEmbedding(
        base_model=base_model,
        task_head=task_head,
        freeze_encoder=config.get('freeze_encoder', False)
    )
    
    # Create datasets
    train_dataset = TradingTaskDataset(train_sequences, train_labels, task_type)
    val_dataset = None
    if val_sequences is not None and val_labels is not None:
        val_dataset = TradingTaskDataset(val_sequences, val_labels, task_type)
    
    # Create trainer and fine-tune
    trainer = FineTuner(model, device)
    history = trainer.fine_tune(train_dataset, val_dataset, config)
    
    # Save configuration
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'finetuning_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(output_path / 'finetuning_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, trainer, history


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate dummy data
    n_samples, seq_len, n_features = 1000, 60, 50
    n_classes = 3
    
    train_sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    train_labels = np.random.randint(0, n_classes, n_samples)
    
    val_sequences = np.random.randn(n_samples // 5, seq_len, n_features).astype(np.float32)
    val_labels = np.random.randint(0, n_classes, n_samples // 5)
    
    # First, we would need a pre-trained model
    # For demo, create a dummy one
    from .autoencoder_trainer import train_autoencoder_embedding
    
    dummy_data = np.random.randn(10000, n_features).astype(np.float32)
    base_model, _, _ = train_autoencoder_embedding(
        dummy_data, 
        config={'max_epochs': 2, 'batch_size': 64, 'seq_len': seq_len}
    )
    
    # Save the base model
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'model_config': {
            'input_dim': n_features,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'seq_len': seq_len,
            'model_type': 'masked'
        }
    }, 'temp_base_model.pt')
    
    # Fine-tune for classification
    model, trainer, history = fine_tune_for_trading_task(
        pretrained_model_path='temp_base_model.pt',
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        task_type='classification',
        num_classes=n_classes,
        config={'max_epochs': 5}
    )
    
    print("Fine-tuning completed!")
    print(f"Final accuracy: {history[-1]['val_metrics']['accuracy']:.4f}")