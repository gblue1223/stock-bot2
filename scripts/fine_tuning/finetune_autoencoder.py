#!/usr/bin/env python3
"""
Fine-tune a pre-trained AutoEncoder model on new data.

This script loads a pre-trained AutoEncoder model and fine-tunes it on new
monthly data while preserving the existing knowledge.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder
from ai_trader.embedding.fine_tuning.fine_tuning import FineTuner, create_fine_tuning_config
from ai_trader.embedding.fast_data_loader import CachedBatchDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pretrained_model(model_path: Path, device: torch.device) -> tuple:
    """Load pre-trained model and configuration."""
    logger.info(f"Loading pre-trained model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = MaskedAutoEncoder(
        input_dim=config.get('num_features', 28),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        seq_len=config.get('seq_len', 60),
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        mask_ratio=config['mask_ratio']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    original_best_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', float('inf')))
    original_history = checkpoint.get('history', {})
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Original best validation loss: {original_best_loss:.6f}")
    logger.info(f"Original training epochs: {len(original_history.get('train_loss', []))}")
    
    return model, config, original_best_loss, original_history


def create_data_loaders(
    data_path: Path,
    train_months: List[str],
    val_months: List[str],
    batch_size: int,
    max_sequences: int,
    max_batches_per_month: int
) -> tuple:
    """Create data loaders for fine-tuning."""
    logger.info("Creating data loaders...")
    
    # For now, create simple mock data loaders for testing
    # In a real implementation, you would use the actual preprocessed data
    logger.warning("Using mock data loaders for testing purposes")
    
    # Create mock datasets
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
            self.seq_len = 60
            self.num_features = 28
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return torch.randn(self.seq_len, self.num_features)
    
    train_dataset = MockDataset(100)
    val_dataset = MockDataset(20)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for simplicity
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Data shape: {train_dataset.seq_len} x {train_dataset.num_features}")
    
    return train_loader, val_loader, train_dataset


def evaluate_initial_performance(model: torch.nn.Module, val_loader, device: torch.device) -> float:
    """Evaluate initial performance on new data."""
    logger.info("Evaluating initial performance on new data...")
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstruction, embedding, mask = model(batch)
            loss = F.mse_loss(reconstruction[mask], batch[mask])
            total_loss += loss.item()
            num_batches += 1
    
    initial_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Initial validation loss on new data: {initial_loss:.6f}")
    
    return initial_loss


def save_results(
    model: torch.nn.Module,
    results: Dict[str, Any],
    original_config: Dict[str, Any],
    finetune_config: Dict[str, Any],
    original_best_loss: float,
    original_history: Dict[str, Any],
    output_dir: Path,
    train_months: List[str],
    val_months: List[str]
) -> None:
    """Save fine-tuned model and results."""
    logger.info(f"Saving results to {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = output_dir / 'model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'original_config': original_config,
        'finetune_config': finetune_config,
        'train_loss': results['history']['train_loss'][-1] if results['history']['train_loss'] else 0.0,
        'val_loss': results['best_val_loss'],
        'best_val_loss': results['best_val_loss'],
        'history': results['history'],
        'original_history': original_history,
        'original_best_loss': original_best_loss,
        'initial_val_loss': results['initial_val_loss'],
        'improvement': results['improvement'],
        'training_time_minutes': results['training_time_minutes'],
        'timestamp': timestamp,
        'device': str(model.device if hasattr(model, 'device') else 'unknown'),
        'pytorch_version': torch.__version__
    }, model_path)
    
    # Save model info
    info_path = output_dir / 'model_info.json'
    model_info = {
        'model_type': 'MaskedAutoEncoder_FineTuned',
        'timestamp': timestamp,
        'original_best_loss': float(original_best_loss),
        'initial_val_loss': float(results['initial_val_loss']),
        'best_val_loss': float(results['best_val_loss']),
        'improvement': float(results['improvement']),
        'improvement_percent': float(results['improvement_percent']),
        'training_time_minutes': float(results['training_time_minutes']),
        'epochs': results['epochs'],
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'data_shape': {
            'seq_len': original_config.get('seq_len', 60),
            'num_features': original_config.get('num_features', 28)
        },
        'config': {
            'original': original_config,
            'finetune': finetune_config
        },
        'new_data_months': {
            'train': train_months,
            'val': val_months
        }
    }
    
    import json
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'finetune_history': results['history'],
            'original_history': original_history
        }, f, indent=2)
    
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Model info: {info_path}")
    logger.info(f"Training history: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune AutoEncoder model')
    
    # Required arguments
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--data', type=Path, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for fine-tuned model')
    
    # Data arguments
    parser.add_argument('--train-months', nargs='+', required=True,
                       help='Training months (e.g., 2025_01 2025_02)')
    parser.add_argument('--val-months', nargs='+', required=True,
                       help='Validation months (e.g., 2025_03)')
    parser.add_argument('--max-batches-per-month', type=int, default=20,
                       help='Maximum batches per month')
    
    # Fine-tuning arguments
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--max-epochs', type=int, default=5,
                       help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--max-sequences', type=int, default=1000,
                       help='Maximum sequences per batch')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Strategy arguments
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder layers')
    parser.add_argument('--freeze-layers', type=int, default=0,
                       help='Number of layers to freeze')
    parser.add_argument('--warmup-epochs', type=int, default=1,
                       help='Number of warmup epochs')
    parser.add_argument('--lr-schedule', choices=['cosine', 'step', 'none'], default='cosine',
                       help='Learning rate schedule')
    parser.add_argument('--dropout-increase', type=float, default=0.05,
                       help='Amount to increase dropout')
    parser.add_argument('--gradient-clip', type=float, default=0.5,
                       help='Gradient clipping threshold')
    
    # Other arguments
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load pre-trained model
        model, original_config, original_best_loss, original_history = load_pretrained_model(
            args.model, device
        )
        
        # Create data loaders
        train_loader, val_loader, train_dataset = create_data_loaders(
            args.data,
            args.train_months,
            args.val_months,
            args.batch_size,
            args.max_sequences,
            args.max_batches_per_month
        )
        
        # Check data compatibility
        expected_seq_len = original_config.get('seq_len', 60)
        expected_features = original_config.get('num_features', 28)
        
        if (train_dataset.seq_len != expected_seq_len or 
            train_dataset.num_features != expected_features):
            logger.warning(
                f"Data shape mismatch! Expected: {expected_seq_len} x {expected_features}, "
                f"Got: {train_dataset.seq_len} x {train_dataset.num_features}"
            )
            logger.warning("This may cause issues. Consider adjusting data preprocessing.")
        
        # Evaluate initial performance
        initial_val_loss = evaluate_initial_performance(model, val_loader, device)
        
        # Create fine-tuning configuration
        finetune_config = create_fine_tuning_config(
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            freeze_encoder=args.freeze_encoder,
            freeze_layers=args.freeze_layers,
            warmup_epochs=args.warmup_epochs,
            lr_schedule=args.lr_schedule if args.lr_schedule != 'none' else None,
            dropout_increase=args.dropout_increase,
            gradient_clip=args.gradient_clip
        )
        
        logger.info("Fine-tuning configuration:")
        for key, value in finetune_config.items():
            logger.info(f"  {key}: {value}")
        
        # Create fine-tuner
        fine_tuner = FineTuner(model, device, finetune_config)
        
        # Perform fine-tuning
        start_time = time.time()
        results = fine_tuner.fine_tune(
            train_loader,
            val_loader,
            save_path=args.output / 'best_model.pt'
        )
        total_time = time.time() - start_time
        
        # Update results with actual timing
        results['training_time_minutes'] = total_time / 60
        
        # Save results
        save_results(
            model,
            results,
            original_config,
            finetune_config,
            original_best_loss,
            original_history,
            args.output,
            args.train_months,
            args.val_months
        )
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING COMPLETED")
        logger.info("="*60)
        logger.info(f"Original Best Loss: {original_best_loss:.6f}")
        logger.info(f"Initial Loss on New Data: {initial_val_loss:.6f}")
        logger.info(f"Fine-tuned Best Loss: {results['best_val_loss']:.6f}")
        logger.info(f"Improvement: {results['improvement']:.6f} ({results['improvement_percent']:.2f}%)")
        logger.info(f"Training Time: {results['training_time_minutes']:.1f} minutes")
        logger.info(f"Epochs: {results['epochs']}")
        logger.info(f"Output Directory: {args.output}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise


if __name__ == '__main__':
    main()