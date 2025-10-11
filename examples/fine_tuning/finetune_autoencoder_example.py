#!/usr/bin/env python3
"""
Example: Fine-tune AutoEncoder on new data

This example demonstrates how to fine-tune a pre-trained AutoEncoder model
on new monthly data while preserving existing knowledge.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder
from ai_trader.embedding.fine_tuning.fine_tuning import FineTuner, create_fine_tuning_config
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune AutoEncoder example')
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to pre-trained model')
    parser.add_argument('--data', type=Path, required=True,
                       help='Path to preprocessed data')
    parser.add_argument('--output', type=Path, default=Path('models/finetuned'),
                       help='Output directory')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load pre-trained model
    logger.info("Loading pre-trained model...")
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint['config']
    
    model = MaskedAutoEncoder(
        input_dim=config.get('num_features', 28),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        seq_len=config.get('seq_len', 60),
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        mask_ratio=config['mask_ratio']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    original_best_loss = checkpoint.get('best_val_loss', float('inf'))
    
    logger.info(f"Model loaded. Original best loss: {original_best_loss:.6f}")
    
    # Create datasets for new data
    logger.info("Creating datasets...")
    
    # Example: Use recent months for fine-tuning
    train_months = ['2024_11', '2024_12']  # Adjust as needed
    val_months = ['2025_01']               # Adjust as needed
    
    # Create mock datasets for example
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return torch.randn(60, 28)  # seq_len=60, num_features=28
    
    train_dataset = MockDataset(100)
    val_dataset = MockDataset(20)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Single-threaded for example
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info("Using mock datasets for example purposes")
    
    # Create fine-tuning configuration
    finetune_config = create_fine_tuning_config(
        learning_rate=5e-5,      # Lower than original training
        max_epochs=3,            # Few epochs for example
        weight_decay=1e-5,
        batch_size=2,
        freeze_layers=1,         # Freeze first layer
        warmup_epochs=1,
        lr_schedule='cosine',
        dropout_increase=0.02,   # Small increase
        gradient_clip=0.5
    )
    
    logger.info("Fine-tuning configuration:")
    for key, value in finetune_config.items():
        logger.info(f"  {key}: {value}")
    
    # Create fine-tuner
    fine_tuner = FineTuner(model, device, finetune_config)
    
    # Perform fine-tuning
    logger.info("Starting fine-tuning...")
    
    try:
        results = fine_tuner.fine_tune(
            train_loader,
            val_loader,
            save_path=args.output / 'best_model.pt'
        )
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("FINE-TUNING RESULTS")
        logger.info("="*50)
        logger.info(f"Initial validation loss: {results['initial_val_loss']:.6f}")
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Improvement: {results['improvement']:.6f} ({results['improvement_percent']:.2f}%)")
        logger.info(f"Training time: {results['training_time_minutes']:.1f} minutes")
        logger.info(f"Epochs completed: {results['epochs']}")
        
        # Save final model with metadata
        args.output.mkdir(parents=True, exist_ok=True)
        
        final_model_path = args.output / 'model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'original_config': config,
            'finetune_config': finetune_config,
            'results': results,
            'original_best_loss': original_best_loss,
            'timestamp': datetime.now().isoformat(),
            'train_months': train_months,
            'val_months': val_months
        }, final_model_path)
        
        logger.info(f"Fine-tuned model saved to: {final_model_path}")
        logger.info("="*50)
        
        # Example: Test the fine-tuned model
        logger.info("Testing fine-tuned model...")
        model.eval()
        
        with torch.no_grad():
            sample_batch = next(iter(val_loader)).to(device)
            reconstruction, embedding, mask = model(sample_batch)
            
            # Calculate reconstruction quality
            recon_loss = torch.nn.functional.mse_loss(
                reconstruction[mask], 
                sample_batch[mask]
            )
            
            logger.info(f"Sample reconstruction loss: {recon_loss.item():.6f}")
            logger.info(f"Embedding shape: {embedding.shape}")
            logger.info(f"Embedding mean: {embedding.mean().item():.4f}")
            logger.info(f"Embedding std: {embedding.std().item():.4f}")
        
        logger.info("Fine-tuning example completed successfully!")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise


if __name__ == '__main__':
    main()