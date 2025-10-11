"""
Example script for training AutoEncoder-based embeddings.
Demonstrates fast training on large datasets as alternative to contrastive learning.
"""

import argparse
import logging
import numpy as np
import duckdb
from pathlib import Path
import time
import json

from ai_trader.embedding.autoencoder_trainer import train_autoencoder_embedding
from ai_trader.embedding.fine_tuning import fine_tune_for_trading_task


def load_data_from_duckdb(db_path: str, limit: int = None) -> tuple:
    """Load training data from DuckDB."""
    conn = duckdb.connect(db_path)
    
    # Get feature columns (excluding metadata)
    query = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'market_data' 
    AND column_name NOT IN ('timestamp', 'stock_code', 'date')
    ORDER BY column_name
    """
    
    feature_columns = [row[0] for row in conn.execute(query).fetchall()]
    
    # Load data
    limit_clause = f"LIMIT {limit}" if limit else ""
    data_query = f"""
    SELECT {', '.join(feature_columns)}
    FROM market_data 
    WHERE {' AND '.join([f'{col} IS NOT NULL' for col in feature_columns])}
    ORDER BY timestamp
    {limit_clause}
    """
    
    data = conn.execute(data_query).fetchnumpy()
    
    # Convert to numpy array
    feature_data = np.column_stack([data[col] for col in feature_columns])
    
    conn.close()
    
    return feature_data.astype(np.float32), feature_columns


def create_trading_labels(data: np.ndarray, seq_len: int = 60, 
                         future_steps: int = 5) -> np.ndarray:
    """
    Create trading labels for fine-tuning.
    
    Args:
        data: Market data (n_samples, n_features)
        seq_len: Sequence length
        future_steps: Steps ahead to predict
    
    Returns:
        labels: Trading signals (0=sell, 1=hold, 2=buy)
    """
    # Assume first feature is price-related
    prices = data[:, 0]
    
    labels = []
    for i in range(len(data) - seq_len - future_steps):
        current_price = prices[i + seq_len - 1]
        future_price = prices[i + seq_len + future_steps - 1]
        
        price_change = (future_price - current_price) / current_price
        
        # Simple thresholding for trading signals
        if price_change > 0.002:  # 0.2% increase -> buy
            label = 2
        elif price_change < -0.002:  # 0.2% decrease -> sell
            label = 0
        else:  # hold
            label = 1
        
        labels.append(label)
    
    return np.array(labels)


def create_sequences(data: np.ndarray, seq_len: int = 60, stride: int = 1) -> np.ndarray:
    """Create sequences from time series data."""
    sequences = []
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i:i + seq_len])
    
    return np.array(sequences)


def main():
    parser = argparse.ArgumentParser(description="Train AutoEncoder embedding model")
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    parser.add_argument("--output-dir", default="models/autoencoder", 
                       help="Output directory for models")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--model-type", choices=['standard', 'masked'], default='masked',
                       help="Model type")
    parser.add_argument("--mask-ratio", type=float, default=0.15, 
                       help="Mask ratio for masked autoencoder")
    parser.add_argument("--fine-tune", action='store_true', 
                       help="Also run fine-tuning for trading task")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda',
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AutoEncoder embedding training")
    logger.info(f"Database: {args.db}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model type: {args.model_type}")
    
    # Load data
    logger.info("Loading data from DuckDB...")
    start_time = time.time()
    
    try:
        data, feature_columns = load_data_from_duckdb(args.db, args.limit)
        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Features: {len(feature_columns)}")
        logger.info(f"Data loading time: {time.time() - start_time:.1f}s")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    logger.info(f"Train data: {train_data.shape}")
    logger.info(f"Validation data: {val_data.shape}")
    
    # Training configuration
    config = {
        'model_type': args.model_type,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'seq_len': args.seq_len,
        'num_layers': 3,
        'dropout': 0.1,
        'mask_ratio': args.mask_ratio,
        'batch_size': args.batch_size,
        'max_epochs': args.epochs,
        'learning_rate': 1e-3,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'output_dir': args.output_dir,
        'stride': 1,
        'save_every': 10
    }
    
    # Train autoencoder
    logger.info("Starting autoencoder training...")
    start_time = time.time()
    
    try:
        model, trainer, history = train_autoencoder_embedding(
            train_data, val_data, config, args.output_dir
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s")
        logger.info(f"Final train loss: {history[-1]['train_loss']:.4f}")
        if history[-1]['val_loss']:
            logger.info(f"Final val loss: {history[-1]['val_loss']:.4f}")
        
        # Save training summary
        summary = {
            'training_time': training_time,
            'data_shape': data.shape,
            'config': config,
            'final_train_loss': history[-1]['train_loss'],
            'final_val_loss': history[-1]['val_loss'],
            'feature_columns': feature_columns
        }
        
        output_path = Path(args.output_dir)
        with open(output_path / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # Fine-tuning for trading task
    if args.fine_tune:
        logger.info("Starting fine-tuning for trading task...")
        
        try:
            # Create sequences and labels
            train_sequences = create_sequences(train_data, args.seq_len, stride=5)
            train_labels = create_trading_labels(train_data, args.seq_len, future_steps=5)
            
            val_sequences = create_sequences(val_data, args.seq_len, stride=5)
            val_labels = create_trading_labels(val_data, args.seq_len, future_steps=5)
            
            # Ensure matching lengths
            min_len = min(len(train_sequences), len(train_labels))
            train_sequences = train_sequences[:min_len]
            train_labels = train_labels[:min_len]
            
            min_len = min(len(val_sequences), len(val_labels))
            val_sequences = val_sequences[:min_len]
            val_labels = val_labels[:min_len]
            
            logger.info(f"Fine-tuning sequences: {train_sequences.shape}")
            logger.info(f"Fine-tuning labels: {train_labels.shape}")
            
            # Fine-tune
            finetuned_model, ft_trainer, ft_history = fine_tune_for_trading_task(
                pretrained_model_path=str(Path(args.output_dir) / 'best_model.pt'),
                train_sequences=train_sequences,
                train_labels=train_labels,
                val_sequences=val_sequences,
                val_labels=val_labels,
                task_type='classification',
                num_classes=3,
                config={
                    'batch_size': 128,
                    'max_epochs': 30,
                    'encoder_lr': 1e-4,
                    'head_lr': 1e-3,
                    'freeze_encoder': False,
                    'output_dir': str(Path(args.output_dir) / 'finetuned')
                }
            )
            
            logger.info("Fine-tuning completed!")
            final_acc = ft_history[-1]['val_metrics']['accuracy']
            logger.info(f"Final validation accuracy: {final_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
    
    logger.info("All training completed successfully!")


if __name__ == "__main__":
    main()