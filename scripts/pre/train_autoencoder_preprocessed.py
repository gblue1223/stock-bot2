#!/usr/bin/env python3
"""
전처리된 HDF5 배치 데이터로 오토인코더 훈련
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import random
from tqdm import tqdm
import argparse

from ai_trader.embedding.autoencoder_model import create_autoencoder_model
from ai_trader.embedding.autoencoder_trainer import AutoEncoderTrainer


class PreprocessedDataset(Dataset):
    """전처리된 HDF5 배치 파일들을 로딩하는 데이터셋"""
    
    def __init__(self, data_dir: str, months: List[str], max_batches_per_month: Optional[int] = None):
        """
        Args:
            data_dir: 전처리된 데이터 디렉토리
            months: 사용할 월 리스트 (예: ['2024_09', '2024_10'])
            max_batches_per_month: 월별 최대 배치 수 (메모리 제한용)
        """
        self.data_dir = Path(data_dir)
        self.batch_files = []
        self.batch_info = {}
        
        # 각 월의 배치 파일들 수집
        for month in months:
            month_dir = self.data_dir / month
            if not month_dir.exists():
                logging.warning(f"Month directory not found: {month_dir}")
                continue
                
            # batch_info.json 로드
            batch_info_file = month_dir / 'batch_info.json'
            if batch_info_file.exists():
                with open(batch_info_file, 'r') as f:
                    info = json.load(f)
                    self.batch_info[month] = info
                    
                    # 배치 파일 목록 생성
                    num_batches = info['num_batches']
                    if max_batches_per_month:
                        num_batches = min(num_batches, max_batches_per_month)
                    
                    for batch_idx in range(num_batches):
                        batch_file = month_dir / f'batch_{batch_idx:06d}.h5'
                        if batch_file.exists():
                            self.batch_files.append(str(batch_file))
        
        # 배치 파일 셔플
        random.shuffle(self.batch_files)
        
        logging.info(f"Found {len(self.batch_files)} batch files from {len(months)} months")
        
        # 첫 번째 배치에서 데이터 형태 확인
        if self.batch_files:
            with h5py.File(self.batch_files[0], 'r') as f:
                sample_data = f['sequences'][:1]
                self.seq_len = sample_data.shape[1]
                self.num_features = sample_data.shape[2]
                logging.info(f"Data shape: seq_len={self.seq_len}, num_features={self.num_features}")
    
    def __len__(self):
        return len(self.batch_files)
    
    def __getitem__(self, idx):
        """배치 파일 하나를 로딩"""
        batch_file = self.batch_files[idx]
        
        with h5py.File(batch_file, 'r') as f:
            sequences = f['sequences'][:]
            # metadata = f['metadata'][:]  # 필요시 사용
        
        # numpy -> torch tensor
        sequences = torch.from_numpy(sequences).float()
        return sequences


class BatchCollator:
    """배치들을 하나의 큰 배치로 합치는 콜레이터"""
    
    def __init__(self, max_sequences_per_batch: int = 1000):
        self.max_sequences_per_batch = max_sequences_per_batch
    
    def __call__(self, batch_list):
        """
        여러 배치 파일의 시퀀스들을 하나로 합침
        
        Args:
            batch_list: List of tensors, each of shape (batch_size, seq_len, num_features)
        
        Returns:
            Combined tensor of shape (total_sequences, seq_len, num_features)
        """
        # 모든 배치를 하나로 합침
        all_sequences = torch.cat(batch_list, dim=0)
        
        # 너무 크면 샘플링
        if len(all_sequences) > self.max_sequences_per_batch:
            indices = torch.randperm(len(all_sequences))[:self.max_sequences_per_batch]
            all_sequences = all_sequences[indices]
        
        return all_sequences


def create_data_loaders(data_dir: str, train_months: List[str], val_months: List[str],
                       batch_size: int = 4, max_batches_per_month: Optional[int] = None,
                       max_sequences_per_batch: int = 1000):
    """데이터 로더 생성"""
    
    # 훈련 데이터셋
    train_dataset = PreprocessedDataset(data_dir, train_months, max_batches_per_month)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=BatchCollator(max_sequences_per_batch),
        pin_memory=True
    )
    
    # 검증 데이터셋
    val_loader = None
    if val_months:
        val_dataset = PreprocessedDataset(data_dir, val_months, max_batches_per_month)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=BatchCollator(max_sequences_per_batch),
            pin_memory=True
        )
    
    return train_loader, val_loader, train_dataset.num_features


class PreprocessedAutoEncoderTrainer(AutoEncoderTrainer):
    """전처리된 데이터용 오토인코더 트레이너"""
    
    def train_epoch_preprocessed(self, dataloader: DataLoader, optimizer, mask_ratio: Optional[float] = None):
        """전처리된 데이터로 한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_sequences = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            batch_size = batch.size(0)
            
            optimizer.zero_grad()
            loss, metrics = self.compute_loss(batch, mask_ratio)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            total_sequences += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'sequences': f"{total_sequences:,}"
            })
        
        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {self.current_epoch}: processed {total_sequences:,} sequences, avg_loss={avg_loss:.4f}")
        return avg_loss
    
    def validate_preprocessed(self, dataloader: DataLoader, mask_ratio: Optional[float] = None):
        """전처리된 데이터로 검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_sequences = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = batch.to(self.device)
                loss, metrics = self.compute_loss(batch, mask_ratio)
                total_loss += loss.item()
                num_batches += 1
                total_sequences += batch.size(0)
        
        avg_loss = total_loss / num_batches
        logging.info(f"Validation: processed {total_sequences:,} sequences, avg_loss={avg_loss:.4f}")
        return avg_loss
    
    def train_preprocessed(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                          config: dict = None):
        """전처리된 데이터로 훈련"""
        if config is None:
            config = {}
        
        # Training parameters
        max_epochs = config.get('max_epochs', 50)
        mask_ratio = config.get('mask_ratio', 0.15) if hasattr(self.model, 'mask_ratio') else None
        
        # Create optimizer
        optimizer, scheduler = self.create_optimizer(config)
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        self.logger.info(f"Using device: {self.device}")
        
        # Training loop
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch_preprocessed(train_loader, optimizer, mask_ratio)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate_preprocessed(val_loader, mask_ratio)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Log progress
            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
            if val_loss:
                log_msg += f", val_loss={val_loss:.4f}"
            log_msg += f", lr={optimizer.param_groups[0]['lr']:.6f}"
            
            self.logger.info(log_msg)
            
            # Save training history
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
            
            # Save best model
            current_loss = val_loss if val_loss else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(config.get('output_dir', 'models'), is_best=True)
            
            # Regular checkpoint
            if epoch % config.get('save_every', 5) == 0:
                self.save_checkpoint(config.get('output_dir', 'models'))
        
        self.logger.info("Training completed!")
        return self.training_history


def main():
    parser = argparse.ArgumentParser(description='Train AutoEncoder on preprocessed data')
    parser.add_argument('--data-dir', required=True, help='Preprocessed data directory')
    parser.add_argument('--output-dir', default='models/autoencoder_preprocessed', help='Output directory')
    parser.add_argument('--train-months', nargs='+', 
                       default=['2024_09', '2024_10', '2024_11', '2024_12', '2025_01', '2025_02'],
                       help='Training months')
    parser.add_argument('--val-months', nargs='+', default=['2025_03'],
                       help='Validation months')
    parser.add_argument('--max-epochs', type=int, default=20, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (number of batch files)')
    parser.add_argument('--max-sequences', type=int, default=2000, help='Max sequences per training batch')
    parser.add_argument('--max-batches-per-month', type=int, default=50, help='Max batches per month')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model-type', choices=['standard', 'masked'], default='masked', help='Model type')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_path / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create data loaders
    logging.info("Creating data loaders...")
    train_loader, val_loader, num_features = create_data_loaders(
        args.data_dir,
        args.train_months,
        args.val_months,
        batch_size=args.batch_size,
        max_batches_per_month=args.max_batches_per_month,
        max_sequences_per_batch=args.max_sequences
    )
    
    logging.info(f"Number of features: {num_features}")
    logging.info(f"Training batches: {len(train_loader)}")
    if val_loader:
        logging.info(f"Validation batches: {len(val_loader)}")
    
    # Model configuration
    config = {
        'model_type': args.model_type,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'seq_len': 60,  # From preprocessed data
        'num_layers': 3,
        'dropout': 0.1,
        'mask_ratio': 0.15,
        'max_epochs': args.max_epochs,
        'learning_rate': args.learning_rate,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'weight_decay': 1e-4,
        'save_every': 5,
        'output_dir': args.output_dir
    }
    
    # Create model
    logging.info("Creating model...")
    model = create_autoencoder_model(num_features, config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    trainer = PreprocessedAutoEncoderTrainer(model, device)
    
    # Save config
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    logging.info("Starting training...")
    history = trainer.train_preprocessed(train_loader, val_loader, config)
    
    # Save training history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info("Training completed successfully!")
    logging.info(f"Best loss: {trainer.best_loss:.4f}")
    logging.info(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()