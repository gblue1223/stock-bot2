"""
분산 훈련용 임베딩 모델 훈련 스크립트
PyTorch DistributedDataParallel (DDP) 사용

A100 40GB x12 GPU 최적화
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from ai_trader.embedding.models import TradingEmbeddingModel
from ai_trader.embedding.losses import InfoNCELoss
from ai_trader.embedding.data import EmbeddingDataLoader
from ai_trader.embedding.evaluation import (
    compute_silhouette_score,
    compute_temporal_coherence
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """분산 훈련 환경 설정"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """분산 훈련 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    """CLI 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='분산 임베딩 모델 훈련',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 기존 train_embedding.py와 동일한 인수들
    parser.add_argument('--db', type=str, required=True, help='DuckDB 데이터베이스 경로')
    parser.add_argument('--table', type=str, required=True, help='테이블명')
    parser.add_argument('--out', type=str, required=True, help='출력 디렉토리')
    parser.add_argument('--seq-len', type=int, default=60, help='시퀀스 길이')
    parser.add_argument('--embedding-dim', type=int, default=128, help='임베딩 차원')
    parser.add_argument('--batch-size', type=int, default=512, help='GPU당 배치 크기')
    parser.add_argument('--epochs', type=int, default=30, help='에포크 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='체크포인트 간격')
    parser.add_argument('--resume', type=str, default=None, help='재개할 체크포인트')
    parser.add_argument('--num-heads', type=int, default=4, help='Attention 헤드 수')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature')
    parser.add_argument('--num-workers', type=int, default=16, help='워커 수')
    parser.add_argument('--positive-time-threshold', type=int, default=10)
    parser.add_argument('--negative-time-threshold', type=int, default=60)
    parser.add_argument('--val-every', type=int, default=3, help='검증 주기')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--stock-codes', type=str, nargs='+', default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    
    # 분산 훈련 전용 인수
    parser.add_argument('--local_rank', type=int, default=0, help='로컬 GPU 랭크')
    
    return parser.parse_args()


def train_epoch_distributed(
    model: DDP,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    rank: int,
    world_size: int
) -> float:
    """분산 훈련 에포크"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        negative_emb = negative_emb.unsqueeze(1)
        
        # 손실 계산
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        total_loss += loss.item()
        num_batches += 1
        
        # 진행 상황 출력 (rank 0만)
        if rank == 0 and (batch_idx + 1) % 100 == 0:
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (GPU: {world_size})"
            )
    
    avg_loss = total_loss / num_batches
    
    # 모든 GPU의 평균 손실 계산
    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    return avg_loss


def main():
    """메인 함수"""
    args = parse_args()
    
    # 분산 환경 설정
    rank, world_size, local_rank = setup_distributed()
    
    # 디바이스 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # Rank 0만 로깅
    if rank == 0:
        logger.info("=" * 80)
        logger.info("분산 임베딩 모델 훈련 시작")
        logger.info("=" * 80)
        logger.info(f"World Size (GPU 개수): {world_size}")
        logger.info(f"Rank: {rank}")
        logger.info(f"Local Rank: {local_rank}")
        logger.info(f"Device: {device}")
        logger.info(f"데이터베이스: {args.db}")
        logger.info(f"배치 크기 (GPU당): {args.batch_size}")
        logger.info(f"총 배치 크기: {args.batch_size * world_size}")
        logger.info("=" * 80)
    
    # 출력 디렉토리 생성 (rank 0만)
    if rank == 0:
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard_dir = output_dir / 'tensorboard_logs'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    # 데이터 로더 초기화 (모든 rank)
    if rank == 0:
        logger.info("데이터 로더 초기화...")
    
    data_loader = EmbeddingDataLoader(
        db_path=args.db,
        table_name=args.table,
        seq_len=args.seq_len,
        start_date=args.start_date,
        end_date=args.end_date,
        stock_codes=args.stock_codes,
        max_samples=args.max_samples
    )
    
    # 데이터 로드 및 분할
    data_loader.load_and_split_data()
    
    # 특징 컬럼 가져오기
    feature_names = data_loader._get_feature_columns()
    input_dim = len(feature_names)
    
    if rank == 0:
        logger.info(f"Input dimension: {input_dim}")
    
    # 분산 샘플러 생성
    train_dataset = data_loader.get_dataset(
        split='train',
        return_metadata=False,
        positive_time_threshold=args.positive_time_threshold,
        negative_time_threshold=args.negative_time_threshold
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        logger.info(f"Train batches per GPU: {len(train_dataloader)}")
        logger.info(f"Total train batches: {len(train_dataloader) * world_size}")
    
    # 모델 초기화
    if rank == 0:
        logger.info("모델 초기화...")
    
    model = TradingEmbeddingModel(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads
    ).to(device)
    
    # DDP로 래핑
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
    
    # 손실 함수 및 Optimizer
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 훈련 루프
    if rank == 0:
        logger.info("훈련 시작...")
    
    for epoch in range(1, args.epochs + 1):
        # 에포크마다 샘플러 셔플
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{args.epochs}")
            logger.info(f"{'='*80}")
        
        # 훈련
        train_loss = train_epoch_distributed(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            rank=rank,
            world_size=world_size
        )
        
        if rank == 0:
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            writer.add_scalar('train/loss', train_loss, epoch)
        
        # 체크포인트 저장 (rank 0만)
        if rank == 0 and epoch % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            
            # DDP 모델의 경우 module.state_dict() 사용
            model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
            
            checkpoint = {
                'state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model.module.get_config() if world_size > 1 else model.get_config(),
                'feature_names': feature_names,
                'training_metadata': {
                    'epoch': epoch,
                    'loss': train_loss,
                    'timestamp': datetime.now().isoformat(),
                    'world_size': world_size
                }
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # 정리
    if rank == 0:
        logger.info("\n" + "="*80)
        logger.info("훈련 완료!")
        logger.info("="*80)
        writer.close()
    
    data_loader.close()
    cleanup_distributed()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        cleanup_distributed()
        sys.exit(1)
