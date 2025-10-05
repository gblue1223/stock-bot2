"""
임베딩 모델 훈련 스크립트

대조 학습을 사용하여 매매 데이터로부터 임베딩 모델을 훈련합니다.

사용 예시:
    python -m ai_trader.embedding.train_embedding \
        --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \
        --table datasets \
        --out models/embedding \
        --seq-len 60 \
        --embedding-dim 128 \
        --batch-size 128 \
        --epochs 50 \
        --lr 1e-4 \
        --device cuda
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
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


def parse_args():
    """CLI 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='임베딩 모델 훈련 - 대조 학습을 통한 매매 데이터 임베딩',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 데이터 관련 인수
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='DuckDB 데이터베이스 경로 (예: datasets_norm_all.duckdb)'
    )
    
    parser.add_argument(
        '--table',
        type=str,
        required=True,
        help='데이터베이스 테이블명 (예: datasets)'
    )
    
    # 출력 관련 인수
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='모델 체크포인트 저장 디렉토리 (예: models/embedding)'
    )
    
    # 모델 하이퍼파라미터
    parser.add_argument(
        '--seq-len',
        type=int,
        default=60,
        help='입력 시퀀스 길이 (타임스텝 수)'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='임베딩 벡터 차원'
    )
    
    # 훈련 하이퍼파라미터
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='배치 크기'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='훈련 에포크 수'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='학습률 (learning rate)'
    )
    
    # 디바이스 설정
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='훈련에 사용할 디바이스'
    )
    
    # 체크포인트 설정
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=5,
        help='체크포인트 저장 간격 (에포크 단위)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='재개할 체크포인트 경로 (선택사항)'
    )
    
    # 모델 하이퍼파라미터 (추가)
    parser.add_argument(
        '--num-heads',
        type=int,
        default=4,
        help='Multi-head Attention의 헤드 수'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help='InfoNCE 손실의 temperature 파라미터'
    )
    
    # 데이터 로더 설정
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='데이터 로더 워커 프로세스 수'
    )
    
    # 대조 학습 쌍 생성 설정
    parser.add_argument(
        '--positive-time-threshold',
        type=int,
        default=10,
        help='긍정 쌍 시간 임계값 (초, 기본값: 10)'
    )
    
    parser.add_argument(
        '--negative-time-threshold',
        type=int,
        default=60,
        help='부정 쌍 시간 임계값 (초, 기본값: 60)'
    )
    
    # 검증 및 체크포인트 설정 (추가)
    parser.add_argument(
        '--val-every',
        type=int,
        default=1,
        help='검증 주기 (에포크 단위, 기본값: 1 = 매 에포크마다 검증)'
    )
    
    return parser.parse_args()


def validate_args(args):
    """인수 유효성 검증"""
    # 데이터베이스 파일 존재 확인
    if not os.path.exists(args.db):
        raise FileNotFoundError(f"데이터베이스 파일을 찾을 수 없습니다: {args.db}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 하이퍼파라미터 범위 검증
    if args.seq_len <= 0:
        raise ValueError(f"seq-len은 양수여야 합니다: {args.seq_len}")
    
    if args.embedding_dim <= 0:
        raise ValueError(f"embedding-dim은 양수여야 합니다: {args.embedding_dim}")
    
    if args.batch_size <= 0:
        raise ValueError(f"batch-size는 양수여야 합니다: {args.batch_size}")
    
    if args.epochs <= 0:
        raise ValueError(f"epochs는 양수여야 합니다: {args.epochs}")
    
    if args.lr <= 0:
        raise ValueError(f"lr은 양수여야 합니다: {args.lr}")
    
    # CUDA 사용 가능 여부 확인
    if args.device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                print("경고: CUDA를 사용할 수 없습니다. CPU로 대체합니다.")
                args.device = 'cpu'
        except ImportError:
            print("경고: PyTorch를 가져올 수 없습니다. 나중에 확인됩니다.")
    
    return args


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    val_loss: float,
    config: Dict[str, Any],
    feature_names: list,
    normalization_stats: Dict[str, Any],
    output_dir: Path
) -> str:
    """
    체크포인트 저장
    
    Args:
        model: 모델
        optimizer: Optimizer
        epoch: 현재 에포크
        loss: 훈련 손실
        val_loss: 검증 손실
        config: 모델 설정
        feature_names: 특징 컬럼명 리스트
        normalization_stats: 정규화 통계
        output_dir: 출력 디렉토리
        
    Returns:
        저장된 체크포인트 경로
    """
    checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'feature_names': feature_names,
        'normalization_stats': normalization_stats,
        'training_metadata': {
            'epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    return str(checkpoint_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Any]:
    """
    체크포인트 로드
    
    Args:
        checkpoint_path: 체크포인트 경로
        model: 모델
        optimizer: Optimizer
        
    Returns:
        체크포인트 메타데이터
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    logger.info(f"Resuming from epoch {checkpoint['training_metadata']['epoch']}")
    
    return checkpoint['training_metadata']


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    한 에포크 훈련
    
    Args:
        model: 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: Optimizer
        device: 디바이스
        epoch: 현재 에포크
        
    Returns:
        평균 손실
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        # 디바이스로 이동
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        # 부정 샘플을 배치 차원으로 변환 (InfoNCE 손실 함수 형식에 맞춤)
        # (batch, embedding_dim) -> (batch, 1, embedding_dim)
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
        
        # 진행 상황 출력
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    검증
    
    Args:
        model: 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 디바이스
        
    Returns:
        (평균 검증 손실, 임베딩 리스트, 종목코드 리스트, 타임스탬프 리스트)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # 임베딩 품질 메트릭 계산을 위한 데이터 수집
    all_embeddings = []
    all_stock_codes = []
    all_timestamps = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 3:
                # 훈련 데이터 형식: (anchor, positive, negative)
                anchor, positive, negative = batch_data
                
                # 디바이스로 이동
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                # Forward pass
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                # 부정 샘플을 배치 차원으로 변환
                negative_emb = negative_emb.unsqueeze(1)
                
                # 손실 계산
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                # 통계 업데이트
                total_loss += loss.item()
                num_batches += 1
                
                # 임베딩 수집 (anchor만 사용)
                all_embeddings.append(anchor_emb.cpu().numpy())
            elif len(batch_data) == 5:
                # 메타데이터 포함 형식: (anchor, positive, negative, stock_codes, timestamps)
                anchor, positive, negative, stock_codes, timestamps = batch_data
                
                # 디바이스로 이동
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                # Forward pass
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                # 부정 샘플을 배치 차원으로 변환
                negative_emb = negative_emb.unsqueeze(1)
                
                # 손실 계산
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                # 통계 업데이트
                total_loss += loss.item()
                num_batches += 1
                
                # 임베딩 및 메타데이터 수집
                all_embeddings.append(anchor_emb.cpu().numpy())
                all_stock_codes.extend(stock_codes)
                all_timestamps.extend(timestamps)
    
    avg_loss = total_loss / num_batches
    
    # 임베딩 배열로 변환
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
    else:
        all_embeddings = np.array([])
    
    return avg_loss, all_embeddings, all_stock_codes, all_timestamps


def compute_normalization_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    정규화 통계 계산
    
    Args:
        data: 데이터 배열 (n_samples, n_features)
        
    Returns:
        정규화 통계 (mean, std)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # std가 0인 경우 1로 대체 (division by zero 방지)
    std = np.where(std == 0, 1.0, std)
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist()
    }


def main():
    """메인 훈련 함수"""
    # CLI 인수 파싱 및 검증
    args = parse_args()
    args = validate_args(args)
    
    # 설정 출력
    logger.info("=" * 80)
    logger.info("임베딩 모델 훈련 시작")
    logger.info("=" * 80)
    logger.info(f"데이터베이스: {args.db}")
    logger.info(f"테이블: {args.table}")
    logger.info(f"출력 디렉토리: {args.out}")
    logger.info(f"시퀀스 길이: {args.seq_len}")
    logger.info(f"임베딩 차원: {args.embedding_dim}")
    logger.info(f"배치 크기: {args.batch_size}")
    logger.info(f"에포크: {args.epochs}")
    logger.info(f"학습률: {args.lr}")
    logger.info(f"디바이스: {args.device}")
    logger.info(f"체크포인트 간격: {args.checkpoint_interval}")
    logger.info("=" * 80)
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 출력 디렉토리 설정
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard 설정
    tensorboard_dir = output_dir / 'tensorboard_logs'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    
    # 데이터 로더 초기화
    logger.info("Initializing data loader...")
    data_loader = EmbeddingDataLoader(
        db_path=args.db,
        table_name=args.table,
        seq_len=args.seq_len
    )
    
    # 데이터 로드 및 분할
    logger.info("Loading and splitting data...")
    data_loader.load_and_split_data()
    
    # 특징 컬럼 가져오기
    feature_names = data_loader._get_feature_columns()
    input_dim = len(feature_names)
    logger.info(f"Input dimension: {input_dim}")
    
    # 정규화 통계 계산 (훈련 세트 기준)
    logger.info("Computing normalization statistics...")
    normalization_stats = compute_normalization_stats(data_loader.data_splits['train']['data'])
    
    # DataLoader 생성
    train_dataloader = data_loader.get_dataloader(
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        return_metadata=False,  # 훈련 시에는 메타데이터 불필요
        positive_time_threshold=args.positive_time_threshold,
        negative_time_threshold=args.negative_time_threshold
    )
    
    val_dataloader = data_loader.get_dataloader(
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        return_metadata=True,  # 검증 시에는 메트릭 계산을 위해 메타데이터 필요
        positive_time_threshold=args.positive_time_threshold,
        negative_time_threshold=args.negative_time_threshold
    )
    
    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Val batches: {len(val_dataloader)}")
    
    # 모델 초기화
    logger.info("Initializing model...")
    model = TradingEmbeddingModel(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 손실 함수 및 Optimizer 초기화
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 체크포인트 재개 (선택사항)
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        metadata = load_checkpoint(args.resume, model, optimizer)
        start_epoch = metadata['epoch'] + 1
    
    # 훈련 루프
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*80}")
        
        # 훈련
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        # TensorBoard 로깅 (훈련 손실은 항상 로깅)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('learning_rate', args.lr, epoch)
        
        # 검증 (val_every 주기마다 실행)
        val_loss = 0.0
        silhouette = 0.0
        temporal_coherence = 0.0
        
        if epoch % args.val_every == 0:
            val_loss, val_embeddings, val_stock_codes, val_timestamps = validate(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device
            )
            
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            # 임베딩 품질 메트릭 계산
            if len(val_embeddings) > 0:
                # Silhouette score 계산 (종목 클러스터링 품질)
                if len(val_stock_codes) > 0:
                    silhouette = compute_silhouette_score(val_embeddings, val_stock_codes)
                    logger.info(f"Epoch {epoch} - Silhouette Score: {silhouette:.4f}")
                
                # Temporal coherence 계산 (시간적 일관성)
                if len(val_timestamps) > 0:
                    temporal_coherence = compute_temporal_coherence(val_embeddings, val_timestamps)
                    logger.info(f"Epoch {epoch} - Temporal Coherence: {temporal_coherence:.4f}")
            
            # TensorBoard 로깅 (검증 메트릭)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/silhouette_score', silhouette, epoch)
            writer.add_scalar('val/temporal_coherence', temporal_coherence, epoch)
            
            # 최고 모델 저장 (검증 손실이 개선된 경우)
            if val_loss > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=train_loss,
                    val_loss=val_loss,
                    config=model.get_config(),
                    feature_names=feature_names,
                    normalization_stats=normalization_stats,
                    output_dir=output_dir
                )
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # 주기적 체크포인트 저장
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                val_loss=val_loss,
                config=model.get_config(),
                feature_names=feature_names,
                normalization_stats=normalization_stats,
                output_dir=output_dir
            )
    
    # 최종 체크포인트 저장
    final_checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        loss=train_loss,
        val_loss=val_loss,
        config=model.get_config(),
        feature_names=feature_names,
        normalization_stats=normalization_stats,
        output_dir=output_dir
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final checkpoint: {final_checkpoint_path}")
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    logger.info(f"{'='*80}")
    
    # 정리
    writer.close()
    data_loader.close()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        sys.exit(1)
