"""
임베딩 모델 훈련 스크립트

대조 학습을 사용하여 매매 데이터로부터 임베딩 모델을 훈련합니다.

사용 예시:
    # 일반 훈련
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
    
    # 사전 계산된 쌍 사용 (빠른 시작)
    python -m ai_trader.embedding.train_embedding \
        --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \
        --table datasets \
        --out models/embedding \
        --precomputed-pairs-dir data/precomputed_pairs \
        --batch-size 128 \
        --epochs 50 \
        --device cuda
    
    # 증분 학습 (전체 데이터 활용)
    python -m ai_trader.embedding.train_embedding \
        --db datasets_norm_all.duckdb \
        --table datasets \
        --out models/embedding_incremental \
        --batch-size 4096 \
        --epochs 10 \
        --lr 5e-5 \
        --max-samples 30000000 \
        --incremental \
        --chunk-size 10000000
    
    # 증분 학습 재개 (중단된 지점부터)
    python -m ai_trader.embedding.train_embedding \
        --db datasets_norm_all.duckdb \
        --table datasets \
        --out models/embedding_incremental \
        --batch-size 4096 \
        --epochs 10 \
        --lr 5e-5 \
        --max-samples 30000000 \
        --incremental \
        --chunk-size 10000000 \
        --resume models/embedding_incremental/checkpoint_chunk2.pt
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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


class EarlyStopping:
    """
    Early Stopping 구현
    
    검증 손실이 개선되지 않으면 훈련을 조기 종료합니다.
    """
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience: 개선이 없어도 기다릴 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        검증 손실을 확인하고 early stopping 여부 결정
        
        Args:
            val_loss: 현재 검증 손실
            
        Returns:
            True if early stopping should be triggered
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


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
        help='체크포인트 저장 간격 (일반 모드: 에포크 단위, 증분 모드: 청크 단위)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='재개할 체크포인트 경로 또는 디렉토리 (디렉토리 지정 시 최신 청크 자동 탐색)'
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
        type=float,
        default=0.05,
        help='긍정 쌍 시간 임계값 (시간_scalar 사용 시 표준화된 값, 기본값: 0.05)'
    )
    
    parser.add_argument(
        '--negative-time-threshold',
        type=float,
        default=0.5,
        help='부정 쌍 시간 임계값 (시간_scalar 사용 시 표준화된 값, 기본값: 0.5)'
    )
    
    # 검증 및 체크포인트 설정 (추가)
    parser.add_argument(
        '--val-every',
        type=int,
        default=1,
        help='검증 주기 (에포크 단위, 기본값: 1 = 매 에포크마다 검증)'
    )
    
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='로그 출력 간격 (배치 단위, 기본값: 10)'
    )
    
    # 데이터 필터링 설정
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='시작 날짜 (YYYYMMDD 형식, 예: 20240901)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='종료 날짜 (YYYYMMDD 형식, 예: 20240930)'
    )
    
    parser.add_argument(
        '--stock-codes',
        type=str,
        nargs='+',
        default=None,
        help='훈련할 종목 코드 리스트 (예: 005930 000660)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='최대 샘플 수 (일반: 로드할 샘플 수, 증분: 전체 샘플 수, 예: 30000000)'
    )
    
    # 증분 학습 설정
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='증분 학습 모드 활성화 (데이터를 청크로 나눠서 학습)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000000,
        help='증분 학습 시 청크 크기 (기본값: 1천만, max-samples를 이 크기로 나눔)'
    )
    
    # 사전 계산된 쌍 설정
    parser.add_argument(
        '--precomputed-pairs-dir',
        type=str,
        default=None,
        help='사전 계산된 긍정/부정 쌍 디렉토리 (예: data/precomputed_pairs)'
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
    epoch: int,
    scaler: torch.cuda.amp.GradScaler = None,
    log_interval: int = 10,
    chunk_info: tuple = None
) -> float:
    """
    한 에포크 훈련 (Mixed Precision 지원)
    
    Args:
        model: 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: Optimizer
        device: 디바이스
        epoch: 현재 에포크
        scaler: GradScaler for mixed precision (optional)
        log_interval: 로그 출력 간격 (배치 단위)
        chunk_info: (chunk_idx, num_chunks) 튜플 (증분 학습용, optional)
        
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
        
        # Mixed Precision Training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                # Forward pass
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                # 부정 샘플을 배치 차원으로 변환
                negative_emb = negative_emb.unsqueeze(1)
                
                # 손실 계산
                loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping (NaN 방지)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training (no mixed precision)
            # Forward pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # 부정 샘플을 배치 차원으로 변환
            negative_emb = negative_emb.unsqueeze(1)
            
            # 손실 계산
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (NaN 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # 통계 업데이트
        loss_value = loss.item()
        
        # NaN 체크
        if np.isnan(loss_value) or np.isinf(loss_value):
            logger.error(f"NaN/Inf loss detected at batch {batch_idx + 1}!")
            logger.error(f"Loss value: {loss_value}")
            logger.error("Training stopped to prevent further issues.")
            raise ValueError(f"NaN/Inf loss detected: {loss_value}")
        
        total_loss += loss_value
        num_batches += 1
        
        # 진행 상황 출력
        if (batch_idx + 1) % log_interval == 0:
            if chunk_info:
                chunk_idx, num_chunks = chunk_info
                logger.info(f"Chunk [{chunk_idx + 1}/{num_chunks}] Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss_value:.4f}")
            else:
                logger.info(f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss_value:.4f}")
    
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
                # timestamps가 Tensor인 경우 리스트로 변환
                if isinstance(timestamps, torch.Tensor):
                    all_timestamps.extend(timestamps.cpu().tolist())
                else:
                    all_timestamps.extend(timestamps)
    
    avg_loss = total_loss / num_batches
    
    # 임베딩 배열로 변환
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
    else:
        all_embeddings = np.array([])
    
    return avg_loss, all_embeddings, all_stock_codes, all_timestamps


def find_latest_checkpoint(resume_path: str) -> Optional[str]:
    """
    최신 체크포인트 파일 찾기
    
    Args:
        resume_path: 체크포인트 파일 경로 또는 디렉토리 경로
        
    Returns:
        최신 체크포인트 파일 경로 또는 None
    """
    import re
    
    resume_path = Path(resume_path)
    
    # 파일이면 그대로 반환
    if resume_path.is_file():
        logger.info(f"Using checkpoint file: {resume_path}")
        return str(resume_path)
    
    # 디렉토리면 최신 청크 파일 찾기
    if resume_path.is_dir():
        checkpoint_pattern = "checkpoint_chunk*.pt"
        checkpoint_files = list(resume_path.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {resume_path}")
            return None
        
        # 청크 번호로 정렬하여 최신 파일 찾기
        def extract_chunk_number(path):
            match = re.search(r'checkpoint_chunk(\d+)\.pt', path.name)
            return int(match.group(1)) if match else 0
        
        latest_checkpoint = max(checkpoint_files, key=extract_chunk_number)
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)
    
    logger.warning(f"Resume path not found: {resume_path}")
    return None


def compute_normalization_stats(data: np.ndarray) -> Dict[str, Any]:
    """
    정규화 통계 계산 (Robust Normalization)
    
    Args:
        data: 입력 데이터 (n_samples, n_features)
        
    Returns:
        정규화 통계 (mean, std)
    """
    # NaN/Inf 체크
    if np.any(np.isnan(data)):
        logger.warning("Data contains NaN values! Replacing with 0.")
        data = np.nan_to_num(data, nan=0.0)
    
    if np.any(np.isinf(data)):
        logger.warning("Data contains Inf values! Clipping to finite range.")
        data = np.nan_to_num(data, posinf=1e10, neginf=-1e10)
    
    # Robust normalization: median과 IQR 사용
    # 극단값에 덜 민감함
    median = np.median(data, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    iqr = q75 - q25
    
    # IQR이 0인 경우 std 사용
    iqr = np.where(iqr == 0, np.std(data, axis=0), iqr)
    
    # 여전히 0이면 1로 대체
    iqr = np.where(iqr == 0, 1.0, iqr)
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    
    # Mean과 Std 대신 Median과 IQR 사용
    mean = median
    std = iqr
    
    logger.info(f"Normalization stats - Median range: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
    logger.info(f"Normalization stats - IQR range: [{np.min(std):.4f}, {np.max(std):.4f}]")
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist()
    }


def train_incremental(args, device, output_dir, writer):
    """
    증분 학습 함수 - 데이터를 청크로 나눠서 순차적으로 학습
    
    증분 학습 방식:
    - 전체 데이터를 청크로 나누어 순차적으로 학습
    - 각 청크마다 args.epochs 만큼 반복 학습 (catastrophic forgetting 방지)
    - 이전 청크의 가중치를 유지하면서 새로운 청크 학습
    
    Args:
        args: CLI 인수
        device: torch device
        output_dir: 출력 디렉토리
        writer: TensorBoard writer
    """
    if not args.max_samples:
        raise ValueError("증분 학습 모드에서는 --max-samples 인수가 필요합니다")
    
    chunk_size = args.chunk_size
    total_samples = args.max_samples  # max_samples가 전체 샘플 수
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    logger.info("=" * 80)
    logger.info("증분 학습 모드")
    logger.info("=" * 80)
    logger.info(f"전체 샘플: {total_samples:,}")
    logger.info(f"청크 크기: {chunk_size:,}")
    logger.info(f"총 청크 수: {num_chunks}")
    logger.info("=" * 80)
    
    model = None
    optimizer = None
    scheduler = None
    normalization_stats = None
    start_chunk_idx = 0
    
    # Resume 체크포인트 로드
    if args.resume:
        # 최신 체크포인트 찾기 (디렉토리 또는 파일)
        checkpoint_path = find_latest_checkpoint(args.resume)
        
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            start_chunk_idx = checkpoint.get('chunk', 0)
            normalization_stats = checkpoint.get('normalization_stats')
            
            logger.info(f"Resuming from chunk {start_chunk_idx + 1}/{num_chunks}")
            
            # 모델 초기화 (첫 청크 데이터로 input_dim 확인 필요)
            # 임시로 데이터 로더 생성
            temp_loader = EmbeddingDataLoader(
                db_path=args.db,
                table_name=args.table,
                seq_len=args.seq_len,
                start_date=args.start_date,
                end_date=args.end_date,
                stock_codes=args.stock_codes,
                max_samples=1000,  # 작은 샘플로 빠르게
                offset=0
            )
            temp_loader.connect()
            feature_names = temp_loader._get_feature_columns()
            input_dim = len(feature_names)
            temp_loader.close()
            
            # 모델 재생성 및 가중치 로드
            model = TradingEmbeddingModel(
                input_dim=input_dim,
                embedding_dim=args.embedding_dim,
                seq_len=args.seq_len,
                num_heads=args.num_heads
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Optimizer 재생성 및 상태 로드
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Scheduler 재생성 및 상태 로드
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs * num_chunks,
                eta_min=args.lr * 0.01
            )
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Loss function 초기화 (빠뜨린 부분)
            criterion = InfoNCELoss(temperature=args.temperature)
            
            logger.info("Checkpoint loaded successfully!")
            logger.info(f"Continuing from chunk {start_chunk_idx + 1}")
            logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            logger.warning("No checkpoint found. Starting from scratch...")
    
    for chunk_idx in range(start_chunk_idx, num_chunks):
        start_sample = chunk_idx * chunk_size
        end_sample = min((chunk_idx + 1) * chunk_size, total_samples)
        current_chunk_size = end_sample - start_sample
        
        logger.info("\n" + "=" * 80)
        logger.info(f"청크 {chunk_idx + 1}/{num_chunks}")
        logger.info(f"샘플 범위: {start_sample:,} ~ {end_sample:,} ({current_chunk_size:,} 샘플)")
        logger.info("=" * 80)
        
        # 데이터 로더 초기화 (OFFSET과 LIMIT 사용)
        data_loader = EmbeddingDataLoader(
            db_path=args.db,
            table_name=args.table,
            seq_len=args.seq_len,
            start_date=args.start_date,
            end_date=args.end_date,
            stock_codes=args.stock_codes,
            max_samples=current_chunk_size,
            offset=start_sample  # 새 파라미터 필요
        )
        
        # 데이터 로드
        data_loader.load_and_split_data()
        
        # 첫 청크에서만 모델 초기화 및 정규화 통계 계산
        if chunk_idx == 0:
            feature_names = data_loader._get_feature_columns()
            input_dim = len(feature_names)
            
            # 정규화 통계 계산
            normalization_stats = compute_normalization_stats(data_loader.data_splits['train']['data'])
            
            # 모델 초기화
            model = TradingEmbeddingModel(
                input_dim=input_dim,
                embedding_dim=args.embedding_dim,
                seq_len=args.seq_len,
                num_heads=args.num_heads
            ).to(device)
            
            # Optimizer 초기화
            criterion = InfoNCELoss(temperature=args.temperature)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs * num_chunks,  # 전체 청크 고려
                eta_min=args.lr * 0.01
            )
            
            logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # DataLoader 생성
        train_dataloader = data_loader.get_dataloader(
            split='train',
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            return_metadata=False,
            positive_time_threshold=args.positive_time_threshold,
            negative_time_threshold=args.negative_time_threshold
        )
        
        # 이 청크에 대해 훈련
        logger.info(f"Training on chunk {chunk_idx + 1}/{num_chunks}...")
        
        # 청크별 훈련 시작 시간 기록
        chunk_start_time = datetime.now()
        
        for epoch in range(1, args.epochs + 1):
            global_epoch = chunk_idx * args.epochs + epoch
            
            logger.info(f"\nChunk {chunk_idx + 1}/{num_chunks}, Epoch {epoch}/{args.epochs} (Global: {global_epoch})")
            
            # 훈련
            train_loss = train_epoch(
                model=model,
                dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=global_epoch,
                scaler=torch.amp.GradScaler('cuda') if device.type == 'cuda' else None,
                log_interval=args.log_interval,
                chunk_info=(chunk_idx, num_chunks)
            )
            
            logger.info(f"Chunk {chunk_idx + 1}, Epoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # TensorBoard 로깅
            writer.add_scalar('train/loss', train_loss, global_epoch)
            writer.add_scalar('train/chunk', chunk_idx + 1, global_epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_epoch)
            
            # Learning rate 업데이트
            scheduler.step()
        
        # 청크 완료 시간 계산
        chunk_end_time = datetime.now()
        chunk_duration = chunk_end_time - chunk_start_time
        logger.info(f"Chunk {chunk_idx + 1} completed in {chunk_duration}")
        
        # 청크 완료 후 체크포인트 저장 (checkpoint_interval 간격으로)
        if (chunk_idx + 1) % args.checkpoint_interval == 0 or (chunk_idx + 1) == num_chunks:
            checkpoint_path = output_dir / f'checkpoint_chunk{chunk_idx + 1}.pt'
            torch.save({
                'chunk': chunk_idx + 1,
                'global_epoch': (chunk_idx + 1) * args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'normalization_stats': normalization_stats,
                'feature_names': feature_names,
                'args': vars(args),
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        else:
            logger.info(f"Skipping checkpoint save for chunk {chunk_idx + 1} (interval: {args.checkpoint_interval})")
        
        # 메모리 정리
        data_loader.close()
        del data_loader
        del train_dataloader
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # 최종 모델 저장
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalization_stats': normalization_stats,
        'feature_names': feature_names,
        'config': {
            'input_dim': input_dim,
            'embedding_dim': args.embedding_dim,
            'seq_len': args.seq_len,
            'num_heads': args.num_heads
        },
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("증분 학습 완료 요약")
    logger.info("=" * 80)
    logger.info(f"총 청크 수: {num_chunks}")
    logger.info(f"총 글로벌 에포크: {num_chunks * args.epochs}")
    logger.info(f"최종 모델: {final_model_path}")
    logger.info("=" * 80)
    
    return model, normalization_stats


def main():
    """메인 훈련 함수"""
    # CLI 인수 파싱 및 검증
    args = parse_args()
    args = validate_args(args)
    
    # 증분 학습 모드 체크
    if args.incremental:
        logger.info("증분 학습 모드로 실행합니다...")
        device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard_dir = output_dir / 'tensorboard_logs'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        
        try:
            train_incremental(args, device, output_dir, writer)
        finally:
            writer.close()
        
        logger.info("증분 학습 완료!")
        return
    
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
        seq_len=args.seq_len,
        start_date=args.start_date,
        end_date=args.end_date,
        stock_codes=args.stock_codes,
        max_samples=args.max_samples
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
    
    # 사전 계산된 쌍 경로 설정
    precomputed_train = None
    precomputed_val = None
    if args.precomputed_pairs_dir:
        precomputed_train = os.path.join(args.precomputed_pairs_dir, 'train_pairs.pkl')
        precomputed_val = os.path.join(args.precomputed_pairs_dir, 'val_pairs.pkl')
        logger.info(f"Using precomputed pairs from: {args.precomputed_pairs_dir}")
    
    # DataLoader 생성
    train_dataloader = data_loader.get_dataloader(
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        positive_time_threshold=args.positive_time_threshold,
        negative_time_threshold=args.negative_time_threshold,
        precomputed_pairs_path=precomputed_train
    )
    
    val_dataloader = data_loader.get_dataloader(
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        positive_time_threshold=args.positive_time_threshold,
        negative_time_threshold=args.negative_time_threshold,
        precomputed_pairs_path=precomputed_val
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning Rate Scheduler 초기화
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    logger.info(f"Learning rate scheduler: CosineAnnealingLR (eta_min={args.lr * 0.01})")
    
    # Mixed Precision Training 초기화
    scaler = None
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Mixed Precision Training enabled (AMP)")
    
    # Early Stopping 초기화
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    logger.info("Early Stopping enabled (patience=5, min_delta=0.001)")
    
    # 체크포인트 재개 (선택사항)
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        metadata = load_checkpoint(args.resume, model, optimizer)
        start_epoch = metadata['epoch'] + 1
    
    # 훈련 루프
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    
    try:
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
                epoch=epoch,
                scaler=scaler,  # Mixed Precision
                log_interval=args.log_interval
            )
            
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Learning rate 업데이트
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # TensorBoard 로깅 (훈련 손실은 항상 로깅)
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('learning_rate', current_lr, epoch)
            writer.flush()  # 즉시 디스크에 기록
            
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
                writer.flush()  # 즉시 디스크에 기록
                
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
                
                # Early Stopping 체크
                if early_stopping(val_loss):
                    logger.info(f"\n{'='*80}")
                    logger.info("Early Stopping triggered!")
                    logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
                    logger.info(f"No improvement for {early_stopping.patience} epochs")
                    logger.info(f"{'='*80}\n")
                    break
            
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
    
    finally:
        # 정리 (에러 발생 시에도 실행)
        logger.info("Closing TensorBoard writer and data loader...")
        writer.close()
        data_loader.close()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"오류 발생: {e}")
        print(f"오류 발생: {e}", file=sys.stderr)
        sys.exit(1)
