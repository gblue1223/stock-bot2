"""
임베딩 모델 평가 스크립트

저장된 임베딩 모델을 로드하고 테스트 데이터에서 평가 메트릭을 계산합니다.

사용 예시:
    python -m ai_trader.embedding.evaluate_embedding \
        --checkpoint models/embedding/checkpoint_epoch50.pt \
        --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \
        --table datasets \
        --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

from ai_trader.embedding.models import TradingEmbeddingModel
from ai_trader.embedding.data import EmbeddingDataLoader
from ai_trader.embedding.evaluation import evaluate_embedding_quality

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """CLI 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='임베딩 모델 평가 - 저장된 모델의 품질 메트릭 계산',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 모델 체크포인트
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='평가할 모델 체크포인트 경로 (예: models/embedding/checkpoint_epoch50.pt)'
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
    
    # 평가 설정
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='평가할 데이터 분할'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='배치 크기'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=1,
        help='Temporal coherence 계산 시 윈도우 크기'
    )
    
    # 디바이스 설정
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='평가에 사용할 디바이스'
    )
    
    # 출력 설정
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='평가 결과를 저장할 JSON 파일 경로 (선택사항)'
    )
    
    return parser.parse_args()


def load_model_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """
    모델 체크포인트 로드
    
    Args:
        checkpoint_path: 체크포인트 경로
        device: 디바이스
        
    Returns:
        (model, config, metadata)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 설정 추출
    config = checkpoint['config']
    metadata = checkpoint.get('training_metadata', {})
    
    # 모델 초기화 및 가중치 로드
    model = TradingEmbeddingModel(
        input_dim=config['input_dim'],
        embedding_dim=config['embedding_dim'],
        seq_len=config['seq_len'],
        num_heads=config.get('num_heads', 4)
    ).to(device)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Training metadata: {metadata}")
    
    return model, config, metadata


def generate_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    데이터로더에서 임베딩 생성
    
    Args:
        model: 임베딩 모델
        dataloader: 데이터 로더
        device: 디바이스
        
    Returns:
        (embeddings, stock_codes, timestamps)
    """
    logger.info("Generating embeddings...")
    
    all_embeddings = []
    all_stock_codes = []
    all_timestamps = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # 배치 데이터 언팩
            if len(batch_data) == 5:
                # 메타데이터 포함: (anchor, positive, negative, stock_codes, timestamps)
                anchor, _, _, stock_codes, timestamps = batch_data
            elif len(batch_data) == 3:
                # 메타데이터 없음: (anchor, positive, negative)
                anchor, _, _ = batch_data
                stock_codes = []
                timestamps = []
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
            
            # 디바이스로 이동
            anchor = anchor.to(device)
            
            # 임베딩 생성
            embeddings = model(anchor)
            
            # CPU로 이동 및 저장
            all_embeddings.append(embeddings.cpu().numpy())
            
            if stock_codes:
                all_stock_codes.extend(stock_codes)
            if timestamps:
                all_timestamps.extend(timestamps)
            
            # 진행 상황 출력
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # 배열로 변환
    all_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Generated {len(all_embeddings)} embeddings")
    
    return all_embeddings, all_stock_codes, all_timestamps


def main():
    """메인 평가 함수"""
    # CLI 인수 파싱
    args = parse_args()
    
    # 설정 출력
    logger.info("=" * 80)
    logger.info("임베딩 모델 평가 시작")
    logger.info("=" * 80)
    logger.info(f"체크포인트: {args.checkpoint}")
    logger.info(f"데이터베이스: {args.db}")
    logger.info(f"테이블: {args.table}")
    logger.info(f"평가 분할: {args.split}")
    logger.info(f"배치 크기: {args.batch_size}")
    logger.info(f"윈도우 크기: {args.window_size}")
    logger.info(f"디바이스: {args.device}")
    logger.info("=" * 80)
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 모델 로드
    model, config, metadata = load_model_checkpoint(args.checkpoint, device)
    
    # 데이터 로더 초기화
    logger.info("Initializing data loader...")
    data_loader = EmbeddingDataLoader(
        db_path=args.db,
        table_name=args.table,
        seq_len=config['seq_len']
    )
    
    # 데이터 로드 및 분할
    logger.info("Loading and splitting data...")
    data_loader.load_and_split_data()
    
    # DataLoader 생성 (메타데이터 포함)
    eval_dataloader = data_loader.get_dataloader(
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        return_metadata=True
    )
    
    logger.info(f"Evaluation batches: {len(eval_dataloader)}")
    
    # 임베딩 생성
    embeddings, stock_codes, timestamps = generate_embeddings(
        model=model,
        dataloader=eval_dataloader,
        device=device
    )
    
    # 평가 메트릭 계산
    logger.info("Computing evaluation metrics...")
    metrics = evaluate_embedding_quality(
        embeddings=embeddings,
        stock_codes=stock_codes,
        timestamps=timestamps,
        window_size=args.window_size
    )
    
    # 결과 출력
    logger.info("\n" + "=" * 80)
    logger.info("평가 결과")
    logger.info("=" * 80)
    logger.info(f"샘플 수: {metrics['num_samples']}")
    logger.info(f"종목 수: {metrics['num_stocks']}")
    logger.info(f"임베딩 차원: {metrics['embedding_dim']}")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"Temporal Coherence: {metrics['temporal_coherence']:.4f}")
    logger.info("=" * 80)
    
    # 결과 저장 (선택사항)
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            'checkpoint': args.checkpoint,
            'split': args.split,
            'metrics': metrics,
            'training_metadata': metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
    
    # 정리
    data_loader.close()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"오류 발생: {e}")
        sys.exit(1)
