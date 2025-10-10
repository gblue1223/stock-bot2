"""
긍정/부정 쌍 사전 계산 스크립트

ContrastiveDataset의 _precompute_pairs 메서드를 독립적으로 실행하여
결과를 파일로 저장합니다. 이를 통해 훈련 시작 시간을 크게 단축할 수 있습니다.

Usage:
    python scripts/precompute_pairs.py \
        --db_path data/trading_data.duckdb \
        --output_dir data/precomputed_pairs \
        --seq_len 60 \
        --positive_threshold 10 \
        --negative_threshold 60
"""

import argparse
import logging
import pickle
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# 프로젝트 루트를 sys.path에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trader.embedding.data import EmbeddingDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _compute_pairs_for_chunk(
    indices_chunk: np.ndarray,
    metadata: np.ndarray,
    stock_time_sorted: Dict[str, List[int]],
    stock_indices: Dict[str, List[int]],
    valid_indices: np.ndarray,
    positive_time_threshold: int,
    negative_time_threshold: int
) -> tuple:
    """
    인덱스 청크에 대한 긍정/부정 쌍 계산 (멀티스레드용)
    
    Args:
        indices_chunk: 처리할 인덱스 청크
        metadata: 메타데이터
        stock_time_sorted: 종목별 시간 정렬된 인덱스
        stock_indices: 종목별 인덱스 매핑
        valid_indices: 전체 유효한 인덱스
        positive_time_threshold: 긍정 쌍 시간 임계값
        negative_time_threshold: 부정 쌍 시간 임계값
    
    Returns:
        (positive_pairs_dict, negative_pairs_dict) 튜플
    """
    positive_pairs = {}
    negative_pairs = {}
    
    for idx in indices_chunk:
        stock_code = metadata[idx, 0]
        anchor_time = float(metadata[idx, 2])
        
        # 긍정 쌍 후보 찾기 (동일 종목, 가까운 시간)
        positive_candidates = []
        for candidate_idx in stock_time_sorted[stock_code]:
            if candidate_idx == idx:
                continue
            time_diff = abs(float(metadata[candidate_idx, 2]) - anchor_time)
            if time_diff < positive_time_threshold:
                positive_candidates.append(candidate_idx)
        
        # 긍정 쌍이 없으면 자기 자신 사용
        if not positive_candidates:
            positive_candidates = [idx]
        
        positive_pairs[idx] = positive_candidates
        
        # 부정 쌍 후보 찾기
        negative_candidates = []
        
        # 전략 1: 다른 종목 (70%)
        other_stocks = [s for s in stock_indices.keys() if s != stock_code]
        if other_stocks:
            for other_stock in other_stocks[:min(5, len(other_stocks))]:
                negative_candidates.extend(stock_indices[other_stock][:100])
        
        # 전략 2: 동일 종목, 먼 시간 (30%)
        for candidate_idx in stock_time_sorted[stock_code]:
            time_diff = abs(float(metadata[candidate_idx, 2]) - anchor_time)
            if time_diff > negative_time_threshold:
                negative_candidates.append(candidate_idx)
                if len(negative_candidates) >= 100:
                    break
        
        # 부정 쌍이 없으면 랜덤 샘플 사용
        if not negative_candidates:
            negative_candidates = list(valid_indices[:100])
        
        negative_pairs[idx] = negative_candidates
    
    return positive_pairs, negative_pairs


def precompute_pairs_for_split(
    data: np.ndarray,
    metadata: np.ndarray,
    valid_indices: np.ndarray,
    stock_indices: Dict[str, List[int]],
    seq_len: int,
    positive_time_threshold: int,
    negative_time_threshold: int,
    num_threads: int = 1,
    checkpoint_path: Path = None,
    checkpoint_interval: int = 10000
) -> tuple:
    """
    특정 데이터 분할에 대한 긍정/부정 쌍 사전 계산
    
    Args:
        data: 특징 데이터
        metadata: 메타데이터
        valid_indices: 유효한 인덱스 배열
        stock_indices: 종목별 인덱스 매핑
        seq_len: 시퀀스 길이
        positive_time_threshold: 긍정 쌍 시간 임계값
        negative_time_threshold: 부정 쌍 시간 임계값
        num_threads: 사용할 스레드 수 (기본값: 1 = 단일 스레드)
        checkpoint_path: 체크포인트 파일 경로 (선택적)
        checkpoint_interval: 체크포인트 저장 간격 (기본값: 10000)
    
    Returns:
        (positive_pairs_cache, negative_pairs_cache) 튜플
    """
    # 체크포인트 로드 (재개 모드)
    positive_pairs_cache = {}
    negative_pairs_cache = {}
    processed_indices = set()
    
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            positive_pairs_cache = checkpoint_data.get('positive_pairs_cache', {})
            negative_pairs_cache = checkpoint_data.get('negative_pairs_cache', {})
            processed_indices = set(checkpoint_data.get('processed_indices', []))
            logger.info(f"Resumed from checkpoint: {len(processed_indices)} indices already processed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.warning("Starting from scratch...")
            positive_pairs_cache = {}
            negative_pairs_cache = {}
            processed_indices = set()
    # 종목별로 시간 정렬된 인덱스 생성
    logger.info("Sorting indices by time for each stock...")
    stock_time_sorted = {}
    for stock_code, indices in tqdm(stock_indices.items(), desc="Sorting stocks"):
        sorted_indices = sorted(indices, key=lambda i: float(metadata[i, 2]))
        stock_time_sorted[stock_code] = sorted_indices
    
    # 각 인덱스에 대해 긍정/부정 쌍 후보 계산
    remaining_indices = [idx for idx in valid_indices if idx not in processed_indices]
    logger.info(f"Computing pairs for {len(remaining_indices)} remaining indices (total: {len(valid_indices)})...")
    
    if len(remaining_indices) == 0:
        logger.info("All indices already processed!")
        return positive_pairs_cache, negative_pairs_cache
    
    if num_threads <= 1:
        # 단일 스레드 모드 (체크포인트 지원)
        processed_count = len(processed_indices)
        
        for i, idx in enumerate(tqdm(remaining_indices, desc="Computing pairs")):
            stock_code = metadata[idx, 0]
            anchor_time = float(metadata[idx, 2])
            
            # 긍정 쌍 후보 찾기 (동일 종목, 가까운 시간)
            positive_candidates = []
            for candidate_idx in stock_time_sorted[stock_code]:
                if candidate_idx == idx:
                    continue
                time_diff = abs(float(metadata[candidate_idx, 2]) - anchor_time)
                if time_diff < positive_time_threshold:
                    positive_candidates.append(candidate_idx)
            
            # 긍정 쌍이 없으면 자기 자신 사용
            if not positive_candidates:
                positive_candidates = [idx]
            
            positive_pairs_cache[idx] = positive_candidates
            
            # 부정 쌍 후보 찾기
            negative_candidates = []
            
            # 전략 1: 다른 종목 (70%)
            other_stocks = [s for s in stock_indices.keys() if s != stock_code]
            if other_stocks:
                for other_stock in other_stocks[:min(5, len(other_stocks))]:
                    negative_candidates.extend(stock_indices[other_stock][:100])
            
            # 전략 2: 동일 종목, 먼 시간 (30%)
            for candidate_idx in stock_time_sorted[stock_code]:
                time_diff = abs(float(metadata[candidate_idx, 2]) - anchor_time)
                if time_diff > negative_time_threshold:
                    negative_candidates.append(candidate_idx)
                    if len(negative_candidates) >= 100:
                        break
            
            # 부정 쌍이 없으면 랜덤 샘플 사용
            if not negative_candidates:
                negative_candidates = list(valid_indices[:100])
            
            negative_pairs_cache[idx] = negative_candidates
            processed_indices.add(idx)
            processed_count += 1
            
            # 체크포인트 저장
            if checkpoint_path and processed_count % checkpoint_interval == 0:
                logger.info(f"\nSaving checkpoint at {processed_count} indices...")
                checkpoint_data = {
                    'positive_pairs_cache': positive_pairs_cache,
                    'negative_pairs_cache': negative_pairs_cache,
                    'processed_indices': list(processed_indices),
                    'total_indices': len(valid_indices)
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # 멀티스레드 모드 (체크포인트 미지원)
        logger.info(f"Using {num_threads} threads for parallel processing...")
        
        # 남은 인덱스를 청크로 분할
        chunk_size = max(1, len(remaining_indices) // (num_threads * 4))  # 스레드당 4개 청크
        chunks = [remaining_indices[i:i + chunk_size] for i in range(0, len(remaining_indices), chunk_size)]
        logger.info(f"Split into {len(chunks)} chunks (chunk size: ~{chunk_size})")
        
        # ThreadPoolExecutor로 병렬 처리
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 각 청크에 대한 작업 제출
            futures = []
            for chunk in chunks:
                future = executor.submit(
                    _compute_pairs_for_chunk,
                    chunk,
                    metadata,
                    stock_time_sorted,
                    stock_indices,
                    valid_indices,
                    positive_time_threshold,
                    negative_time_threshold
                )
                futures.append(future)
            
            # 완료된 작업 수집 (진행 상황 표시)
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                pos_pairs, neg_pairs = future.result()
                positive_pairs_cache.update(pos_pairs)
                negative_pairs_cache.update(neg_pairs)
        
        logger.warning("Note: Checkpointing is not supported in multi-threaded mode")
    
    # 최종 체크포인트 삭제 (완료 시)
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Removing checkpoint file: {checkpoint_path}")
        checkpoint_path.unlink()
    
    return positive_pairs_cache, negative_pairs_cache


def compute_valid_indices(data: np.ndarray, metadata: np.ndarray, seq_len: int) -> np.ndarray:
    """시퀀스 생성이 가능한 유효한 인덱스 계산"""
    n_samples = len(data)
    valid_indices = []
    
    logger.info(f"Computing valid indices for {n_samples} samples...")
    for i in tqdm(range(n_samples - seq_len + 1), desc="Finding valid sequences"):
        stock_codes = metadata[i:i+seq_len, 0]
        dates = metadata[i:i+seq_len, 1]
        
        if len(np.unique(stock_codes)) == 1 and len(np.unique(dates)) == 1:
            valid_indices.append(i)
    
    return np.array(valid_indices)


def build_stock_index(valid_indices: np.ndarray, metadata: np.ndarray) -> Dict[str, List[int]]:
    """종목별 인덱스 매핑 생성"""
    stock_indices = {}
    logger.info("Building stock index...")
    for idx in tqdm(valid_indices, desc="Indexing stocks"):
        stock_code = metadata[idx, 0]
        if stock_code not in stock_indices:
            stock_indices[stock_code] = []
        stock_indices[stock_code].append(idx)
    
    return stock_indices


def main():
    parser = argparse.ArgumentParser(description='Precompute positive/negative pairs for contrastive learning')
    parser.add_argument('--db_path', type=str, required=True, help='Path to DuckDB database')
    parser.add_argument('--table_name', type=str, default='datasets', help='Table name')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for precomputed pairs')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--positive_threshold', type=float, default=0.05, help='Positive pair time threshold (default: 0.05, same as train_embedding.py)')
    parser.add_argument('--negative_threshold', type=float, default=0.5, help='Negative pair time threshold (default: 0.5, same as train_embedding.py)')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--start_date', type=str, default=None, help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--stock_codes', type=str, nargs='+', default=None, help='Stock codes to filter')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='Process data in chunks to avoid OOM (default: auto-detect based on max_samples)'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=None,
        help='Number of threads for parallel processing (default: CPU count)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint (skip already completed splits)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10000,
        help='Save intermediate checkpoint every N indices (default: 10000)'
    )
    parser.add_argument(
        '--date-chunk-days',
        type=int,
        default=None,
        help='Process data in date chunks (e.g., 30 days per chunk) instead of row-based chunks'
    )
    parser.add_argument(
        '--skip-clipping',
        action='store_true',
        help='Skip data clipping to speed up processing (use if data is already normalized)'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Precomputing Positive/Negative Pairs")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Sequence length: {args.seq_len}")
    logger.info(f"Positive threshold: {args.positive_threshold}")
    logger.info(f"Negative threshold: {args.negative_threshold}")
    
    # 스레드 수 설정 (기본값: CPU 코어 수)
    if args.num_threads is None:
        num_threads = multiprocessing.cpu_count()
        logger.info(f"Number of threads: {num_threads} (auto-detected)")
    else:
        num_threads = args.num_threads
        logger.info(f"Number of threads: {num_threads}")
    
    # 날짜 기반 청크 처리 우선 사용 (성능 최적화)
    use_date_chunks = args.date_chunk_days is not None
    
    if use_date_chunks:
        logger.info(f"Using date-based chunking: {args.date_chunk_days} days per chunk")
        logger.info("This avoids OFFSET and significantly improves query performance")
    else:
        # 청크 크기 자동 설정 (OOM 방지)
        chunk_size = args.chunk_size
        if chunk_size is None and args.max_samples:
            # max_samples가 크면 청크로 분할
            if args.max_samples > 5_000_000:
                chunk_size = 2_000_000
                logger.info(f"Auto-detected chunk size: {chunk_size:,} (max_samples: {args.max_samples:,})")
            elif args.max_samples > 2_000_000:
                chunk_size = 1_000_000
                logger.info(f"Auto-detected chunk size: {chunk_size:,} (max_samples: {args.max_samples:,})")
        
        if chunk_size:
            logger.info(f"Using row-based chunking with chunk size: {chunk_size:,}")
            logger.info("This will process data in multiple passes to avoid OOM errors")
    
    # 데이터 로더 생성 및 데이터 로드
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading data from database")
    logger.info("=" * 80)
    
    # 날짜 기반 청크 처리
    if use_date_chunks:
        from datetime import datetime, timedelta
        
        # 날짜 범위 계산
        start = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        
        if not start or not end:
            logger.error("--start_date and --end_date are required for date-based chunking")
            return
        
        # 날짜 청크 생성
        date_chunks = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=args.date_chunk_days), end)
            date_chunks.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
            current = chunk_end
        
        logger.info(f"Processing in {len(date_chunks)} date chunks")
        
        all_data_splits = {'train': {'data': [], 'metadata': []},
                          'val': {'data': [], 'metadata': []},
                          'test': {'data': [], 'metadata': []}}
        
        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks):
            logger.info(f"\nProcessing chunk {chunk_idx + 1}/{len(date_chunks)} (dates: {chunk_start} to {chunk_end})")
            
            data_loader = EmbeddingDataLoader(
                db_path=args.db_path,
                table_name=args.table_name,
                seq_len=args.seq_len,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                start_date=chunk_start,
                end_date=chunk_end,
                stock_codes=args.stock_codes,
                max_samples=args.max_samples,
                skip_clipping=args.skip_clipping
            )
            
            data_loader.connect()
            chunk_splits = data_loader.load_and_split_data()
            data_loader.close()
            
            # 청크 데이터 병합
            for split_name in ['train', 'val', 'test']:
                all_data_splits[split_name]['data'].append(chunk_splits[split_name]['data'])
                all_data_splits[split_name]['metadata'].append(chunk_splits[split_name]['metadata'])
            
            logger.info(f"Chunk {chunk_idx + 1} loaded successfully")
        
        # 모든 청크 병합
        logger.info("\nMerging all chunks...")
        data_splits = {}
        for split_name in ['train', 'val', 'test']:
            data_splits[split_name] = {
                'data': np.vstack(all_data_splits[split_name]['data']),
                'metadata': np.vstack(all_data_splits[split_name]['metadata'])
            }
            logger.info(f"{split_name}: {len(data_splits[split_name]['data']):,} samples")
    
    # 행 기반 청크 처리
    elif chunk_size and args.max_samples and args.max_samples > chunk_size:
        logger.info(f"Processing in chunks: {args.max_samples // chunk_size + 1} chunks")
        
        # 청크별로 처리
        num_chunks = (args.max_samples + chunk_size - 1) // chunk_size
        all_data_splits = {'train': {'data': [], 'metadata': []},
                          'val': {'data': [], 'metadata': []},
                          'test': {'data': [], 'metadata': []}}
        
        for chunk_idx in range(num_chunks):
            offset = chunk_idx * chunk_size
            current_chunk_size = min(chunk_size, args.max_samples - offset)
            
            logger.info(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (offset: {offset:,}, size: {current_chunk_size:,})")
            
            data_loader = EmbeddingDataLoader(
                db_path=args.db_path,
                table_name=args.table_name,
                seq_len=args.seq_len,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                start_date=args.start_date,
                end_date=args.end_date,
                stock_codes=args.stock_codes,
                max_samples=current_chunk_size,
                offset=offset,
                skip_clipping=args.skip_clipping
            )
            
            data_loader.connect()
            chunk_splits = data_loader.load_and_split_data()
            data_loader.close()
            
            # 청크 데이터 병합
            for split_name in ['train', 'val', 'test']:
                all_data_splits[split_name]['data'].append(chunk_splits[split_name]['data'])
                all_data_splits[split_name]['metadata'].append(chunk_splits[split_name]['metadata'])
            
            logger.info(f"Chunk {chunk_idx + 1} loaded successfully")
        
        # 모든 청크 병합
        logger.info("\nMerging all chunks...")
        data_splits = {}
        for split_name in ['train', 'val', 'test']:
            data_splits[split_name] = {
                'data': np.vstack(all_data_splits[split_name]['data']),
                'metadata': np.vstack(all_data_splits[split_name]['metadata'])
            }
            logger.info(f"{split_name}: {len(data_splits[split_name]['data']):,} samples")
    else:
        # 일반 처리 (청크 불필요)
        data_loader = EmbeddingDataLoader(
            db_path=args.db_path,
            table_name=args.table_name,
            seq_len=args.seq_len,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            start_date=args.start_date,
            end_date=args.end_date,
            stock_codes=args.stock_codes,
            max_samples=args.max_samples,
            skip_clipping=args.skip_clipping
        )
        
        data_loader.connect()
        data_splits = data_loader.load_and_split_data()
        data_loader.close()
    
    # 각 분할(train, val, test)에 대해 쌍 계산
    for split_name in ['train', 'val', 'test']:
        output_file = output_dir / f"{split_name}_pairs.pkl"
        
        # 재개 모드: 이미 완료된 분할 건너뛰기
        if args.resume and output_file.exists():
            logger.info("\n" + "=" * 80)
            logger.info(f"Step 2: Skipping {split_name} split (already exists)")
            logger.info("=" * 80)
            logger.info(f"Found existing file: {output_file}")
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            continue
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Step 2: Processing {split_name} split")
        logger.info("=" * 80)
        
        split_data = data_splits[split_name]
        data = split_data['data']
        metadata = split_data['metadata']
        
        logger.info(f"Split size: {len(data)} samples")
        
        # 유효한 인덱스 계산
        valid_indices = compute_valid_indices(data, metadata, args.seq_len)
        logger.info(f"Valid sequences: {len(valid_indices)}")
        
        # 종목별 인덱스 매핑 생성
        stock_indices = build_stock_index(valid_indices, metadata)
        logger.info(f"Number of stocks: {len(stock_indices)}")
        
        # 체크포인트 경로 설정
        checkpoint_file = output_dir / f"{split_name}_checkpoint.pkl"
        
        # 긍정/부정 쌍 계산
        positive_pairs, negative_pairs = precompute_pairs_for_split(
            data=data,
            metadata=metadata,
            valid_indices=valid_indices,
            stock_indices=stock_indices,
            seq_len=args.seq_len,
            positive_time_threshold=args.positive_threshold,
            negative_time_threshold=args.negative_threshold,
            num_threads=num_threads,
            checkpoint_path=checkpoint_file if args.resume else None,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # 결과 저장
        logger.info(f"Saving precomputed pairs to {output_file}...")
        
        pairs_data = {
            'valid_indices': valid_indices,
            'stock_indices': stock_indices,
            'positive_pairs_cache': positive_pairs,
            'negative_pairs_cache': negative_pairs,
            'seq_len': args.seq_len,
            'positive_threshold': args.positive_threshold,
            'negative_threshold': args.negative_threshold,
            'metadata_shape': metadata.shape,
            'data_shape': data.shape,
            'completed': True  # 완료 플래그
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(pairs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 파일 크기 출력
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {output_file.name} ({file_size_mb:.2f} MB)")
        
        # 통계 출력
        avg_positive = np.mean([len(v) for v in positive_pairs.values()])
        avg_negative = np.mean([len(v) for v in negative_pairs.values()])
        logger.info(f"Average positive pairs per sample: {avg_positive:.2f}")
        logger.info(f"Average negative pairs per sample: {avg_negative:.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Precomputation Complete!")
    logger.info("=" * 80)
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("\nTo use precomputed pairs in training, pass the following argument:")
    logger.info(f"  --precomputed_pairs_dir {args.output_dir}")


if __name__ == '__main__':
    main()
