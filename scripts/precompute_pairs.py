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
    num_threads: int = 1
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
    
    Returns:
        (positive_pairs_cache, negative_pairs_cache) 튜플
    """
    # 종목별로 시간 정렬된 인덱스 생성
    logger.info("Sorting indices by time for each stock...")
    stock_time_sorted = {}
    for stock_code, indices in tqdm(stock_indices.items(), desc="Sorting stocks"):
        sorted_indices = sorted(indices, key=lambda i: float(metadata[i, 2]))
        stock_time_sorted[stock_code] = sorted_indices
    
    # 각 인덱스에 대해 긍정/부정 쌍 후보 계산
    logger.info(f"Computing pairs for {len(valid_indices)} valid indices...")
    
    if num_threads <= 1:
        # 단일 스레드 모드 (기존 방식)
        positive_pairs_cache = {}
        negative_pairs_cache = {}
        
        for idx in tqdm(valid_indices, desc="Computing pairs"):
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
    else:
        # 멀티스레드 모드
        logger.info(f"Using {num_threads} threads for parallel processing...")
        
        # 인덱스를 청크로 분할
        chunk_size = max(1, len(valid_indices) // (num_threads * 4))  # 스레드당 4개 청크
        chunks = [valid_indices[i:i + chunk_size] for i in range(0, len(valid_indices), chunk_size)]
        logger.info(f"Split into {len(chunks)} chunks (chunk size: ~{chunk_size})")
        
        positive_pairs_cache = {}
        negative_pairs_cache = {}
        
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
    parser.add_argument('--positive_threshold', type=int, default=10, help='Positive pair time threshold')
    parser.add_argument('--negative_threshold', type=int, default=60, help='Negative pair time threshold')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--start_date', type=str, default=None, help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--stock_codes', type=str, nargs='+', default=None, help='Stock codes to filter')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    parser.add_argument(
        '--num_threads',
        type=int,
        default=None,
        help='Number of threads for parallel processing (default: CPU count)'
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
    
    # 스레드 수 설정
    num_threads = args.num_threads if args.num_threads else multiprocessing.cpu_count()
    logger.info(f"Number of threads: {num_threads}")
    
    # 데이터 로더 생성 및 데이터 로드
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading data from database")
    logger.info("=" * 80)
    
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
        max_samples=args.max_samples
    )
    
    data_loader.connect()
    data_splits = data_loader.load_and_split_data()
    
    # 각 분할(train, val, test)에 대해 쌍 계산
    for split_name in ['train', 'val', 'test']:
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
        
        # 긍정/부정 쌍 계산
        positive_pairs, negative_pairs = precompute_pairs_for_split(
            data=data,
            metadata=metadata,
            valid_indices=valid_indices,
            stock_indices=stock_indices,
            seq_len=args.seq_len,
            positive_time_threshold=args.positive_threshold,
            negative_time_threshold=args.negative_threshold,
            num_threads=num_threads
        )
        
        # 결과 저장
        output_file = output_dir / f"{split_name}_pairs.pkl"
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
            'data_shape': data.shape
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
    
    data_loader.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("Precomputation Complete!")
    logger.info("=" * 80)
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("\nTo use precomputed pairs in training, pass the following argument:")
    logger.info(f"  --precomputed_pairs_dir {args.output_dir}")


if __name__ == '__main__':
    main()
