"""
사전 계산된 긍정/부정 쌍 사용 예제

이 예제는 다음을 보여줍니다:
1. 긍정/부정 쌍 사전 계산
2. 사전 계산된 쌍을 사용한 훈련
3. 성능 비교
"""

import time
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trader.embedding.data import EmbeddingDataLoader


def example_without_precomputed_pairs():
    """사전 계산 없이 데이터셋 생성 (느림)"""
    print("=" * 80)
    print("Example 1: Without Precomputed Pairs (Slow)")
    print("=" * 80)
    
    # 데이터 로더 생성
    data_loader = EmbeddingDataLoader(
        db_path='data/trading_data.duckdb',
        seq_len=60,
        max_samples=100000  # 예제를 위해 샘플 수 제한
    )
    
    # 데이터 로드
    print("\nLoading data from database...")
    data_loader.connect()
    data_loader.load_and_split_data()
    
    # Dataset 생성 (쌍 계산 포함)
    print("\nCreating dataset (computing pairs on-the-fly)...")
    start_time = time.time()
    
    train_dataset = data_loader.get_dataset(
        split='train',
        positive_time_threshold=10,
        negative_time_threshold=60
    )
    
    elapsed_time = time.time() - start_time
    print(f"Dataset creation time: {elapsed_time:.2f} seconds")
    print(f"Number of valid sequences: {len(train_dataset)}")
    
    data_loader.close()
    
    return elapsed_time


def example_with_precomputed_pairs():
    """사전 계산된 쌍 사용 (빠름)"""
    print("\n" + "=" * 80)
    print("Example 2: With Precomputed Pairs (Fast)")
    print("=" * 80)
    
    # 데이터 로더 생성
    data_loader = EmbeddingDataLoader(
        db_path='data/trading_data.duckdb',
        seq_len=60,
        max_samples=100000  # 예제를 위해 샘플 수 제한
    )
    
    # 데이터 로드
    print("\nLoading data from database...")
    data_loader.connect()
    data_loader.load_and_split_data()
    
    # Dataset 생성 (사전 계산된 쌍 로드)
    print("\nCreating dataset (loading precomputed pairs)...")
    start_time = time.time()
    
    train_dataset = data_loader.get_dataset(
        split='train',
        positive_time_threshold=10,
        negative_time_threshold=60,
        precomputed_pairs_path='data/precomputed_pairs/train_pairs.pkl'
    )
    
    elapsed_time = time.time() - start_time
    print(f"Dataset creation time: {elapsed_time:.2f} seconds")
    print(f"Number of valid sequences: {len(train_dataset)}")
    
    data_loader.close()
    
    return elapsed_time


def example_dataloader_with_precomputed_pairs():
    """DataLoader 생성 시 사전 계산된 쌍 사용"""
    print("\n" + "=" * 80)
    print("Example 3: DataLoader with Precomputed Pairs")
    print("=" * 80)
    
    # 데이터 로더 생성
    data_loader = EmbeddingDataLoader(
        db_path='data/trading_data.duckdb',
        seq_len=60
    )
    
    # 데이터 로드
    print("\nLoading data from database...")
    data_loader.connect()
    data_loader.load_and_split_data()
    
    # 사전 계산된 쌍을 사용하여 DataLoader 생성
    print("\nCreating DataLoaders with precomputed pairs...")
    
    train_loader = data_loader.get_dataloader(
        split='train',
        batch_size=128,
        shuffle=True,
        num_workers=4,
        precomputed_pairs_path='data/precomputed_pairs/train_pairs.pkl'
    )
    
    val_loader = data_loader.get_dataloader(
        split='val',
        batch_size=128,
        shuffle=False,
        num_workers=4,
        precomputed_pairs_path='data/precomputed_pairs/val_pairs.pkl'
    )
    
    test_loader = data_loader.get_dataloader(
        split='test',
        batch_size=128,
        shuffle=False,
        num_workers=4,
        precomputed_pairs_path='data/precomputed_pairs/test_pairs.pkl'
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 첫 번째 배치 가져오기
    print("\nFetching first batch...")
    anchor, positive, negative = next(iter(train_loader))
    
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")
    
    data_loader.close()


def example_verify_pairs():
    """사전 계산된 쌍 검증"""
    print("\n" + "=" * 80)
    print("Example 4: Verify Precomputed Pairs")
    print("=" * 80)
    
    import pickle
    
    pairs_path = 'data/precomputed_pairs/train_pairs.pkl'
    
    print(f"\nLoading precomputed pairs from {pairs_path}...")
    with open(pairs_path, 'rb') as f:
        pairs_data = pickle.load(f)
    
    print("\nPrecomputed pairs information:")
    print(f"  Sequence length: {pairs_data['seq_len']}")
    print(f"  Positive threshold: {pairs_data['positive_threshold']}")
    print(f"  Negative threshold: {pairs_data['negative_threshold']}")
    print(f"  Data shape: {pairs_data['data_shape']}")
    print(f"  Metadata shape: {pairs_data['metadata_shape']}")
    print(f"  Valid indices: {len(pairs_data['valid_indices'])}")
    print(f"  Number of stocks: {len(pairs_data['stock_indices'])}")
    print(f"  Positive pairs cache size: {len(pairs_data['positive_pairs_cache'])}")
    print(f"  Negative pairs cache size: {len(pairs_data['negative_pairs_cache'])}")
    
    # 통계 계산
    import numpy as np
    
    positive_counts = [len(v) for v in pairs_data['positive_pairs_cache'].values()]
    negative_counts = [len(v) for v in pairs_data['negative_pairs_cache'].values()]
    
    print("\nPositive pairs statistics:")
    print(f"  Mean: {np.mean(positive_counts):.2f}")
    print(f"  Median: {np.median(positive_counts):.2f}")
    print(f"  Min: {np.min(positive_counts)}")
    print(f"  Max: {np.max(positive_counts)}")
    
    print("\nNegative pairs statistics:")
    print(f"  Mean: {np.mean(negative_counts):.2f}")
    print(f"  Median: {np.median(negative_counts):.2f}")
    print(f"  Min: {np.min(negative_counts)}")
    print(f"  Max: {np.max(negative_counts)}")


def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("Precomputed Pairs Usage Examples")
    print("=" * 80)
    
    # 예제 선택
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--example',
        type=str,
        choices=['all', 'without', 'with', 'dataloader', 'verify'],
        default='all',
        help='Which example to run'
    )
    args = parser.parse_args()
    
    try:
        if args.example in ['all', 'without']:
            time_without = example_without_precomputed_pairs()
        
        if args.example in ['all', 'with']:
            time_with = example_with_precomputed_pairs()
        
        if args.example in ['all', 'dataloader']:
            example_dataloader_with_precomputed_pairs()
        
        if args.example in ['all', 'verify']:
            example_verify_pairs()
        
        # 성능 비교
        if args.example == 'all':
            print("\n" + "=" * 80)
            print("Performance Comparison")
            print("=" * 80)
            print(f"Without precomputed pairs: {time_without:.2f} seconds")
            print(f"With precomputed pairs: {time_with:.2f} seconds")
            print(f"Speedup: {time_without / time_with:.2f}x")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the following command first to precompute pairs:")
        print("  python scripts/precompute_pairs.py \\")
        print("    --db_path data/trading_data.duckdb \\")
        print("    --output_dir data/precomputed_pairs")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
