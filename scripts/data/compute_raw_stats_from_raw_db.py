"""
원본 데이터베이스에서 정규화 전 통계 계산

datasets_raw_all.duckdb (정규화되지 않은 원본 데이터)에서
정규화 전략에 맞는 통계를 계산합니다.

Usage:
    python scripts/data/compute_raw_stats_from_raw_db.py \
        --db "C:\\Users\\user\\Workspace\\datasets@20251016\\datasets_raw_all.duckdb" \
        --output models/grpo_scalping@2025120/raw_normalization_stats.json \
        --sample-size 100000
"""

import argparse
import json
import numpy as np
import duckdb
from pathlib import Path
import sys
from tqdm import tqdm

# lib 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.normalization import FEATURE_NAMES, get_normalization_strategy, signed_log1p


def compute_raw_statistics(db_path: str, sample_size: int = 100000) -> dict:
    """
    원본 데이터베이스에서 정규화 전 통계 계산
    
    정규화 전략별로:
    1. log_std: 로그 변환 후의 mean/std 계산
    2. std_only: 원본 값의 mean/std 계산
    3. derived: 원본 값의 mean/std 계산 (이미 정규화됨)
    """
    
    print(f"Opening database: {db_path}")
    conn = duckdb.connect(db_path, read_only=True)
    
    # 전체 데이터 수 확인
    total_count = conn.execute("SELECT COUNT(*) FROM datasets_raw").fetchone()[0]
    print(f"Total samples in database: {total_count:,}")
    
    # 샘플링 (너무 크면 일부만 사용)
    if total_count > sample_size:
        print(f"Sampling {sample_size:,} rows for statistics computation...")
        sample_query = f"SELECT * FROM datasets_raw USING SAMPLE {sample_size} ROWS"
    else:
        print(f"Using all {total_count:,} rows...")
        sample_query = "SELECT * FROM datasets_raw"
    
    # 데이터 로드
    print("Loading data...")
    df = conn.execute(sample_query).fetchdf()
    conn.close()
    
    print(f"Loaded {len(df):,} samples")
    print("=" * 80)
    
    # 특징별 통계 계산
    feature_cols = list(FEATURE_NAMES)
    means = []
    stds = []
    
    print(f"Computing statistics for {len(feature_cols)} features...")
    print("=" * 80)
    
    for i, col in enumerate(tqdm(feature_cols, desc="Processing features"), 1):
        strategy = get_normalization_strategy(col)
        
        if col not in df.columns:
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: MISSING - using defaults")
            means.append(0.0)
            stds.append(1.0)
            continue
        
        # 원본 값
        raw_values = df[col].values
        
        # NaN 제거
        valid_mask = ~np.isnan(raw_values)
        if not valid_mask.any():
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: ALL NaN - using defaults")
            means.append(0.0)
            stds.append(1.0)
            continue
        
        raw_values = raw_values[valid_mask]
        
        try:
            if strategy == "derived":
                # 파생 피처는 이미 정규화되어 있으므로 그대로 사용
                mean = float(np.mean(raw_values))
                std = float(np.std(raw_values, ddof=0))
                
            elif strategy == "log_std":
                # 로그 변환 후 통계 계산
                log_values = signed_log1p(raw_values)
                mean = float(np.mean(log_values))
                std = float(np.std(log_values, ddof=0))
                
            elif strategy == "std_only":
                # 원본 값의 통계 계산
                mean = float(np.mean(raw_values))
                std = float(np.std(raw_values, ddof=0))
                
            else:
                # 알 수 없는 전략 - 기본값 사용
                mean = 0.0
                std = 1.0
            
            # std가 0이면 1.0으로 설정
            if std == 0 or std < 1e-8 or not np.isfinite(std):
                std = 1.0
            
            # mean이 inf/nan이면 0.0으로 설정
            if not np.isfinite(mean):
                mean = 0.0
            
            means.append(mean)
            stds.append(std)
            
            # 로그 출력
            strategy_str = f"[{strategy:8s}]"
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s} {strategy_str}: "
                  f"mean={mean:12.4f}, std={std:12.4f}")
        
        except Exception as e:
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: ERROR - {e}")
            means.append(0.0)
            stds.append(1.0)
    
    print("=" * 80)
    
    return {
        'features': feature_cols,
        'mean': means,
        'std': stds,
        'num_features': len(feature_cols),
        'sample_size': len(df),
        'note': 'Statistics computed from raw (unnormalized) data'
    }


def main():
    parser = argparse.ArgumentParser(
        description='원본 데이터에서 정규화 통계 계산'
    )
    parser.add_argument('--db', required=True, help='Raw DuckDB database path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--sample-size', type=int, default=100000,
                       help='Number of samples to use for statistics (default: 100000)')
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not Path(args.db).exists():
        print(f"❌ Error: Database file not found: {args.db}")
        sys.exit(1)
    
    # 통계 계산
    print("\n" + "=" * 80)
    print("RAW NORMALIZATION STATISTICS COMPUTATION")
    print("=" * 80)
    
    stats = compute_raw_statistics(args.db, args.sample_size)
    
    # 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Raw normalization stats saved to: {output_path}")
    print(f"   Features: {stats['num_features']}")
    print(f"   Sample size: {stats['sample_size']:,}")
    print(f"   Mean range: [{min(stats['mean']):.4f}, {max(stats['mean']):.4f}]")
    print(f"   Std range: [{min(stats['std']):.4f}, {max(stats['std']):.4f}]")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print("=" * 80)
    
    # 주요 특징 통계 출력
    print("\nKey Features (After Transformation):")
    key_features = ['등락률', '누적거래대금', '거래회전율', '체결강도']
    for feat in key_features:
        if feat in stats['features']:
            idx = stats['features'].index(feat)
            strategy = get_normalization_strategy(feat)
            print(f"  {feat:15s} [{strategy:8s}]: mean={stats['mean'][idx]:12.4f}, std={stats['std'][idx]:12.4f}")
    
    print("\n✅ Done!")
    print("\n💡 Next steps:")
    print(f"   1. Update config/trading_config.json:")
    print(f'      "normalization_stats_path": "{args.output}"')
    print(f"   2. Update scripts/live/live_trading.py to enable training stats")
    print(f"   3. Restart live trading with the new statistics")


if __name__ == '__main__':
    main()
