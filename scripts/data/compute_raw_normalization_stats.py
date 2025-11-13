"""
정규화된 학습 데이터에서 원본 데이터의 정규화 통계를 계산

DuckDB에 저장된 데이터는 이미 정규화되어 있습니다.
이 스크립트는 정규화된 데이터를 역변환하여 원본 데이터의 통계를 계산합니다.

Usage:
    python scripts/data/compute_raw_normalization_stats.py \
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


def inverse_signed_log1p(y: np.ndarray) -> np.ndarray:
    """
    signed_log1p의 역변환
    
    x = sign(y) * (exp(|y|) - 1)
    """
    return np.sign(y) * (np.exp(np.abs(y)) - 1)


def compute_raw_statistics_from_normalized(db_path: str, sample_size: int = 100000) -> dict:
    """
    정규화된 데이터베이스에서 원본 데이터의 통계 계산
    
    정규화 과정:
    1. log_std: x -> log1p(|x|) * sign(x) -> z-score
    2. std_only: x -> z-score
    3. derived: 이미 정규화됨 (변환 없음)
    
    역변환 과정:
    1. log_std: z-score -> inverse_z -> inverse_log1p -> x
    2. std_only: z-score -> inverse_z -> x
    3. derived: 그대로 사용
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
    
    # 특징별 원본 통계 계산
    feature_cols = list(FEATURE_NAMES)
    raw_means = []
    raw_stds = []
    
    print(f"Computing raw statistics for {len(feature_cols)} features...")
    print("=" * 80)
    
    for i, col in enumerate(tqdm(feature_cols, desc="Processing features"), 1):
        strategy = get_normalization_strategy(col)
        
        if col not in df.columns:
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: MISSING - using defaults")
            raw_means.append(0.0)
            raw_stds.append(1.0)
            continue
        
        # 정규화된 값
        normalized_values = df[col].values
        
        # NaN 제거
        valid_mask = ~np.isnan(normalized_values)
        if not valid_mask.any():
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: ALL NaN - using defaults")
            raw_means.append(0.0)
            raw_stds.append(1.0)
            continue
        
        normalized_values = normalized_values[valid_mask]
        
        try:
            if strategy == "derived":
                # 파생 피처는 이미 정규화되어 있으므로 그대로 사용
                raw_mean = float(np.mean(normalized_values))
                raw_std = float(np.std(normalized_values, ddof=0))
                
            elif strategy == "log_std":
                # 역변환: z-score -> log-transformed -> raw
                # 정규화된 데이터의 통계를 사용하여 역변환
                norm_mean = float(np.mean(normalized_values))
                norm_std = float(np.std(normalized_values, ddof=0))
                
                # 로그 변환된 값으로 복원 (z-score 역변환)
                log_values = normalized_values * norm_std + norm_mean
                
                # 로그 역변환
                raw_values = inverse_signed_log1p(log_values)
                
                # 원본 통계 계산
                raw_mean = float(np.mean(raw_values))
                raw_std = float(np.std(raw_values, ddof=0))
                
            elif strategy == "std_only":
                # 역변환: z-score -> raw
                norm_mean = float(np.mean(normalized_values))
                norm_std = float(np.std(normalized_values, ddof=0))
                
                # z-score 역변환
                raw_values = normalized_values * norm_std + norm_mean
                
                # 원본 통계 계산
                raw_mean = float(np.mean(raw_values))
                raw_std = float(np.std(raw_values, ddof=0))
                
            else:
                # 알 수 없는 전략 - 기본값 사용
                raw_mean = 0.0
                raw_std = 1.0
            
            # std가 0이면 1.0으로 설정
            if raw_std == 0 or raw_std < 1e-8:
                raw_std = 1.0
            
            raw_means.append(raw_mean)
            raw_stds.append(raw_std)
            
            # 로그 출력
            strategy_str = f"[{strategy:8s}]"
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s} {strategy_str}: "
                  f"mean={raw_mean:12.4f}, std={raw_std:12.4f}")
        
        except Exception as e:
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: ERROR - {e}")
            raw_means.append(0.0)
            raw_stds.append(1.0)
    
    print("=" * 80)
    
    return {
        'features': feature_cols,
        'mean': raw_means,
        'std': raw_stds,
        'num_features': len(feature_cols),
        'sample_size': len(df),
        'note': 'Raw statistics computed from normalized data via inverse transformation'
    }


def main():
    parser = argparse.ArgumentParser(
        description='정규화된 데이터에서 원본 통계 계산'
    )
    parser.add_argument('--db', required=True, help='Normalized DuckDB database path')
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
    
    stats = compute_raw_statistics_from_normalized(args.db, args.sample_size)
    
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
    print("\nKey Features (Raw Statistics):")
    key_features = ['등락률', '누적거래대금', '거래회전율', '체결강도']
    for feat in key_features:
        if feat in stats['features']:
            idx = stats['features'].index(feat)
            print(f"  {feat:15s}: mean={stats['mean'][idx]:12.4f}, std={stats['std'][idx]:12.4f}")
    
    print("\n✅ Done!")
    print("\n💡 Next steps:")
    print(f"   1. Update config/trading_config.json:")
    print(f'      "normalization_stats_path": "{args.output}"')
    print(f"   2. Restart live trading with the new statistics")


if __name__ == '__main__':
    main()
