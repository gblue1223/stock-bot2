"""
정규화 통계 검증 및 분석

생성된 정규화 통계가 실제 데이터를 잘 대표하는지 검증합니다.

Usage:
    python scripts/data/validate_normalization_stats.py \
        --db datasets_raw_all.duckdb \
        --stats models/grpo_scalping@2025120/raw_normalization_stats.json \
        --validation-samples 50000
"""

import argparse
import json
import numpy as np
import duckdb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.normalization import FEATURE_NAMES, get_normalization_strategy, signed_log1p


def validate_statistics(db_path: str, stats_path: str, validation_samples: int = 50000):
    """
    정규화 통계 검증
    
    1. 다른 샘플로 통계 재계산하여 비교
    2. 종목별/시간대별 분포 분석
    3. 정규화 후 값의 분포 확인
    """
    
    print("=" * 80)
    print("NORMALIZATION STATISTICS VALIDATION")
    print("=" * 80)
    
    # 저장된 통계 로드
    print(f"\n1. Loading saved statistics from: {stats_path}")
    with open(stats_path, 'r', encoding='utf-8') as f:
        saved_stats = json.load(f)
    
    saved_means = np.array(saved_stats['mean'])
    saved_stds = np.array(saved_stats['std'])
    
    print(f"   Features: {saved_stats['num_features']}")
    print(f"   Original sample size: {saved_stats['sample_size']:,}")
    
    # 데이터베이스 연결
    print(f"\n2. Connecting to database: {db_path}")
    conn = duckdb.connect(db_path, read_only=True)
    
    total_count = conn.execute("SELECT COUNT(*) FROM datasets_raw").fetchone()[0]
    print(f"   Total samples: {total_count:,}")
    
    # 검증용 샘플 로드 (다른 샘플)
    print(f"\n3. Loading validation samples: {validation_samples:,}")
    validation_query = f"SELECT * FROM datasets_raw USING SAMPLE {validation_samples} ROWS"
    df = conn.execute(validation_query).fetchdf()
    print(f"   Loaded: {len(df):,} samples")
    
    # 검증용 통계 계산
    print("\n4. Computing validation statistics...")
    print("=" * 80)
    
    feature_cols = list(FEATURE_NAMES)
    validation_means = []
    validation_stds = []
    differences = []
    
    for i, col in enumerate(feature_cols, 1):
        strategy = get_normalization_strategy(col)
        
        if col not in df.columns:
            validation_means.append(0.0)
            validation_stds.append(1.0)
            differences.append(0.0)
            continue
        
        raw_values = df[col].values
        valid_mask = ~np.isnan(raw_values)
        
        if not valid_mask.any():
            validation_means.append(0.0)
            validation_stds.append(1.0)
            differences.append(0.0)
            continue
        
        raw_values = raw_values[valid_mask]
        
        # 전략별 통계 계산
        if strategy == "derived":
            val_mean = float(np.mean(raw_values))
            val_std = float(np.std(raw_values, ddof=0))
        elif strategy == "log_std":
            log_values = signed_log1p(raw_values)
            val_mean = float(np.mean(log_values))
            val_std = float(np.std(log_values, ddof=0))
        elif strategy == "std_only":
            val_mean = float(np.mean(raw_values))
            val_std = float(np.std(raw_values, ddof=0))
        else:
            val_mean = 0.0
            val_std = 1.0
        
        if val_std == 0 or val_std < 1e-8 or not np.isfinite(val_std):
            val_std = 1.0
        if not np.isfinite(val_mean):
            val_mean = 0.0
        
        validation_means.append(val_mean)
        validation_stds.append(val_std)
        
        # 차이 계산 (상대 오차)
        saved_mean = saved_means[i-1]
        saved_std = saved_stds[i-1]
        
        if saved_std > 0:
            mean_diff = abs(val_mean - saved_mean) / saved_std * 100  # %
            std_diff = abs(val_std - saved_std) / saved_std * 100  # %
        else:
            mean_diff = 0.0
            std_diff = 0.0
        
        differences.append(max(mean_diff, std_diff))
        
        # 차이가 큰 경우 경고
        if mean_diff > 20 or std_diff > 20:
            status = "⚠️ LARGE DIFF"
        elif mean_diff > 10 or std_diff > 10:
            status = "⚠️ DIFF"
        else:
            status = "✅ OK"
        
        print(f"  [{i:2d}] {col:20s} [{strategy:8s}] {status}")
        print(f"       Saved:      mean={saved_mean:10.4f}, std={saved_std:10.4f}")
        print(f"       Validation: mean={val_mean:10.4f}, std={val_std:10.4f}")
        print(f"       Difference: mean={mean_diff:6.2f}%, std={std_diff:6.2f}%")
    
    print("=" * 80)
    
    # 종목별 분포 분석
    print("\n5. Stock-level distribution analysis...")
    print("=" * 80)
    
    # 랜덤 종목 5개 선택
    stock_query = """
        SELECT DISTINCT 종목코드
        FROM datasets_raw
        ORDER BY RANDOM()
        LIMIT 5
    """
    stocks = conn.execute(stock_query).fetchdf()['종목코드'].tolist()
    
    print(f"   Analyzing {len(stocks)} random stocks...")
    
    for stock_code in stocks:
        stock_query = f"""
            SELECT 등락률, 누적거래대금, 거래회전율, 체결강도
            FROM datasets_raw
            WHERE 종목코드 = ?
            LIMIT 1000
        """
        stock_df = conn.execute(stock_query, [stock_code]).fetchdf()
        
        if len(stock_df) > 0:
            print(f"\n   Stock: {stock_code} (samples: {len(stock_df)})")
            
            # 등락률
            rate = stock_df['등락률'].dropna()
            if len(rate) > 0:
                print(f"     등락률:      mean={rate.mean():8.2f}, std={rate.std():8.2f}")
            
            # 누적거래대금 (로그 변환)
            volume = stock_df['누적거래대금'].dropna()
            if len(volume) > 0:
                log_volume = signed_log1p(volume.values)
                print(f"     누적거래대금:  mean={np.mean(log_volume):8.2f}, std={np.std(log_volume):8.2f} (log)")
    
    print("=" * 80)
    
    # 시간대별 분포 분석
    print("\n6. Time-based distribution analysis...")
    print("=" * 80)
    
    time_query = """
        SELECT 
            CAST(시간 / 10000 AS INTEGER) as hour,
            AVG(등락률) as avg_rate,
            STDDEV(등락률) as std_rate,
            COUNT(*) as count
        FROM datasets_raw
        WHERE 시간 IS NOT NULL
        GROUP BY hour
        ORDER BY hour
    """
    time_df = conn.execute(time_query).fetchdf()
    
    print("   Hourly statistics (등락률):")
    for _, row in time_df.iterrows():
        hour = int(row['hour'])
        avg_rate = row['avg_rate']
        std_rate = row['std_rate']
        count = int(row['count'])
        print(f"     {hour:02d}:00 - mean={avg_rate:8.2f}, std={std_rate:8.2f}, samples={count:,}")
    
    print("=" * 80)
    
    # 정규화 후 분포 확인
    print("\n7. Normalized distribution check...")
    print("=" * 80)
    
    # 샘플 데이터로 정규화 테스트
    test_samples = 1000
    test_query = f"SELECT * FROM datasets_raw USING SAMPLE {test_samples} ROWS"
    test_df = conn.execute(test_query).fetchdf()
    
    print(f"   Testing normalization on {len(test_df)} samples...")
    
    key_features = ['등락률', '누적거래대금', '거래회전율', '체결강도']
    for feat in key_features:
        if feat not in test_df.columns:
            continue
        
        strategy = get_normalization_strategy(feat)
        idx = feature_cols.index(feat)
        
        raw_values = test_df[feat].dropna().values
        if len(raw_values) == 0:
            continue
        
        # 정규화 적용
        if strategy == "log_std":
            transformed = signed_log1p(raw_values)
        elif strategy == "std_only":
            transformed = raw_values
        else:
            transformed = raw_values
        
        normalized = (transformed - saved_means[idx]) / saved_stds[idx]
        normalized = np.clip(normalized, -5, 5)
        
        print(f"\n   {feat} [{strategy}]:")
        print(f"     Raw:        min={raw_values.min():10.2f}, max={raw_values.max():10.2f}, mean={raw_values.mean():10.2f}")
        print(f"     Normalized: min={normalized.min():10.2f}, max={normalized.max():10.2f}, mean={normalized.mean():10.2f}")
        print(f"     Clipped:    {(np.abs(normalized) >= 5).sum()} / {len(normalized)} ({(np.abs(normalized) >= 5).sum()/len(normalized)*100:.1f}%)")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    
    print(f"Average difference: {avg_diff:.2f}%")
    print(f"Maximum difference: {max_diff:.2f}%")
    
    if avg_diff < 5:
        print("✅ Statistics are HIGHLY CONSISTENT")
    elif avg_diff < 10:
        print("✅ Statistics are CONSISTENT")
    elif avg_diff < 20:
        print("⚠️ Statistics show MODERATE VARIATION")
    else:
        print("❌ Statistics show HIGH VARIATION - consider using more samples")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='정규화 통계 검증')
    parser.add_argument('--db', required=True, help='Raw DuckDB database path')
    parser.add_argument('--stats', required=True, help='Normalization stats JSON file')
    parser.add_argument('--validation-samples', type=int, default=50000,
                       help='Number of samples for validation (default: 50000)')
    args = parser.parse_args()
    
    if not Path(args.db).exists():
        print(f"❌ Error: Database not found: {args.db}")
        sys.exit(1)
    
    if not Path(args.stats).exists():
        print(f"❌ Error: Stats file not found: {args.stats}")
        sys.exit(1)
    
    validate_statistics(args.db, args.stats, args.validation_samples)


if __name__ == '__main__':
    main()
