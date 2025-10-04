"""
Training issue diagnosis script
Checks for data leakage, distribution issues, and normalization problems
"""
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def check_data_leakage(db_path: str, table: str = "datasets"):
    """Check if future information is leaking into features"""
    print("\n" + "="*80)
    print("1. Data Leakage Check")
    print("="*80)
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Check if there are any rows where 시가 = 현재가 (suspicious)
    query = f"""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN 시가 = 현재가 THEN 1 ELSE 0 END) as same_open_close,
        SUM(CASE WHEN 고가 = 저가 THEN 1 ELSE 0 END) as same_high_low,
        SUM(CASE WHEN ABS(시가 - 현재가) < 1e-6 THEN 1 ELSE 0 END) as near_same_open_close
    FROM {table}
    """
    result = conn.execute(query).fetchdf()
    print(f"\n시가 = 현재가 rows: {result['same_open_close'].iloc[0]:,} / {result['total_rows'].iloc[0]:,}")
    print(f"고가 = 저가 rows: {result['same_high_low'].iloc[0]:,} / {result['total_rows'].iloc[0]:,}")
    print(f"시가 ≈ 현재가 rows: {result['near_same_open_close'].iloc[0]:,} / {result['total_rows'].iloc[0]:,}")
    
    if result['same_open_close'].iloc[0] > result['total_rows'].iloc[0] * 0.5:
        print("⚠️  WARNING: Too many rows with 시가 = 현재가 (possible data leakage)")
    
    conn.close()


def check_normalization_by_group(db_path: str, table: str = "datasets", sample_size: int = 100000):
    """Check if normalization was done per group (날짜, 종목명)"""
    print("\n" + "="*80)
    print("2. Group-wise Normalization Check")
    print("="*80)
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Sample data
    query = f"""
    SELECT 날짜, 종목명_scalar, 현재가, 거래량, 체결강도, 등락률
    FROM {table}
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """
    df = conn.execute(query).fetchdf()
    
    # Check stats per group
    grouped = df.groupby(['날짜', '종목명_scalar'])
    
    stats = grouped.agg({
        '현재가': ['mean', 'std'],
        '거래량': ['mean', 'std'],
        '체결강도': ['mean', 'std']
    })
    
    print(f"\nSampled {len(grouped)} groups")
    print(f"\n현재가 stats per group:")
    print(f"  Mean of means: {stats[('현재가', 'mean')].mean():.4f}")
    print(f"  Mean of stds: {stats[('현재가', 'std')].mean():.4f}")
    print(f"  Std of means: {stats[('현재가', 'mean')].std():.4f}")
    
    print(f"\n체결강도 stats per group:")
    print(f"  Mean of means: {stats[('체결강도', 'mean')].mean():.4f}")
    print(f"  Mean of stds: {stats[('체결강도', 'std')].mean():.4f}")
    print(f"  Std of means: {stats[('체결강도', 'mean')].std():.4f}")
    
    # Check if 체결강도 has proper range
    print(f"\n체결강도 global stats:")
    print(f"  Min: {df['체결강도'].min():.4f}")
    print(f"  Max: {df['체결강도'].max():.4f}")
    print(f"  Mean: {df['체결강도'].mean():.4f}")
    print(f"  Std: {df['체결강도'].std():.4f}")
    
    if df['체결강도'].mean() > 0.1 or df['체결강도'].std() < 0.8:
        print("⚠️  WARNING: 체결강도 not properly normalized")
    
    conn.close()


def check_target_distribution(db_path: str, table: str = "datasets", horizon: int = 20):
    """Check target variable distribution and direction3 labels"""
    print("\n" + "="*80)
    print("3. Target Distribution Check")
    print("="*80)
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Get sample with future prices
    query = f"""
    WITH ordered AS (
        SELECT 
            날짜, 종목명_scalar, 번호, 현재가,
            ROW_NUMBER() OVER (PARTITION BY 날짜, 종목명_scalar ORDER BY 번호) as rn
        FROM {table}
    )
    SELECT 
        a.현재가 as current_price,
        b.현재가 as future_price,
        (b.현재가 - a.현재가) as price_change
    FROM ordered a
    LEFT JOIN ordered b 
        ON a.날짜 = b.날짜 
        AND a.종목명_scalar = b.종목명_scalar 
        AND b.rn = a.rn + {horizon}
    WHERE b.현재가 IS NOT NULL
    LIMIT 50000
    """
    df = conn.execute(query).fetchdf()
    
    print(f"\nSampled {len(df)} rows with future prices")
    print(f"\nPrice change distribution:")
    print(f"  Mean: {df['price_change'].mean():.6f}")
    print(f"  Std: {df['price_change'].std():.6f}")
    print(f"  Min: {df['price_change'].min():.6f}")
    print(f"  Max: {df['price_change'].max():.6f}")
    
    # Calculate direction3 labels with threshold
    threshold = 0.015
    df['direction3'] = 1  # neutral
    df.loc[df['price_change'] > threshold, 'direction3'] = 2  # up
    df.loc[df['price_change'] < -threshold, 'direction3'] = 0  # down
    
    print(f"\nDirection3 distribution (threshold={threshold}):")
    print(df['direction3'].value_counts(normalize=True).sort_index())
    
    # Check if distribution is too imbalanced
    dist = df['direction3'].value_counts(normalize=True)
    if dist.max() > 0.8:
        print("⚠️  WARNING: Highly imbalanced classes (max class > 80%)")
    
    conn.close()


def check_sequence_continuity(db_path: str, table: str = "datasets"):
    """Check if sequences are continuous within groups"""
    print("\n" + "="*80)
    print("4. Sequence Continuity Check")
    print("="*80)
    
    conn = duckdb.connect(db_path, read_only=True)
    
    query = f"""
    WITH gaps AS (
        SELECT 
            날짜, 종목명_scalar,
            번호,
            LAG(번호) OVER (PARTITION BY 날짜, 종목명_scalar ORDER BY 번호) as prev_번호,
            번호 - LAG(번호) OVER (PARTITION BY 날짜, 종목명_scalar ORDER BY 번호) as gap
        FROM {table}
    )
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN gap IS NULL THEN 1 ELSE 0 END) as first_rows,
        SUM(CASE WHEN gap = 1 THEN 1 ELSE 0 END) as continuous_rows,
        SUM(CASE WHEN gap > 1 THEN 1 ELSE 0 END) as gap_rows,
        AVG(CASE WHEN gap > 1 THEN gap ELSE NULL END) as avg_gap_size
    FROM gaps
    """
    result = conn.execute(query).fetchdf()
    
    print(f"\nTotal rows: {result['total_rows'].iloc[0]:,}")
    print(f"First rows in groups: {result['first_rows'].iloc[0]:,}")
    print(f"Continuous rows (gap=1): {result['continuous_rows'].iloc[0]:,}")
    print(f"Rows with gaps (gap>1): {result['gap_rows'].iloc[0]:,}")
    if result['avg_gap_size'].iloc[0]:
        print(f"Average gap size: {result['avg_gap_size'].iloc[0]:.2f}")
    
    if result['gap_rows'].iloc[0] > 0:
        print("⚠️  WARNING: Sequences have gaps (may cause issues with sliding windows)")
    
    conn.close()


def check_feature_variance(db_path: str, table: str = "datasets", sample_size: int = 50000):
    """Check if features have sufficient variance"""
    print("\n" + "="*80)
    print("5. Feature Variance Check")
    print("="*80)
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Get all numeric columns
    query = f"SELECT * FROM {table} LIMIT 1"
    sample = conn.execute(query).fetchdf()
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude metadata columns
    exclude = ['날짜', '번호', '종목명_scalar', '종목코드']
    feature_cols = [c for c in numeric_cols if c not in exclude]
    
    # Sample and calculate variance
    query = f"""
    SELECT {', '.join(feature_cols)}
    FROM {table}
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """
    df = conn.execute(query).fetchdf()
    
    variances = df.var()
    low_variance = variances[variances < 0.01]
    
    print(f"\nChecked {len(feature_cols)} features")
    print(f"Low variance features (var < 0.01): {len(low_variance)}")
    
    if len(low_variance) > 0:
        print("\nLow variance features:")
        for col, var in low_variance.items():
            print(f"  {col}: {var:.6f}")
        print("⚠️  WARNING: Some features have very low variance")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Diagnose training issues")
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    parser.add_argument("--table", default="datasets", help="Table name")
    parser.add_argument("--sample-size", type=int, default=50000, help="Sample size for checks")
    parser.add_argument("--horizon", type=int, default=20, help="Prediction horizon")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Training Issue Diagnosis")
    print("="*80)
    print(f"Database: {args.db}")
    print(f"Table: {args.table}")
    
    check_data_leakage(args.db, args.table)
    check_normalization_by_group(args.db, args.table, args.sample_size)
    check_target_distribution(args.db, args.table, args.horizon)
    check_sequence_continuity(args.db, args.table)
    check_feature_variance(args.db, args.table, args.sample_size)
    
    print("\n" + "="*80)
    print("Diagnosis Complete")
    print("="*80)


if __name__ == "__main__":
    main()
