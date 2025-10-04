"""
Check if normalization is being applied per-group (날짜, 종목명) or globally
"""
import duckdb
import numpy as np
import pandas as pd
import argparse


def check_group_normalization(db_path: str, table: str = "datasets"):
    """
    Check if data is normalized per-group or globally by examining statistics
    """
    print("="*80)
    print("Group-wise vs Global Normalization Check")
    print("="*80)
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Sample multiple groups
    query = f"""
    WITH group_samples AS (
        SELECT 날짜, 종목명_scalar, 
               COUNT(*) as group_size
        FROM {table}
        GROUP BY 날짜, 종목명_scalar
        HAVING COUNT(*) >= 60
        ORDER BY RANDOM()
        LIMIT 20
    )
    SELECT t.*
    FROM {table} t
    INNER JOIN group_samples g 
        ON t.날짜 = g.날짜 AND t.종목명_scalar = g.종목명_scalar
    ORDER BY t.날짜, t.종목명_scalar, t.번호
    """
    
    df = conn.execute(query).fetchdf()
    print(f"\nSampled {len(df)} rows from {df.groupby(['날짜', '종목명_scalar']).ngroups} groups")
    
    # Check key features
    features_to_check = ['현재가', '거래량', '체결강도', '등락률']
    
    print("\n" + "="*80)
    print("Per-Group Statistics")
    print("="*80)
    
    for feature in features_to_check:
        if feature not in df.columns:
            continue
            
        print(f"\n{feature}:")
        print("-" * 40)
        
        # Global stats
        global_mean = df[feature].mean()
        global_std = df[feature].std()
        print(f"Global: mean={global_mean:.4f}, std={global_std:.4f}")
        
        # Per-group stats
        grouped = df.groupby(['날짜', '종목명_scalar'])[feature]
        group_means = grouped.mean()
        group_stds = grouped.std()
        
        print(f"\nPer-group means:")
        print(f"  Mean of means: {group_means.mean():.4f}")
        print(f"  Std of means: {group_means.std():.4f}")
        print(f"  Range: [{group_means.min():.4f}, {group_means.max():.4f}]")
        
        print(f"\nPer-group stds:")
        print(f"  Mean of stds: {group_stds.mean():.4f}")
        print(f"  Std of stds: {group_stds.std():.4f}")
        print(f"  Range: [{group_stds.min():.4f}, {group_stds.max():.4f}]")
        
        # Diagnosis
        if abs(global_mean) < 0.1 and abs(global_std - 1.0) < 0.1:
            if group_means.std() > 0.3:
                print(f"⚠️  WARNING: Global normalization detected!")
                print(f"   - Global stats look normalized (mean≈0, std≈1)")
                print(f"   - But per-group means vary widely (std={group_means.std():.4f})")
                print(f"   - This means normalization was done GLOBALLY, not per-group")
            else:
                print(f"✅ GOOD: Per-group normalization detected")
                print(f"   - Both global and per-group stats are normalized")
        elif group_means.std() < 0.1 and group_stds.mean() > 0.8:
            print(f"✅ GOOD: Per-group normalization detected")
            print(f"   - Per-group means are consistent (std={group_means.std():.4f})")
            print(f"   - Per-group stds are close to 1 (mean={group_stds.mean():.4f})")
        else:
            print(f"⚠️  UNCLEAR: Normalization pattern unclear")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Check group-wise normalization")
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    parser.add_argument("--table", default="datasets", help="Table name")
    
    args = parser.parse_args()
    
    check_group_normalization(args.db, args.table)


if __name__ == "__main__":
    main()
