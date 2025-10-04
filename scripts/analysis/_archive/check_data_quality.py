"""
데이터 품질 및 정규화 상태 확인
"""
import duckdb
import numpy as np
import pandas as pd

def check_data_quality(db_path: str, table: str = "datasets", limit: int = 10000):
    """데이터 품질 확인"""
    
    conn = duckdb.connect(db_path)
    
    # Sample data
    query = f"SELECT * FROM {table} LIMIT {limit}"
    df = conn.execute(query).df()
    
    print("=" * 80)
    print("Data Quality Check")
    print("=" * 80)
    print(f"Total columns: {len(df.columns)}")
    print(f"Sample size: {len(df)}")
    print()
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Check for NaN/Inf
    print("\n" + "=" * 80)
    print("NaN/Inf Check")
    print("=" * 80)
    
    for col in numeric_cols[:20]:  # Check first 20 numeric columns
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"{col}:")
            print(f"  NaN: {nan_count} ({nan_count/len(df)*100:.2f}%)")
            print(f"  Inf: {inf_count} ({inf_count/len(df)*100:.2f}%)")
    
    # Check value ranges (normalization check)
    print("\n" + "=" * 80)
    print("Value Range Check (Normalization)")
    print("=" * 80)
    
    key_features = ['현재가', '등락률', '거래회전율', '체결강도', '누적거래대금']
    
    for col in key_features:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Min: {values.min():.6f}")
                print(f"  Max: {values.max():.6f}")
                print(f"  Mean: {values.mean():.6f}")
                print(f"  Std: {values.std():.6f}")
                
                # Check if normalized
                if abs(values.mean()) > 1.0 or values.std() > 10.0:
                    print(f"  ⚠️  WARNING: Not properly normalized!")
                
                # Check for constant values
                if values.std() < 1e-6:
                    print(f"  ⚠️  WARNING: Nearly constant values!")
    
    # Check target column distribution
    print("\n" + "=" * 80)
    print("Target Column (현재가) Analysis")
    print("=" * 80)
    
    if '현재가' in df.columns:
        prices = df['현재가'].dropna()
        print(f"Non-null values: {len(prices)}")
        print(f"Unique values: {prices.nunique()}")
        print(f"Value range: [{prices.min():.6f}, {prices.max():.6f}]")
        
        # Check for zero variance
        if prices.std() < 1e-6:
            print("⚠️  CRITICAL: Target has zero variance!")
        
        # Sample values
        print(f"\nSample values (first 20):")
        print(prices.head(20).values)
    
    # Check direction3 labels
    print("\n" + "=" * 80)
    print("Direction3 Label Distribution (Simulated)")
    print("=" * 80)
    
    if '현재가' in df.columns:
        # Simulate direction3 calculation
        prices = df['현재가'].values
        seq_len = 60
        horizon = 20
        threshold = 0.015
        
        labels = []
        for i in range(len(prices) - seq_len - horizon):
            p_now = prices[i + seq_len - 1]
            p_future = prices[i + seq_len + horizon - 1]
            
            if p_now == 0:
                continue
            
            chg = (p_future - p_now) / p_now
            
            if chg > threshold:
                labels.append(2)
            elif chg < -threshold:
                labels.append(0)
            else:
                labels.append(1)
        
        if labels:
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            
            print(f"Total labels: {total}")
            for cls, cnt in zip(unique, counts):
                print(f"  Class {cls}: {cnt} ({cnt/total*100:.2f}%)")
            
            # Check for severe imbalance
            max_prop = max(counts) / total
            if max_prop > 0.80:
                print(f"\n⚠️  WARNING: Severe class imbalance! Majority class: {max_prop*100:.1f}%")
    
    conn.close()


if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Workspace\datasets@20251002\datasets_norm_all.duckdb"
    
    check_data_quality(db_path)
