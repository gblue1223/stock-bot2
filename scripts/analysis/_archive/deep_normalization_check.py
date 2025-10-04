"""
Deep check of normalization to understand the exact issue
"""
import duckdb
import numpy as np
import pandas as pd
import argparse


def deep_check(db_path: str, table: str = "datasets"):
    """
    Detailed check of a few specific groups
    """
    print("="*80)
    print("Deep Normalization Check - Individual Groups")
    print("="*80)
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Get 5 specific groups with enough data
    query = f"""
    SELECT 날짜, 종목명_scalar, COUNT(*) as cnt
    FROM {table}
    GROUP BY 날짜, 종목명_scalar
    HAVING COUNT(*) >= 100
    ORDER BY RANDOM()
    LIMIT 5
    """
    groups = conn.execute(query).fetchdf()
    
    print(f"\nAnalyzing {len(groups)} groups in detail:\n")
    
    for idx, row in groups.iterrows():
        date = row['날짜']
        stock_scalar = row['종목명_scalar']
        cnt = row['cnt']
        
        # Get data for this group
        query = f"""
        SELECT 현재가, 거래량, 체결강도, 등락률
        FROM {table}
        WHERE 날짜 = ? AND 종목명_scalar = ?
        ORDER BY 번호
        """
        df = conn.execute(query, [date, stock_scalar]).fetchdf()
        
        print(f"Group {idx+1}: 날짜={date}, 종목명_scalar={stock_scalar:.4f}, rows={cnt}")
        print("-" * 60)
        
        for col in ['현재가', '거래량', '체결강도', '등락률']:
            if col in df.columns:
                values = df[col].values
                print(f"  {col}:")
                print(f"    mean={values.mean():.6f}, std={values.std():.6f}")
                print(f"    min={values.min():.6f}, max={values.max():.6f}")
                print(f"    sample: {values[:5].tolist()}")
        print()
    
    conn.close()
    
    # Now check if the issue is in how training loads data
    print("="*80)
    print("Checking Training Data Loading")
    print("="*80)
    print("""
The issue might be in how train_supervised.py loads and batches data:

1. If data from different groups is mixed in same batch:
   - Each group has mean=0, std=1
   - But mixing them creates inconsistent scales
   - Model sees different distributions in each batch

2. If sliding windows cross group boundaries:
   - Window might contain data from multiple normalizations
   - Creates discontinuities in the sequence

Check ai_trader/ml/data.py to see how sequences are created!
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="datasets")
    args = parser.parse_args()
    
    deep_check(args.db, args.table)


if __name__ == "__main__":
    main()
