import sys
import os
import time
import duckdb
import pandas as pd
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.normalization import compute_stock_name_scalar_batch

def run_extraction():
    src_db = 'C:/Users/user/Workspace/datasets@raw/datasets_all.duckdb'
    dst_dir = 'C:/Users/user/Workspace/datasets@20260811'
    dst_db = os.path.join(dst_dir, 'datasets_260811.duckdb')
    
    print("=" * 80)
    print("DuckDB High-Performance Dataset Extraction Tool")
    print(f"Source DB      : {src_db}")
    print(f"Destination DB : {dst_db}")
    print("Time Window    : 09:00:00.000 ~ 11:00:00.000 (90000000 ~ 110000000)")
    print("=" * 80)
    
    if not os.path.exists(src_db):
        raise FileNotFoundError(f"Source DB not found: {src_db}")
        
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(dst_db):
        print(f"Removing existing destination DB: {dst_db}")
        os.remove(dst_db)
        
    start_time = time.time()
    
    # 1. Connect to Destination DB and attach Source DB
    print("[STEP 1/4] Connecting to DBs and building stock name encoding map...")
    conn = duckdb.connect(dst_db)
    
    # Increase memory limit & thread count for max speed
    conn.execute("SET threads TO 8")
    conn.execute(f"ATTACH '{src_db}' AS src (READ_ONLY)")
    
    # Compute stock_name_scalar mapping for unique stock names
    unique_names_df = conn.execute("SELECT DISTINCT 종목명 FROM src.datasets WHERE 종목명 IS NOT NULL").fetchdf()
    unique_names = unique_names_df['종목명']
    scalars = compute_stock_name_scalar_batch(unique_names)
    stock_map_df = pd.DataFrame({'종목명': unique_names, '종목명_scalar': scalars})
    
    conn.register('stock_map', stock_map_df)
    print(f"[OK] Stock name scalar map created for {len(stock_map_df)} unique stocks.")
    
    # 2. Execute High-Performance Streaming SQL Extraction
    print("[STEP 2/4] Executing streaming extraction & feature calculations in DuckDB...")
    
    sql = """
    CREATE TABLE datasets AS
    WITH filtered AS (
        SELECT 
            날짜,
            종목코드,
            종목명,
            시간,
            COALESCE(TRY_CAST(현재가 AS DOUBLE), 0.0) / 1000000.0 AS 현재가,
            COALESCE(TRY_CAST(등락률 AS DOUBLE), 0.0) AS 등락률,
            COALESCE(TRY_CAST(누적거래대금 AS DOUBLE), 0.0) AS 누적거래대금,
            
            -- 매도대기금액 1~10 (백만원 단위 = 호가 * 수량 / 1,000,000)
            (COALESCE(TRY_CAST(매도호가1 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량1 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액1,
            (COALESCE(TRY_CAST(매도호가2 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량2 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액2,
            (COALESCE(TRY_CAST(매도호가3 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량3 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액3,
            (COALESCE(TRY_CAST(매도호가4 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량4 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액4,
            (COALESCE(TRY_CAST(매도호가5 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량5 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액5,
            (COALESCE(TRY_CAST(매도호가6 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량6 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액6,
            (COALESCE(TRY_CAST(매도호가7 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량7 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액7,
            (COALESCE(TRY_CAST(매도호가8 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량8 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액8,
            (COALESCE(TRY_CAST(매도호가9 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량9 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액9,
            (COALESCE(TRY_CAST(매도호가10 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매도호가수량10 AS DOUBLE), 0.0)) / 1000000.0 AS 매도대기금액10,
            
            -- 매수대기금액 1~10 (백만원 단위 = 호가 * 수량 / 1,000,000)
            (COALESCE(TRY_CAST(매수호가1 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량1 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액1,
            (COALESCE(TRY_CAST(매수호가2 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량2 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액2,
            (COALESCE(TRY_CAST(매수호가3 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량3 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액3,
            (COALESCE(TRY_CAST(매수호가4 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량4 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액4,
            (COALESCE(TRY_CAST(매수호가5 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량5 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액5,
            (COALESCE(TRY_CAST(매수호가6 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량6 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액6,
            (COALESCE(TRY_CAST(매수호가7 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량7 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액7,
            (COALESCE(TRY_CAST(매수호가8 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량8 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액8,
            (COALESCE(TRY_CAST(매수호가9 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량9 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액9,
            (COALESCE(TRY_CAST(매수호가10 AS DOUBLE), 0.0) * COALESCE(TRY_CAST(매수호가수량10 AS DOUBLE), 0.0)) / 1000000.0 AS 매수대기금액10,
            
            LPAD(SPLIT_PART(CAST(시간 AS VARCHAR), '.', 1), 9, '0') AS t_padded
        FROM src.datasets
        WHERE TRY_CAST(시간 AS DOUBLE) >= 90000000.0
          AND TRY_CAST(시간 AS DOUBLE) <= 110000000.0
    ),
    computed AS (
        SELECT
            f.*,
            CAST(SUBSTR(t_padded, 1, 2) AS INT) * 3600 +
            CAST(SUBSTR(t_padded, 3, 2) AS INT) * 60 +
            CAST(SUBSTR(t_padded, 5, 2) AS INT) AS secs
        FROM filtered f
    )
    SELECT 
        c.날짜,
        c.종목코드,
        c.종목명,
        c.시간,
        c.현재가,
        c.등락률,
        c.누적거래대금,
        c.매도대기금액1, c.매도대기금액2, c.매도대기금액3, c.매도대기금액4, c.매도대기금액5,
        c.매도대기금액6, c.매도대기금액7, c.매도대기금액8, c.매도대기금액9, c.매도대기금액10,
        c.매수대기금액1, c.매수대기금액2, c.매수대기금액3, c.매수대기금액4, c.매수대기금액5,
        c.매수대기금액6, c.매수대기금액7, c.매수대기금액8, c.매수대기금액9, c.매수대기금액10,
        COALESCE(m.종목명_scalar, 0.0) AS 종목명_scalar,
        SIN(2.0 * PI() * secs / 86400.0) AS 시간_sin,
        COS(2.0 * PI() * secs / 86400.0) AS 시간_cos,
        (GREATEST(0.0, secs - 32400.0) - 10800.0) / 10800.0 AS 시간_scalar
    FROM computed c
    LEFT JOIN stock_map m ON c.종목명 = m.종목명
    """
    
    conn.execute(sql)
    print("[OK] Table datasets created successfully.")
    
    # 3. Create index
    print("[STEP 3/4] Creating index on (종목코드, 날짜, 시간)...")
    conn.execute("CREATE INDEX idx_datasets_code_date_time ON datasets(종목코드, 날짜, 시간)")
    print("[OK] Index created.")
    
    # 4. Verify & Statistics
    print("[STEP 4/4] Verifying extracted database...")
    count = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    sample_df = conn.execute("SELECT * FROM datasets LIMIT 1").fetchdf()
    conn.close()
    
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[SUCCESS] Extracted DB created at: {dst_db}")
    print(f"Total Saved Rows : {count:,}")
    print(f"Total Columns    : {len(sample_df.columns)}")
    print(f"Elapsed Time     : {elapsed:.2f} seconds")
    print("=" * 80)
    print("\n--- [Output Schema & First Record Sample] ---")
    record = sample_df.to_dict(orient='records')[0]
    for k, v in record.items():
        print(f"  {k}: {v}")
    print("=" * 80)

if __name__ == '__main__':
    run_extraction()
