"""
데이터베이스의 날짜 범위 확인 스크립트
"""
import duckdb
import sys

if len(sys.argv) < 2:
    print("Usage: python check_db_dates.py <db_path> [table_name]")
    sys.exit(1)

db_path = sys.argv[1]
table_name = sys.argv[2] if len(sys.argv) > 2 else 'datasets'

try:
    conn = duckdb.connect(db_path, read_only=True)
    
    # 날짜 범위 확인
    query = f"SELECT MIN(날짜) as min_date, MAX(날짜) as max_date, COUNT(*) as total_rows FROM {table_name}"
    result = conn.execute(query).fetchdf()
    
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    print("=" * 80)
    print(f"Date range: {result['min_date'].iloc[0]} to {result['max_date'].iloc[0]}")
    print(f"Total rows: {result['total_rows'].iloc[0]:,}")
    
    # 날짜별 행 수 확인 (상위 10개)
    print("\nTop 10 dates by row count:")
    date_count_query = f"SELECT 날짜, COUNT(*) as count FROM {table_name} GROUP BY 날짜 ORDER BY 날짜 DESC LIMIT 10"
    date_counts = conn.execute(date_count_query).fetchdf()
    for _, row in date_counts.iterrows():
        print(f"  {row['날짜']}: {row['count']:,} rows")
    
    # 종목 수 확인
    stock_count_query = f"SELECT COUNT(DISTINCT 종목코드) as stock_count FROM {table_name}"
    stock_count = conn.execute(stock_count_query).fetchdf()
    print(f"\nTotal stocks: {stock_count['stock_count'].iloc[0]:,}")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
