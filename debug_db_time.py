
import duckdb
import pandas as pd

db_path = r"C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb"
table_name = "datasets"

try:
    conn = duckdb.connect(db_path, read_only=True)
    print(f"Connected to {db_path}")
    
    # Check table existence
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Tables: {tables}")
    
    if (table_name,) in tables or [table_name] in tables or (table_name) in [t[0] for t in tables]:
        # Sample '시간' column
        query = f"SELECT 시간 FROM {table_name} LIMIT 20"
        times = conn.execute(query).fetchdf()
        print("\nSample '시간' values:")
        print(times)
        
        # Check min/max
        min_max = conn.execute(f"SELECT MIN(시간), MAX(시간) FROM {table_name}").fetchall()
        print(f"\nMin Time: {min_max[0][0]}, Max Time: {min_max[0][1]}")
        
        # Check type
        schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        time_col = schema[schema['column_name'] == '시간']
        print("\n'시간' column schema:")
        print(time_col)
        
    else:
        print(f"Table '{table_name}' not found.")

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
