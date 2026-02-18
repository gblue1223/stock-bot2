
import duckdb
import os
import argparse

def inspect_db(db_path):
    print(f"Connecting to {db_path}...")
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        # List tables
        tables = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
        print(f"Tables found: {tables}")
        
        table_name = tables[0][0] if tables else 'datasets'
        print(f"Using table: {table_name}")
        
        # Suspects
        suspects = [
            ('064400', '20250623'),
            ('052690', '20250617')
        ]
        
        for code, date in suspects:
            print(f"Checking {code} / {date}...")
            # Count rows
            count = con.execute(f"SELECT count(*) FROM {table_name} WHERE 종목코드=? AND 날짜=?", [code, date]).fetchone()[0]
            print(f"  -> {count} rows")
            
            # If huge, memory allocation would fail
            # Expected rows for a day: ~29400 (if 1s data from 9:00 to 15:30)
            # If millions -> ERROR
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    db_path = r"C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb"
    inspect_db(db_path)
