import duckdb
import json
import os

parquet_path = r"C:\Users\user\Workspace\datasets@20260117\embeddings_v3"
cache_path = os.path.join(parquet_path, "key_index_v3.json")

print(f"Building index from {parquet_path} using DuckDB...")
query = f"""
SELECT code, date, filename as file_path, count(*) as cnt
FROM read_parquet('{os.path.join(parquet_path, "*.parquet")}', filename=True)
GROUP BY code, date, filename
"""

print("Executing query... This might take a few minutes depending on data size.")
con = duckdb.connect()
res = con.execute(query).fetchall()

key_index = {}
for code, date, file_path, cnt in res:
    key = (code, str(date))  # Ensure date is string if it's not
    # 윈도우 경로 구분자 정규화
    file_path = os.path.normpath(file_path)
    if key in key_index:
        existing = key_index[key]
        if isinstance(existing, list):
            existing.append((file_path, cnt))
        else:
            key_index[key] = [existing, (file_path, cnt)]
    else:
        key_index[key] = (file_path, cnt)

# JSON 키는 문자열이어야 함
with open(cache_path, 'w', encoding='utf-8') as f:
    json.dump({str(k): v for k, v in key_index.items()}, f)

print(f"Saved key index cache with {len(key_index)} entries to {cache_path}")
