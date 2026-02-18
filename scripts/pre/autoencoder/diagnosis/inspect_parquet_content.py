
import pandas as pd
import os

files = [
    r"C:\Users\user\Workspace\datasets@20260117\embeddings_v2\embeddings_0313.parquet",
    r"C:\Users\user\Workspace\datasets@20260117\embeddings_v2\embeddings_0703.parquet"
]

for f in files:
    print(f"Inspecting {os.path.basename(f)}...")
    if not os.path.exists(f):
        print("File not found! Did you delete it?")
        continue

    try:
        df = pd.read_parquet(f)
        print(f"  Shape: {df.shape}")
        
        # Check specific codes
        if 'embeddings_0313' in f:
            target = '052690'
        elif 'embeddings_0703' in f:
            target = '064400'
            
        subset = df[df['code'] == target]
        print(f"  Subset shape for {target}: {subset.shape}")
        
        if not subset.empty:
            print(f"  Columns: {subset.columns.tolist()}")
            # Print sample embedding shape
            print(f"  Embedding sample: {subset['embedding'].iloc[0][:5]} (len={len(subset['embedding'].iloc[0])})")
            
            # Check for duplicates
            dupes = subset.duplicated(subset=['time'])
            print(f"  Duplicate times: {dupes.sum()}")
            
    except Exception as e:
        print(f"  Error reading: {e}")
