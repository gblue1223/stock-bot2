
import pandas as pd
import os
import gc

def clean_duplicates():
    files = [
        r"C:\Users\user\Workspace\datasets@20260117\embeddings_v2\embeddings_0313.parquet",
        r"C:\Users\user\Workspace\datasets@20260117\embeddings_v2\embeddings_0703.parquet"
    ]
    
    for f in files:
        if not os.path.exists(f):
            print(f"Skipping {f} (not found)")
            continue
            
        print(f"Processing {os.path.basename(f)}...")
        try:
            df = pd.read_parquet(f)
            orig_len = len(df)
            
            # Check duplicates on time for each (code, date) group
            # Actually, (code, date, time) should be unique
            df_dedup = df.drop_duplicates(subset=['code', 'date', 'time'], keep='last')
            new_len = len(df_dedup)
            
            dropped = orig_len - new_len
            print(f"  Shape: {orig_len} -> {new_len} (Dropped {dropped} duplicates)")
            
            if dropped > 0:
                print("  Saving deduped file...")
                df_dedup.to_parquet(f, index=False)
                print("  [OK] Saved.")
            else:
                print("  No duplicates found. Skipping save.")
                
            del df
            del df_dedup
            gc.collect()
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    clean_duplicates()
