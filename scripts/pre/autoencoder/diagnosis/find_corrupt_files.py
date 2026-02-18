
import os
import glob
import pandas as pd
from tqdm import tqdm
import sys

def find_corrupt_files(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    print(f"Scanning {len(files)} files for corruption (064400, 052690, or huge size)...")
    
    corrupt_candidates = []
    
    for f in tqdm(files):
        try:
            # Metadata read first (fast)
            df_meta = pd.read_parquet(f, columns=['code', 'date'])
            codes = set(df_meta['code'].unique())
            
            suspect = False
            if '064400' in codes:
                print(f"[FOUND] {os.path.basename(f)} contains 064400")
                suspect = True
            if '052690' in codes:
                print(f"[FOUND] {os.path.basename(f)} contains 052690")
                suspect = True
                
            if suspect:
                # deeper check
                try:
                    df = pd.read_parquet(f)
                    mem = df.memory_usage(deep=True).sum()
                    print(f"  -> Size: {mem / 1024**3:.2f} GB")
                    if mem > 1 * 1024**3: # > 1GB
                        print(f"  -> MARKED AS CORRUPT (Huge Size)")
                        corrupt_candidates.append(f)
                except Exception as e:
                    print(f"  -> READ FAILED: {e}")
                    corrupt_candidates.append(f)

        except Exception as e:
            print(f"Metadata read failed for {os.path.basename(f)}: {e}")
            corrupt_candidates.append(f)
            
    print(f"\nFound {len(corrupt_candidates)} corrupt files.")
    for c in corrupt_candidates:
        print(f"DELETE: {c}")

if __name__ == "__main__":
    directory = r"C:\Users\user\Workspace\datasets@20260117\embeddings_v2"
    find_corrupt_files(directory)
