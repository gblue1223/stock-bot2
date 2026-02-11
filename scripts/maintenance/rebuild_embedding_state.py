import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

def rebuild_state(output_dir, state_file):
    print(f"Scanning {output_dir} for parquet files...")
    files = glob.glob(os.path.join(output_dir, "embeddings_*.parquet"))
    
    if not files:
        print("No files found.")
        return

    existing_stocks = set()
    
    print(f"Found {len(files)} files. Reading stock codes...")
    for f in tqdm(files):
        try:
            # We only need the 'code' column.
            # Using read_parquet with columns argument is faster.
            df = pd.read_parquet(f, columns=['code'])
            codes = df['code'].unique()
            existing_stocks.update(codes)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            print(f"!!! WARNING: File {f} seems corrupt or unreadable. You should probably delete it.")
    
    print(f"Total unique stocks found in Parquet files: {len(existing_stocks)}")
    
    # Backup old state file if exists
    if os.path.exists(state_file):
        backup_path = state_file + ".bak"
        import shutil
        shutil.copy(state_file, backup_path)
        print(f"Backed up existing state file to {backup_path}")
        
    # Write new state file
    with open(state_file, 'w') as f:
        for s in sorted(list(existing_stocks)):
            f.write(f"{s}\n")
            
    print(f"Successfully rebuilt {state_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help="Path to embeddings directory")
    parser.add_argument('--state_file', type=str, default='processed_stocks.txt', help="Path to state file")
    args = parser.parse_args()
    
    rebuild_state(args.output_dir, args.state_file)
