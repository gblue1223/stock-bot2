
import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_coverage(parquet_dir, processed_list_path):
    if not os.path.exists(parquet_dir):
        logger.error(f"Directory not found: {parquet_dir}")
        return
        
    if not os.path.exists(processed_list_path):
        logger.error(f"Processed list not found: {processed_list_path}")
        return

    # 1. Load processed stocks list
    with open(processed_list_path, 'r') as f:
        processed_stocks = set(line.strip() for line in f if line.strip())
    
    logger.info(f"Loaded {len(processed_stocks)} stocks from {os.path.basename(processed_list_path)}")

    # 2. Scan parquet files for actual stocks
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    found_stocks = set()
    
    logger.info(f"Scanning {len(parquet_files)} parquet files...")
    
    for file_path in tqdm(parquet_files, desc="Scanning files"):
        try:
            # Read only 'code' column to be fast
            df = pd.read_parquet(file_path, columns=['code'])
            if not df.empty:
                # Add unique codes from this file
                unique_codes = df['code'].unique()
                found_stocks.update(unique_codes)
        except Exception as e:
            logger.warning(f"Failed to read {os.path.basename(file_path)}: {e}")

    logger.info(f"Found {len(found_stocks)} unique stocks in parquet files.")
    
    # 3. Compare
    missing_in_parquet = processed_stocks - found_stocks
    extra_in_parquet = found_stocks - processed_stocks
    
    logger.info("="*50)
    logger.info("Comparison Result:")
    
    if not missing_in_parquet:
        logger.info("[OK] All stocks in processed_stocks.txt are present in parquet files.")
    else:
        logger.error(f"[FAIL] {len(missing_in_parquet)} stocks are listed as processed but MISSING in parquet:")
        # Print first 10
        sorted_missing = sorted(list(missing_in_parquet))
        for s in sorted_missing[:10]:
            logger.info(f"  - {s}")
        if len(sorted_missing) > 10:
            logger.info(f"  ... and {len(sorted_missing)-10} more.")
            
    if extra_in_parquet:
        logger.warning(f"[NOTE] {len(extra_in_parquet)} stocks found in parquet but NOT in processed list (maybe from previous interrupted runs?):")
        sorted_extra = sorted(list(extra_in_parquet))
        for s in sorted_extra[:10]:
            logger.info(f"  - {s}")
        if len(sorted_extra) > 10:
            logger.info(f"  ... and {len(sorted_extra)-10} more.")
            
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify stock coverage.")
    parser.add_argument("--dir", type=str, required=True, help="Parquet directory")
    parser.add_argument("--list", type=str, required=True, help="Path to processed_stocks.txt")
    args = parser.parse_args()

    verify_coverage(args.dir, args.list)
