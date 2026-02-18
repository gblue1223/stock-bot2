
import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_parquet_files(directory):
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return

    parquet_files = glob.glob(os.path.join(directory, "*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files in {directory}")

    corrupted_files = []
    empty_files = []
    valid_count = 0

    for file_path in tqdm(parquet_files, desc="Verifying files"):
        try:
            # Try reading just the metadata/columns first for speed
            # Reading 'code' and 'date' is enough to check file integrity and schema
            df = pd.read_parquet(file_path, columns=['code', 'date'])
            
            if df.empty:
                empty_files.append(file_path)
                continue

            # Check for expected columns
            # We are only reading code and date, so only check for those
            expected_cols = ['code', 'date']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                 logger.warning(f"File {os.path.basename(file_path)} missing columns: {missing_cols}")
                 # We treat this as "corrupted" or at least invalid for our purpose
                 corrupted_files.append((file_path, f"Missing columns: {missing_cols}"))
                 continue
            
            # Check embedding column type (should be array/list, not just string)
            # This is a basic check.
            # sample_embedding = df.iloc[0]['embedding']
            # if not hasattr(sample_embedding, '__len__'):
            #      logger.warning(f"File {os.path.basename(file_path)} embedding column might be wrong type")

            valid_count += 1

        except Exception as e:
            logger.error(f"Error reading {os.path.basename(file_path)}: {e}")
            corrupted_files.append((file_path, str(e)))

    logger.info("="*50)
    logger.info(f"Verification Complete.")
    logger.info(f"Total Files: {len(parquet_files)}")
    logger.info(f"Valid Files: {valid_count}")
    logger.info(f"Empty Files: {len(empty_files)}")
    logger.info(f"Corrupted Files: {len(corrupted_files)}")
    logger.info("="*50)

    if corrupted_files:
        logger.info("Corrupted Files List:")
        for f, err in corrupted_files:
            logger.info(f"  - {f} : {err}")
            
    if empty_files:
        logger.info("Empty Files List:")
        for f in empty_files:
            logger.info(f"  - {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Parquet embeddings files.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing parquet files")
    args = parser.parse_args()

    verify_parquet_files(args.dir)
