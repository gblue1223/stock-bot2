#!/usr/bin/env python3
"""
데이터 임베딩 일괄 생성 (Parallel & Resume & IPC optimized & Memory Efficient)

1. Resume 기능: processed_stocks.txt에 완료된 종목 기록 및 로드
2. 병렬 처리: Producer-Consumer 패턴 (ProcessPoolExecutor)
   - Worker: DB Fetch -> Normalize -> Tensor 변환 (Chunk 단위) -> Temp File Save
   - Main: Load Temp File -> GPU Inference -> Save

수정 내역:
- Memory Error (Unable to allocate 20GB+) 방지를 위해 
  Worker 내부에서 데이터를 한 번에 거대한 3D Array로 만들지 않고, 
  작은 청크 단위로 나누어 임시 파일에 저장하도록 변경.
"""

import os
import sys
import gc
import argparse
import duckdb
import torch
import numpy as np
import pandas as pd
import time
import uuid
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# TPU 지원 (torch_xla)
try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder

# --- Worker Function ---
def process_stock_data(stock_code, db_path, table_name, feature_cols, means, stds, seq_len, col_map, temp_dir):
    """
    Worker Process에서 실행되는 함수
    메모리 효율성을 위해 데이터를 청크 파일로 분할 저장합니다.
    """
    try:
        con = duckdb.connect(db_path, read_only=True, config={'access_mode': 'READ_ONLY'})
        
        select_cols = [f"\"{col_map['date']}\"", f"\"{col_map['time']}\"", f"\"{col_map['code']}\""] + \
                      [f"\"{c}\"" for c in feature_cols]
        select_clause = ", ".join(select_cols)
        
        query = f"SELECT {select_clause} FROM {table_name} WHERE \"{col_map['code']}\" = '{stock_code}' ORDER BY \"{col_map['date']}\", \"{col_map['time']}\""
        df = con.sql(query).df()
        con.close()
        
        if len(df) < seq_len:
            return None
            
        # Meta Data
        meta = df.iloc[seq_len-1:][[col_map['date'], col_map['time'], col_map['code']]].reset_index(drop=True)
        meta.columns = ['date', 'time', 'code']
        
        # Features & Norm
        feats = df[feature_cols].values.astype(np.float32)
        feats = (feats - means) / stds
        
        # 시퀀스 데이터 생성 (Chunk 단위 처리)
        # 전체를 (N, 120, 28)로 만들면 20GB 넘게 필요하므로 절대 금지.
        # 대신 (chunk_size, 120, 28)씩 잘라서 저장.
        
        num_samples = len(feats) - seq_len + 1
        chunk_size = 10000  # 한 번에 처리할 샘플 수 (약 130MB 메모리 사용)
        
        file_paths = []
        unique_id = str(uuid.uuid4())
        
        # Meta 데이터도 분할해서 저장해야 할까? 
        # Meta는 (N, 3)이라 작음. 그냥 한 번에 저장해도 됨.
        meta_path = os.path.join(temp_dir, f"{stock_code}_{unique_id}_meta.parquet")
        meta.to_parquet(meta_path, engine='pyarrow', index=False)
        
        # Sequence 데이터 분할 저장
        try:
            for i in range(0, num_samples, chunk_size):
                end = min(i + chunk_size, num_samples)
                
                slice_start = i
                slice_end = end + seq_len - 1
                
                if slice_end > len(feats):
                    slice_end = len(feats)
                    
                sub_feats = feats[slice_start:slice_end]
                
                sub_num_samples = len(sub_feats) - seq_len + 1
                if sub_num_samples <= 0: continue
                
                shape = (sub_num_samples, seq_len, sub_feats.shape[1])
                strides = (sub_feats.strides[0], sub_feats.strides[0], sub_feats.strides[1])
                
                sub_seqs = np.lib.stride_tricks.as_strided(sub_feats, shape=shape, strides=strides, writeable=False)
                
                sub_seqs_contig = np.ascontiguousarray(sub_seqs)
                
                chunk_path = os.path.join(temp_dir, f"{stock_code}_{unique_id}_seq_{i}.npy")
                np.save(chunk_path, sub_seqs_contig)
                file_paths.append(chunk_path)
                
                del sub_seqs
                del sub_seqs_contig
                
            return (stock_code, meta_path, file_paths)
            
        except Exception as e:
            # 에러 발생 시(디스크 부족 등) 생성된 안쓰는 파일 즉시 삭제
            for p in file_paths:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            if os.path.exists(meta_path):
                try: os.remove(meta_path)
                except: pass
            raise e
        
    except Exception as e:
        return (stock_code, Exception(f"[{stock_code}] {e}"))

# --- Main Script ---

def get_column_names(con, table_name):
    df = con.sql(f"DESCRIBE {table_name}").df()
    return df['column_name'].tolist()

def get_global_stats(con, table_name, feature_col_names):
    print("Calculating global statistics...")
    avg_query = ", ".join([f"AVG(\"{col}\")" for col in feature_col_names])
    std_query = ", ".join([f"STDDEV(\"{col}\")" for col in feature_col_names])
    stats_df = con.sql(f"SELECT {avg_query}, {std_query} FROM {table_name}").df()
    n = len(feature_col_names)
    means = stats_df.iloc[0, :n].values.astype(np.float32)
    stds = stats_df.iloc[0, n:].values.astype(np.float32)
    stds[stds == 0] = 1.0
    return means, stds

# 전역 카운터
file_counter = 0

def save_chunk(df, output_dir):
    global file_counter
    filename = f"embeddings_{file_counter:04d}.parquet"
    path = os.path.join(output_dir, filename)
    df.to_parquet(path, engine='pyarrow', index=False)
    file_counter += 1

def main():
    parser = argparse.ArgumentParser(description='Generate Embeddings (Parallel IPC Safe Memory Efficient)')
    parser.add_argument('--db_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--table_name', type=str, default='datasets')
    parser.add_argument('--output_dir', type=str, default='data/embeddings')
    parser.add_argument('--state_file', type=str, default='processed_stocks.txt')
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu, or tpu')
    
    args = parser.parse_args()
    
    # Init
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    if args.device == 'tpu':
        if not XLA_AVAILABLE:
            print("Error: torch_xla not found. Please install it (e.g. pip install torch-xla)")
            sys.exit(1)
        device = xm.xla_device()
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
    print(f"Device: {device}, Workers: {args.num_workers}")

    # Initialize file_counter based on existing files to avoid overwriting
    existing_files = [f for f in os.listdir(args.output_dir) if f.startswith('embeddings_') and f.endswith('.parquet')]
    max_idx = -1
    for f in existing_files:
        try:
            # Extract number from embeddings_XXXX.parquet
            part = f.split('_')[1].split('.')[0]
            idx = int(part)
            if idx > max_idx:
                max_idx = idx
        except:
            pass
    
    global file_counter
    file_counter = max_idx + 1
    print(f"Resuming file counter from {file_counter} (Found {len(existing_files)} existing files)")
    
    # Load Completed Stocks
    processed_stocks = set()
    if os.path.exists(args.state_file):
        with open(args.state_file, 'r') as f:
            processed_stocks = set(line.strip() for line in f if line.strip())
    print(f"Resuming... {len(processed_stocks)} stocks already processed.")
    
    # DB Setup
    con = duckdb.connect(args.db_path, read_only=True)
    all_cols = get_column_names(con, args.table_name)
    
    col_map = {
        'date': all_cols[0],
        'code': all_cols[2],
        'time': all_cols[4]
    }
    feature_cols = all_cols[5:33]
    
    means, stds = get_global_stats(con, args.table_name, feature_cols)
    
    all_stocks = [s[0] for s in con.sql(f"SELECT DISTINCT \"{col_map['code']}\" FROM {args.table_name}").fetchall()]
    stocks_to_process = [s for s in all_stocks if s not in processed_stocks]
    print(f"Total: {len(all_stocks)}, To Process: {len(stocks_to_process)}")
    con.close()
    
    # Model
    model = MaskedAutoEncoder(input_dim=len(feature_cols), embedding_dim=128)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Clean up old temp files
    print(f"Cleaning up temp dir: {temp_dir}")
    for f in os.listdir(temp_dir):
        fp = os.path.join(temp_dir, f)
        if os.path.isfile(fp):
            try: os.remove(fp)
            except: pass
            
    # --- Parallel Processing Loop ---
    
    # --- Manual Task Submission Loop ---
    
    buffer_meta = []
    MAX_TEMP_SIZE = 500 * 1024 * 1024 * 1024 
    buffer_limit = args.batch_size * 5 
    
    # Stocks to process queue
    stock_queue = list(stocks_to_process)
    active_futures = {} # {future: stock_code}
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        pbar = tqdm(total=len(stocks_to_process), desc="Processing")
        
        # Initial submission (up to num_workers * 2)
        initial_batch = min(len(stock_queue), args.num_workers * 2)
        for _ in range(initial_batch):
            stock = stock_queue.pop(0)
            fut = executor.submit(process_stock_data, stock, args.db_path, args.table_name, feature_cols, means, stds, args.seq_len, col_map, temp_dir)
            active_futures[fut] = stock
            
        while active_futures:
            # Fallback if torch.concurrent not available or standard usage
            from concurrent.futures import wait, FIRST_COMPLETED
            dones, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in dones:
                stock_code = active_futures.pop(future)
                pbar.update(1)
                
                try:
                    res = future.result()
                    if res is None: continue
                    if len(res) == 2 and isinstance(res[1], Exception):
                        tqdm.write(f"Error: {res[1]}")
                        continue
                        
                    _, meta_path, seq_file_paths = res
                    
                    # Consume Logic 
                    try:
                        # 1. Meta Load
                        meta = pd.read_parquet(meta_path)
                        os.remove(meta_path)
                        
                        # 2. Sequence Load & Inference
                        embeddings_list = []
                        for seq_path in seq_file_paths:
                            chunk_seqs = np.load(seq_path)
                            os.remove(seq_path)
                            
                            with torch.no_grad():
                                for i in range(0, len(chunk_seqs), args.batch_size):
                                    batch = torch.from_numpy(chunk_seqs[i : i + args.batch_size]).to(device)
                                    out = model(batch)
                                    if isinstance(out, tuple): out = out[1] if len(out) == 3 else out[1]
                                    embeddings_list.append(out.cpu().numpy())
                            
                            if args.device == 'tpu':
                                xm.mark_step() # TPU 실행 트리거
                                
                            del chunk_seqs
                            
                        # 3. Merge
                        stock_embeddings = np.concatenate(embeddings_list, axis=0)
                        if len(stock_embeddings) != len(meta):
                            min_len = min(len(meta), len(stock_embeddings))
                            meta = meta.iloc[:min_len]
                            stock_embeddings = stock_embeddings[:min_len]
                        
                        meta['embedding'] = list(stock_embeddings)
                        buffer_meta.append(meta)
                        
                        # Check Buffer & Save
                        current_rows = sum(len(df) for df in buffer_meta)
                        if current_rows >= buffer_limit:
                            all_meta = pd.concat(buffer_meta, ignore_index=True)
                            save_chunk(all_meta, args.output_dir)
                            # tqdm.write(f"Saved chunk with {len(all_meta)} rows.")
                            
                            buffer_meta = []
                            gc.collect()
                            
                        # Save state immediately for this stock
                        with open(args.state_file, 'a') as f:
                            f.write(f"{stock_code}\n")

                    except Exception as e:
                        tqdm.write(f"Error processing {stock_code}: {e}")
                
                except Exception as e:
                    tqdm.write(f"Worker Error {stock_code}: {e}")

            # Check Disk Space before submitting new tasks
            paused = False
            try:
                total_size = sum(os.path.getsize(os.path.join(temp_dir, f)) for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f)))
                if total_size > MAX_TEMP_SIZE:
                    paused = True
                    tqdm.write(f"WARNING: Temp dir size {total_size / (1024**3):.2f} GB. Pausing submissions...")
            except: pass
            
            # Submit new tasks if queue not empty and slots available AND NOT PAUSED
            if not paused:
                while stock_queue and len(active_futures) < args.num_workers * 2:
                    stock = stock_queue.pop(0)
                    fut = executor.submit(process_stock_data, stock, args.db_path, args.table_name, feature_cols, means, stds, args.seq_len, col_map, temp_dir)
                    active_futures[fut] = stock
            else:
                 # If paused and no active futures, we are stuck (orphaned files from *current* run?)
                 # But we just cleaned at startup.
                 # This would only happen if the currently running N workers produced > 500GB.
                 # 500GB / 16 workers = 31GB per worker. Unlikely for one stock.
                 pass

        # Final Flush
        if buffer_meta:
            all_meta = pd.concat(buffer_meta, ignore_index=True)
            save_chunk(all_meta, args.output_dir)
            
            with open(args.state_file, 'a') as f:
                for s in stock_batch_completed:
                    f.write(f"{s}\n")
                    
    # Remove temp dir
    try:
        os.rmdir(temp_dir)
    except: pass

    print(f"Done. Files saved in {args.output_dir}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
