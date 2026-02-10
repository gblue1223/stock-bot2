#!/usr/bin/env python3
"""
데이터 임베딩 일괄 생성 (Parallel & Resume)

1. Resume 기능: processed_stocks.txt에 완료된 종목 기록 및 로드
2. 병렬 처리: Producer-Consumer 패턴 (ProcessPoolExecutor)
   - Worker: DB Fetch -> Normalize -> Tensor 변환
   - Main: GPU Inference -> Save

수정 내역:
- DuckDB Concurrency Issue 해결을 위해 read_only=True 명시 및 설정 변경
- Worker 프로세스에서 DB 연결 시 설정 추가 ('duckdb.connect(..., config={"access_mode": "READ_ONLY"})')
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
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder

# --- Worker Function ---
def process_stock_data(stock_code, db_path, table_name, feature_cols, means, stds, seq_len, col_map):
    """
    Worker Process에서 실행되는 함수
    """
    try:
        # 각 프로세스마다 별도 DB 연결 (Read Only)
        # config dictionary를 사용하여 명시적으로 읽기 전용 설정
        con = duckdb.connect(db_path, read_only=True, config={'access_mode': 'READ_ONLY'})
        
        # 쿼리 구성
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
        
        # 시퀀스 데이터 생성
        num_samples = len(feats) - seq_len + 1
        
        shape = (num_samples, seq_len, feats.shape[1])
        strides = (feats.strides[0], feats.strides[0], feats.strides[1])
        sequences = np.lib.stride_tricks.as_strided(feats, shape=shape, strides=strides, writeable=False)
        
        # copy to make it contiguous and independent
        return (stock_code, meta, np.ascontiguousarray(sequences))
        
    except Exception as e:
        # 에러 메시지에 stock_code 포함
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
    parser = argparse.ArgumentParser(description='Generate Embeddings (Parallel)')
    parser.add_argument('--db_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--table_name', type=str, default='datasets')
    parser.add_argument('--output_dir', type=str, default='data/embeddings')
    parser.add_argument('--state_file', type=str, default='processed_stocks.txt')
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Init
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Workers: {args.num_workers}")
    
    # Load Completed Stocks
    processed_stocks = set()
    if os.path.exists(args.state_file):
        with open(args.state_file, 'r') as f:
            processed_stocks = set(line.strip() for line in f if line.strip())
    print(f"Resuming... {len(processed_stocks)} stocks already processed.")
    
    # DB Setup (Main Process)
    # 메인 프로세스도 read_only로 염
    con = duckdb.connect(args.db_path, read_only=True)
    all_cols = get_column_names(con, args.table_name)
    
    # Column Mapping
    col_map = {
        'date': all_cols[0],
        'code': all_cols[2],
        'time': all_cols[4]
    }
    feature_cols = all_cols[5:33]
    
    # Stats
    means, stds = get_global_stats(con, args.table_name, feature_cols)
    
    # Stock List
    all_stocks = [s[0] for s in con.sql(f"SELECT DISTINCT \"{col_map['code']}\" FROM {args.table_name}").fetchall()]
    stocks_to_process = [s for s in all_stocks if s not in processed_stocks]
    print(f"Total: {len(all_stocks)}, To Process: {len(stocks_to_process)}")
    
    # 메인 프로세스의 DB 연결 종료 (Worker들과 충돌 방지 위해)
    con.close()
    
    # Model
    model = MaskedAutoEncoder(input_dim=len(feature_cols), embedding_dim=128)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # --- Parallel Processing Loop ---
    
    buffer_meta = []
    buffer_sequences = []
    buffer_limit = args.batch_size * 20 
    
    total_saved = 0
    
    # max_workers=1로 테스트 해보는 것도 방법 (디버깅용), 하지만 병렬성을 위해 args.num_workers 사용
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_stock_data, 
                stock, 
                args.db_path, 
                args.table_name, 
                feature_cols, 
                means, stds, 
                args.seq_len, 
                col_map
            ): stock for stock in stocks_to_process
        }
        
        pbar = tqdm(total=len(stocks_to_process), desc="Processing")
        
        stock_batch_completed = [] 
        
        for future in as_completed(futures):
            res = future.result()
            
            pbar.update(1)
            
            if res is None: continue 
            if len(res) == 2 and isinstance(res[1], Exception):
                # 에러 발생 시 출력하고 계속 진행 (파일 잠금 등 일시적 오류일 수 있음)
                # 단, 너무 많이 발생하면 문제.
                tqdm.write(f"Error: {res[1]}")
                continue
                
            stock_code, meta, sequences = res
            
            buffer_meta.append(meta)
            buffer_sequences.append(sequences)
            stock_batch_completed.append(stock_code)
            
            current_samples = sum(len(s) for s in buffer_sequences)
            
            if current_samples >= buffer_limit:
                all_seqs = np.concatenate(buffer_sequences, axis=0)
                all_meta = pd.concat(buffer_meta, ignore_index=True)
                
                embeddings_list = []
                dataset_size = len(all_seqs)
                
                with torch.no_grad():
                    for i in range(0, dataset_size, args.batch_size):
                        batch = torch.from_numpy(all_seqs[i : i + args.batch_size]).to(device)
                        out = model(batch)
                        if isinstance(out, tuple): out = out[1] if len(out) == 3 else out[1]
                        embeddings_list.append(out.cpu().numpy())
                
                embeddings_array = np.concatenate(embeddings_list, axis=0)
                all_meta['embedding'] = list(embeddings_array)
                
                save_chunk(all_meta, args.output_dir)
                
                with open(args.state_file, 'a') as f:
                    for s in stock_batch_completed:
                        f.write(f"{s}\n")
                        
                buffer_meta = []
                buffer_sequences = []
                stock_batch_completed = []
                gc.collect()

        if buffer_sequences:
            all_seqs = np.concatenate(buffer_sequences, axis=0)
            all_meta = pd.concat(buffer_meta, ignore_index=True)
            
            embeddings_list = []
            for i in range(0, len(all_seqs), args.batch_size):
                batch = torch.from_numpy(all_seqs[i : i + args.batch_size]).to(device)
                out = model(batch)
                if isinstance(out, tuple): out = out[1] if len(out) == 3 else out[1]
                embeddings_list.append(out.cpu().numpy())
                
            embeddings_array = np.concatenate(embeddings_list, axis=0)
            all_meta['embedding'] = list(embeddings_array)
            save_chunk(all_meta, args.output_dir)
            
            with open(args.state_file, 'a') as f:
                for s in stock_batch_completed:
                    f.write(f"{s}\n")

    print(f"Done. Files saved in {args.output_dir}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
