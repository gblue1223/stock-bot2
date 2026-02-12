#!/usr/bin/env python3
"""
Colab 최적화 임베딩 생성 스크립트 (Single Process, No Temp Files)

기존 generate_embeddings_v3_parallel.py는 Producer-Consumer 패턴으로
temp 파일을 대량 생성하여 로컬 PC에서 효과적이지만,
Colab에서는 디스크/메모리 제한으로 인해 성능이 크게 저하됩니다.

이 스크립트는:
1. 단일 프로세스로 동작 (멀티프로세스 오버헤드 제거)
2. Temp 파일 없이 메모리 내에서 직접 GPU 인퍼런스 수행
3. 종목별로 순차 처리하여 메모리 사용량 최소화
4. Resume 기능 유지
"""

import os
import sys
import gc
import argparse
import duckdb
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder

# 전역 카운터
file_counter = 0

def save_chunk(df, output_dir):
    global file_counter
    filename = f"embeddings_{file_counter:04d}.parquet"
    path = os.path.join(output_dir, filename)
    df.to_parquet(path, engine='pyarrow', index=False)
    file_counter += 1

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

def process_stock_inline(stock_code, con, table_name, feature_cols, means, stds, seq_len, col_map, model, device, batch_size):
    """
    단일 종목을 메모리 내에서 처리: DB 읽기 → 정규화 → GPU 인퍼런스 → 결과 반환
    Temp 파일을 사용하지 않음.
    """
    select_cols = [f"\"{col_map['date']}\"", f"\"{col_map['time']}\"", f"\"{col_map['code']}\""] + \
                  [f"\"{c}\"" for c in feature_cols]
    select_clause = ", ".join(select_cols)
    
    query = f"SELECT {select_clause} FROM {table_name} WHERE \"{col_map['code']}\" = '{stock_code}' ORDER BY \"{col_map['date']}\", \"{col_map['time']}\""
    df = con.sql(query).df()
    
    if len(df) < seq_len:
        return None
    
    # Meta Data
    meta = df.iloc[seq_len-1:][[col_map['date'], col_map['time'], col_map['code']]].reset_index(drop=True)
    meta.columns = ['date', 'time', 'code']
    
    # Features & Norm
    feats = df[feature_cols].values.astype(np.float32)
    del df
    feats = (feats - means) / stds
    
    # 시퀀스 생성 & GPU 인퍼런스 (청크 단위, 파일 저장 없이)
    num_samples = len(feats) - seq_len + 1
    chunk_size = 10000
    
    embeddings_list = []
    
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        slice_end = end + seq_len - 1
        if slice_end > len(feats):
            slice_end = len(feats)
        
        sub_feats = feats[i:slice_end]
        sub_num = len(sub_feats) - seq_len + 1
        if sub_num <= 0:
            continue
        
        # Sliding window (메모리 효율적)
        shape = (sub_num, seq_len, sub_feats.shape[1])
        strides_val = (sub_feats.strides[0], sub_feats.strides[0], sub_feats.strides[1])
        sub_seqs = np.lib.stride_tricks.as_strided(sub_feats, shape=shape, strides=strides_val, writeable=False)
        sub_seqs = np.ascontiguousarray(sub_seqs)
        
        # GPU Inference (배치 단위)
        with torch.no_grad():
            for j in range(0, len(sub_seqs), batch_size):
                batch = torch.from_numpy(sub_seqs[j:j+batch_size]).to(device)
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[1] if len(out) == 3 else out[1]
                embeddings_list.append(out.cpu().numpy())
        
        del sub_seqs
    
    del feats
    
    if not embeddings_list:
        return None
    
    stock_embeddings = np.concatenate(embeddings_list, axis=0)
    del embeddings_list
    
    # 길이 맞추기
    if len(stock_embeddings) != len(meta):
        min_len = min(len(meta), len(stock_embeddings))
        meta = meta.iloc[:min_len]
        stock_embeddings = stock_embeddings[:min_len]
    
    meta['embedding'] = list(stock_embeddings)
    return meta


def main():
    parser = argparse.ArgumentParser(description='Generate Embeddings (Colab Optimized - No Temp Files)')
    parser.add_argument('--db_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--table_name', type=str, default='datasets')
    parser.add_argument('--output_dir', type=str, default='data/embeddings')
    parser.add_argument('--state_file', type=str, default='processed_stocks.txt')
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--buffer_size', type=int, default=20000, help='Number of rows before flushing to parquet')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    if args.device == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except ImportError:
            print("torch_xla not found, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # File counter
    existing_files = [f for f in os.listdir(args.output_dir) if f.startswith('embeddings_') and f.endswith('.parquet')]
    max_idx = -1
    for f in existing_files:
        try:
            idx = int(f.split('_')[1].split('.')[0])
            if idx > max_idx: max_idx = idx
        except: pass
    
    global file_counter
    file_counter = max_idx + 1
    print(f"Resuming file counter from {file_counter} (Found {len(existing_files)} existing files)")
    
    # Processed stocks
    processed_stocks = set()
    if os.path.exists(args.state_file):
        with open(args.state_file, 'r') as f:
            processed_stocks = set(line.strip() for line in f if line.strip())
    print(f"Resuming... {len(processed_stocks)} stocks already processed.")
    
    # DB
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
    
    # Model
    model = MaskedAutoEncoder(input_dim=len(feature_cols), embedding_dim=128)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Process
    buffer_meta = []
    pbar = tqdm(stocks_to_process, desc="Processing")
    
    for stock_code in pbar:
        pbar.set_postfix(stock=stock_code)
        
        try:
            result = process_stock_inline(
                stock_code, con, args.table_name, feature_cols, 
                means, stds, args.seq_len, col_map, model, device, args.batch_size
            )
            
            if result is not None:
                buffer_meta.append(result)
            
            # Save state
            with open(args.state_file, 'a') as f:
                f.write(f"{stock_code}\n")
            
            # Flush buffer
            current_rows = sum(len(df) for df in buffer_meta)
            if current_rows >= args.buffer_size:
                all_meta = pd.concat(buffer_meta, ignore_index=True)
                save_chunk(all_meta, args.output_dir)
                tqdm.write(f"Saved chunk {file_counter-1}: {len(all_meta)} rows")
                buffer_meta = []
                gc.collect()
                
        except Exception as e:
            tqdm.write(f"Error [{stock_code}]: {e}")
            continue
    
    # Final flush
    if buffer_meta:
        all_meta = pd.concat(buffer_meta, ignore_index=True)
        save_chunk(all_meta, args.output_dir)
        tqdm.write(f"Saved final chunk: {len(all_meta)} rows")
    
    con.close()
    print(f"Done. Files saved in {args.output_dir}")

if __name__ == "__main__":
    main()
