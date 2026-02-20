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
    parser.add_argument('--temp_dir', type=str, default=None, help='Temp dir for intermediate files (default: output_dir/temp). Use local disk on Colab!')
    
    args = parser.parse_args()
    
    # Init
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = args.temp_dir if args.temp_dir else os.path.join(args.output_dir, "temp")
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
    checkpoint = torch.load(args.model_path, map_location='cpu')
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

    # ── 자동 병합: embedding 생성 완료 후 DuckDB 데이터를 parquet에 병합 ──
    print("\n" + "=" * 60)
    print("[AUTO-MERGE] Parquet 생성 완료 → DuckDB 데이터 병합 시작...")
    print("=" * 60)
    merge_main(
        parquet_dir=args.output_dir,
        db_path=args.db_path,
        table_name=args.table_name,
        output_dir=args.output_dir,  # 같은 폴더에 덮어쓰기
        workers=args.num_workers,
    )


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --mode merge: 병합만 단독 실행
    if '--mode' in sys.argv:
        idx = sys.argv.index('--mode')
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == 'merge':
            merge_main()
            sys.exit(0)

    main()


# =============================================================================
# DuckDB → Parquet 병합 (merge mode)
#
# 사용법:
#   python scripts/pre/parquet/generate_embeddings_v3_parallel.py --mode merge \
#     --parquet_dir "C:/Users/user/Workspace/datasets@20260117/embeddings_v2" \
#     --db_path    "C:/Users/user/Workspace/datasets@20260117/datasets_raw_09_11.duckdb" \
#     --output_dir "C:/Users/user/Workspace/datasets@20260117/embeddings_v3" \
#     --workers 4
#
# 현재 Parquet 컬럼: date, time, code, embedding
# DuckDB 컬럼:      날짜, 번호, 종목코드, 종목명, 시간, 종목명_scalar, 시간_sin/cos/scalar,
#                   등락률, 누적거래대금, 거래회전율, 체결강도, 매도/매수대기금액1~10
# Join key: (date, code, time)
# =============================================================================

import logging as _logging
import glob as _glob
from pathlib import Path as _Path
from concurrent.futures import ProcessPoolExecutor as _PPE, as_completed as _as_completed

_MERGE_DONE_SUFFIX = '.merged'


def _merge_one_file(args_tuple):
    """단일 parquet 파일에 DuckDB 데이터를 병합 (ProcessPoolExecutor용)."""
    import time as _time
    import os as _os
    import pandas as _pd
    import duckdb as _duckdb
    from pathlib import Path as _P

    parquet_path, db_path, table_name, output_dir = args_tuple
    file_name = _os.path.basename(parquet_path)
    out_path = _os.path.join(output_dir, file_name)
    done_marker = out_path + _MERGE_DONE_SUFFIX

    if _os.path.exists(done_marker):
        return ('skipped', parquet_path, 0)

    t0 = _time.time()
    try:
        pq_df = _pd.read_parquet(parquet_path)
        if pq_df.empty:
            return ('empty', parquet_path, 0)

        dates_str = ', '.join(f"'{d}'" for d in pq_df['date'].unique())
        codes_str = ', '.join(f"'{c}'" for c in pq_df['code'].unique())

        query = f"""
            SELECT
                날짜 AS date, 종목코드 AS code, 시간 AS time,
                번호, 종목명, 종목명_scalar,
                시간_sin, 시간_cos, 시간_scalar,
                등락률, 누적거래대금, 거래회전율, 체결강도,
                매도대기금액1, 매도대기금액2, 매도대기금액3, 매도대기금액4, 매도대기금액5,
                매도대기금액6, 매도대기금액7, 매도대기금액8, 매도대기금액9, 매도대기금액10,
                매수대기금액1, 매수대기금액2, 매수대기금액3, 매수대기금액4, 매수대기금액5,
                매수대기금액6, 매수대기금액7, 매수대기금액8, 매수대기금액9, 매수대기금액10
            FROM {table_name}
            WHERE 날짜 IN ({dates_str}) AND 종목코드 IN ({codes_str})
        """
        conn = _duckdb.connect(database=db_path, read_only=True)
        db_df = conn.execute(query).fetchdf()
        conn.close()

        if db_df.empty:
            return ('no_db_data', parquet_path, 0)

        merged = pq_df.merge(db_df, on=['date', 'code', 'time'], how='left')
        matched = merged['등락률'].notna().sum()
        total = len(merged)

        _os.makedirs(output_dir, exist_ok=True)
        merged.to_parquet(out_path, index=False, compression='snappy')
        _P(done_marker).touch()

        return ('ok', parquet_path, _time.time() - t0, matched, total)

    except Exception as e:
        return ('error', parquet_path, str(e))


def merge_main(*, parquet_dir=None, db_path=None, table_name='datasets',
               output_dir=None, workers=4, overwrite=False):
    """
    DuckDB 데이터를 Parquet 파일에 병합.
    
    직접 인자를 전달하거나, 인자 없이 호출하면 argparse로 CLI에서 읽음.
    """
    import time as _time, os as _os, sys as _sys

    _logging.basicConfig(
        level=_logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            _logging.StreamHandler(_sys.stdout),
            _logging.FileHandler('merge_embeddings.log', encoding='utf-8'),
        ],
        force=True,
    )
    log = _logging.getLogger('merge')

    # CLI fallback: 인자가 주어지지 않은 경우 argparse 사용
    if parquet_dir is None or db_path is None:
        import argparse as _ap
        parser = _ap.ArgumentParser(description='DuckDB → Parquet 병합')
        parser.add_argument('--parquet_dir', required=True,  help='원본 parquet 폴더')
        parser.add_argument('--db_path',     required=True,  help='DuckDB 파일 경로')
        parser.add_argument('--table_name',  default='datasets')
        parser.add_argument('--output_dir',  default=None,   help='출력 폴더 (기본=parquet_dir)')
        parser.add_argument('--workers',     type=int, default=4)
        parser.add_argument('--overwrite',   action='store_true')
        parser.add_argument('--mode',        default='merge')
        args = parser.parse_args()
        parquet_dir = args.parquet_dir
        db_path     = args.db_path
        table_name  = args.table_name
        output_dir  = args.output_dir
        workers     = args.workers
        overwrite   = args.overwrite

    output_dir = output_dir or parquet_dir

    if not _os.path.exists(db_path):
        log.error(f"DuckDB not found: {db_path}"); return

    pq_files = sorted(_glob.glob(_os.path.join(parquet_dir, '*.parquet')))
    if not pq_files:
        log.error(f"No parquet files in {parquet_dir}"); return

    log.info(f"파일 수: {len(pq_files)}, 출력: {output_dir}, Workers: {workers}")

    if not overwrite:
        pending = [
            f for f in pq_files
            if not _os.path.exists(_os.path.join(output_dir, _os.path.basename(f)) + _MERGE_DONE_SUFFIX)
        ]
        log.info(f"완료: {len(pq_files)-len(pending)}개 (스킵), 처리 대상: {len(pending)}개")
        pq_files = pending

    if not pq_files:
        log.info("처리할 파일 없음. 완료!"); return

    tasks = [(f, db_path, table_name, output_dir) for f in pq_files]
    ok_count = error_count = total_rows = 0
    t_start = _time.time()

    with _PPE(max_workers=workers) as executor:
        futures = {executor.submit(_merge_one_file, t): t[0] for t in tasks}
        for i, future in enumerate(_as_completed(futures), 1):
            result = future.result()
            status, path = result[0], result[1]
            fname = _os.path.basename(path)

            if status == 'ok':
                _, _, elapsed, matched, total = result
                ok_count += 1; total_rows += total
                eta = (len(tasks) - i) * (_time.time() - t_start) / i
                log.info(f"[{i}/{len(tasks)}] OK {fname} ({matched}/{total} matched, {elapsed:.1f}s) ETA:{eta/60:.1f}m")
            elif status == 'skipped':
                ok_count += 1
            elif status == 'error':
                error_count += 1
                log.error(f"[{i}/{len(tasks)}] ERROR {fname}: {result[2]}")
            else:
                log.warning(f"[{i}/{len(tasks)}] {status.upper()} {fname}")

    elapsed_total = _time.time() - t_start
    log.info(f"완료: {ok_count}개 성공, {error_count}개 실패, 총 {total_rows:,}행, {elapsed_total/60:.1f}분")
