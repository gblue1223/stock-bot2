import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import duckdb
from torch.utils.tensorboard import SummaryWriter

from .data import load_real_dataframe
from .models import CNNLSTMAttn, ModelConfig


def get_requested_features() -> list[str]:
    return [
        "날짜", "등락률", "누적거래대금", "거래회전율", "체결강도",
        "매도호가수량1", "매도호가수량2", "매도호가수량3", "매도호가수량4", "매도호가수량5",
        "매도호가수량6", "매도호가수량7", "매도호가수량8", "매도호가수량9", "매도호가수량10",
        "매수호가수량1", "매수호가수량2", "매수호가수량3", "매수호가수량4", "매수호가수량5",
        "매수호가수량6", "매수호가수량7", "매수호가수량8", "매수호가수량9", "매수호가수량10",
        "매도호가총잔량", "매수호가총잔량",
        "매도거래원수량1", "매도거래원수량2", "매도거래원수량3", "매도거래원수량4", "매도거래원수량5",
        "매도거래원별증감1", "매도거래원별증감2", "매도거래원별증감3", "매도거래원별증감4", "매도거래원별증감5",
        "매수거래원수량1", "매수거래원수량2", "매수거래원수량3", "매수거래원수량4", "매수거래원수량5",
        "매수거래원별증감1", "매수거래원별증감2", "매수거래원별증감3", "매수거래원별증감4", "매수거래원별증감5",
        "매도거래원1_scalar", "매도거래원2_scalar", "매도거래원3_scalar", "매도거래원4_scalar", "매도거래원5_scalar",
        "매수거래원1_scalar", "매수거래원2_scalar", "매수거래원3_scalar", "매수거래원4_scalar", "매수거래원5_scalar",
        "종목명_scalar", "시간_scalar",
    ]


def convert_date_to_month_inplace(df: pd.DataFrame) -> None:
    if "날짜" in df.columns:
        month_series = df["날짜"].astype(str).str[4:6]
        df["날짜"] = pd.to_numeric(month_series, errors="coerce").fillna(0).astype(np.int32)


def list_duckdb_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = [os.path.join(path, n) for n in os.listdir(path) if n.lower().endswith((".duckdb", ".db"))]
        files.sort()
        if not files:
            raise ValueError(f"No DuckDB files (*.duckdb|*.db) found in directory: {path}")
        return files
    return [path]


def save_checkpoint(path: str,
                    state_dict: dict,
                    config_dict: dict,
                    feature_names: list[str],
                    target_col: str,
                    horizon: int,
                    aux_task: str,
                    extra: dict | None = None) -> None:
    payload = {
        "state_dict": state_dict,
        "config": config_dict,
        "feature_names": list(feature_names),
        "target_col": target_col,
        "horizon": horizon,
        "aux_task": aux_task,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint file in the given directory"""
    if not os.path.isdir(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            if os.path.isfile(filepath):
                # Get modification time
                mtime = os.path.getmtime(filepath)
                checkpoint_files.append((mtime, filepath, filename))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(reverse=True)
    latest_file = checkpoint_files[0][1]
    latest_name = checkpoint_files[0][2]
    
    print(f"Found {len(checkpoint_files)} checkpoint(s) in {checkpoint_dir}")
    print(f"Latest checkpoint: {latest_name}")
    
    return latest_file


def load_checkpoint(path: str, device: str) -> dict:
    """Load checkpoint and return its contents"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    print(f"Loaded checkpoint from: {path}")
    
    # Print all available keys for debugging
    print(f"  - Available keys: {list(checkpoint.keys())}")
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    if "chunk" in checkpoint:
        print(f"  - Chunk: {checkpoint['chunk']}")
    if "trained_chunks" in checkpoint:
        print(f"  - Trained Chunks: {checkpoint['trained_chunks']}")
    if "val_loss" in checkpoint:
        print(f"  - Val Loss: {checkpoint['val_loss']:.6f}")
    if "file" in checkpoint:
        print(f"  - File: {checkpoint['file']}")
    if "last_no" in checkpoint:
        print(f"  - Last No: {checkpoint['last_no']}")
    
    return checkpoint


def _fmt_int(n: int) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def log_startup_summary(
    mode: str,
    device: str,
    db_path: str,
    table: str,
    files: list[str] | None,
    per_file_rows: list[int] | None,
    total_rows: int | None,
    available_features: list[str] | None,
    target_col: str | None,
    seq_len: int,
    horizon: int,
    batch_size: int,
    epochs: int,
    chunk_size: int | None,
    checkpoint_every_chunks: int | None,
    checkpoint_every_epochs: int | None,
    checkpoint_epochs: str | None,
    progress_every: int | None,
) -> None:
    print("================= Training Startup Summary =================")
    print(f"mode           : {mode}")
    print(f"device         : {device}")
    print(f"db_path        : {db_path}")
    print(f"table          : {table}")
    if files is not None:
        print(f"files          : {len(files)} file(s)")
        if per_file_rows is not None:
            for fp, cnt in zip(files, per_file_rows):
                print(f"  - {os.path.basename(fp)} rows={_fmt_int(cnt)}")
    if total_rows is not None:
        print(f"total rows     : {_fmt_int(total_rows)}")
    if available_features is not None:
        print(f"features       : {len(available_features)} -> {', '.join(available_features)}")
    if target_col is not None:
        print(f"target_col     : {target_col}")
    print(f"seq_len        : {seq_len}")
    print(f"horizon        : {horizon}")
    print(f"batch_size     : {batch_size}")
    print(f"epochs         : {epochs}")
    if chunk_size is not None:
        print(f"chunk_size     : {chunk_size}")
    print(f"ckpt-every-chunks : {checkpoint_every_chunks}")
    print(f"ckpt-every-epochs  : {checkpoint_every_epochs}")
    print(f"ckpt-epochs        : {checkpoint_epochs}")
    if progress_every is not None:
        print(f"progress-every : {progress_every}")
    print("============================================================")


def train(
    db_path: str,
    output_dir: str,
    table: str = "datasets",
    code: Optional[str] = None,
    date: Optional[str] = None,
    seq_len: int = 60,
    horizon: int = 10,
    target_col: Optional[str] = None,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-4,
    device: Optional[str] = None,
    aux_task: str = "regression",  # one of {"regression", "direction", "volatility"}
    checkpoint_every_chunks: Optional[int] = None,  # save checkpoint every N chunks
    checkpoint_every_epochs: Optional[int] = None,
    checkpoint_epochs: Optional[str] = None,  # comma-separated list, e.g., "5,10,20"
    chunk_size: int = 1000,
    progress_every: int = 10,
    resume_from: Optional[str] = None,  # path to checkpoint to resume from
    enable_tensorboard: bool = True,  # enable TensorBoard logging
    loss_type: str = "mse",          # regression loss: {"mse", "huber"}
    huber_delta: float = 1.0,         # Huber delta
    weight_decay: float = 1e-4,       # AdamW weight decay
):
    """
    SQLite에 저장된 시계열 실수(REAL) 컬럼 데이터로 CNN+LSTM+어텐션 모델을 지도학습합니다.

    Parameters
    - db_path (str): 데이터셋이 저장된 SQLite DB 파일 경로.
    - output_dir (str): 체크포인트(`model.pt`)와 학습 메타(`config.json`)를 저장할 디렉터리.
    - table (str, default="datasets"): DB에서 읽어올 테이블명.
    - code (Optional[str], default=None): 특정 종목 코드로 데이터 필터링(예: "005930"). None이면 전체(환경/데이터 로직에 따름).
    - date (Optional[str], default=None): 특정 일자(YYYYMMDD)로 데이터 필터링. None이면 전체 사용.
    - seq_len (int, default=60): 모델 입력으로 사용할 시퀀스(윈도우) 길이.
    - horizon (int, default=10): 예측 시점까지의 간격(몇 스텝 뒤를 예측할지). direction/volatility에서도 사용.
    - target_col (Optional[str], default=None): 예측 대상 컬럼명. None일 경우 첫 번째 feature를 사용합니다.
    - batch_size (int, default=128): 학습 배치 크기.
    - epochs (int, default=20): 최대 학습 에폭 수(얼리 스탑 적용).
    - lr (float, default=1e-4): AdamW 옵티마이저의 학습률.
    - device (Optional[str], default=None): "cuda"/"cpu" 등 장치 지정. None이면 가능 시 CUDA 사용, 아니면 CPU.
    - aux_task (str, default="regression"): 보조 학습 목표. {regression, direction, volatility}
    - checkpoint_every_chunks (Optional[int], default=None): N 청크마다 체크포인트 저장 (예: 100). 미지정 시 비활성화
    - checkpoint_every (Optional[int], default=None): N 에폭마다 체크포인트 저장 (예: 5). 미지정 시 비활성화
    - checkpoint_epochs (Optional[str], default=None): 지정 에폭에서 체크포인트 저장 (쉼표 구분, 예: '5,10,20')
    - chunk_size (int, default=1000): DuckDB에서 한 번에 읽을 레코드 수. 기본값: 1000. 0 또는 음수면 전체 로드
    - progress_every (int, default=10): 청크 진행 로그 출력 주기(청크 단위). 0이면 비활성화
    - resume_from (Optional[str], default=None): 재시작할 체크포인트 파일 경로 또는 폴더 경로. 폴더 지정 시 가장 최신 체크포인트 자동 선택. None이면 처음부터 시작
    - enable_tensorboard (bool, default=True): TensorBoard 로깅 활성화 여부
    - loss_type (str, default="mse"): 회귀 손실 함수 선택: mse 또는 huber
    - huber_delta (float, default=1.0): Huber 손실의 delta (허용 오차)
    - weight_decay (float, default=1e-4): AdamW weight decay (L2 정규화 강도)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint directory for periodic/specified-epoch saves
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if (checkpoint_every_epochs is not None or 
        (checkpoint_epochs is not None and checkpoint_epochs.strip() != "") or
        checkpoint_every_chunks is not None):
        os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = None
    if enable_tensorboard:
        log_dir = os.path.join(output_dir, "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"To view logs, run: tensorboard --logdir {log_dir}")

    # Parse checkpoint epochs list
    epoch_save_set = set()
    if checkpoint_epochs:
        for tok in checkpoint_epochs.split(','):
            tok = tok.strip()
            if tok:
                try:
                    epoch_save_set.add(int(tok))
                except ValueError:
                    pass

    # STREAMING TRAINING PATH (chunked reading) ---------------------------------
    if chunk_size is not None and chunk_size > 0:
        # Determine global feature columns on first chunk; others will be aligned to this set/order
        global_features: Optional[list[str]] = None
        tgt_col: Optional[str] = None

        # Initialize model/opt/loss lazily when we know feature dimension
        model: Optional[torch.nn.Module] = None
        opt = None
        if aux_task == "direction":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.HuberLoss(delta=huber_delta) if (loss_type == "huber") else nn.MSELoss()

        # Training loop over epochs and files/chunks
        best_val = float("inf")
        patience = 5
        no_improve = 0
        global_chunk_count = 0  # Track total chunks processed across all epochs
        global_trained_chunk_count = 0  # Track chunks that actually produced sequences/training
        global_processed_rows = 0  # Track total rows processed across all files
        start_epoch = 1
        resume_chunk_count = 0

        # Load checkpoint if resuming
        resume_last_no = None
        if resume_from:
            checkpoint_path = resume_from
            
            # If resume_from is a directory, find the latest checkpoint
            if os.path.isdir(resume_from):
                latest_checkpoint = find_latest_checkpoint(resume_from)
                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint
                    resume_from = latest_checkpoint  # Update resume_from for later use
                    print(f"Auto-selected latest checkpoint from directory: {resume_from}")
                else:
                    print(f"WARNING: No checkpoint files found in directory: {resume_from}")
                    print("Starting from scratch...")
                    checkpoint_path = None
                    resume_from = None
            elif not os.path.exists(resume_from):
                print(f"WARNING: Resume checkpoint not found: {resume_from}")
                print("Starting from scratch...")
                checkpoint_path = None
                resume_from = None
            
            if checkpoint_path:
                try:
                    checkpoint = load_checkpoint(checkpoint_path, device)
                    start_epoch = checkpoint.get("epoch", 1)
                    # Try both keys for backward compatibility
                    resume_chunk_count = checkpoint.get("trained_chunks", checkpoint.get("chunk", 0))
                    global_chunk_count = resume_chunk_count
                    global_trained_chunk_count = resume_chunk_count
                    # Estimate processed rows (chunk_size * chunks)
                    global_processed_rows = resume_chunk_count * chunk_size
                    best_val = checkpoint.get("val_loss", float("inf"))
                    resume_last_no = checkpoint.get("last_no")
                    print(f"=== RESUME INFO ===")
                    print(f"Resuming from epoch {start_epoch}")
                    print(f"Resume trained chunk count: {resume_chunk_count}")
                    print(f"Best validation loss: {best_val:.6f}")
                    if resume_last_no:
                        print(f"Resume last 번호: {resume_last_no}")
                    print(f"===================")
                    resume_file = checkpoint.get("file")
                    if resume_file is not None:
                        print(f"Resume hint -> file: {resume_file}, last_no: {resume_last_no}")
                    else:
                        print("Resume hint -> no keyset info saved; will skip by counting trained chunks")
                except Exception as e:
                    print(f"Failed to load checkpoint: {e}")
                    print("Starting from scratch...")

        duckdb_files = list_duckdb_files(db_path)

        # Startup summary prepass: count rows per file and detect available features superset
        per_file_rows: list[int] = []
        sup_features: set[str] = set()
        total_rows = 0
        has_no_by_file: list[bool] = []
        for fpath in duckdb_files:
            conn = duckdb.connect(fpath)
            try:
                schema = list(conn.execute(f"PRAGMA table_info('{table}')").fetchall())
                col_names = [row[1] for row in schema]
                has_no = ("번호" in col_names)
                has_no_by_file.append(has_no)
                req = get_requested_features()
                avail = [c for c in req if c in col_names]
                sup_features.update(avail)
                where_clauses = []
                params = []
                if code:
                    where_clauses.append('"종목코드" = ?')
                    params.append(code)
                if date:
                    where_clauses.append('"날짜" = ?')
                    params.append(date)
                where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                cnt = conn.execute(f"SELECT COUNT(*) FROM {table}{where_sql}", params).fetchone()[0]
                per_file_rows.append(int(cnt))
                total_rows += int(cnt)
            finally:
                conn.close()

        sup_features_list = sorted(list(sup_features))
        # Tentative target for summary only (real tgt will be finalized when model is built)
        tgt_for_summary = target_col or ("현재가" if "현재가" in sup_features_list else (sup_features_list[0] if sup_features_list else None))
        log_startup_summary(
            mode="streaming",
            device=device,
            db_path=db_path,
            table=table,
            files=duckdb_files,
            per_file_rows=per_file_rows,
            total_rows=total_rows,
            available_features=sup_features_list,
            target_col=tgt_for_summary,
            seq_len=seq_len,
            horizon=horizon,
            batch_size=batch_size,
            epochs=epochs,
            chunk_size=chunk_size,
            checkpoint_every_chunks=checkpoint_every_chunks,
            checkpoint_every_epochs=checkpoint_every_epochs,
            checkpoint_epochs=checkpoint_epochs,
            progress_every=progress_every,
        )

        for epoch in range(start_epoch, epochs + 1):
            tr_loss_epoch_sum = 0.0
            tr_samples = 0
            va_loss_epoch_sum = 0.0
            va_samples = 0
            # For validation metrics aggregation across chunks
            va_preds_all: list[float] = []
            va_targets_all: list[float] = []

            # Determine/ensure loss fn per epoch (already set above)

            # Determine starting file and keyset if resuming
            resume_file = None
            resume_last_no = None
            if resume_from and os.path.exists(resume_from):
                try:
                    ck = torch.load(resume_from, map_location=device)
                    resume_file = ck.get("file")
                    resume_last_no = ck.get("last_no")
                except Exception:
                    resume_file = None
                    resume_last_no = None

            for fpath in duckdb_files:
                # If resuming from a particular file, skip until we reach it
                if resume_file is not None and os.path.basename(fpath) != os.path.basename(resume_file):
                    print(f"[resume-skip-file] skipping file until {os.path.basename(resume_file)}: {os.path.basename(fpath)}")
                    continue
                # Once we hit the resume file, clear the filter for subsequent files
                resume_file = None

                conn = duckdb.connect(fpath)
                try:
                    # Inspect schema to know if "번호" exists for ordering and which requested features are present
                    schema = list(conn.execute(f"PRAGMA table_info('{table}')").fetchall())
                    col_names = [row[1] for row in schema]
                    has_no = ("번호" in col_names)
                    # Build WHERE and COUNT
                    where_clauses = []
                    params = []
                    if code:
                        where_clauses.append('"종목코드" = ?')
                        params.append(code)
                    if date:
                        where_clauses.append('"날짜" = ?')
                        params.append(date)
                    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                    total = conn.execute(f"SELECT COUNT(*) FROM {table}{where_sql}", params).fetchone()[0]

                    # Columns to select: available requested features + optional 번호 for ordering
                    req = get_requested_features()
                    avail_feats = [c for c in req if c in col_names]
                    select_cols = avail_feats.copy()
                    if has_no:
                        select_cols = ["번호", *select_cols]
                    col_sql = ", ".join([f'"{c}"' for c in select_cols])

                    prev_tail_feats = None  # carry-over for sequence continuity between chunks
                    step = max(1, int(chunk_size))
                    processed = 0
                    chunk_idx = 0

                    # Define per-chunk processing to ensure training/checkpoint per fetched chunk
                    def _process_chunk(df_chunk: pd.DataFrame) -> tuple[bool, int | None]:
                        nonlocal prev_tail_feats, global_features, tgt_col, model, opt
                        nonlocal tr_loss_epoch_sum, tr_samples, va_loss_epoch_sum, va_samples
                        nonlocal global_trained_chunk_count

                        if df_chunk.empty:
                            return (False, None)
                        # Transform 날짜 -> month
                        convert_date_to_month_inplace(df_chunk)
                        # Keep only feature columns (exclude 번호)
                        feats = df_chunk[[c for c in avail_feats if c in df_chunk.columns]].copy()
                        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                        # Lazy init model and features
                        if global_features is None:
                            if feats.empty:
                                return (False, None)
                            global_features = list(feats.columns)
                            tgt = target_col or ("현재가" if "현재가" in global_features else global_features[0])
                            if tgt not in global_features:
                                raise ValueError(f"target_col '{tgt}' not in feature columns")
                            tgt_col = tgt
                            cfg = ModelConfig(input_features=len(global_features), seq_len=seq_len)
                            model = CNNLSTMAttn(cfg).to(device)
                            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                opt, mode='min', factor=0.5, patience=3, min_lr=1e-7
                            )
                            
                            # Load model state if resuming (only once when model is first created)
                            if resume_from and os.path.exists(resume_from):
                                try:
                                    checkpoint = load_checkpoint(resume_from, device)
                                    model.load_state_dict(checkpoint["state_dict"])
                                    # Load scheduler state if available
                                    if "scheduler_state" in checkpoint:
                                        scheduler.load_state_dict(checkpoint["scheduler_state"])
                                        print("Model and scheduler state loaded from checkpoint")
                                    else:
                                        print("Model state loaded from checkpoint (no scheduler state)")
                                except Exception as e:
                                    print(f"Failed to load model state: {e}")
                                    print("Using fresh model...")
                        # Align to global feature order
                        feats = feats.reindex(columns=global_features, fill_value=0.0)
                        
                        # IMPORTANT: Do NOT re-normalize here. Datasets are already normalized
                        # by scripts/normalize_datasets.py. Per-chunk normalization causes scale
                        # drift across chunks and within concatenated tails, destabilizing loss.
                        # Use features as-is.
                        feats_all = pd.concat([prev_tail_feats, feats], axis=0, ignore_index=True) if prev_tail_feats is not None else feats

                        feat_vals = feats_all.to_numpy(dtype=np.float32)
                        prices = feats_all[tgt_col].to_numpy(dtype=np.float32)
                        T = len(feats_all)
                        if T < (seq_len + horizon):
                            prev_tail_feats = feats_all.tail(seq_len + horizon - 1)
                            return (False, None)

                        X_list: list[np.ndarray] = []
                        y_list: list[float] = []
                        for i in range(T - seq_len - horizon + 1):
                            xw = feat_vals[i : i + seq_len]
                            if aux_task == "regression":
                                yv = float(prices[i + seq_len + horizon - 1])
                            elif aux_task == "direction":
                                p_now = float(prices[i + seq_len - 1])
                                p_future = float(prices[i + seq_len + horizon - 1])
                                yv = 1.0 if (p_future - p_now) > 0 else 0.0
                            elif aux_task == "volatility":
                                p0 = float(prices[i + seq_len - 1])
                                if p0 <= 0:
                                    yv = 0.0
                                else:
                                    future = prices[i + seq_len : i + seq_len + horizon]
                                    rets = []
                                    prev = p0
                                    for p in future:
                                        rets.append((float(p) / float(prev) - 1.0) if prev != 0 else 0.0)
                                        prev = float(p)
                                    yv = float(np.std(rets, dtype=np.float32)) if len(rets) > 0 else 0.0
                            else:
                                raise ValueError(f"Unknown aux_task: {aux_task}")
                            X_list.append(xw)
                            y_list.append(yv)

                        if not X_list:
                            prev_tail_feats = feats_all.tail(seq_len + horizon - 1)
                            return (False, None)

                        # We produced sequences => a trained chunk
                        global_trained_chunk_count += 1

                        X = np.stack(X_list, axis=0).astype(np.float32)
                        y_arr = np.array(y_list, dtype=np.float32)

                        # split
                        n = len(X)
                        n_val = max(1, int(n * 0.2))
                        n_tr = n - n_val
                        X_tr, y_tr = X[:n_tr], y_arr[:n_tr]
                        X_va, y_va = X[n_tr:], y_arr[n_tr:]

                        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
                        val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
                        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

                        # train
                        model.train()
                        batch_count = 0
                        for xb, yb in train_loader:
                            xb = xb.to(device)
                            yb = yb.to(device)
                            opt.zero_grad()
                            pred = model(xb)
                            loss = loss_fn(pred, yb)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                            opt.step()
                            tr_loss_epoch_sum += loss.item() * len(xb)
                            tr_samples += len(xb)
                            
                            # Log batch-level metrics to TensorBoard
                            if writer and batch_count % 10 == 0:  # Log every 10 batches
                                global_step = (epoch - 1) * 1000 + global_trained_chunk_count * 10 + batch_count
                                writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                                writer.add_scalar('Learning_Rate', opt.param_groups[0]['lr'], global_step)
                            batch_count += 1

                        # val
                        model.eval()
                        with torch.no_grad():
                            for xb, yb in val_loader:
                                xb = xb.to(device)
                                yb = yb.to(device)
                                pred = model(xb)
                                loss = loss_fn(pred, yb)
                                va_loss_epoch_sum += loss.item() * len(xb)
                                va_samples += len(xb)
                                # accumulate for metrics
                                va_preds_all.extend(pred.detach().cpu().float().tolist())
                                va_targets_all.extend(yb.detach().cpu().float().tolist())

                        # update tail
                        prev_tail_feats = feats_all.tail(seq_len + horizon - 1)

                        # return trained True and end 번호 if present
                        end_no = None
                        if "번호" in df_chunk.columns:
                            try:
                                end_no = int(df_chunk["번호"].iloc[-1])
                            except Exception:
                                end_no = None
                        return (True, end_no)

                    if has_no:
                        # Keyset pagination using 번호 to avoid huge OFFSET scans
                        # Seed last_no from resume if provided
                        last_no = int(resume_last_no) if resume_last_no is not None else None
                        # Clear after first use
                        resume_last_no = None
                        
                        # If resuming with keyset pagination, estimate current position
                        estimated_processed = 0
                        started_with_keyset = False
                        if resume_chunk_count > 0 and last_no is not None:
                            started_with_keyset = True
                            # Estimate how many rows we've processed based on last_no
                            try:
                                # Build the count query with proper WHERE/AND logic
                                if where_sql:
                                    count_sql = f"SELECT COUNT(*) FROM {table}{where_sql} AND \"번호\" <= ?"
                                else:
                                    count_sql = f"SELECT COUNT(*) FROM {table} WHERE \"번호\" <= ?"
                                count_params = list(params) + [last_no]
                                estimated_processed = conn.execute(count_sql, count_params).fetchone()[0]
                                processed = estimated_processed
                                print(f"[KEYSET-RESUME] file={os.path.basename(fpath)} resuming from 번호={last_no}, estimated_processed={estimated_processed}/{total}")
                            except Exception as e:
                                print(f"[KEYSET-RESUME] Failed to estimate position: {e}")
                        
                        # If resuming but no keyset info, calculate chunks to skip
                        elif resume_chunk_count > 0:
                            remaining_chunks_to_skip = resume_chunk_count - global_chunk_count
                            if remaining_chunks_to_skip > 0:
                                # Calculate how many chunks this file can contribute to skipping
                                max_chunks_in_file = (total + step - 1) // step  # ceiling division
                                chunks_to_skip_in_file = min(remaining_chunks_to_skip, max_chunks_in_file)
                                
                                if chunks_to_skip_in_file > 0:
                                    # Skip chunks by using OFFSET
                                    skip_offset = chunks_to_skip_in_file * step
                                    processed += skip_offset
                                    chunk_idx += chunks_to_skip_in_file
                                    global_chunk_count += chunks_to_skip_in_file
                                    
                                    print(f"[FAST-SKIP] file={os.path.basename(fpath)} skipped {chunks_to_skip_in_file} chunks (offset={skip_offset}) chunk={global_chunk_count}/{resume_chunk_count}")
                                    
                                    # If we've skipped all chunks in this file, continue to next file
                                    if chunks_to_skip_in_file >= max_chunks_in_file:
                                        continue
                                    
                                    # Start from the offset position
                                    if last_no is None:
                                        # Use OFFSET to jump to the right position
                                        skip_sql = f"SELECT \"번호\" FROM {table}{where_sql} ORDER BY \"번호\" ASC LIMIT 1 OFFSET {skip_offset - 1}"
                                        try:
                                            result = conn.execute(skip_sql, params).fetchone()
                                            if result:
                                                last_no = int(result[0])
                                        except:
                                            pass
                        
                        while True:
                            base_sql = f"SELECT {col_sql} FROM {table}{where_sql} ORDER BY \"번호\" ASC LIMIT {step}"
                            if last_no is None:
                                sql = base_sql
                                params2 = list(params)
                            else:
                                extra = (" AND " if where_sql else " WHERE ") + '"번호" > ?'
                                sql = f"SELECT {col_sql} FROM {table}{where_sql}{extra} ORDER BY \"번호\" ASC LIMIT {step}"
                                params2 = list(params) + [last_no]
                            df_chunk = conn.execute(sql, params2).df()
                            if df_chunk.empty:
                                break
                            processed += len(df_chunk)
                            chunk_idx += 1
                            global_chunk_count += 1

                            # Skip training if resuming and haven't reached resume trained chunk yet
                            # But if we started with keyset info, we're already at the right position
                            if resume_chunk_count > 0 and global_chunk_count <= resume_chunk_count and not started_with_keyset:
                                if progress_every > 0 and (chunk_idx % progress_every == 0 or processed >= total):
                                    print(f"[SKIP-TRAINED] file={os.path.basename(fpath)} epoch={epoch} chunk={global_chunk_count}/{resume_chunk_count} processed={processed}/{total} (skipping already trained chunk)")
                                if "번호" in df_chunk.columns:
                                    last_no = int(df_chunk["번호"].iloc[-1])
                                continue

                            if progress_every > 0 and (chunk_idx % progress_every == 0 or processed >= total):
                                print(f"[TRAINING] file={os.path.basename(fpath)} epoch={epoch} chunk={global_chunk_count} processed={processed}/{total} (training chunk)")

                            # update last_no for next page
                            if "번호" in df_chunk.columns:
                                last_no = int(df_chunk["번호"].iloc[-1])

                            # Process this chunk immediately
                            trained, end_no = _process_chunk(df_chunk)

                            # per N trained chunks checkpoint (after processing so we can include file and end_no)
                            if trained and (checkpoint_every_chunks is not None and checkpoint_every_chunks > 0 and
                                global_trained_chunk_count % checkpoint_every_chunks == 0):
                                chunk_ckpt_path = os.path.join(ckpt_dir, f"model_chunk_trained{global_trained_chunk_count}.pt")
                                cfg_dict = model.cfg.__dict__ if hasattr(model, 'cfg') else {"input_features": len(global_features), "seq_len": seq_len}
                                save_checkpoint(
                                    chunk_ckpt_path,
                                    state_dict=model.state_dict(),
                                    config_dict=cfg_dict,
                                    feature_names=list(global_features or []),
                                    target_col=tgt_col,
                                    horizon=horizon,
                                    aux_task=aux_task,
                                    # Save both keys for compatibility + keyset hint
                                    extra={
                                        "epoch": epoch,
                                        "trained_chunks": global_trained_chunk_count,
                                        "chunk": global_trained_chunk_count,
                                        "file": os.path.basename(fpath),
                                        "last_no": end_no,
                                    },
                                )
                                print(f"Saved chunk checkpoint: {chunk_ckpt_path} (trained_chunks {global_trained_chunk_count})")
                    else:
                        # Fallback to OFFSET pagination (slower). Consider reducing chunk_size if slow.
                        
                        # If resuming, calculate starting offset to skip already processed chunks
                        start_offset = 0
                        if resume_chunk_count > 0:
                            remaining_chunks_to_skip = resume_chunk_count - global_chunk_count
                            if remaining_chunks_to_skip > 0:
                                max_chunks_in_file = (total + step - 1) // step
                                chunks_to_skip_in_file = min(remaining_chunks_to_skip, max_chunks_in_file)
                                
                                if chunks_to_skip_in_file > 0:
                                    start_offset = chunks_to_skip_in_file * step
                                    processed += start_offset
                                    chunk_idx += chunks_to_skip_in_file
                                    global_chunk_count += chunks_to_skip_in_file
                                    
                                    print(f"[FAST-SKIP-OFFSET] file={os.path.basename(fpath)} skipped {chunks_to_skip_in_file} chunks (offset={start_offset}) chunk={global_chunk_count}/{resume_chunk_count}")
                                    
                                    # If we've skipped all chunks in this file, continue to next file
                                    if start_offset >= total:
                                        continue
                        
                        for offset in range(start_offset, total, step):
                            order_sql = ""  # no stable key; OFFSET may be expensive
                            sql = f"SELECT {col_sql} FROM {table}{where_sql}{order_sql} LIMIT {step} OFFSET {offset}"
                            df_chunk = conn.execute(sql, params).df()
                            if df_chunk.empty:
                                continue
                            processed += len(df_chunk)
                            chunk_idx += 1
                            global_chunk_count += 1

                            # Skip training if resuming and haven't reached resume trained chunk yet
                            if resume_chunk_count > 0 and global_chunk_count <= resume_chunk_count:
                                if progress_every > 0 and (chunk_idx % progress_every == 0 or processed >= total):
                                    print(f"[SKIP-TRAINED-OFFSET] file={os.path.basename(fpath)} epoch={epoch} chunk={global_chunk_count}/{resume_chunk_count} processed={processed}/{total} (skipping already trained chunk)")
                                continue

                            if progress_every > 0 and (chunk_idx % progress_every == 0 or processed >= total):
                                print(f"[TRAINING-OFFSET] file={os.path.basename(fpath)} epoch={epoch} chunk={global_chunk_count} processed={processed}/{total} (training chunk)")

                            # Process this chunk immediately
                            trained, _ = _process_chunk(df_chunk)
                            if trained and (checkpoint_every_chunks is not None and checkpoint_every_chunks > 0 and
                                global_trained_chunk_count % checkpoint_every_chunks == 0):
                                chunk_ckpt_path = os.path.join(ckpt_dir, f"model_chunk_trained{global_trained_chunk_count}.pt")
                                cfg_dict = model.cfg.__dict__ if hasattr(model, 'cfg') else {"input_features": len(global_features), "seq_len": seq_len}
                                save_checkpoint(
                                    chunk_ckpt_path,
                                    state_dict=model.state_dict(),
                                    config_dict=cfg_dict,
                                    feature_names=list(global_features or []),
                                    target_col=tgt_col,
                                    horizon=horizon,
                                    aux_task=aux_task,
                                    extra={
                                        "epoch": epoch,
                                        "trained_chunks": global_trained_chunk_count,
                                        "chunk": global_trained_chunk_count,
                                        "file": os.path.basename(fpath),
                                        "last_no": None,
                                    },
                                )
                                print(f"Saved chunk checkpoint: {chunk_ckpt_path} (trained_chunks {global_trained_chunk_count})")
                finally:
                    conn.close()

            # End of epoch: compute averages and handle checkpoints/early stopping
            tr_loss_avg = tr_loss_epoch_sum / max(1, tr_samples)
            va_loss_avg = va_loss_epoch_sum / max(1, va_samples)
            # Compute validation metrics
            va_r = 0.0
            va_r2 = 0.0
            if va_targets_all and aux_task == "regression":
                import numpy as _np
                y_true = _np.array(va_targets_all, dtype=_np.float32)
                y_pred = _np.array(va_preds_all, dtype=_np.float32)
                if y_true.size > 1:
                    # Pearson r
                    yt = y_true - y_true.mean()
                    yp = y_pred - y_pred.mean()
                    denom = (yt.std() * yp.std())
                    va_r = float((yt * yp).mean() / denom) if denom != 0 else 0.0
                    # R^2
                    ss_res = float(((y_true - y_pred) ** 2).sum())
                    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
                    va_r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
            
            # Update learning rate scheduler
            if 'scheduler' in locals():
                old_lr = opt.param_groups[0]['lr']
                scheduler.step(va_loss_avg)
                current_lr = opt.param_groups[0]['lr']
                if current_lr != old_lr:
                    print(f"Learning rate reduced from {old_lr:.2e} to {current_lr:.2e}")
                print(f"Epoch {epoch}/{epochs} - train_loss={tr_loss_avg:.6f} val_loss={va_loss_avg:.6f} lr={current_lr:.2e}")
            else:
                print(f"Epoch {epoch}/{epochs} - train_loss={tr_loss_avg:.6f} val_loss={va_loss_avg:.6f}")

            # Log epoch-level metrics to TensorBoard
            if writer:
                writer.add_scalar('Loss/Train_Epoch', tr_loss_avg, epoch)
                writer.add_scalar('Loss/Validation_Epoch', va_loss_avg, epoch)
                writer.add_scalar('Metrics/Best_Validation_Loss', best_val, epoch)
                writer.add_scalar('Metrics/No_Improve_Count', no_improve, epoch)
                writer.add_scalar('Metrics/Trained_Chunks', global_trained_chunk_count, epoch)
                writer.add_scalar('Metrics/Total_Samples_Trained', tr_samples, epoch)
                writer.add_scalar('Metrics/Total_Samples_Validated', va_samples, epoch)
                if 'scheduler' in locals():
                    writer.add_scalar('Learning_Rate/Epoch', opt.param_groups[0]['lr'], epoch)
                if aux_task == 'regression':
                    writer.add_scalar('Metrics/Val_Pearson_r', va_r, epoch)
                    writer.add_scalar('Metrics/Val_R2', va_r2, epoch)
            
            if va_loss_avg + 1e-9 < best_val:
                best_val = va_loss_avg
                no_improve = 0
                # save best checkpoint
                ckpt_path = os.path.join(output_dir, "model.pt")
                extra_data = {}
                if 'scheduler' in locals():
                    extra_data["scheduler_state"] = scheduler.state_dict()
                save_checkpoint(
                    ckpt_path,
                    state_dict=model.state_dict(),
                    config_dict=model.cfg.__dict__,
                    feature_names=list(global_features or []),
                    target_col=tgt_col,
                    horizon=horizon,
                    aux_task=aux_task,
                    extra=extra_data,
                )
                if writer:
                    writer.add_scalar('Checkpoints/Best_Model_Saved', best_val, epoch)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping")
                    if writer:
                        writer.add_text('Training/Early_Stop', f'Early stopping at epoch {epoch}', epoch)
                    break

            # Optional epoch checkpoints
            save_by_interval = checkpoint_every_epochs is not None and checkpoint_every_epochs > 0 and (epoch % checkpoint_every_epochs == 0)
            save_by_list = epoch in epoch_save_set
            if save_by_interval or save_by_list:
                epoch_ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pt")
                extra_data = {"epoch": epoch, "val_loss": va_loss_avg}
                if 'scheduler' in locals():
                    extra_data["scheduler_state"] = scheduler.state_dict()
                save_checkpoint(
                    epoch_ckpt_path,
                    state_dict=model.state_dict(),
                    config_dict=model.cfg.__dict__,
                    feature_names=list(global_features or []),
                    target_col=tgt_col,
                    horizon=horizon,
                    aux_task=aux_task,
                    extra=extra_data,
                )
                print(f"Saved epoch checkpoint: {epoch_ckpt_path}")

        # Save training metadata after streaming training completes
        meta = {
            "db_path": db_path,
            "table": table,
            "code": code,
            "date": date,
            "seq_len": seq_len,
            "horizon": horizon,
            "target_col": tgt_col,
            "feature_names": list(global_features or []),
            "best_val_mse": best_val,
            "aux_task": aux_task,
        }
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # Close TensorBoard writer
        if writer:
            writer.close()
            print("TensorBoard logging completed.")
        return

    # 1) db_path가 폴더라면 폴더 내의 모든 DuckDB 파일을 로드하여 concat
    #    파일 기준: *.duckdb (필요 시 *.db 도 포함)
    dfs = []
    if os.path.isdir(db_path):
        duckdb_files = []
        for name in os.listdir(db_path):
            if name.lower().endswith(".duckdb") or name.lower().endswith(".db"):
                duckdb_files.append(os.path.join(db_path, name))
        duckdb_files.sort()
        if not duckdb_files:
            raise ValueError(f"No DuckDB files (*.duckdb|*.db) found in directory: {db_path}")
        for f in duckdb_files:
            df_part, _ = load_real_dataframe(f, table=table, code=code, date=date)
            if not df_part.empty:
                dfs.append(df_part)
        if not dfs:
            raise ValueError("No data loaded from DuckDB files in the folder.")
        df = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        df, _ = load_real_dataframe(db_path, table=table, code=code, date=date)

    # 2) 사용자 지정 feature 목록 구성 ('날짜'는 month만 사용)
    requested_features = get_requested_features()

    # 실제 존재하는 컬럼만 사용
    available_features = [c for c in requested_features if c in df.columns]
    if not available_features:
        raise ValueError("None of the requested feature columns exist in the loaded DataFrame.")

    feats = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    real_cols = list(feats.columns)

    # Target setup (기존 로직 유지: 지정 없고 '현재가'가 없으면 첫 feature 사용됨)
    tgt_col = target_col or ("현재가" if "현재가" in real_cols else real_cols[0])
    if tgt_col not in real_cols:
        raise ValueError(f"target_col '{tgt_col}' not in feature columns")

    # '날짜'를 month(01~12) 정수로 변환하여 덮어쓰기
    convert_date_to_month_inplace(df)

    feat_vals = feats.to_numpy(dtype=np.float32)
    prices = feats[tgt_col].to_numpy(dtype=np.float32)
    T = len(feats)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for i in range(T - seq_len - horizon + 1):
        xw = feat_vals[i : i + seq_len]
        # Compute targets depending on aux_task
        if aux_task == "regression":
            yv = float(prices[i + seq_len + horizon - 1])
        elif aux_task == "direction":
            p_now = float(prices[i + seq_len - 1])
            p_future = float(prices[i + seq_len + horizon - 1])
            yv = 1.0 if (p_future - p_now) > 0 else 0.0
        elif aux_task == "volatility":
            # std of simple returns over future horizon window
            p0 = float(prices[i + seq_len - 1])
            if p0 <= 0:
                yv = 0.0
            else:
                future = prices[i + seq_len : i + seq_len + horizon]
                # simple returns relative to previous step to avoid div by zero
                rets = []
                prev = p0
                for p in future:
                    rets.append((float(p) / float(prev) - 1.0) if prev != 0 else 0.0)
                    prev = float(p)
                yv = float(np.std(rets, dtype=np.float32)) if len(rets) > 0 else 0.0
        else:
            raise ValueError(f"Unknown aux_task: {aux_task}")
        X_list.append(xw)
        y_list.append(yv)

    if not X_list:
        raise ValueError("Not enough rows to create sequences. Reduce seq_len/horizon or load more data.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N, T, F]
    y = np.array(y_list, dtype=np.float32)           # [N]

    # train/val split (time-ordered)
    n = len(X)
    n_val = max(1, int(n * 0.2))
    n_tr = n - n_val
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr:], y[n_tr:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    cfg = ModelConfig(input_features=len(real_cols), seq_len=seq_len)
    model = CNNLSTMAttn(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    if aux_task == "direction":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.HuberLoss(delta=huber_delta) if (loss_type == "huber") else nn.MSELoss()

    # Non-streaming startup summary
    log_startup_summary(
        mode="full-load",
        device=device,
        db_path=db_path,
        table=table,
        files=None,
        per_file_rows=None,
        total_rows=len(df),
        available_features=available_features,
        target_col=tgt_col,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        epochs=epochs,
        chunk_size=None,
        checkpoint_every_chunks=None,
        checkpoint_every_epochs=checkpoint_every_epochs,
        checkpoint_epochs=checkpoint_epochs,
        progress_every=None,
    )

    best_val = float("inf")
    patience = 5
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        batch_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
            n_tr += len(xb)
            
            # Log batch-level metrics to TensorBoard
            if writer and batch_count % 10 == 0:  # Log every 10 batches
                global_step = (epoch - 1) * len(train_loader) + batch_count
                writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                writer.add_scalar('Learning_Rate', opt.param_groups[0]['lr'], global_step)
            batch_count += 1
        tr_loss /= max(1, n_tr)

        model.eval()
        va_loss = 0.0
        n_va = 0
        va_preds_list = []
        va_targets_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * len(xb)
                n_va += len(xb)
                va_preds_list.extend(pred.detach().cpu().float().tolist())
                va_targets_list.extend(yb.detach().cpu().float().tolist())
        va_loss /= max(1, n_va)
        
        # Compute validation metrics
        va_r = 0.0
        va_r2 = 0.0
        if aux_task == "regression" and len(va_targets_list) > 1:
            import numpy as _np
            y_true = _np.array(va_targets_list, dtype=_np.float32)
            y_pred = _np.array(va_preds_list, dtype=_np.float32)
            yt = y_true - y_true.mean()
            yp = y_pred - y_pred.mean()
            denom = (yt.std() * yp.std())
            va_r = float((yt * yp).mean() / denom) if denom != 0 else 0.0
            ss_res = float(((y_true - y_pred) ** 2).sum())
            ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
            va_r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        
        # Update learning rate scheduler
        old_lr = opt.param_groups[0]['lr']
        scheduler.step(va_loss)
        current_lr = opt.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.2e} to {current_lr:.2e}")
        print(f"Epoch {epoch}/{epochs} - train_loss={tr_loss:.6f} val_loss={va_loss:.6f} lr={current_lr:.2e}")
        
        # Log epoch-level metrics to TensorBoard
        if writer:
            writer.add_scalar('Loss/Train_Epoch', tr_loss, epoch)
            writer.add_scalar('Loss/Validation_Epoch', va_loss, epoch)
            writer.add_scalar('Metrics/Best_Validation_Loss', best_val, epoch)
            writer.add_scalar('Metrics/No_Improve_Count', no_improve, epoch)
            writer.add_scalar('Metrics/Total_Samples_Trained', n_tr, epoch)
            writer.add_scalar('Metrics/Total_Samples_Validated', n_va, epoch)
            writer.add_scalar('Learning_Rate/Epoch', current_lr, epoch)
            if aux_task == 'regression':
                writer.add_scalar('Metrics/Val_Pearson_r', va_r, epoch)
                writer.add_scalar('Metrics/Val_R2', va_r2, epoch)
        
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            no_improve = 0
            # save best checkpoint
            ckpt_path = os.path.join(output_dir, "model.pt")
            save_checkpoint(
                ckpt_path,
                state_dict=model.state_dict(),
                config_dict=cfg.__dict__,
                feature_names=list(real_cols),
                target_col=tgt_col,
                horizon=horizon,
                aux_task=aux_task,
                extra={"scheduler_state": scheduler.state_dict()},
            )
            if writer:
                writer.add_scalar('Checkpoints/Best_Model_Saved', best_val, epoch)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                if writer:
                    writer.add_text('Training/Early_Stop', f'Early stopping at epoch {epoch}', epoch)
                break

        # Optional epoch checkpoints
        save_by_interval = checkpoint_every_epochs is not None and checkpoint_every_epochs > 0 and (epoch % checkpoint_every_epochs == 0)
        save_by_list = epoch in epoch_save_set
        if save_by_interval or save_by_list:
            epoch_ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pt")
            save_checkpoint(
                epoch_ckpt_path,
                state_dict=model.state_dict(),
                config_dict=cfg.__dict__,
                feature_names=list(real_cols),
                target_col=tgt_col,
                horizon=horizon,
                aux_task=aux_task,
                extra={"epoch": epoch, "val_loss": va_loss, "scheduler_state": scheduler.state_dict()},
            )
            print(f"Saved epoch checkpoint: {epoch_ckpt_path}")

    # Save training metadata
    meta = {
        "db_path": db_path,
        "table": table,
        "code": code,
        "date": date,
        "seq_len": seq_len,
        "horizon": horizon,
        "target_col": tgt_col,
        "feature_names": list(real_cols),
        "best_val_mse": best_val,
        "aux_task": aux_task,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    # Close TensorBoard writer
    if writer:
        writer.close()
        print("TensorBoard logging completed.")


def main():
    p = argparse.ArgumentParser(description="SQLite REAL 컬럼 기반 CNN+LSTM+어텐션 지도학습")
    p.add_argument("--db", default="datasets/datasets.db", help="데이터셋 SQLite DB 파일 경로. 기본값: datasets/datasets.db")
    p.add_argument("--table", default="datasets", help="DB에서 사용할 테이블명. 기본값: datasets")
    p.add_argument("--code", default=None, help="특정 종목 코드로 필터링(예: 005930). 미지정 시 전체/로직에 따름")
    p.add_argument("--date", default=None, help="특정 일자(YYYYMMDD)로 필터링. 미지정 시 전체/로직에 따름")
    p.add_argument("--out", default="models/supervised", help="출력 디렉터리(체크포인트와 설정 저장). 기본값: models/supervised")
    p.add_argument("--seq-len", type=int, default=60, help="입력 시퀀스(윈도우) 길이. 기본값: 60")
    p.add_argument("--horizon", type=int, default=10, help="예측 시점까지의 간격(몇 스텝 뒤를 예측할지). 기본값: 1")
    p.add_argument("--target-col", default=None, help="예측 대상 컬럼명. 미지정 시 첫 번째 feature 사용")
    p.add_argument("--batch-size", type=int, default=128, help="학습 배치 크기. 기본값: 128")
    p.add_argument("--epochs", type=int, default=20, help="최대 학습 에폭 수(얼리 스탑 적용). 기본값: 20")
    p.add_argument("--lr", type=float, default=1e-4, help="학습률(AdamW). 기본값: 1e-4")
    p.add_argument("--device", default=None, help="장치 지정: cuda/cpu. 미지정 시 가능하면 CUDA 사용, 아니면 CPU")
    p.add_argument("--aux-task", choices=["regression", "direction", "volatility"], default="regression", help="보조 학습 목표")
    p.add_argument("--ckpt-every-chunks", type=int, default=None, help="N 청크마다 체크포인트 저장 (예: 100). 미지정 시 비활성화")
    p.add_argument("--ckpt-every-epochs", type=int, default=None, help="N 에폭마다 체크포인트 저장 (예: 5). 미지정 시 비활성화")
    p.add_argument("--ckpt-epochs", default=None, help="지정 에폭에서 체크포인트 저장 (쉼표 구분, 예: '5,10,20')")
    p.add_argument("--resume-from", default=None, help="재시작할 체크포인트 파일 경로 또는 체크포인트 폴더 경로. 폴더 지정 시 가장 최신 체크포인트 자동 선택. 미지정 시 처음부터 시작")
    p.add_argument("--chunk-size", type=int, default=1000, help="DuckDB에서 한 번에 읽을 레코드 수. 기본값: 1000. 0 또는 음수면 전체 로드")
    p.add_argument("--progress-every", type=int, default=10, help="청크 진행 로그 출력 주기(청크 단위). 0이면 비활성화")
    p.add_argument("--no-tensorboard", action="store_true", help="TensorBoard 로깅 비활성화")
    p.add_argument("--loss", choices=["mse", "huber"], default="mse", help="회귀 손실 함수 선택: mse 또는 huber")
    p.add_argument("--huber-delta", type=float, default=1.0, help="Huber 손실의 delta (허용 오차)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (L2 정규화 강도)")
    args = p.parse_args()

    train(
        db_path=args.db,
        output_dir=args.out,
        table=args.table,
        code=args.code,
        date=args.date,
        seq_len=args.seq_len,
        horizon=args.horizon,
        target_col=args.target_col,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        aux_task=args.aux_task,
        checkpoint_every_chunks=args.ckpt_every_chunks,
        checkpoint_every_epochs=args.ckpt_every_epochs,
        checkpoint_epochs=args.ckpt_epochs,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
        resume_from=args.resume_from,
        enable_tensorboard=not args.no_tensorboard,
        loss_type=args.loss,
        huber_delta=args.huber_delta,
        weight_decay=args.weight_decay,
    )


#
# python -m ai_trader.rl.train_rl --algo ppo --db models/test_datasets2.db --out models/rl_ppo --seq-len 60 --target-col 현재가 --total-timesteps 200000
#
# python -m ai_trader.rl.infer_rl --algo ppo --db models/test_datasets2.db --model models/rl/ppo_model.zip --seq-len 60 --target-col 현재가
#
if __name__ == "__main__":
    main()
