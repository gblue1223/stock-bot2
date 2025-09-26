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

from .data import load_real_dataframe
from .models import CNNLSTMAttn, ModelConfig


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
    lr: float = 1e-3,
    device: Optional[str] = None,
    aux_task: str = "regression",  # one of {"regression", "direction", "volatility"}
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
    - lr (float, default=1e-3): AdamW 옵티마이저의 학습률.
    - device (Optional[str], default=None): "cuda"/"cpu" 등 장치 지정. None이면 가능 시 CUDA 사용, 아니면 CPU.
    - aux_task (str, default="regression"): 보조 학습 목표. {regression, direction, volatility}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
    requested_features = [
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

    # '날짜'를 month(01~12) 정수로 변환하여 덮어쓰기
    if "날짜" in df.columns:
        # YYYYMMDD 형태 가정. 문자열로 변환 후 [4:6] 슬라이스.
        month_series = df["날짜"].astype(str).str[4:6]
        # 숫자로 안전 변환 (비정상 값은 0 처리)
        month_vals = pd.to_numeric(month_series, errors="coerce").fillna(0).astype(np.int32)
        df["날짜"] = month_vals

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
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if aux_task == "direction":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience = 5
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
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
        tr_loss /= max(1, n_tr)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * len(xb)
                n_va += len(xb)
        va_loss /= max(1, n_va)

        print(f"Epoch {epoch}/{epochs} - train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            no_improve = 0
            # save checkpoint
            ckpt_path = os.path.join(output_dir, "model.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "feature_names": list(real_cols),
                "target_col": tgt_col,
                "horizon": horizon,
                "aux_task": aux_task,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

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


def main():
    p = argparse.ArgumentParser(description="SQLite REAL 컬럼 기반 CNN+LSTM+어텐션 지도학습")
    p.add_argument("--db", default="datasets/datasets.db", help="데이터셋 SQLite DB 경로. 기본값: datasets/datasets.db")
    p.add_argument("--table", default="datasets", help="DB에서 사용할 테이블명. 기본값: datasets")
    p.add_argument("--code", default=None, help="특정 종목 코드로 필터링(예: 005930). 미지정 시 전체/로직에 따름")
    p.add_argument("--date", default=None, help="특정 일자(YYYYMMDD)로 필터링. 미지정 시 전체/로직에 따름")
    p.add_argument("--out", default="models/supervised", help="출력 디렉터리(체크포인트와 설정 저장). 기본값: models/supervised")
    p.add_argument("--seq-len", type=int, default=60, help="입력 시퀀스(윈도우) 길이. 기본값: 60")
    p.add_argument("--horizon", type=int, default=10, help="예측 시점까지의 간격(몇 스텝 뒤를 예측할지). 기본값: 1")
    p.add_argument("--target-col", default=None, help="예측 대상 컬럼명. 미지정 시 첫 번째 feature 사용")
    p.add_argument("--batch-size", type=int, default=128, help="학습 배치 크기. 기본값: 128")
    p.add_argument("--epochs", type=int, default=20, help="최대 학습 에폭 수(얼리 스탑 적용). 기본값: 20")
    p.add_argument("--lr", type=float, default=1e-3, help="학습률(AdamW). 기본값: 1e-3")
    p.add_argument("--device", default=None, help="장치 지정: cuda/cpu. 미지정 시 가능하면 CUDA 사용, 아니면 CPU")
    p.add_argument("--aux-task", choices=["regression", "direction", "volatility"], default="regression", help="보조 학습 목표")
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
    )


#
# python -m ai_trader.rl.train_rl --algo ppo --db models/test_datasets2.db --out models/rl_ppo --seq-len 60 --target-col 현재가 --total-timesteps 200000
#
# python -m ai_trader.rl.infer_rl --algo ppo --db models/test_datasets2.db --model models/rl/ppo_model.zip --seq-len 60 --target-col 현재가
#
if __name__ == "__main__":
    main()
