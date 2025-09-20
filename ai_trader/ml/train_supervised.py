import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_real_dataframe, make_sequences, train_val_split
from .models import CNNLSTMAttn, ModelConfig


def train(
    db_path: str,
    output_dir: str,
    table: str = "datasets",
    code: Optional[str] = None,
    date: Optional[str] = None,
    seq_len: int = 60,
    horizon: int = 1,
    target_col: Optional[str] = None,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[str] = None,
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
    - horizon (int, default=1): 예측 시점까지의 간격(타깃을 몇 스텝 뒤로 볼지).
    - target_col (Optional[str], default=None): 예측 대상 컬럼명. None일 경우 첫 번째 feature를 사용합니다.
    - batch_size (int, default=128): 학습 배치 크기.
    - epochs (int, default=20): 최대 학습 에폭 수(얼리 스탑 적용).
    - lr (float, default=1e-3): AdamW 옵티마이저의 학습률.
    - device (Optional[str], default=None): "cuda"/"cpu" 등 장치 지정. None이면 가능 시 CUDA 사용, 아니면 CPU.

    동작
    - DB에서 시계열 데이터를 로드하고, `seq_len`과 `horizon`에 맞춰 시퀀스를 생성합니다.
    - 학습/검증 세트로 분할하여 `CNNLSTMAttn` 모델을 학습합니다.
    - 검증 손실(`val_loss`)이 개선될 때마다 `model.pt` 체크포인트를 저장합니다.
    - 최종으로 구성/특징/최적 검증 손실 등을 `config.json`에 기록합니다.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df, real_cols = load_real_dataframe(db_path, table=table, code=code, date=date)
    seq = make_sequences(df, real_cols, seq_len=seq_len, horizon=horizon, target_col=target_col)
    train_seq, val_seq = train_val_split(seq, val_ratio=0.2)

    train_ds = TensorDataset(torch.from_numpy(train_seq.X), torch.from_numpy(train_seq.y))
    val_ds = TensorDataset(torch.from_numpy(val_seq.X), torch.from_numpy(val_seq.y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    cfg = ModelConfig(input_features=len(seq.feature_names), seq_len=seq_len)
    model = CNNLSTMAttn(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
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
                "feature_names": seq.feature_names,
                "target_col": target_col or seq.feature_names[0],
                "horizon": horizon,
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
        "target_col": target_col or seq.feature_names[0],
        "feature_names": seq.feature_names,
        "best_val_mse": best_val,
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
    p.add_argument("--horizon", type=int, default=1, help="예측 시점까지의 간격(몇 스텝 뒤를 예측할지). 기본값: 1")
    p.add_argument("--target-col", default=None, help="예측 대상 컬럼명. 미지정 시 첫 번째 feature 사용")
    p.add_argument("--batch-size", type=int, default=128, help="학습 배치 크기. 기본값: 128")
    p.add_argument("--epochs", type=int, default=20, help="최대 학습 에폭 수(얼리 스탑 적용). 기본값: 20")
    p.add_argument("--lr", type=float, default=1e-3, help="학습률(AdamW). 기본값: 1e-3")
    p.add_argument("--device", default=None, help="장치 지정: cuda/cpu. 미지정 시 가능하면 CUDA 사용, 아니면 CPU")
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
    )


#
# python -m ai_trader.rl.train_rl --algo ppo --db models/test_datasets2.db --out models/rl_ppo --seq-len 60 --target-col 현재가 --total-timesteps 200000
#
# python -m ai_trader.rl.infer_rl --algo ppo --db models/test_datasets2.db --model models/rl/ppo_model.zip --seq-len 60 --target-col 현재가
#
if __name__ == "__main__":
    main()
