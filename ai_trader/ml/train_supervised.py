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
    p = argparse.ArgumentParser(description="Train CNN+LSTM+MHA on SQLite REAL columns")
    p.add_argument("--db", default="datasets/datasets.db")
    p.add_argument("--table", default="datasets")
    p.add_argument("--code", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--out", default="models/supervised")
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--target-col", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
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


# python -m ai_trader.ml.train_supervised --db models/test_datasets2.db --table datasets --out models/supervised --seq-len 60 --horizon 1 --target-col 현재가
# python -m ai_trader.ml.train_supervised --db models/datasets.db --table datasets --out models/supervised --seq-len 60 --horizon 1 --target-col 현재가
if __name__ == "__main__":
    main()
