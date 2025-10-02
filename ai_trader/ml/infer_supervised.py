import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .data import load_real_dataframe, make_sequences
from .models import CNNLSTMAttn, ModelConfig


def infer(
    db_path: str,
    ckpt_dir: str,
    table: str = "datasets",
    code: Optional[str] = None,
    date: Optional[str] = None,
    seq_len: Optional[int] = None,
    device: Optional[str] = None,
    top_k: int = 10,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**{**ckpt["config"]})
    if seq_len is not None and seq_len != cfg.seq_len:
        # Allow override of seq_len if needed
        cfg.seq_len = seq_len
    feature_names = ckpt["feature_names"]
    target_col = ckpt.get("target_col", feature_names[0])

    model = CNNLSTMAttn(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    df, real_cols = load_real_dataframe(db_path, table=table, code=code, date=date)
    # Align to training features intersection
    real_cols = [c for c in feature_names if c in real_cols]
    if len(real_cols) == 0:
        raise ValueError("No overlap between training features and DB REAL columns")

    seq = make_sequences(df, real_cols, seq_len=cfg.seq_len, horizon=1, target_col=target_col)

    X = torch.from_numpy(seq.X[-top_k:]).to(device) if top_k > 0 else torch.from_numpy(seq.X).to(device)
    with torch.no_grad():
        logits = model(X).cpu()

    # Classification vs regression handling
    if getattr(model.cfg, 'num_classes', 1) and int(model.cfg.num_classes) >= 2:
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=-1)
        print("Class probabilities (most recent first): [down, flat, up]")
        for i, (p, c) in enumerate(zip(probs[::-1], preds[::-1]), 1):
            print(f"- t-{i}: class={int(c)} probs=" + ", ".join(f"{v:.4f}" for v in p.tolist()))
    else:
        y_pred = logits.numpy()
        print("Predictions (most recent first):")
        for i, v in enumerate(y_pred[::-1], 1):
            print(f"- t-{i}: {float(v):.6f}")


def main():
    p = argparse.ArgumentParser(description="Run inference with trained CNN+LSTM+MHA model")
    p.add_argument("--db", default="datasets/datasets.db")
    p.add_argument("--ckpt", default="models/supervised")
    p.add_argument("--table", default="datasets")
    p.add_argument("--code", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    infer(
        db_path=args.db,
        ckpt_dir=args.ckpt,
        table=args.table,
        code=args.code,
        date=args.date,
        seq_len=args.seq_len,
        device=args.device,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
