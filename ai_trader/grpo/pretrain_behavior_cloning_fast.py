#!/usr/bin/env python3
"""Small-batch distillation entry point using the verified shared training path."""
from pathlib import Path
import argparse
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ai_trader.grpo.pretrain_behavior_cloning import OfflineEpisodeDataset, pretrain


def fast_forward_student(student, x):
    """Keep inference and distillation on the same policy forward implementation."""
    return student(x)


def pretrain_fast(teacher_path: str, extracted_dir: str, output_path: str,
                  epochs: int = 3, batch_size: int = 32, lr: float = 2e-4,
                  temperature: float = 2.0, seq_len=None, step_size: int = 2000,
                  device: str = "cuda", train_end_date=None, validation_end_date=None,
                  embargo_dates: int = 0):
    # Sparse sampling reduces work without changing the model or compiling a
    # different JIT graph. Sequence length is inherited from the teacher schema.
    return pretrain(teacher_path, extracted_dir, output_path, epochs, batch_size,
                    lr, temperature, device, seq_len, step_size,
                    train_end_date, validation_end_date, embargo_dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse-window GRU to xLSTM distillation")
    parser.add_argument("--teacher_policy", required=True)
    parser.add_argument("--extracted_dir", default="data/extracted_episodes")
    parser.add_argument("--output_path", default="models/pretrain/xlstm_distilled.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seq_len", type=int, default=None, help="Must match the teacher checkpoint")
    parser.add_argument("--step_size", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_end_date", default=None)
    parser.add_argument("--validation_end_date", default=None)
    parser.add_argument("--embargo_dates", type=int, default=0)
    args = parser.parse_args()
    pretrain_fast(args.teacher_policy, args.extracted_dir, args.output_path, args.epochs,
                  args.batch_size, args.lr, args.temperature, args.seq_len, args.step_size, args.device,
                  args.train_end_date, args.validation_end_date, args.embargo_dates)
