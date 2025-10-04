---
inclusion: always
---

# Technology Stack

## Core Technologies

- **Language**: Python 3.10+
- **Database**: DuckDB (monthly-sharded files: `datasets_YYYYMM.duckdb`)
- **Deep Learning**: PyTorch 2.2+, TensorBoard
- **RL Framework**: Gymnasium, Stable-Baselines3 (PPO, A2C)
- **Data Processing**: pandas, numpy, pyarrow, duckdb

## Key Libraries

- `torch`, `torchvision` - Neural network training
- `stable-baselines3` - RL algorithms
- `gymnasium` - RL environment interface
- `duckdb` - Analytical database
- `tensorboard` - Training visualization
- `pydantic` - Data validation
- `requests` - HTTP client for Kiwoom API

## Common Commands

### Data Preparation
```bash
# Generate datasets from CSV
python scripts/generate_datasets.py <input_folder> -o <output.duckdb> --workers 12

# Merge multiple databases
python scripts/merge_datasets.py <input_folder> --out <merged.duckdb> --threads 2

# Normalize datasets
python scripts/normalize_datasets.py <input_folder> -o <output.duckdb> --workers 12 --checkpoint-interval 50
```

### Training

#### Supervised Learning (Classification)
```bash
python -m ai_trader.ml.train_supervised \
  --db <path_to_db> \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 20 \
  --batch-size 128 \
  --epochs 10 \
  --lr 5e-5 \
  --aux-task direction3 \
  --loss focal \
  --direction3-threshold 0.015 \
  --use-weighted-sampler \
  --device cuda
```

#### Reinforcement Learning
```bash
python -m ai_trader.rl.train_rl \
  --algo ppo \
  --db <path_to_db> \
  --out models/rl_ppo \
  --seq-len 60 \
  --target-col 현재가 \
  --policy cnn \
  --obs-format matrix \
  --device cpu
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir models/supervised/tensorboard_logs

# Custom monitoring script
python scripts/check_tensorboard.py \
  --logdir models/supervised/tensorboard_logs \
  --watch --interval 10 \
  --plot-last 400 \
  --save-png models/supervised/plots
```

### Testing
```bash
# Run pytest
pytest

# Run specific test
pytest tests/test_specific.py
```

## Platform Notes

- **Windows**: Primary development platform (cmd shell)
- **GPU**: CUDA support for PyTorch training (optional, CPU fallback available)
- **Multiprocessing**: Uses spawn mode for Windows compatibility
