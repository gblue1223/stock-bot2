# Technology Stack

## Core Technologies

- **Python 3.10+**: Primary development language
- **PyTorch**: Deep learning framework for CNN+LSTM+Attention models
- **Stable-Baselines3**: Reinforcement learning algorithms (PPO, A2C)
- **DuckDB**: High-performance analytical database for time series data
- **Gymnasium**: RL environment framework
- **Pandas/NumPy**: Data manipulation and numerical computing

## Key Dependencies

- **Data & ML**: `pandas>=2.0`, `numpy>=1.26`, `scikit-learn>=1.4`, `pyarrow>=14.0`
- **Deep Learning**: `torch>=2.2`, `torchvision>=0.17`, `tensorboard>=2.14`
- **RL**: `gymnasium>=0.29`, `stable-baselines3>=2.3.0`
- **Database**: `duckdb>=1.4.0`
- **Validation**: `pydantic>=2.0`, `pytest>=8.0`

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Data Processing
```bash
# Generate datasets
python scripts/generate_datasets.py

# Normalize datasets  
python scripts/normalize_datasets.py

# Merge datasets
python scripts/merge_datasets.py
```

### Training
```bash
# Supervised learning
python -m ai_trader.ml.train_supervised --db models/datasets.duckdb --table datasets --out models/supervised

# Reinforcement learning
python -m ai_trader.rl.train_rl --algo ppo --db models/datasets.duckdb --out models/rl_ppo

# Pipeline execution
python -m ai_trader.pipeline --model models/rl/ppo_model
```

### Testing
```bash
pytest -q --maxfail=1
```

## Platform Notes

- **Windows-focused**: Includes Windows-specific utilities and Kiwoom API integration
- **CPU-optimized**: Recommends CPU execution for most RL training scenarios
- **CUDA optional**: PyTorch CPU wheels recommended for Windows without CUDA