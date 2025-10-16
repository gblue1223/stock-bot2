# Technology Stack

## Core Technologies

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch 2.2+, TorchVision 0.17+
- **Reinforcement Learning**: Gymnasium 0.29+, Stable-Baselines3 2.3.0+
- **Data Processing**: DuckDB 1.4.0+, Pandas 2.0+, NumPy 1.26+, PyArrow 14.0+
- **Scientific Computing**: SciPy 1.10+, scikit-learn 1.4+
- **Monitoring**: TensorBoard 2.14+
- **Environment**: python-dotenv 1.0+

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (CUDA 11.8+)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: SSD recommended for DuckDB operations

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Development install
pip install -e .
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_grpo.py -v

# Run with coverage
python -m pytest tests/ --cov=ai_trader --cov-report=html

# Performance benchmarks
python -m pytest tests/test_performance_benchmark.py -v -s
```

### Data Pipeline
```bash
# Generate datasets from raw CSV files
python scripts/data/generate_datasets.py <input_dir> -o <output.duckdb> --workers 12

# Normalize datasets
python scripts/data/normalize_datasets.py <input_dir> -o <output.duckdb> --workers 12

# Merge monthly sharded databases
python scripts/data/merge_datasets.py <input_dir> --out <merged.duckdb> --threads 4
```

### Training
```bash
# AutoEncoder pre-training (use Jupyter notebooks in examples/)
jupyter notebook examples/autoencoder_training_example.py

# GRPO training
python -m ai_trader.grpo.train_grpo \
    --embedding-model models/autoencoder/best_model.pt \
    --db datasets_norm_all.duckdb \
    --out models/grpo_scalping \
    --episodes-per-group 12 \
    --device cuda

# Quick training script
python scripts/grpo/quick_train_grpo.py
```

### Inference & Evaluation
```bash
# Run inference example
python examples/grpo_inference_example.py

# Backtesting
python examples/grpo_backtest_example.py

# Generate HTML report
python examples/generate_html_report_example.py
```

### Monitoring
```bash
# Launch TensorBoard
tensorboard --logdir runs/sb3

# Check TensorBoard logs
python scripts/check_tensorboard.py
```

## Development Tools

- **Testing**: pytest 8.0+
- **Version Control**: Git
- **IDE**: VSCode (configuration in .vscode/)
- **Virtual Environment**: venv or conda recommended
