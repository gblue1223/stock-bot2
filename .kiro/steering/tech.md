# Technology Stack

## Core Technologies

- **Python 3.8+**: Primary development language
- **PyTorch 2.2+**: Deep learning framework for embedding models and GRPO
- **DuckDB 1.4+**: High-performance analytical database for market data
- **Stable Baselines3 2.3+**: Reinforcement learning algorithms
- **Gymnasium 0.29+**: RL environment interface

## Key Libraries

### Data Processing
- **pandas 2.0+**: Data manipulation and analysis
- **numpy 1.26+**: Numerical computing
- **pyarrow 14.0+**: Columnar data processing
- **scikit-learn 1.4+**: Machine learning utilities

### Deep Learning & RL
- **torch**: Neural networks, optimizers, CUDA support
- **torchvision**: Vision utilities and transforms
- **tensorboard**: Training visualization and logging
- **stable-baselines3**: PPO, GRPO implementations

### Development & Testing
- **pytest 8.0+**: Testing framework
- **pydantic 2.0+**: Data validation and settings
- **python-dotenv**: Environment configuration

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_grpo.py

# Run with verbose output
pytest -v

# Run tests excluding regression suite
pytest -m "not regression"
```

### Data Processing
```bash
# Generate normalized datasets
python scripts/normalize_datasets.py <input_folder> -o <output.duckdb> --workers 8

# Merge multiple datasets
python scripts/merge_datasets.py <input_folder> --out <merged.duckdb> --threads 4

# Precompute stock pairs
python scripts/precompute_pairs.py
```

### Training
```bash
# Train embedding model
python -m ai_trader.embedding.train_embedding --db <path> --out <model_dir>

# Train GRPO agent
python -m ai_trader.grpo.train_grpo --embedding-model <path> --db <path> --out <model_dir>

# Complete workflow
python examples/complete_workflow_example.py --db <path> --out <model_dir>
```

### Evaluation & Backtesting
```bash
# Evaluate embedding model
python examples/embedding_evaluation_example.py --checkpoint <path> --db <path>

# Run GRPO backtest
python examples/grpo_backtest_example.py --embedding-model <path> --policy <path> --db <path>

# Generate HTML report
python examples/generate_html_report_example.py --results <path>
```

### Monitoring
```bash
# Check TensorBoard logs
python scripts/check_tensorboard.py

# View training progress
tensorboard --logdir runs/
```

## Development Environment

### GPU Support
- CUDA-compatible GPU recommended for training
- CPU fallback available for inference and small-scale testing
- Use `--device cuda` or `--device cpu` in training scripts

### Memory Requirements
- Minimum 16GB RAM for training
- 32GB+ recommended for large datasets
- SSD storage recommended for DuckDB performance

### Windows Compatibility
- All scripts designed for Windows paths and cmd/PowerShell
- Multiprocessing uses spawn mode for Windows compatibility
- Path separators handled automatically