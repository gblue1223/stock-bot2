# Technology Stack

## Core Technologies

- **Python 3.8+**: Primary development language
- **PyTorch 2.2+**: Deep learning framework for AutoEncoder and GRPO models
- **DuckDB**: High-performance analytical database for time series data
- **Gymnasium**: RL environment framework
- **Stable Baselines3**: Reinforcement learning algorithms

## Key Libraries

### Deep Learning & ML
- `torch>=2.2` - Neural networks and GPU acceleration
- `torchvision>=0.17` - Vision utilities
- `tensorboard>=2.14` - Training visualization and logging
- `stable-baselines3>=2.3.0` - RL algorithms
- `gymnasium>=0.29` - RL environments
- `scikit-learn>=1.4` - Traditional ML utilities
- `numpy>=1.26`, `scipy>=1.10` - Numerical computing

### Data Processing
- `duckdb>=1.4.0` - Analytical database for large datasets
- `pandas>=2.0` - Data manipulation
- `pyarrow>=14.0` - Columnar data format

### Development & Testing
- `pytest>=8.0` - Testing framework
- `pydantic>=2.0` - Data validation
- `python-dotenv>=1.0` - Environment configuration

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (tested with CUDA 11.8)
- **RAM**: 16GB minimum, 32GB recommended for large datasets
- **Storage**: SSD recommended for DuckDB performance

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# GPU support (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Data Processing
```bash
# Generate normalized datasets
python scripts/normalize_datasets.py <input_folder> -o <output.duckdb> --workers 8

# Merge multiple datasets
python scripts/merge_datasets.py <input_folder> --out <merged.duckdb> --threads 4
```

### Training
```bash
# AutoEncoder training
python examples/autoencoder_training_example.py --db <path.duckdb> --output-dir models/autoencoder

# GRPO training (after AutoEncoder)
python examples/grpo_training_example.py --embedding-model models/autoencoder/best_model.pt
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Performance benchmarks
python -m pytest tests/test_performance_benchmark.py -v -s

# Specific test suites
python -m pytest tests/test_autoencoder_*.py -v
python -m pytest tests/test_grpo_*.py -v
```

### Performance Monitoring
```bash
# TensorBoard logging
tensorboard --logdir runs/

# Benchmark AutoEncoder
python scripts/benchmark_autoencoder.py --model-path models/autoencoder/best_model.pt
```

## Configuration

- Environment variables in `.env` file
- Model configurations in JSON format
- DuckDB paths and device settings configurable
- Supports both CPU and CUDA execution