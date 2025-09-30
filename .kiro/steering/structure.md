# Project Structure

## Core Modules

### `ai_trader/`
Main ML/RL trading system with two primary sub-modules:

- **`ml/`**: Supervised learning components
  - `data.py`: DuckDB data loading with sliding window generation
  - `models.py`: CNN+LSTM+Attention model implementations
  - `train_supervised.py`: Training CLI for supervised models
  - `infer_supervised.py`: Inference CLI for predictions

- **`rl/`**: Reinforcement learning components  
  - `env.py`: Trading environment with customizable rewards
  - `policies.py`: Custom CNN extractors for time series data
  - `train_rl.py`: RL training CLI (PPO/A2C)
  - `infer_rl.py`: RL inference and evaluation

- `pipeline.py`: Integration pipeline combining supervised + RL models

### `koapys/`
REST API client for Kiwoom trading platform:
- `client.py`: Main REST client with compatibility layer
- `endpoints.py`: API endpoint definitions
- `types.py`: Trading data types and enums
- `docs/`: API documentation

### `lib/`
Shared utility modules:
- `event_emitter.py`: Event handling system
- `io.py`, `pd.py`: Data I/O and pandas utilities
- `plot.py`: Visualization helpers
- `win32.py`: Windows-specific integrations
- `thread_safe_dict.py`: Concurrent data structures

### `scripts/`
Data processing utilities:
- `generate_datasets.py`: Dataset creation from raw data
- `normalize_datasets.py`: Data normalization and cleaning
- `merge_datasets.py`: Dataset combination utilities

## Data Organization

### `models/`
- `datasets/`: Raw dataset storage
- `supervised/`: Trained supervised model checkpoints
- `rl_*/`: RL model outputs and checkpoints
- `*.db`: DuckDB database files with trading data

### `logs/`
Application logs with timestamp-based naming

### `runs/`
Training run outputs and TensorBoard logs

## Legacy Components

### `koapys_openAPI/`
Original OpenAPI implementation (being replaced by REST client)

### `koapys_test/`
Test files for API integration

## Conventions

- **Module imports**: Use relative imports within packages, absolute for cross-package
- **CLI modules**: Implement as `python -m package.module` pattern
- **Configuration**: Use dataclasses for structured config objects
- **Database**: DuckDB files use `.db` extension, expect `datasets` table
- **Models**: Save as `.pt` files with config metadata included
- **Logging**: Timestamp-based log files in `logs/` directory