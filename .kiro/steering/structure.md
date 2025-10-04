---
inclusion: always
---

# Project Structure

## Root Directory

- `requirements.txt` - Python dependencies
- `pytest.ini` - Pytest configuration
- `.env` - Environment variables (API keys, credentials)
- `README.md` - Main documentation with training examples

## Core Modules

### `ai_trader/`

ML and RL training/inference modules

- `ml/` - Supervised learning
  - `data.py` - DuckDB data loader, sliding window generation
  - `models.py` - CNN+LSTM+Attention model architecture
  - `train_supervised.py` - Training CLI with chunked streaming
  - `infer_supervised.py` - Inference CLI
- `rl/` - Reinforcement learning
  - `env.py` - Custom Gymnasium trading environment
  - `policies.py` - CNN feature extractors for SB3
  - `train_rl.py` - PPO/A2C training CLI
  - `infer_rl.py` - RL inference CLI
- `pipeline.py` - End-to-end integration pipeline

### `koapys/`

Kiwoom REST API client (compatibility layer)

- `client.py` - Main REST client (`KoapyRestSimple`)
- `endpoints.py` - API endpoint definitions
- `types.py` - Order types, enums
- `docs/` - API documentation

### `koapys_openAPI/`

Legacy OpenAPI integration (desktop-based)

- `koa_proxy.py` - Proxy server for OpenAPI
- `koa_proxy_parser.py` - Response parser
- `order.py` - Order management
- `simulator.py` - Trading simulator
- `price_unit.py` - Price unit calculations

### `scripts/`

Data processing and analysis utilities

- `generate_datasets.py` - CSV to DuckDB conversion
- `merge_datasets.py` - Merge multiple DuckDB files
- `normalize_datasets.py` - Feature normalization with monthly sharding
- `normalize_datasets_from_csv.py` - Direct CSV normalization
- `auto_retrain_supervised_ml.py` - Automated retraining pipeline
- `check_tensorboard.py` - TensorBoard log monitoring
- `analysis/` - Analysis scripts (active)
  - `diagnose_training_issues.py` - Training issue diagnostics
  - `training_analyzer.py` - Comprehensive training analysis
  - `README.md` - Analysis tools documentation
  - `_archive/` - Archived analysis scripts
    - `analyze_training_loss.py` - Training loss analysis
    - `check_data_quality.py` - Data quality validation
    - `check_group_normalization.py` - Group normalization verification
    - `deep_normalization_check.py` - Deep normalization diagnostics
    - `verify_normalization.py` - Normalization verification

### `lib/`

Shared utility modules

- `event_emitter.py` - Event handling
- `io.py` - I/O utilities
- `pd.py` - Pandas helpers
- `plot.py` - Plotting utilities
- `process.py` - Process management
- `thread_safe_dict.py` - Thread-safe data structures
- `utils.py` - General utilities
- `win32.py` - Windows-specific utilities

## Data Directories

### `models/`

Trained model checkpoints and datasets

- `supervised/` - Supervised learning models
  - `checkpoints/` - Training checkpoints
  - `tensorboard_logs/` - TensorBoard logs
  - `plots/` - Training plots
- `rl_ppo_scalp/` - RL models
- `datasets/` - Processed datasets
- `test_datasets.db` - Test database

### `logs/`

Application logs

### `prompt_logs/`

Training documentation and results

- `*.md` - Training notes and fixes
- `train_result*.html` - Training result reports

### `runs/`

TensorBoard run data

- `sb3/` - Stable-Baselines3 logs

## Conventions

### File Naming

- CSV input: `{code}_{name}_{type}_{date}.csv` (e.g., `005930_SAMSUNG_execution_20240115.csv`)
- DuckDB output: `datasets_YYYYMM.duckdb` (monthly sharding)
- Checkpoints: `checkpoint_epoch{N}_chunk{M}.pt`

### Database Schema

- Table name: `datasets` (default)
- Key columns: `날짜`, `종목코드`, `번호`
- Feature columns: 60+ REAL/FLOAT columns (Korean names)
- Derived features: `*_scalar`, `시간_sin`, `시간_cos`

### Code Organization

- CLI entry points use `argparse`
- Module execution: `python -m ai_trader.ml.train_supervised`
- Multiprocessing workers defined at module top-level (Windows spawn compatibility)
- Checkpoint format: PyTorch dict with `state_dict`, `config`, `feature_names`, metadata
