# Project Structure

## Core Package: `ai_trader/`

Main Python package containing all AI trading components.

### `ai_trader/embedding/`
AutoEncoder-based embedding models for time-series feature extraction.

- `autoencoder_model.py`: AutoEncoder and MaskedAutoEncoder implementations
- `autoencoder_trainer.py`: Training pipeline for embeddings
- `fine_tuning.py`: Task-specific fine-tuning system
- `data.py`: Data loaders for embedding training
- `evaluation.py`: Embedding quality evaluation metrics
- `fast_data_loader.py`: Optimized data loading for large datasets
- `fine_tuning/`: Fine-tuning modules and task heads

### `ai_trader/grpo/`
GRPO (Group Relative Policy Optimization) reinforcement learning system.

- `grpo.py`: Core GRPO algorithm implementation
- `train_scalping.py`: Scalping-specific training script
- `train_direct_features.py`: Direct feature training (without embeddings)
- `backtest.py`: Backtesting framework
- `evaluation.py`: Performance evaluation metrics
- `alerts.py`: Alert system for trading signals
- `environments/`: Trading environment implementations
- `policies/`: Policy network architectures
- `inference/`: Real-time inference engines

### `ai_trader/lstm/`
Legacy LSTM-based models (deprecated in favor of AutoEncoder + GRPO).

### `ai_trader/reporting/`
Report generation and visualization.

- `html_report.py`: HTML report generator
- `report_generator.py`: Report generation utilities

## Scripts: `scripts/`

Utility scripts for data processing, training, and analysis.

### `scripts/data/`
Data pipeline scripts for preprocessing and normalization.

- `generate_datasets.py`: Convert raw CSV to DuckDB
- `normalize_datasets.py`: Normalize and clean datasets
- `merge_datasets.py`: Merge monthly sharded databases
- `export_datasets.py`: Export filtered datasets

### `scripts/grpo/`
GRPO training and evaluation scripts.

- `quick_train_grpo.py`: Quick training script with defaults

### `scripts/analysis/`
Analysis and diagnostic scripts.

### `scripts/colab/`
Google Colab notebooks for cloud training.

- `grpo_training_complete.ipynb`: Complete GRPO training workflow

### `scripts/pre/`
Pre-training data preparation scripts.

## Examples: `examples/`

End-to-end usage examples demonstrating workflows.

- `autoencoder_training_example.py`: AutoEncoder training
- `complete_workflow_example.py`: Full pipeline from data to inference
- `grpo_training_example.py`: GRPO training
- `grpo_inference_example.py`: Real-time inference
- `grpo_backtest_example.py`: Backtesting
- `grpo_evaluation_example.py`: Model evaluation
- `generate_html_report_example.py`: Report generation
- `fine_tuning/`: Fine-tuning examples

## Tests: `tests/`

Comprehensive test suite (38+ unit tests).

- `test_grpo*.py`: GRPO component tests
- `test_embedding*.py`: Embedding model tests
- `test_autoencoder*.py`: AutoEncoder tests
- `test_performance_benchmark.py`: Performance benchmarks
- `test_backtest.py`: Backtesting tests

## Data & Models

### `models/`
Trained model checkpoints (gitignored).

- `autoencoder/`: Pre-trained AutoEncoder models
- `grpo_*/`: GRPO policy checkpoints
- `datasets/`: Preprocessed datasets for training

### `logs/`
Application logs and alert logs.

- `alerts.jsonl`: Trading alert logs
- `*.log`: Application logs

### `runs/`
TensorBoard logs for training monitoring.

## Configuration

### `config/`
Configuration files.

- `alert_config.json`: Alert system configuration

### `.env`
Environment variables (gitignored).

- Database paths
- Model directories
- Device configuration (cuda/cpu)

## Documentation: `docs/`

Detailed documentation for specific features.

- `GRPO_*.md`: GRPO-related documentation
- `TASK_*.md`: Implementation summaries for specific tasks
- `*_QUICK_REFERENCE.md`: Quick reference guides

## External Integrations

### `koapys/`
Korean stock market API client (REST-based).

### `koapys_openAPI/`
Korean stock market OpenAPI integration (legacy).

### `lib/`
Shared utility libraries.

- `event_emitter.py`: Event handling
- `io.py`, `pd.py`, `plot.py`: Data I/O and visualization
- `utils.py`: General utilities

## Naming Conventions

- **Python files**: snake_case (e.g., `autoencoder_model.py`)
- **Classes**: PascalCase (e.g., `GRPOTrainer`, `AutoEncoderEmbedding`)
- **Functions/methods**: snake_case (e.g., `train_grpo`, `compute_advantage`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_GRAD_NORM`)
- **Private members**: Leading underscore (e.g., `_compute_loss`)

## File Organization Patterns

- Each major component has its own subdirectory under `ai_trader/`
- Related functionality grouped in modules (e.g., all GRPO components in `grpo/`)
- Examples are standalone scripts demonstrating specific workflows
- Tests mirror the structure of the main package
- Documentation uses descriptive names with task/feature identifiers
