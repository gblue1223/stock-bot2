# Project Structure

## Core Architecture

The project follows a modular architecture with clear separation between embedding models, reinforcement learning, and supporting utilities.

```
ai_trader/                  # Main package
├── embedding/              # AutoEncoder embedding models
│   ├── autoencoder_model.py    # Model definitions (AutoEncoder, MaskedAutoEncoder)
│   ├── autoencoder_trainer.py  # Training pipeline
│   ├── fine_tuning/            # Trading-specific fine-tuning
│   ├── data.py                 # Data loaders and datasets
│   └── evaluation.py           # Model evaluation utilities
├── grpo/                   # GRPO reinforcement learning
│   ├── env.py                  # Scalping environment (GRPOScalpingEnv)
│   ├── grpo.py                 # GRPO algorithm implementation
│   ├── policy.py               # Policy networks
│   ├── backtest.py             # Backtesting framework
│   └── train_grpo.py           # Training scripts
├── inference/              # Real-time inference
│   └── infer_grpo.py           # Fast inference engine (2.87ms)
└── reporting/              # Analysis and reporting
    └── html_report.py          # HTML report generation
```

## Supporting Directories

```
examples/                   # Usage examples and demos
├── autoencoder_training_example.py
├── grpo_training_example.py
└── complete_workflow_example.py

scripts/                    # Data processing and utilities
├── normalize_datasets.py       # DuckDB data normalization
├── merge_datasets.py          # Dataset merging
├── benchmark_autoencoder.py   # Performance benchmarking
└── analysis/                  # Analysis scripts

tests/                      # Comprehensive test suite
├── test_autoencoder_*.py      # AutoEncoder tests
├── test_grpo_*.py             # GRPO tests
├── test_performance_benchmark.py
└── autoencoder/               # Integration test results

models/                     # Trained models and checkpoints
├── autoencoder/               # AutoEncoder models
├── grpo_*/                    # GRPO trained policies
└── datasets/                  # Preprocessed datasets

config/                     # Configuration files
docs/                       # Documentation and guides
logs/                       # Application logs
runs/                       # TensorBoard logs
```

## Code Organization Patterns

### Module Structure
- Each major component (embedding, grpo) has its own package
- Clear separation between model definitions, training, and inference
- Shared utilities in `lib/` directory

### Naming Conventions
- **Files**: Snake_case (e.g., `autoencoder_model.py`)
- **Classes**: PascalCase (e.g., `GRPOScalpingEnv`, `AutoEncoderEmbedding`)
- **Functions**: Snake_case (e.g., `train_autoencoder_embedding`)
- **Constants**: UPPER_CASE (e.g., `TRANSACTION_COST_RATE`)

### Import Patterns
```python
# Absolute imports from ai_trader package
from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding
from ai_trader.grpo.env import GRPOScalpingEnv

# Relative imports within modules
from .data import AutoEncoderDataLoader
from .fine_tuning import fine_tune_for_trading_task
```

### Configuration Management
- Environment variables in `.env` file
- Model configs as JSON files in `config/` or passed as arguments
- Database paths and device settings configurable via environment

### Data Flow
1. **Raw CSV** → `scripts/normalize_datasets.py` → **DuckDB**
2. **DuckDB** → `embedding/data.py` → **AutoEncoder Training**
3. **Trained AutoEncoder** → `fine_tuning/` → **Task-specific Model**
4. **Fine-tuned Model** → `grpo/train_grpo.py` → **GRPO Policy**
5. **GRPO Policy** → `inference/infer_grpo.py` → **Real-time Trading**

### Testing Structure
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance benchmarks with specific targets
- Test data in `test_models/` and temporary directories

### Documentation
- Docstrings in Korean and English
- README files in major directories
- Implementation summaries in `docs/TASK_*.md`
- Quick reference guides for key features