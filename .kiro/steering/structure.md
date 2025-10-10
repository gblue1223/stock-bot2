# Project Structure

## Core Architecture

The project follows a modular architecture with clear separation between data processing, model training, evaluation, and utilities.

```
ai_trader/                 # Main package
├── embedding/            # Embedding model components
├── grpo/                # GRPO reinforcement learning
├── lstm/                # LSTM models (legacy)
└── reporting/           # Report generation

examples/                 # Usage examples and workflows
scripts/                 # Data processing and utilities
tests/                   # Test suite
models/                  # Trained model storage
docs/                    # Documentation
config/                  # Configuration files
```

## Module Organization

### ai_trader/ - Core Package
- **embedding/**: Embedding model training, evaluation, and inference
- **grpo/**: GRPO agent training, environment, evaluation, backtesting
- **lstm/**: Legacy LSTM implementations
- **reporting/**: HTML report generation and visualization

### examples/ - Usage Examples
- Complete workflow examples showing end-to-end pipelines
- Individual component examples (training, evaluation, inference)
- Integration examples combining multiple components
- All examples include comprehensive CLI argument parsing

### scripts/ - Data Processing
- **normalize_datasets.py**: Convert raw CSV to normalized DuckDB
- **merge_datasets.py**: Combine multiple DuckDB files
- **precompute_pairs.py**: Generate stock pair correlations
- **generate_datasets.py**: Dataset generation utilities

### tests/ - Test Suite
- Comprehensive test coverage for all components
- Integration tests for end-to-end workflows
- Performance and regression tests
- Uses pytest framework with custom markers

## File Naming Conventions

### Python Modules
- Use snake_case for all Python files and directories
- Prefix test files with `test_`
- Use descriptive names indicating functionality

### Model Files
- Checkpoints: `checkpoint_epoch{N}.pt` or `checkpoint_step{N}.pt`
- Final models: `model_final.pt`
- Store in dated subdirectories: `models/grpo_YYYYMMDD/`

### Data Files
- DuckDB files: `datasets_YYYYMM.duckdb` (monthly sharding)
- Logs: `YYYYMMDD_HHMMSS_component.log`
- Results: `results_YYYYMMDD.json`

## Configuration Management

### Environment Variables
- Use `.env` file for local configuration
- Database paths, model directories, API keys
- Load with `python-dotenv`

### JSON Configuration
- `config/alert_config.json`: Performance alert thresholds
- Component-specific configs in respective modules
- Use Pydantic for validation

## Data Flow Architecture

### Input Data
```
Raw CSV files → normalize_datasets.py → DuckDB (monthly shards)
```

### Training Pipeline
```
DuckDB → Embedding Training → Embedding Model
DuckDB + Embedding Model → GRPO Training → Policy Model
```

### Evaluation Pipeline
```
Models + Test Data → Evaluation → Metrics + Reports
```

### Inference Pipeline
```
Live Data → Embedding → Policy → Trading Decision
```

## Development Guidelines

### Code Organization
- Keep modules focused and single-purpose
- Use clear imports and avoid circular dependencies
- Follow Python PEP 8 style guidelines
- Include comprehensive docstrings

### Error Handling
- Use structured logging with appropriate levels
- Implement graceful degradation for non-critical failures
- Provide clear error messages with context

### Performance Considerations
- Use GPU when available (`--device cuda`)
- Implement batch processing for large datasets
- Cache frequently accessed data
- Monitor memory usage in long-running processes

### Testing Strategy
- Unit tests for individual components
- Integration tests for workflows
- Performance tests for latency requirements
- Use pytest markers for test categorization

## Deployment Structure

### Model Storage
```
models/
├── embedding/
│   ├── checkpoint_epoch50.pt
│   └── model_config.json
├── grpo/
│   ├── checkpoint_step1000000.pt
│   └── policy_config.json
└── combined_training_report.html
```

### Log Management
```
logs/
├── YYYYMMDD_HHMMSS_training.log
├── alerts.jsonl
└── trades.csv
```

### Temporary Files
- Use system temp directories or `--tmp-dir` parameter
- Clean up intermediate files automatically
- Implement resume capability for long-running processes