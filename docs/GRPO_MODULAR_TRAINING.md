# GRPO Modular Training System

## Overview

The GRPO training system has been modularized to support multiple environments and policies. This allows for easy experimentation and comparison between different approaches.

## Structure

### 1. Modularized Environments

Located in `ai_trader/grpo/environments/`:

- **`normalized_feature_env.py`**: Environment using normalized features with statistical pattern extraction
  - Feature dimension: 29 (current values, short/medium-term changes, volatility, relative positions, trends)
  - Optimized for normalized data analysis
  - Sequence length: 30 steps
  
- **`direct_feature_env.py`**: Environment using raw features directly without embedding
  - Uses flattened sequences as input
  - Bypasses embedding quality issues
  - Sequence length: 30 steps (configurable)

### 2. Modularized Policies

Located in `ai_trader/grpo/policies/`:

- **`normalized_feature_policy.py`**: Neural network for normalized feature environment
  - Input: 29-dimensional feature vector
  - Hidden dimension: 64 (default)
  - Action dimension: 3 (hold, buy, sell)
  
- **`direct_feature_policy.py`**: Neural network for direct feature environment
  - Input: flattened sequence (seq_len × features)
  - Projects to hidden dimension for processing
  - Action dimension: 3 (hold, buy, sell)

### 3. Training Scripts

#### Unified Modular Script: `scripts/grpo/train.py`

Command-line interface for selecting environments and policies:

```bash
# Example: Train with normalized features
python scripts/grpo/train.py \
    --env normalized \
    --policy normalized \
    --db /path/to/datasets_norm.duckdb \
    --seq_len 30 \
    --features 28 \
    --episode_steps 150 \
    --lr 0.0005 \
    --output_dir models/grpo_normalized

# Example: Train with direct features
python scripts/grpo/train.py \
    --env direct \
    --policy direct \
    --db /path/to/datasets_raw.duckdb \
    --seq_len 30 \
    --features 24 \
    --episode_steps 100 \
    --lr 0.001 \
    --output_dir models/grpo_direct

# Example: Train with scalping environment (requires embedding model)
python scripts/grpo/train.py \
    --env scalping \
    --policy grpo \
    --db /path/to/datasets.duckdb \
    --embedding_model models/autoencoder.pt \
    --embedding_dim 128 \
    --seq_len 60 \
    --features 28 \
    --episode_steps 200 \
    --output_dir models/grpo_scalping
```

#### Standalone Scripts

- **`scripts/grpo/normalized_features_train.py`**: Dedicated script for normalized features training
  - Pre-configured with optimal settings
  - No command-line arguments needed (paths hardcoded)
  
- **`scripts/grpo/direct_features_train.py`**: Dedicated script for direct features training
  - Pre-configured with conservative settings
  - No command-line arguments needed (paths hardcoded)

## Command-Line Arguments

### Environment Selection
- `--env {direct, normalized, scalping}`: Choose environment type
- `--policy {direct, normalized, grpo}`: Choose policy type

### Data Configuration
- `--db DB_PATH`: Database path (required)
- `--table TABLE_NAME`: Table name (default: datasets)
- `--seq_len SEQ_LEN`: Sequence length (default: 30)
- `--features FEATURES`: Number of features (default: 28)
- `--episode_steps EPISODE_STEPS`: Max episode steps (default: 150)

### Embedding Model (for scalping environment)
- `--embedding_model PATH`: Path to embedding model checkpoint
- `--embedding_dim DIM`: Embedding dimension (default: 128)
- `--quick_exit_mode {penalty_only, force_close}`: Quick exit mode

### Policy Architecture
- `--hidden_dim HIDDEN_DIM`: Hidden layer dimension (default: 64)
- `--action_dim ACTION_DIM`: Action dimension (default: 3)

### Training Hyperparameters
- `--episodes_per_group N`: Episodes per group (default: 6)
- `--num_groups N`: Number of groups (default: 3)
- `--lr LR`: Learning rate (default: 5e-4)
- `--gamma GAMMA`: Discount factor (default: 0.99)
- `--clip CLIP`: Clip epsilon (default: 0.2)
- `--kl_target KL_TARGET`: KL target (default: 0.01)
- `--entropy_coef COEF`: Entropy coefficient (default: 0.05)
- `--value_coef COEF`: Value coefficient (default: 0.5)
- `--max_grad_norm NORM`: Max gradient norm (default: 0.5)

### Training Control
- `--total_timesteps N`: Total timesteps (default: 10000)
- `--checkpoint_interval N`: Checkpoint interval (default: 10)
- `--output_dir PATH`: Output directory (default: models/grpo_modular)

## Environment-Policy Compatibility

| Environment | Compatible Policies | Notes |
|-------------|-------------------|-------|
| `normalized` | `normalized` | Uses 29-dimensional normalized features |
| `direct` | `direct` | Uses flattened raw sequences |
| `scalping` | `grpo` | Requires embedding model |

## Benefits of Modularization

1. **Code Reusability**: Environment and policy classes can be imported and reused
2. **Easy Comparison**: Switch between different approaches with a single command-line argument
3. **Maintainability**: Changes to environments/policies are centralized
4. **Flexibility**: Mix and match different components
5. **No Code Duplication**: Class definitions exist in only one place

## Example Usage

### Quick Start with Normalized Features

```bash
source .venv64/Scripts/activate
python scripts/grpo/normalized_features_train.py
```

### Quick Start with Direct Features

```bash
source .venv64/Scripts/activate
python scripts/grpo/direct_features_train.py
```

### Advanced Usage with Custom Configuration

```bash
source .venv64/Scripts/activate
python scripts/grpo/train.py \
    --env normalized \
    --policy normalized \
    --db C:/Users/user/Workspace/datasets@20251013/datasets_norm_all.duckdb \
    --seq_len 30 \
    --features 28 \
    --episode_steps 150 \
    --hidden_dim 128 \
    --episodes_per_group 8 \
    --num_groups 4 \
    --lr 0.001 \
    --total_timesteps 20000 \
    --output_dir models/grpo_normalized_custom
```

## Monitoring Training

All training scripts create TensorBoard logs:

```bash
tensorboard --logdir models/grpo_normalized/tensorboard_logs
```

## Checkpoints

Checkpoints are saved periodically during training:

- Location: `{output_dir}/checkpoints/checkpoint_iter{N}.pt`
- Final model: `{output_dir}/{env}_{policy}_model.pt`

## Next Steps

1. **Experiment**: Try different environment-policy combinations
2. **Tune**: Adjust hyperparameters for better performance
3. **Evaluate**: Use the backtest module to evaluate trained models
4. **Compare**: Compare performance across different approaches

## Troubleshooting

### Import Errors
Ensure the project root is in your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### Database Not Found
Check the database path and ensure it exists:
```bash
ls -l /path/to/database.duckdb
```

### GPU/CUDA Issues
The system automatically falls back to CPU if CUDA is not available. To force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/grpo/train.py ...
```
