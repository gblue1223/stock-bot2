# AutoEncoder Fine-tuning Guide

This guide explains how to fine-tune pre-trained AutoEncoder models on new data while preserving existing knowledge.

## Overview

Fine-tuning allows you to adapt a pre-trained model to new data patterns without starting from scratch. This is particularly useful when:

- New monthly data becomes available
- Market conditions change
- You want to improve performance on specific time periods
- You need to adapt to new stock symbols or features

## Benefits

- **Time Efficient**: 80%+ faster than full retraining
- **Knowledge Preservation**: Maintains existing learned patterns
- **Improved Performance**: Better adaptation to new data
- **Resource Efficient**: Lower memory and compute requirements

## Quick Start

### 1. Using the CLI Script

```bash
python scripts/pre/finetune_autoencoder.py \
    --model models/pretrained/model.pt \
    --data data/preprocessed \
    --output models/finetuned \
    --train-months 2025_01 2025_02 \
    --val-months 2025_03 \
    --learning-rate 1e-4 \
    --max-epochs 5
```

### 2. Using Python API

```python
from ai_trader.embedding.fine_tuning import FineTuner, create_fine_tuning_config
from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder

# Load pre-trained model
checkpoint = torch.load('models/pretrained/model.pt')
model = MaskedAutoEncoder(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Create fine-tuning configuration
config = create_fine_tuning_config(
    learning_rate=1e-4,
    max_epochs=5,
    freeze_layers=1
)

# Fine-tune
fine_tuner = FineTuner(model, device, config)
results = fine_tuner.fine_tune(train_loader, val_loader)
```

### 3. Using Google Colab

Use the provided Colab notebook for cloud-based fine-tuning:

- `scripts/colab/autoencoder_finetuning.ipynb`

## Configuration Options

### Basic Settings

| Parameter       | Default | Description                                  |
| --------------- | ------- | -------------------------------------------- |
| `learning_rate` | 1e-4    | Learning rate (lower than original training) |
| `max_epochs`    | 5       | Maximum number of epochs                     |
| `batch_size`    | 2       | Batch size for training                      |
| `weight_decay`  | 1e-5    | Weight decay for regularization              |

### Fine-tuning Strategy

| Parameter          | Default  | Description                 |
| ------------------ | -------- | --------------------------- |
| `freeze_encoder`   | False    | Freeze entire encoder       |
| `freeze_layers`    | 0        | Number of layers to freeze  |
| `warmup_epochs`    | 1        | Number of warmup epochs     |
| `lr_schedule`      | 'cosine' | Learning rate schedule      |
| `dropout_increase` | 0.05     | Amount to increase dropout  |
| `gradient_clip`    | 0.5      | Gradient clipping threshold |

### Data Settings

| Parameter               | Default | Description                 |
| ----------------------- | ------- | --------------------------- |
| `max_sequences`         | 1000    | Maximum sequences per batch |
| `max_batches_per_month` | 20      | Maximum batches per month   |

## Fine-tuning Strategies

### 1. Conservative Fine-tuning

- Low learning rate (1e-5 to 1e-4)
- Freeze early layers
- Small dropout increase
- Few epochs (3-5)

```python
config = create_fine_tuning_config(
    learning_rate=5e-5,
    max_epochs=3,
    freeze_layers=2,
    dropout_increase=0.02
)
```

### 2. Aggressive Fine-tuning

- Higher learning rate (1e-4 to 5e-4)
- No layer freezing
- More epochs (5-10)
- Larger dropout increase

```python
config = create_fine_tuning_config(
    learning_rate=2e-4,
    max_epochs=8,
    freeze_layers=0,
    dropout_increase=0.1
)
```

### 3. Encoder-only Fine-tuning

- Freeze encoder completely
- Only train decoder
- Very low learning rate

```python
config = create_fine_tuning_config(
    learning_rate=1e-5,
    freeze_encoder=True,
    max_epochs=5
)
```

## Best Practices

### Data Preparation

1. **Consistent Format**: Ensure new data matches original preprocessing
2. **Quality Check**: Verify data quality and completeness
3. **Balanced Sampling**: Use representative samples from new months
4. **Validation Split**: Keep separate validation data

### Training Process

1. **Start Conservative**: Begin with low learning rates
2. **Monitor Closely**: Watch for overfitting or degradation
3. **Early Stopping**: Stop if validation loss increases
4. **Save Checkpoints**: Save best models during training

### Evaluation

1. **Compare Performance**: Test on both old and new data
2. **Check Embeddings**: Verify embedding quality
3. **Validate Improvements**: Ensure actual performance gains
4. **A/B Testing**: Compare with original model

## Monitoring and Debugging

### Key Metrics to Watch

- **Validation Loss**: Should decrease or stabilize
- **Training Loss**: Should decrease smoothly
- **Learning Rate**: Should follow schedule
- **Gradient Norms**: Should be stable

### Common Issues

#### Overfitting

- **Symptoms**: Training loss decreases, validation loss increases
- **Solutions**: Reduce learning rate, increase dropout, early stopping

#### Catastrophic Forgetting

- **Symptoms**: Performance degrades on original data
- **Solutions**: Lower learning rate, freeze more layers, reduce epochs

#### No Improvement

- **Symptoms**: Validation loss doesn't improve
- **Solutions**: Increase learning rate, unfreeze layers, more epochs

#### Memory Issues

- **Symptoms**: CUDA out of memory errors
- **Solutions**: Reduce batch size, reduce max_sequences, use gradient checkpointing

## File Structure

After fine-tuning, the output directory contains:

```
models/finetuned_YYYYMMDD_HHMMSS/
├── model.pt                 # Fine-tuned model
├── model_info.json         # Model metadata
├── training_history.json   # Training history
└── results.png            # Results visualization (if generated)
```

## Integration with GRPO

Fine-tuned embeddings can be used directly with GRPO training:

```python
# Load fine-tuned model
checkpoint = torch.load('models/finetuned/model.pt')
embedding_model = MaskedAutoEncoder(**checkpoint['original_config'])
embedding_model.load_state_dict(checkpoint['model_state_dict'])

# Use with GRPO
from ai_trader.grpo.train_grpo import train_grpo
train_grpo(
    embedding_model=embedding_model,
    data_path='data/preprocessed',
    output_dir='models/grpo_finetuned'
)
```

## Performance Expectations

Typical fine-tuning results:

- **Time Savings**: 80-90% reduction in training time
- **Performance**: 5-15% improvement on new data
- **Stability**: Maintains 95%+ performance on original data
- **Memory**: 50-70% reduction in memory usage

## Troubleshooting

### Data Compatibility Issues

```python
# Check data shapes
print(f"Original: {original_config['seq_len']} x {original_config['num_features']}")
print(f"New: {dataset.seq_len} x {dataset.num_features}")
```

### Model Loading Issues

```python
# Verify checkpoint contents
checkpoint = torch.load('model.pt', map_location='cpu')
print("Available keys:", checkpoint.keys())
print("Config:", checkpoint.get('config', 'Not found'))
```

### Memory Optimization

```python
# Reduce memory usage
config = create_fine_tuning_config(
    batch_size=1,
    max_sequences=500,
    max_batches_per_month=10
)
```

## Advanced Usage

### Custom Loss Functions

```python
class CustomFineTuner(FineTuner):
    def compute_loss(self, reconstruction, target, embedding, mask):
        recon_loss = F.mse_loss(reconstruction[mask], target[mask])
        # Add custom regularization
        custom_reg = 0.01 * torch.norm(embedding - self.target_embedding, dim=1).mean()
        return recon_loss + custom_reg
```

### Progressive Unfreezing

```python
# Start with frozen layers, gradually unfreeze
for epoch in range(max_epochs):
    if epoch % 2 == 0 and epoch > 0:
        # Unfreeze one more layer
        unfreeze_layer(model, epoch // 2)

    train_epoch(...)
```

### Multi-stage Fine-tuning

```python
# Stage 1: Conservative fine-tuning
stage1_config = create_fine_tuning_config(learning_rate=1e-5, freeze_layers=3)
fine_tuner = FineTuner(model, device, stage1_config)
fine_tuner.fine_tune(train_loader, val_loader)

# Stage 2: More aggressive fine-tuning
stage2_config = create_fine_tuning_config(learning_rate=5e-5, freeze_layers=1)
fine_tuner = FineTuner(model, device, stage2_config)
fine_tuner.fine_tune(train_loader, val_loader)
```

## References

- [AutoEncoder Training Guide](README.md)
- [GRPO Training Guide](../grpo/README.md)
- [Data Preprocessing Guide](../preprocessing/README.md)
- [Colab Notebooks](../../scripts/colab/)
