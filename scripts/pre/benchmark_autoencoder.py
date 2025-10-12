"""
Benchmark script to compare AutoEncoder vs Contrastive Learning training speed.
Demonstrates the performance advantage of AutoEncoder approach.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt

from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder
from ai_trader.embedding.autoencoder_trainer import TimeSeriesDataset


class SimpleContrastiveModel(nn.Module):
    """Simple contrastive learning model for comparison."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # Simple pooling over sequence
        pooled = x.mean(dim=1)
        return self.encoder(pooled)


class TimeSeriesSequenceDataset(Dataset):
    """Dataset for contrastive learning with positive/negative pairs."""
    
    def __init__(self, data: np.ndarray, seq_len: int = 60):
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.n_sequences = len(data) - seq_len + 1
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Anchor
        anchor = self.data[idx:idx + self.seq_len]
        
        # Positive (nearby sequence)
        pos_offset = np.random.randint(1, min(10, self.n_sequences - idx))
        positive = self.data[idx + pos_offset:idx + pos_offset + self.seq_len]
        
        # Negative (random distant sequence)
        neg_idx = np.random.randint(0, max(1, idx - 50)) if idx > 50 else np.random.randint(idx + 50, self.n_sequences)
        neg_idx = min(neg_idx, self.n_sequences - self.seq_len)
        negative = self.data[neg_idx:neg_idx + self.seq_len]
        
        return anchor, positive, negative


def contrastive_loss(anchor, positive, negative, temperature=0.1):
    """Compute contrastive loss."""
    # Normalize embeddings
    anchor = nn.functional.normalize(anchor, dim=1)
    positive = nn.functional.normalize(positive, dim=1)
    negative = nn.functional.normalize(negative, dim=1)
    
    # Compute similarities
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature
    
    # Contrastive loss
    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
    return loss.mean()


def benchmark_autoencoder(data: np.ndarray, config: dict, device: str = 'cuda'):
    """Benchmark AutoEncoder training."""
    
    input_dim = data.shape[1]
    seq_len = config['seq_len']
    
    # Create model
    if config['model_type'] == 'masked':
        model = MaskedAutoEncoder(
            input_dim=input_dim,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            seq_len=seq_len,
            mask_ratio=config['mask_ratio']
        ).to(device)
    else:
        model = AutoEncoderEmbedding(
            input_dim=input_dim,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            seq_len=seq_len
        ).to(device)
    
    # Create dataset and dataloader
    dataset = TimeSeriesDataset(data, seq_len, stride=config['stride'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    model.train()
    start_time = time.time()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, MaskedAutoEncoder):
                reconstruction, embedding, mask = model(batch)
                loss = nn.functional.mse_loss(reconstruction[mask], batch[mask])
            else:
                reconstruction, embedding = model(batch)
                loss = nn.functional.mse_loss(reconstruction, batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        total_loss += epoch_loss / epoch_batches
        num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"AutoEncoder Epoch {epoch + 1}: Loss = {epoch_loss / epoch_batches:.4f}")
    
    training_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    return {
        'training_time': training_time,
        'avg_loss': avg_loss,
        'samples_per_second': len(dataset) * config['epochs'] / training_time,
        'model_params': sum(p.numel() for p in model.parameters())
    }


def benchmark_contrastive(data: np.ndarray, config: dict, device: str = 'cuda'):
    """Benchmark Contrastive Learning training."""
    
    input_dim = data.shape[1]
    seq_len = config['seq_len']
    
    # Create model
    model = SimpleContrastiveModel(
        input_dim=input_dim,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Create dataset and dataloader
    dataset = TimeSeriesSequenceDataset(data, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    model.train()
    start_time = time.time()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute contrastive loss
            loss = contrastive_loss(anchor_emb, positive_emb, negative_emb)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        total_loss += epoch_loss / epoch_batches
        num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Contrastive Epoch {epoch + 1}: Loss = {epoch_loss / epoch_batches:.4f}")
    
    training_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    return {
        'training_time': training_time,
        'avg_loss': avg_loss,
        'samples_per_second': len(dataset) * config['epochs'] / training_time,
        'model_params': sum(p.numel() for p in model.parameters())
    }


def run_benchmark(data_sizes: list, config: dict, output_dir: str = 'benchmark_results'):
    """Run comprehensive benchmark comparing both approaches."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = {
        'autoencoder': [],
        'contrastive': [],
        'data_sizes': data_sizes,
        'config': config
    }
    
    for data_size in data_sizes:
        print(f"\n=== Benchmarking with {data_size:,} samples ===")
        
        # Generate synthetic data
        n_features = config['input_dim']
        data = np.random.randn(data_size, n_features).astype(np.float32)
        
        # Benchmark AutoEncoder
        print("Running AutoEncoder benchmark...")
        ae_results = benchmark_autoencoder(data, config, device)
        results['autoencoder'].append(ae_results)
        
        print(f"AutoEncoder - Time: {ae_results['training_time']:.1f}s, "
              f"Speed: {ae_results['samples_per_second']:.0f} samples/s")
        
        # Benchmark Contrastive Learning
        print("Running Contrastive Learning benchmark...")
        cl_results = benchmark_contrastive(data, config, device)
        results['contrastive'].append(cl_results)
        
        print(f"Contrastive - Time: {cl_results['training_time']:.1f}s, "
              f"Speed: {cl_results['samples_per_second']:.0f} samples/s")
        
        # Speed comparison
        speedup = cl_results['training_time'] / ae_results['training_time']
        print(f"AutoEncoder is {speedup:.1f}x faster than Contrastive Learning")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    create_benchmark_plots(results, output_path)
    
    return results


def create_benchmark_plots(results: dict, output_path: Path):
    """Create visualization plots for benchmark results."""
    
    data_sizes = results['data_sizes']
    ae_times = [r['training_time'] for r in results['autoencoder']]
    cl_times = [r['training_time'] for r in results['contrastive']]
    ae_speeds = [r['samples_per_second'] for r in results['autoencoder']]
    cl_speeds = [r['samples_per_second'] for r in results['contrastive']]
    
    # Training time comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(data_sizes, ae_times, 'o-', label='AutoEncoder', linewidth=2)
    plt.plot(data_sizes, cl_times, 's-', label='Contrastive Learning', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Training speed comparison
    plt.subplot(1, 2, 2)
    plt.plot(data_sizes, ae_speeds, 'o-', label='AutoEncoder', linewidth=2)
    plt.plot(data_sizes, cl_speeds, 's-', label='Contrastive Learning', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('Samples per Second')
    plt.title('Training Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Speedup plot
    speedups = [cl_times[i] / ae_times[i] for i in range(len(data_sizes))]
    
    plt.figure(figsize=(8, 6))
    plt.plot(data_sizes, speedups, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Dataset Size')
    plt.ylabel('Speedup Factor (AutoEncoder vs Contrastive)')
    plt.title('AutoEncoder Training Speedup')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Add speedup annotations
    for i, (size, speedup) in enumerate(zip(data_sizes, speedups)):
        plt.annotate(f'{speedup:.1f}x', (size, speedup), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig(output_path / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AutoEncoder vs Contrastive Learning")
    parser.add_argument("--data-sizes", nargs='+', type=int, 
                       default=[10000, 50000, 100000, 500000],
                       help="Dataset sizes to benchmark")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--input-dim", type=int, default=50, help="Input feature dimension")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'seq_len': args.seq_len,
        'input_dim': args.input_dim,
        'learning_rate': 1e-3,
        'model_type': 'masked',
        'mask_ratio': 0.15,
        'stride': 1
    }
    
    print("Starting benchmark comparison...")
    print(f"Configuration: {config}")
    print(f"Data sizes: {args.data_sizes}")
    
    # Run benchmark
    results = run_benchmark(args.data_sizes, config, args.output_dir)
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    for i, size in enumerate(args.data_sizes):
        ae_time = results['autoencoder'][i]['training_time']
        cl_time = results['contrastive'][i]['training_time']
        speedup = cl_time / ae_time
        
        print(f"Dataset size: {size:,}")
        print(f"  AutoEncoder: {ae_time:.1f}s")
        print(f"  Contrastive: {cl_time:.1f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print()
    
    avg_speedup = np.mean([results['contrastive'][i]['training_time'] / results['autoencoder'][i]['training_time'] 
                          for i in range(len(args.data_sizes))])
    print(f"Average speedup: {avg_speedup:.1f}x")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()