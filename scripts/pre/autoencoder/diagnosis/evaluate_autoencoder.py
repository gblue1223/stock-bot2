"""
AutoEncoder 임베딩 모델 평가 스크립트

모델 품질을 다각도로 평가:
1. 재구성 오차 (Reconstruction Error)
2. 임베딩 품질 (Silhouette Score, Temporal Coherence)
3. 임베딩 분포 분석
4. 종목별 임베딩 시각화
"""

import sys
import logging
from pathlib import Path
import argparse
import json
from datetime import datetime
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 프로젝트 루트를 경로에 추가
# 스크립트 위치: scripts/pre/autoencoder/diagnosis/evaluate_autoencoder.py
# 프로젝트 루트: 3단계 위 (diagnosis -> pre -> scripts -> root)
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent

# 프로젝트 루트가 sys.path에 없으면 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 작업 디렉토리를 프로젝트 루트로 변경
os.chdir(project_root)

from ai_trader.embedding.autoencoder_model import (
    AutoEncoderEmbedding,
    MaskedAutoEncoder,
)
from ai_trader.embedding.data import AutoEncoderDataLoader
from ai_trader.embedding.evaluation import (
    compute_silhouette_score,
    compute_temporal_coherence,
    evaluate_embedding_quality,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 경로 정보 로깅
logger.info(f"Project root: {project_root}")
logger.info(f"Working directory: {os.getcwd()}")


class AutoEncoderEvaluator:
    """AutoEncoder 모델 평가기"""
    
    def __init__(
        self,
        model_path: str,
        db_path: str,
        device: str = 'cuda',
        output_dir: str = 'scripts/pre/autoencoder/diagnosis/results'
    ):
        self.model_path = Path(model_path)
        self.db_path = Path(db_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.data_loader = None
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Output: {self.output_dir}")
    
    def load_model(self):
        """모델 로드"""
        logger.info("Loading model...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 체크포인트 구조 확인
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # 모델 설정 추출 - 다양한 형식 지원
        config = None
        if 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("Found 'config' in checkpoint")
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
            logger.info("Found 'model_config' in checkpoint")
        
        # config에서 파라미터 추출
        if config:
            input_dim = config.get('input_dim', None)
            embedding_dim = config.get('embedding_dim', 128)
            hidden_dim = config.get('hidden_dim', 256)
            seq_len = config.get('seq_len', 60)
            num_layers = config.get('num_layers', 3)
            dropout = config.get('dropout', 0.1)
            model_type = config.get('model_type', 'standard')
            
            # input_dim이 없으면 state_dict에서 추론
            if input_dim is None:
                logger.warning("input_dim not found in config, inferring from state_dict")
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                # encoder.input_projection.weight의 shape에서 추론
                if 'encoder.input_projection.weight' in state_dict:
                    input_dim = state_dict['encoder.input_projection.weight'].shape[1]
                    logger.info(f"Inferred input_dim: {input_dim}")
                else:
                    logger.warning("Cannot infer input_dim, using default: 50")
                    input_dim = 50
        else:
            # config가 없으면 state_dict에서 추론
            logger.warning("No config found in checkpoint, inferring from state_dict")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # input_dim 추론
            if 'encoder.input_projection.weight' in state_dict:
                input_dim = state_dict['encoder.input_projection.weight'].shape[1]
                hidden_dim = state_dict['encoder.input_projection.weight'].shape[0]
                logger.info(f"Inferred input_dim: {input_dim}, hidden_dim: {hidden_dim}")
            else:
                logger.warning("Cannot infer dimensions, using defaults")
                input_dim = 50
                hidden_dim = 256
            
            # embedding_dim 추론
            if 'encoder.bottleneck.0.weight' in state_dict:
                embedding_dim = state_dict['encoder.bottleneck.0.weight'].shape[0]
                logger.info(f"Inferred embedding_dim: {embedding_dim}")
            else:
                embedding_dim = 128
            
            # 기본값
            seq_len = 60
            num_layers = 3
            dropout = 0.1
            model_type = 'standard'
        
        # 모델 생성
        if model_type == 'masked':
            self.model = MaskedAutoEncoder(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            self.model = AutoEncoderEmbedding(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # 가중치 로드 - 다양한 형식 지원
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info("Loading from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info("Loading from 'state_dict'")
        else:
            # checkpoint 자체가 state_dict인 경우
            state_dict = checkpoint
            logger.info("Loading checkpoint as state_dict directly")
        
        try:
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.info("Attempting to load with strict=False")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded: {model_type}")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Seq len: {seq_len}")
        logger.info(f"  Num layers: {num_layers}")
        
        # 반환할 config 구성
        return_config = {
            'input_dim': input_dim,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'seq_len': seq_len,
            'num_layers': num_layers,
            'dropout': dropout,
            'model_type': model_type
        }
        
        return return_config
    
    def load_data(self, config: dict, max_samples: int = 10000):
        """데이터 로드"""
        logger.info(f"Loading data (max {max_samples} samples)...")
        
        self.data_loader = AutoEncoderDataLoader(
            db_path=str(self.db_path),
            seq_len=config['seq_len'],
            max_samples=max_samples
        )
        
        self.data_loader.connect()
        self.data_loader.load_and_split_data()
        
        logger.info("Data loaded successfully")
    
    def compute_reconstruction_error(self, split: str = 'test') -> dict:
        """재구성 오차 계산"""
        logger.info(f"Computing reconstruction error on {split} set...")
        
        dataset = self.data_loader.get_dataset(split, return_metadata=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0
        )
        
        mse_errors = []
        mae_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 4:
                    sequences, _, _, _ = batch
                else:
                    sequences = batch
                
                sequences = sequences.to(self.device)
                
                # 재구성
                if isinstance(self.model, MaskedAutoEncoder):
                    reconstruction, _, _ = self.model(sequences, mask_ratio=0.0)
                else:
                    reconstruction, _ = self.model(sequences)
                
                # 오차 계산
                mse = F.mse_loss(reconstruction, sequences, reduction='none')
                mae = F.l1_loss(reconstruction, sequences, reduction='none')
                
                mse_errors.append(mse.mean(dim=(1, 2)).cpu().numpy())
                mae_errors.append(mae.mean(dim=(1, 2)).cpu().numpy())
        
        mse_errors = np.concatenate(mse_errors)
        mae_errors = np.concatenate(mae_errors)
        
        results = {
            'mse_mean': float(mse_errors.mean()),
            'mse_std': float(mse_errors.std()),
            'mse_median': float(np.median(mse_errors)),
            'mae_mean': float(mae_errors.mean()),
            'mae_std': float(mae_errors.std()),
            'mae_median': float(np.median(mae_errors)),
            'num_samples': len(mse_errors)
        }
        
        logger.info(f"Reconstruction Error:")
        logger.info(f"  MSE: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
        logger.info(f"  MAE: {results['mae_mean']:.6f} ± {results['mae_std']:.6f}")
        
        return results, mse_errors, mae_errors
    
    def extract_embeddings(self, split: str = 'test', max_samples: int = 5000):
        """임베딩 추출"""
        logger.info(f"Extracting embeddings from {split} set...")
        
        dataset = self.data_loader.get_dataset(split, return_metadata=True)
        
        # 샘플 수 제한
        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0
        )
        
        embeddings = []
        stock_codes = []
        dates = []
        times = []
        
        with torch.no_grad():
            for batch in dataloader:
                sequences, stock_code, date, time = batch
                sequences = sequences.to(self.device)
                
                # 임베딩 추출
                embedding = self.model.encode(sequences)
                
                embeddings.append(embedding.cpu().numpy())
                stock_codes.extend(stock_code)
                dates.extend(date)
                times.extend([float(t) for t in time])
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        logger.info(f"Extracted {len(embeddings)} embeddings")
        
        return embeddings, stock_codes, dates, times
    
    def evaluate_embedding_quality(self, embeddings, stock_codes, times):
        """임베딩 품질 평가"""
        logger.info("Evaluating embedding quality...")
        
        metrics = evaluate_embedding_quality(
            embeddings=embeddings,
            stock_codes=stock_codes,
            timestamps=times,
            window_size=1
        )
        
        logger.info(f"Embedding Quality Metrics:")
        logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"  Temporal Coherence: {metrics['temporal_coherence']:.4f}")
        logger.info(f"  Num Samples: {metrics['num_samples']}")
        logger.info(f"  Num Stocks: {metrics['num_stocks']}")
        logger.info(f"  Embedding Dim: {metrics['embedding_dim']}")
        
        return metrics
    
    def analyze_embedding_distribution(self, embeddings):
        """임베딩 분포 분석"""
        logger.info("Analyzing embedding distribution...")
        
        # 통계량 계산
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        
        # L2 norm 분포
        norms = np.linalg.norm(embeddings, axis=1)
        
        stats = {
            'mean_norm': float(norms.mean()),
            'std_norm': float(norms.std()),
            'min_norm': float(norms.min()),
            'max_norm': float(norms.max()),
            'mean_mean': float(mean.mean()),
            'mean_std': float(std.mean())
        }
        
        logger.info(f"Embedding Distribution:")
        logger.info(f"  Mean L2 norm: {stats['mean_norm']:.4f} ± {stats['std_norm']:.4f}")
        logger.info(f"  Norm range: [{stats['min_norm']:.4f}, {stats['max_norm']:.4f}]")
        
        return stats, norms
    
    def visualize_embeddings(self, embeddings, stock_codes, method='tsne'):
        """임베딩 시각화 (t-SNE 또는 PCA)"""
        logger.info(f"Visualizing embeddings with {method.upper()}...")
        
        # 종목 코드를 숫자 레이블로 변환
        unique_stocks = sorted(list(set(stock_codes)))
        stock_to_idx = {stock: idx for idx, stock in enumerate(unique_stocks)}
        labels = np.array([stock_to_idx[stock] for stock in stock_codes])
        
        # 차원 축소
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        
        # 상위 10개 종목만 표시
        top_stocks = sorted(unique_stocks[:10])
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_stocks)))
        
        for idx, stock in enumerate(top_stocks):
            mask = np.array(stock_codes) == stock
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx]],
                label=stock,
                alpha=0.6,
                s=20
            )
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Embedding Visualization ({method.upper()})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = self.output_dir / f'embeddings_{method}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {output_path}")
    
    def plot_reconstruction_errors(self, mse_errors, mae_errors):
        """재구성 오차 분포 시각화"""
        logger.info("Plotting reconstruction error distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MSE 분포
        axes[0].hist(mse_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(mse_errors.mean(), color='r', linestyle='--', label=f'Mean: {mse_errors.mean():.6f}')
        axes[0].axvline(np.median(mse_errors), color='g', linestyle='--', label=f'Median: {np.median(mse_errors):.6f}')
        axes[0].set_xlabel('MSE')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Reconstruction MSE Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # MAE 분포
        axes[1].hist(mae_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1].axvline(mae_errors.mean(), color='r', linestyle='--', label=f'Mean: {mae_errors.mean():.6f}')
        axes[1].axvline(np.median(mae_errors), color='g', linestyle='--', label=f'Median: {np.median(mae_errors):.6f}')
        axes[1].set_xlabel('MAE')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Reconstruction MAE Distribution')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'reconstruction_errors.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
    
    def plot_embedding_norms(self, norms):
        """임베딩 L2 norm 분포 시각화"""
        logger.info("Plotting embedding norm distribution...")
        
        plt.figure(figsize=(10, 6))
        plt.hist(norms, bins=50, alpha=0.7, edgecolor='black', color='purple')
        plt.axvline(norms.mean(), color='r', linestyle='--', label=f'Mean: {norms.mean():.4f}')
        plt.axvline(np.median(norms), color='g', linestyle='--', label=f'Median: {np.median(norms):.4f}')
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        plt.title('Embedding L2 Norm Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'embedding_norms.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
    
    def run_full_evaluation(self, max_samples: int = 10000):
        """전체 평가 실행"""
        logger.info("=" * 60)
        logger.info("Starting Full Evaluation")
        logger.info("=" * 60)
        
        # 1. 모델 로드
        config = self.load_model()
        
        # 2. 데이터 로드
        self.load_data(config, max_samples=max_samples)
        
        # 3. 재구성 오차 계산
        recon_results, mse_errors, mae_errors = self.compute_reconstruction_error('test')
        
        # 4. 임베딩 추출
        embeddings, stock_codes, dates, times = self.extract_embeddings('test', max_samples=5000)
        
        # 5. 임베딩 품질 평가
        quality_metrics = self.evaluate_embedding_quality(embeddings, stock_codes, times)
        
        # 6. 임베딩 분포 분석
        dist_stats, norms = self.analyze_embedding_distribution(embeddings)
        
        # 7. 시각화
        self.plot_reconstruction_errors(mse_errors, mae_errors)
        self.plot_embedding_norms(norms)
        self.visualize_embeddings(embeddings, stock_codes, method='tsne')
        self.visualize_embeddings(embeddings, stock_codes, method='pca')
        
        # 8. 결과 저장
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'db_path': str(self.db_path),
            'config': config,
            'reconstruction_error': recon_results,
            'embedding_quality': quality_metrics,
            'embedding_distribution': dist_stats
        }
        
        output_path = self.output_dir / 'evaluation_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results: {output_path}")
        
        # 9. 요약 출력
        logger.info("=" * 60)
        logger.info("Evaluation Summary")
        logger.info("=" * 60)
        logger.info(f"Reconstruction MSE: {recon_results['mse_mean']:.6f}")
        logger.info(f"Reconstruction MAE: {recon_results['mae_mean']:.6f}")
        logger.info(f"Silhouette Score: {quality_metrics['silhouette_score']:.4f}")
        logger.info(f"Temporal Coherence: {quality_metrics['temporal_coherence']:.4f}")
        logger.info(f"Mean Embedding Norm: {dist_stats['mean_norm']:.4f}")
        logger.info("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='AutoEncoder 임베딩 모델 평가')
    parser.add_argument(
        '--model',
        type=str,
        default=r'C:\Users\user\Workspace\datasets@20260109\autoencoder\model.pt',
        help='모델 파일 경로'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=r'C:\Users\user\Workspace\datasets@20260109\datasets_norm_all.duckdb',
        help='DuckDB 데이터베이스 경로'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='디바이스 (cuda/cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scripts/pre/autoencoder/diagnosis/results',
        help='결과 저장 디렉토리'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='평가에 사용할 최대 샘플 수'
    )
    
    args = parser.parse_args()
    
    # 평가 실행
    evaluator = AutoEncoderEvaluator(
        model_path=args.model,
        db_path=args.db,
        device=args.device,
        output_dir=args.output
    )
    
    evaluator.run_full_evaluation(max_samples=args.max_samples)
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
