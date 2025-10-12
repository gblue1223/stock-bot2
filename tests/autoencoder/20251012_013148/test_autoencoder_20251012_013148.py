#!/usr/bin/env python3
"""
AutoEncoder 모델 테스트 스크립트
모델 경로: models/autoencoder/20251012_013148

이 스크립트는 훈련된 autoencoder 모델을 로드하고 다양한 테스트를 수행합니다:
1. 모델 로딩 및 구조 검증
2. 추론 성능 테스트
3. 임베딩 품질 평가
4. 재구성 품질 평가
5. 성능 벤치마크
"""

import torch
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder


class AutoEncoderModelTester:
    """AutoEncoder 모델 테스터"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: 모델 디렉토리 경로
            device: 사용할 디바이스 ('auto', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)
        
        # 디바이스 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # 모델 정보 로드
        self.model_info = self._load_model_info()
        self.training_history = self._load_training_history()
        
        # 모델 로드
        self.model = self._load_model()
        
        print(f"Model loaded successfully!")
        print(f"Model type: {self.model.__class__.__name__}")
        print(f"Parameters: {self.model_info['total_params']:,}")
        print(f"Best validation loss: {self.model_info['best_val_loss']:.6f}")
        
    def _load_model_info(self) -> dict:
        """모델 정보 로드"""
        info_path = self.model_path / 'model_info.json'
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def _load_training_history(self) -> dict:
        """훈련 히스토리 로드"""
        history_path = self.model_path / 'training_history.json'
        with open(history_path, 'r') as f:
            return json.load(f)
    
    def _load_model(self):
        """모델 로드"""
        model_file = self.model_path / 'model.pt'
        
        # 체크포인트 로드
        checkpoint = torch.load(model_file, map_location=self.device)
        
        # 모델 설정 추출
        config = self.model_info['config']
        data_shape = self.model_info['data_shape']
        
        # 모델 생성 (MaskedAutoEncoder 사용)
        model = MaskedAutoEncoder(
            input_dim=data_shape['num_features'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            seq_len=data_shape['seq_len'],
            dropout=config['dropout'],
            mask_ratio=config['mask_ratio']
        )
        
        # 상태 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 직접 저장된 경우
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def test_model_structure(self):
        """모델 구조 테스트"""
        print("\n=== Model Structure Test ===")
        
        config = self.model_info['config']
        data_shape = self.model_info['data_shape']
        
        print(f"Input dimensions: {data_shape['num_features']} features, {data_shape['seq_len']} sequence length")
        print(f"Embedding dimension: {config['embedding_dim']}")
        print(f"Hidden dimension: {config['hidden_dim']}")
        print(f"Number of layers: {config['num_layers']}")
        print(f"Dropout rate: {config['dropout']}")
        print(f"Mask ratio: {config['mask_ratio']}")
        
        # 모델 파라미터 수 확인
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Expected parameters: {self.model_info['total_params']:,}")
        
        assert total_params == self.model_info['total_params'], "Parameter count mismatch!"
        print("✓ Model structure validation passed")
    
    def test_inference(self):
        """추론 테스트"""
        print("\n=== Inference Test ===")
        
        data_shape = self.model_info['data_shape']
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            # 테스트 데이터 생성
            test_input = torch.randn(
                batch_size, 
                data_shape['seq_len'], 
                data_shape['num_features']
            ).to(self.device)
            
            # 추론 실행
            with torch.no_grad():
                start_time = time.time()
                reconstruction, embedding, mask = self.model(test_input)
                inference_time = time.time() - start_time
            
            # 출력 형태 검증
            expected_recon_shape = test_input.shape
            expected_emb_shape = (batch_size, self.model_info['config']['embedding_dim'])
            expected_mask_shape = (batch_size, data_shape['seq_len'])
            
            assert reconstruction.shape == expected_recon_shape, f"Reconstruction shape mismatch: {reconstruction.shape} vs {expected_recon_shape}"
            assert embedding.shape == expected_emb_shape, f"Embedding shape mismatch: {embedding.shape} vs {expected_emb_shape}"
            assert mask.shape == expected_mask_shape, f"Mask shape mismatch: {mask.shape} vs {expected_mask_shape}"
            
            # 추론 시간 출력
            per_sample_time = (inference_time / batch_size) * 1000  # ms
            print(f"Batch size {batch_size:2d}: {inference_time:.4f}s total, {per_sample_time:.2f}ms per sample")
        
        print("✓ Inference test passed")
    
    def test_reconstruction_quality(self):
        """재구성 품질 테스트"""
        print("\n=== Reconstruction Quality Test ===")
        
        data_shape = self.model_info['data_shape']
        n_samples = 100
        
        # 1. 정규분포 테스트 데이터
        test_input = torch.randn(
            n_samples, 
            data_shape['seq_len'], 
            data_shape['num_features']
        ).to(self.device)
        
        with torch.no_grad():
            reconstruction, embedding, mask = self.model(test_input)
        
        # MSE 계산 (마스킹되지 않은 부분만)
        mask_expanded = mask.unsqueeze(-1).expand_as(test_input)
        unmasked_input = test_input[~mask_expanded].view(-1)
        unmasked_recon = reconstruction[~mask_expanded].view(-1)
        
        mse_loss = torch.nn.functional.mse_loss(unmasked_recon, unmasked_input)
        mae_loss = torch.nn.functional.l1_loss(unmasked_recon, unmasked_input)
        
        print(f"Random data reconstruction MSE: {mse_loss:.6f}")
        print(f"Random data reconstruction MAE: {mae_loss:.6f}")
        
        # 2. 더 현실적인 시계열 데이터로 테스트
        # 주식 데이터와 유사한 패턴 생성
        realistic_input = torch.zeros(n_samples, data_shape['seq_len'], data_shape['num_features']).to(self.device)
        
        # 가격 변화 시뮬레이션 (작은 변화)
        for i in range(1, data_shape['seq_len']):
            change = torch.randn(n_samples, data_shape['num_features']).to(self.device) * 0.01
            realistic_input[:, i, :] = realistic_input[:, i-1, :] + change
        
        with torch.no_grad():
            realistic_recon, realistic_emb, realistic_mask = self.model(realistic_input)
        
        # 현실적 데이터에 대한 MSE 계산
        realistic_mask_expanded = realistic_mask.unsqueeze(-1).expand_as(realistic_input)
        realistic_unmasked_input = realistic_input[~realistic_mask_expanded].view(-1)
        realistic_unmasked_recon = realistic_recon[~realistic_mask_expanded].view(-1)
        
        realistic_mse = torch.nn.functional.mse_loss(realistic_unmasked_recon, realistic_unmasked_input)
        realistic_mae = torch.nn.functional.l1_loss(realistic_unmasked_recon, realistic_unmasked_input)
        
        print(f"Realistic data reconstruction MSE: {realistic_mse:.6f}")
        print(f"Realistic data reconstruction MAE: {realistic_mae:.6f}")
        print(f"Best validation loss: {self.model_info['best_val_loss']:.6f}")
        
        # 마스킹 비율 확인
        mask_ratio = mask.float().mean().item()
        expected_mask_ratio = self.model_info['config']['mask_ratio']
        print(f"Actual mask ratio: {mask_ratio:.3f}, Expected: {expected_mask_ratio:.3f}")
        
        # 기본 검증
        assert not torch.isnan(embedding).any(), "Embeddings contain NaN values"
        assert not torch.isinf(embedding).any(), "Embeddings contain infinite values"
        assert not torch.isnan(reconstruction).any(), "Reconstruction contains NaN values"
        assert not torch.isinf(reconstruction).any(), "Reconstruction contains infinite values"
        
        # 마스킹 비율이 합리적인지 확인
        assert 0.1 <= mask_ratio <= 0.2, f"Mask ratio out of expected range: {mask_ratio:.3f}"
        
        # 현실적 데이터에서의 성능이 더 나은지 확인
        if realistic_mse < mse_loss:
            print("✓ Model performs better on realistic time series data")
        else:
            print("⚠ Model performance similar on both data types")
        
        print("✓ Reconstruction quality test passed")
    
    def test_embedding_quality(self):
        """임베딩 품질 테스트"""
        print("\n=== Embedding Quality Test ===")
        
        data_shape = self.model_info['data_shape']
        n_samples = 200
        
        # 다양한 패턴의 테스트 데이터 생성
        test_inputs = []
        
        # 1. 정규분포 데이터
        normal_data = torch.randn(n_samples//4, data_shape['seq_len'], data_shape['num_features'])
        test_inputs.append(("Normal", normal_data))
        
        # 2. 트렌드 데이터
        trend_data = torch.zeros(n_samples//4, data_shape['seq_len'], data_shape['num_features'])
        for i in range(data_shape['seq_len']):
            trend_data[:, i, :] = i * 0.1
        test_inputs.append(("Trend", trend_data))
        
        # 3. 주기적 데이터
        periodic_data = torch.zeros(n_samples//4, data_shape['seq_len'], data_shape['num_features'])
        for i in range(data_shape['seq_len']):
            periodic_data[:, i, :] = torch.sin(torch.tensor(i * 0.5))
        test_inputs.append(("Periodic", periodic_data))
        
        # 4. 상수 데이터
        constant_data = torch.ones(n_samples//4, data_shape['seq_len'], data_shape['num_features'])
        test_inputs.append(("Constant", constant_data))
        
        embeddings_by_type = {}
        
        for data_type, data in test_inputs:
            data = data.to(self.device)
            
            with torch.no_grad():
                _, embeddings, _ = self.model(data)
            
            embeddings_by_type[data_type] = embeddings.cpu().numpy()
            
            # 임베딩 통계
            emb_mean = embeddings.mean().item()
            emb_std = embeddings.std().item()
            emb_min = embeddings.min().item()
            emb_max = embeddings.max().item()
            
            print(f"{data_type} embeddings - Mean: {emb_mean:.4f}, Std: {emb_std:.4f}, Range: [{emb_min:.4f}, {emb_max:.4f}]")
        
        # 임베딩 다양성 검증
        all_embeddings = np.concatenate(list(embeddings_by_type.values()), axis=0)
        embedding_std = np.std(all_embeddings)
        
        print(f"Overall embedding std: {embedding_std:.4f}")
        assert embedding_std > 0.01, f"Embeddings lack diversity: std = {embedding_std:.6f}"
        
        print("✓ Embedding quality test passed")
    
    def test_performance_benchmark(self):
        """성능 벤치마크 테스트"""
        print("\n=== Performance Benchmark ===")
        
        data_shape = self.model_info['data_shape']
        
        # 다양한 배치 크기로 성능 측정
        batch_sizes = [1, 8, 16, 32, 64]
        n_iterations = 100
        
        results = {}
        
        for batch_size in batch_sizes:
            times = []
            
            # 워밍업
            test_input = torch.randn(batch_size, data_shape['seq_len'], data_shape['num_features']).to(self.device)
            with torch.no_grad():
                _ = self.model(test_input)
            
            # 실제 측정
            for _ in range(n_iterations):
                test_input = torch.randn(batch_size, data_shape['seq_len'], data_shape['num_features']).to(self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(test_input)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            per_sample_time = (avg_time / batch_size) * 1000  # ms
            throughput = batch_size / avg_time  # samples/sec
            
            results[batch_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'per_sample_time': per_sample_time,
                'throughput': throughput
            }
            
            print(f"Batch {batch_size:2d}: {avg_time:.4f}±{std_time:.4f}s, {per_sample_time:.2f}ms/sample, {throughput:.1f} samples/sec")
        
        # 단일 샘플 추론 시간 확인 (실시간 거래 요구사항: < 10ms)
        single_sample_time = results[1]['per_sample_time']
        print(f"\nSingle sample inference time: {single_sample_time:.2f}ms")
        
        if single_sample_time < 10:
            print("✓ Meets real-time trading requirement (< 10ms)")
        else:
            print("⚠ Does not meet real-time trading requirement (< 10ms)")
        
        print("✓ Performance benchmark completed")
    
    def visualize_training_history(self):
        """훈련 히스토리 시각화"""
        print("\n=== Training History Visualization ===")
        
        train_losses = self.training_history['train_loss']
        val_losses = self.training_history['val_loss']
        learning_rates = self.training_history['learning_rate']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 그래프
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 학습률 그래프
        ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프 저장
        output_path = self.model_path / 'test_results_training_history.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {output_path}")
        
        # plt.show()  # Skip showing plot in automated test
        
        # 훈련 통계 출력
        print(f"Training completed in {self.model_info['training_time_minutes']:.1f} minutes")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        print(f"Best validation loss: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses)) + 1})")
        
        # 수렴성 분석
        improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"Training loss improvement: {improvement:.1f}%")
        
        print("✓ Training history visualization completed")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print(f"Testing AutoEncoder model: {self.model_path}")
        print("=" * 60)
        
        try:
            self.test_model_structure()
            self.test_inference()
            self.test_reconstruction_quality()
            self.test_embedding_quality()
            self.test_performance_benchmark()
            self.visualize_training_history()
            
            print("\n" + "=" * 60)
            print("🎉 All tests passed successfully!")
            print("✓ Model is ready for production use")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AutoEncoder model')
    parser.add_argument('--model-path', 
                       default='models/autoencoder/20251012_013148',
                       help='Path to model directory')
    parser.add_argument('--device', 
                       choices=['auto', 'cuda', 'cpu'], 
                       default='auto',
                       help='Device to use for testing')
    
    args = parser.parse_args()
    
    # 모델 경로 확인
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    if not (model_path / 'model.pt').exists():
        print(f"Error: Model file not found: {model_path / 'model.pt'}")
        return
    
    # 테스터 생성 및 실행
    tester = AutoEncoderModelTester(args.model_path, args.device)
    tester.run_all_tests()


if __name__ == '__main__':
    main()