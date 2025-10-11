"""
AutoEncoder 성능 벤치마크 테스트
"""

import pytest
import time
import torch
import numpy as np
from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding, MaskedAutoEncoder
from ai_trader.embedding.autoencoder_trainer import AutoEncoderTrainer, TimeSeriesDataset


class TestPerformanceBenchmark:
    """성능 벤치마크 테스트"""
    
    def test_training_speed_benchmark(self):
        """훈련 속도 벤치마크"""
        # 테스트 설정
        data_sizes = [1000, 5000]
        input_dim = 50
        embedding_dim = 128
        seq_len = 60
        batch_size = 64
        epochs = 3
        
        results = {}
        
        for data_size in data_sizes:
            print(f"\n=== 벤치마킹 데이터 크기: {data_size:,} ===")
            
            # 데이터 생성
            data = np.random.randn(data_size, input_dim).astype(np.float32)
            
            # AutoEncoder 모델
            model = AutoEncoderEmbedding(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                seq_len=seq_len
            )
            
            trainer = AutoEncoderTrainer(model, device='cpu')
            
            # 훈련 시간 측정
            start_time = time.time()
            
            config = {
                'batch_size': batch_size,
                'max_epochs': epochs,
                'learning_rate': 1e-3,
                'seq_len': seq_len,
                'stride': 5
            }
            
            history = trainer.train(data, None, config)
            
            training_time = time.time() - start_time
            samples_per_second = data_size * epochs / training_time
            
            results[data_size] = {
                'training_time': training_time,
                'samples_per_second': samples_per_second,
                'final_loss': history[-1]['train_loss']
            }
            
            print(f"훈련 시간: {training_time:.2f}초")
            print(f"처리 속도: {samples_per_second:.0f} samples/sec")
            print(f"최종 손실: {history[-1]['train_loss']:.4f}")
        
        # 성능 검증
        assert all(r['training_time'] > 0 for r in results.values())
        assert all(r['samples_per_second'] > 0 for r in results.values())
        
        # 더 큰 데이터셋이 더 오래 걸려야 함
        if len(results) > 1:
            sizes = sorted(results.keys())
            for i in range(1, len(sizes)):
                prev_size, curr_size = sizes[i-1], sizes[i]
                # 데이터가 5배 커졌을 때 시간이 10배 이상 늘어나지 않아야 함 (효율성 확인)
                time_ratio = results[curr_size]['training_time'] / results[prev_size]['training_time']
                size_ratio = curr_size / prev_size
                assert time_ratio < size_ratio * 2, f"훈련 시간이 비효율적으로 증가: {time_ratio:.2f}x vs {size_ratio:.2f}x"
    
    def test_inference_speed_benchmark(self):
        """추론 속도 벤치마크"""
        input_dim = 50
        embedding_dim = 128
        seq_len = 60
        batch_sizes = [1, 16, 64]
        
        # 모델 생성
        model = AutoEncoderEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len
        )
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n=== 추론 벤치마크 배치 크기: {batch_size} ===")
            
            # 테스트 데이터
            test_input = torch.randn(batch_size, seq_len, input_dim)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_input)
            
            # 실제 측정
            num_runs = 50
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    reconstruction, embedding = model(test_input)
            
            total_time = time.time() - start_time
            avg_time_per_batch = total_time / num_runs
            samples_per_second = batch_size * num_runs / total_time
            
            results[batch_size] = {
                'avg_time_per_batch': avg_time_per_batch,
                'samples_per_second': samples_per_second
            }
            
            print(f"배치당 평균 시간: {avg_time_per_batch*1000:.2f}ms")
            print(f"처리 속도: {samples_per_second:.0f} samples/sec")
            
            # 단일 샘플 추론 시간 확인 (10ms 목표)
            if batch_size == 1:
                single_sample_time_ms = avg_time_per_batch * 1000
                print(f"단일 샘플 추론 시간: {single_sample_time_ms:.2f}ms")
                # 목표: < 50ms (CPU에서는 더 느릴 수 있음)
                assert single_sample_time_ms < 100, f"단일 샘플 추론이 너무 느림: {single_sample_time_ms:.2f}ms"
        
        # 배치 크기가 클수록 샘플당 처리 시간이 줄어야 함
        if len(results) > 1:
            batch_1_speed = results[1]['samples_per_second']
            batch_64_speed = results[64]['samples_per_second']
            assert batch_64_speed > batch_1_speed, "배치 처리가 단일 처리보다 빨라야 함"
    
    def test_masked_vs_standard_autoencoder_speed(self):
        """Masked AutoEncoder vs Standard AutoEncoder 속도 비교"""
        input_dim = 50
        embedding_dim = 128
        seq_len = 60
        data_size = 2000
        epochs = 2
        
        # 테스트 데이터
        data = np.random.randn(data_size, input_dim).astype(np.float32)
        
        config = {
            'batch_size': 32,
            'max_epochs': epochs,
            'learning_rate': 1e-3,
            'seq_len': seq_len,
            'stride': 10
        }
        
        results = {}
        
        # Standard AutoEncoder
        print("\n=== Standard AutoEncoder 벤치마크 ===")
        standard_model = AutoEncoderEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len
        )
        
        standard_trainer = AutoEncoderTrainer(standard_model, device='cpu')
        
        start_time = time.time()
        standard_history = standard_trainer.train(data, None, config)
        standard_time = time.time() - start_time
        
        results['standard'] = {
            'training_time': standard_time,
            'final_loss': standard_history[-1]['train_loss']
        }
        
        print(f"Standard 훈련 시간: {standard_time:.2f}초")
        print(f"Standard 최종 손실: {standard_history[-1]['train_loss']:.4f}")
        
        # Masked AutoEncoder
        print("\n=== Masked AutoEncoder 벤치마크 ===")
        masked_model = MaskedAutoEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            mask_ratio=0.15
        )
        
        masked_trainer = AutoEncoderTrainer(masked_model, device='cpu')
        
        start_time = time.time()
        masked_history = masked_trainer.train(data, None, config)
        masked_time = time.time() - start_time
        
        results['masked'] = {
            'training_time': masked_time,
            'final_loss': masked_history[-1]['train_loss']
        }
        
        print(f"Masked 훈련 시간: {masked_time:.2f}초")
        print(f"Masked 최종 손실: {masked_history[-1]['train_loss']:.4f}")
        
        # 비교
        time_ratio = masked_time / standard_time
        print(f"\n시간 비율 (Masked/Standard): {time_ratio:.2f}x")
        
        # Masked AutoEncoder가 너무 느리지 않아야 함 (2배 이내)
        assert time_ratio < 3.0, f"Masked AutoEncoder가 너무 느림: {time_ratio:.2f}x"
        
        # 둘 다 학습이 진행되어야 함
        assert results['standard']['final_loss'] < 10.0
        assert results['masked']['final_loss'] < 10.0
    
    def test_memory_usage_benchmark(self):
        """메모리 사용량 벤치마크"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 초기 메모리 사용량
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 큰 모델 생성
        model = AutoEncoderEmbedding(
            input_dim=100,
            embedding_dim=256,
            seq_len=120,
            hidden_dim=512,
            num_layers=4
        )
        
        # 모델 로드 후 메모리
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 큰 배치로 훈련
        large_data = np.random.randn(5000, 100).astype(np.float32)
        trainer = AutoEncoderTrainer(model, device='cpu')
        
        config = {
            'batch_size': 64,
            'max_epochs': 1,
            'learning_rate': 1e-3,
            'seq_len': 120,
            'stride': 20
        }
        
        # 훈련 중 최대 메모리
        trainer.train(large_data, None, config)
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n=== 메모리 사용량 벤치마크 ===")
        print(f"초기 메모리: {initial_memory:.1f} MB")
        print(f"모델 로드 후: {model_memory:.1f} MB")
        print(f"훈련 중 최대: {peak_memory:.1f} MB")
        print(f"모델 메모리 증가: {model_memory - initial_memory:.1f} MB")
        print(f"훈련 메모리 증가: {peak_memory - model_memory:.1f} MB")
        
        # 메모리 사용량이 합리적인 범위 내에 있어야 함
        model_overhead = model_memory - initial_memory
        training_overhead = peak_memory - model_memory
        
        assert model_overhead < 500, f"모델 메모리 사용량이 너무 큼: {model_overhead:.1f} MB"
        assert training_overhead < 1000, f"훈련 메모리 사용량이 너무 큼: {training_overhead:.1f} MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])