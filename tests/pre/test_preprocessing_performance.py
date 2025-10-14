"""
사전 작업 스크립트들의 성능 테스트

실제 데이터를 사용하여 전처리 및 데이터 로딩 성능을 벤치마크합니다.
기존 방식 vs 최적화 방식의 성능 비교를 제공합니다.
"""

import pytest
import time
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import psutil
import torch
from torch.utils.data import DataLoader

# 테스트 대상 모듈들
from ai_trader.embedding.data import AutoEncoderDataLoader, TimeSeriesSequenceDataset
from ai_trader.embedding.fast_data_loader import FastAutoEncoderDataLoader, CachedBatchDataset
from scripts.preprocess_for_autoencoder import AutoEncoderPreprocessor

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """성능 벤치마크 클래스"""
    
    def __init__(self):
        self.results = {}
        self.temp_dirs = []
    
    def cleanup(self):
        """임시 디렉토리 정리"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def create_test_database(self, num_samples: int = 100000) -> str:
        """테스트용 DuckDB 생성"""
        import duckdb
        
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        db_path = temp_dir / "test_data.duckdb"
        
        # 테스트 데이터 생성
        np.random.seed(42)
        
        # 메타데이터
        dates = np.random.choice(['20241001', '20241002', '20241003'], num_samples)
        stock_codes = np.random.choice(['005930', '000660', '035420'], num_samples)
        times = np.random.randint(90000000, 153000000, num_samples)  # 9시~15시30분
        
        # 특징 데이터 (60개 컬럼)
        feature_data = np.random.randn(num_samples, 60).astype(np.float32)
        
        # DuckDB에 저장
        conn = duckdb.connect(str(db_path))
        
        # 테이블 생성
        columns = ['날짜', '종목코드', '시간'] + [f'feature_{i}' for i in range(60)]
        
        # 데이터 결합
        all_data = np.column_stack([
            dates, stock_codes, times, feature_data
        ])
        
        # DataFrame으로 변환하여 저장
        import pandas as pd
        df = pd.DataFrame(all_data, columns=columns)
        
        # 특징 컬럼들을 float로 변환
        for i in range(60):
            df[f'feature_{i}'] = df[f'feature_{i}'].astype(float)
        
        conn.execute("CREATE TABLE datasets AS SELECT * FROM df")
        conn.close()
        
        logger.info(f"Created test database with {num_samples} samples: {db_path}")
        return str(db_path)
    
    def measure_time_and_memory(self, func, *args, **kwargs) -> Tuple[float, Dict, any]:
        """시간과 메모리 사용량 측정"""
        process = psutil.Process()
        
        # 시작 메모리
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 시간 측정
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 종료 메모리
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            'elapsed_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'peak_memory': end_memory
        }
        
        return metrics['elapsed_time'], metrics, result


class TestPreprocessingPerformance:
    """전처리 성능 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.benchmark = PerformanceBenchmark()
        logging.basicConfig(level=logging.INFO)
    
    def teardown_method(self):
        """테스트 정리"""
        self.benchmark.cleanup()
    
    def test_basic_vs_preprocessed_loading(self):
        """기본 로딩 vs 전처리된 로딩 성능 비교"""
        # 테스트 데이터 생성
        db_path = self.benchmark.create_test_database(50000)  # 5만 샘플
        
        # 1. 기본 방식 (DuckDB 직접 로딩)
        print("\n=== 기본 방식 (DuckDB 직접 로딩) ===")
        
        def basic_loading():
            loader = AutoEncoderDataLoader(
                db_path=db_path,
                seq_len=60,
                max_samples=10000
            )
            loader.load_and_split_data()
            dataset = loader.get_dataset('train', stride=1)
            
            # 데이터 로딩 테스트
            samples = []
            for i in range(min(1000, len(dataset))):
                samples.append(dataset[i])
            
            return len(samples)
        
        basic_time, basic_metrics, basic_result = self.benchmark.measure_time_and_memory(basic_loading)
        
        # 2. 전처리 방식
        print("\n=== 전처리 방식 ===")
        
        # 2-1. 전처리 수행
        temp_dir = Path(tempfile.mkdtemp())
        self.benchmark.temp_dirs.append(temp_dir)
        
        def preprocessing():
            preprocessor = AutoEncoderPreprocessor(db_path, str(temp_dir))
            preprocessor.connect()
            
            try:
                # 정규화 파라미터 계산
                preprocessor.compute_normalization_params(sample_size=10000)
                
                # 배치 생성
                batch_info = preprocessor.create_sequence_batches(
                    seq_len=60,
                    stride=1,
                    batch_size=1000,
                    max_samples=10000
                )
                return batch_info
            finally:
                preprocessor.close()
        
        preprocess_time, preprocess_metrics, batch_info = self.benchmark.measure_time_and_memory(preprocessing)
        
        # 2-2. 전처리된 데이터 로딩
        def preprocessed_loading():
            fast_loader = FastAutoEncoderDataLoader(
                preprocessed_dir=str(temp_dir),
                use_cached_batches=True
            )
            dataset = fast_loader.get_dataset('train')
            
            # 데이터 로딩 테스트
            samples = []
            for i in range(min(1000, len(dataset))):
                samples.append(dataset[i])
            
            return len(samples)
        
        preprocessed_time, preprocessed_metrics, preprocessed_result = self.benchmark.measure_time_and_memory(preprocessed_loading)
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"성능 비교 결과 (샘플 수: {basic_result})")
        print(f"{'='*60}")
        print(f"기본 방식:")
        print(f"  - 로딩 시간: {basic_time:.2f}초")
        print(f"  - 메모리 사용: {basic_metrics['memory_used']:.1f}MB")
        print(f"  - 초당 샘플: {basic_result/basic_time:.0f} samples/sec")
        
        print(f"\n전처리 방식:")
        print(f"  - 전처리 시간: {preprocess_time:.2f}초 (1회만)")
        print(f"  - 로딩 시간: {preprocessed_time:.2f}초")
        print(f"  - 메모리 사용: {preprocessed_metrics['memory_used']:.1f}MB")
        print(f"  - 초당 샘플: {preprocessed_result/preprocessed_time:.0f} samples/sec")
        
        # 성능 개선도 계산
        speedup = basic_time / preprocessed_time
        memory_reduction = (basic_metrics['memory_used'] - preprocessed_metrics['memory_used']) / basic_metrics['memory_used'] * 100
        
        print(f"\n개선도:")
        print(f"  - 로딩 속도: {speedup:.1f}배 빠름")
        print(f"  - 메모리 절약: {memory_reduction:.1f}%")
        
        # 전처리 오버헤드 고려
        total_preprocessed_time = preprocess_time + preprocessed_time
        if total_preprocessed_time < basic_time:
            print(f"  - 전처리 포함 총 시간도 {basic_time/total_preprocessed_time:.1f}배 빠름")
        else:
            break_even = basic_time / preprocessed_time
            print(f"  - {break_even:.0f}회 이상 사용 시 전처리 비용 회수")
        
        # 검증
        assert speedup > 1.5, f"로딩 속도가 1.5배 미만 개선됨: {speedup:.1f}배"
        assert preprocessed_result == basic_result, "로딩된 샘플 수가 다름"
    
    def test_dataloader_performance(self):
        """DataLoader 성능 비교"""
        # 테스트 데이터 생성
        db_path = self.benchmark.create_test_database(30000)
        
        # 전처리 수행
        temp_dir = Path(tempfile.mkdtemp())
        self.benchmark.temp_dirs.append(temp_dir)
        
        preprocessor = AutoEncoderPreprocessor(db_path, str(temp_dir))
        preprocessor.connect()
        
        try:
            preprocessor.compute_normalization_params(sample_size=5000)
            preprocessor.create_sequence_batches(
                seq_len=60,
                stride=1,
                batch_size=500,
                max_samples=5000
            )
        finally:
            preprocessor.close()
        
        # 1. 기본 DataLoader
        print("\n=== DataLoader 성능 비교 ===")
        
        def basic_dataloader_test():
            loader = AutoEncoderDataLoader(
                db_path=db_path,
                seq_len=60,
                max_samples=5000
            )
            loader.load_and_split_data()
            
            dataloader = loader.get_dataloader(
                split='train',
                batch_size=64,
                shuffle=False,
                num_workers=0  # 단일 프로세스로 공정한 비교
            )
            
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 20:  # 20 배치만 테스트
                    break
            
            return batch_count
        
        basic_dl_time, basic_dl_metrics, basic_batches = self.benchmark.measure_time_and_memory(basic_dataloader_test)
        
        # 2. 고속 DataLoader
        def fast_dataloader_test():
            fast_loader = FastAutoEncoderDataLoader(
                preprocessed_dir=str(temp_dir),
                use_cached_batches=True
            )
            
            dataloader = fast_loader.get_dataloader(
                split='train',
                batch_size=64,
                shuffle=False,
                num_workers=0  # 단일 프로세스로 공정한 비교
            )
            
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 20:  # 20 배치만 테스트
                    break
            
            return batch_count
        
        fast_dl_time, fast_dl_metrics, fast_batches = self.benchmark.measure_time_and_memory(fast_dataloader_test)
        
        # 결과 출력
        print(f"\nDataLoader 성능 비교 (20 배치):")
        print(f"기본 DataLoader: {basic_dl_time:.2f}초 ({20/basic_dl_time:.1f} batches/sec)")
        print(f"고속 DataLoader: {fast_dl_time:.2f}초 ({20/fast_dl_time:.1f} batches/sec)")
        
        speedup = basic_dl_time / fast_dl_time
        print(f"성능 개선: {speedup:.1f}배 빠름")
        
        # 검증
        assert speedup > 2.0, f"DataLoader 속도가 2배 미만 개선됨: {speedup:.1f}배"
        assert basic_batches == fast_batches, "배치 수가 다름"
    
    def test_memory_mapping_performance(self):
        """메모리 맵핑 성능 테스트"""
        print("\n=== 메모리 맵핑 성능 테스트 ===")
        
        # 큰 테스트 데이터 생성
        temp_dir = Path(tempfile.mkdtemp())
        self.benchmark.temp_dirs.append(temp_dir)
        
        # 메모리 맵핑 파일 생성
        n_sequences, seq_len, n_features = 10000, 60, 50
        sequences = np.random.randn(n_sequences, seq_len, n_features).astype(np.float32)
        
        # 1. 일반 numpy 배열 로딩
        def numpy_loading():
            # 파일 저장/로드
            np_file = temp_dir / 'sequences.npy'
            np.save(np_file, sequences)
            
            loaded_data = np.load(np_file)
            
            # 랜덤 액세스 테스트
            indices = np.random.randint(0, len(loaded_data), 1000)
            samples = [loaded_data[i] for i in indices]
            
            return len(samples)
        
        numpy_time, numpy_metrics, numpy_result = self.benchmark.measure_time_and_memory(numpy_loading)
        
        # 2. 메모리 맵핑 로딩
        def memmap_loading():
            # 메모리 맵핑 파일 생성
            memmap_file = temp_dir / 'sequences.dat'
            fp = np.memmap(memmap_file, dtype='float32', mode='w+', shape=sequences.shape)
            fp[:] = sequences[:]
            del fp
            
            # 메타데이터 저장
            metadata = {
                'shape': sequences.shape,
                'dtype': 'float32',
                'filename': 'sequences.dat'
            }
            with open(temp_dir / 'sequences.dat.meta', 'w') as f:
                json.dump(metadata, f)
            
            # 메모리 맵핑으로 로드
            loaded_data = np.memmap(
                memmap_file,
                dtype='float32',
                mode='r',
                shape=sequences.shape
            )
            
            # 랜덤 액세스 테스트
            indices = np.random.randint(0, len(loaded_data), 1000)
            samples = [loaded_data[i].copy() for i in indices]  # copy() 필요
            
            return len(samples)
        
        memmap_time, memmap_metrics, memmap_result = self.benchmark.measure_time_and_memory(memmap_loading)
        
        # 결과 출력
        print(f"\n메모리 맵핑 성능 비교:")
        print(f"NumPy 배열:")
        print(f"  - 시간: {numpy_time:.2f}초")
        print(f"  - 메모리: {numpy_metrics['memory_used']:.1f}MB")
        
        print(f"메모리 맵핑:")
        print(f"  - 시간: {memmap_time:.2f}초")
        print(f"  - 메모리: {memmap_metrics['memory_used']:.1f}MB")
        
        memory_reduction = (numpy_metrics['memory_used'] - memmap_metrics['memory_used']) / numpy_metrics['memory_used'] * 100
        print(f"메모리 절약: {memory_reduction:.1f}%")
        
        # 검증
        assert memmap_result == numpy_result, "로딩된 샘플 수가 다름"
        if memory_reduction > 0:
            print(f"✓ 메모리 사용량 {memory_reduction:.1f}% 절약")
    
    def test_batch_size_optimization(self):
        """배치 크기 최적화 테스트"""
        print("\n=== 배치 크기 최적화 테스트 ===")
        
        # 테스트 데이터 생성
        db_path = self.benchmark.create_test_database(20000)
        
        # 전처리
        temp_dir = Path(tempfile.mkdtemp())
        self.benchmark.temp_dirs.append(temp_dir)
        
        preprocessor = AutoEncoderPreprocessor(db_path, str(temp_dir))
        preprocessor.connect()
        
        try:
            preprocessor.compute_normalization_params(sample_size=5000)
            preprocessor.create_sequence_batches(
                seq_len=60,
                stride=1,
                batch_size=1000,
                max_samples=5000
            )
        finally:
            preprocessor.close()
        
        # 다양한 배치 크기 테스트
        batch_sizes = [32, 64, 128, 256, 512]
        results = {}
        
        for batch_size in batch_sizes:
            def test_batch_size():
                fast_loader = FastAutoEncoderDataLoader(
                    preprocessed_dir=str(temp_dir),
                    use_cached_batches=True
                )
                
                dataloader = fast_loader.get_dataloader(
                    split='train',
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )
                
                batch_count = 0
                total_samples = 0
                
                for batch in dataloader:
                    batch_count += 1
                    total_samples += batch.shape[0]
                    if batch_count >= 10:  # 10 배치만 테스트
                        break
                
                return total_samples
            
            batch_time, batch_metrics, samples = self.benchmark.measure_time_and_memory(test_batch_size)
            
            throughput = samples / batch_time
            results[batch_size] = {
                'time': batch_time,
                'throughput': throughput,
                'memory': batch_metrics['memory_used']
            }
            
            print(f"배치 크기 {batch_size:3d}: {throughput:6.0f} samples/sec, {batch_time:.2f}초, {batch_metrics['memory_used']:4.1f}MB")
        
        # 최적 배치 크기 찾기
        best_batch_size = max(results.keys(), key=lambda x: results[x]['throughput'])
        best_throughput = results[best_batch_size]['throughput']
        
        print(f"\n최적 배치 크기: {best_batch_size} (처리량: {best_throughput:.0f} samples/sec)")
        
        # 검증: 큰 배치가 일반적으로 더 효율적이어야 함
        assert results[256]['throughput'] > results[32]['throughput'], "큰 배치 크기가 더 느림"


def run_comprehensive_benchmark():
    """종합 성능 벤치마크 실행"""
    print("="*80)
    print("AutoEncoder 전처리 종합 성능 벤치마크")
    print("="*80)
    
    # 테스트 실행
    test_class = TestPreprocessingPerformance()
    
    try:
        test_class.setup_method()
        
        print("\n1. 기본 vs 전처리 로딩 성능 비교")
        test_class.test_basic_vs_preprocessed_loading()
        
        print("\n2. DataLoader 성능 비교")
        test_class.test_dataloader_performance()
        
        print("\n3. 메모리 맵핑 성능 테스트")
        test_class.test_memory_mapping_performance()
        
        print("\n4. 배치 크기 최적화 테스트")
        test_class.test_batch_size_optimization()
        
        print("\n" + "="*80)
        print("✓ 모든 성능 테스트 완료!")
        print("="*80)
        
    finally:
        test_class.teardown_method()


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 종합 벤치마크 실행
    run_comprehensive_benchmark()