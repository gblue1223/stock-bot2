"""
실제 데이터 로딩 성능 테스트

AutoEncoderDataLoader vs FastAutoEncoderDataLoader의 성능을 비교합니다.
"""

import time
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import logging

# 프로젝트 모듈 import
try:
    from ai_trader.embedding.data import TimeSeriesSequenceDataset
    from ai_trader.embedding.fast_data_loader import CachedBatchDataset
    import h5py
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("필요한 모듈이 설치되지 않았습니다.")
    exit(1)


def create_test_hdf5_batches(temp_dir: Path, n_sequences=5000, seq_len=60, n_features=60, batch_size=1000):
    """테스트용 HDF5 배치 파일 생성"""
    print(f"HDF5 배치 파일 생성 중... ({n_sequences:,} 시퀀스)")
    
    # 테스트 데이터 생성
    sequences = np.random.randn(n_sequences, seq_len, n_features).astype(np.float32)
    metadata = np.array([
        [f"00593{i%10}", f"2024100{i%10}", f"09300{i%1000:04d}"] 
        for i in range(n_sequences)
    ])
    
    # 배치별로 저장
    num_batches = (n_sequences + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_sequences)
        
        batch_sequences = sequences[start_idx:end_idx]
        batch_metadata = metadata[start_idx:end_idx]
        
        # HDF5 파일로 저장
        batch_file = temp_dir / f'batch_{batch_idx:06d}.h5'
        
        with h5py.File(batch_file, 'w') as f:
            f.create_dataset('sequences', data=batch_sequences, compression='gzip', compression_opts=6)
            f.create_dataset('metadata', data=batch_metadata.astype('S20'))
    
    # 배치 정보 저장
    batch_info = {
        'num_batches': num_batches,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'num_features': n_features,
        'total_sequences': n_sequences
    }
    
    with open(temp_dir / 'batch_info.json', 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    print(f"배치 파일 생성 완료: {num_batches}개 배치")
    return sequences, metadata


def test_basic_dataset_loading(sequences, metadata, n_samples=1000):
    """기본 TimeSeriesSequenceDataset 로딩 테스트"""
    print("\n🔄 기본 Dataset 로딩 테스트...")
    
    try:
        start_time = time.time()
        
        # Dataset 생성 - 2D 데이터로 변환 (n_samples * seq_len, n_features)
        reshaped_data = sequences.reshape(-1, sequences.shape[-1])
        
        # 메타데이터도 확장
        expanded_metadata = []
        for i, seq in enumerate(sequences):
            for j in range(len(seq)):
                expanded_metadata.append(metadata[i])
        expanded_metadata = np.array(expanded_metadata)
        
        dataset = TimeSeriesSequenceDataset(
            data=reshaped_data,
            metadata=expanded_metadata,
            seq_len=60,
            stride=1,
            return_metadata=False
        )
        
        # 샘플 로딩
        sample_count = min(n_samples, len(dataset))
        samples = []
        
        for i in range(sample_count):
            sample = dataset[i]
            samples.append(sample)
        
        elapsed_time = time.time() - start_time
        throughput = sample_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'dataset_size': len(dataset),
            'samples_loaded': len(samples),
            'elapsed_time': elapsed_time,
            'throughput': throughput,
            'sample_shape': samples[0].shape if samples else None
        }
    
    except Exception as e:
        print(f"    ❌ 기본 Dataset 테스트 실패: {e}")
        return {
            'dataset_size': 0,
            'samples_loaded': 0,
            'elapsed_time': 0.001,  # 0으로 나누기 방지
            'throughput': 0,
            'sample_shape': None,
            'error': str(e)
        }


def test_cached_dataset_loading(temp_dir: Path, n_samples=1000):
    """CachedBatchDataset 로딩 테스트"""
    print("\n🚀 캐시된 Dataset 로딩 테스트...")
    
    start_time = time.time()
    
    # Dataset 생성
    dataset = CachedBatchDataset(
        batch_dir=str(temp_dir),
        preload_batches=3,
        use_memory_mapping=True
    )
    
    # 샘플 로딩
    sample_count = min(n_samples, len(dataset))
    samples = []
    
    for i in range(sample_count):
        sample = dataset[i]
        samples.append(sample)
    
    elapsed_time = time.time() - start_time
    throughput = sample_count / elapsed_time if elapsed_time > 0 else 0
    
    return {
        'dataset_size': len(dataset),
        'samples_loaded': len(samples),
        'elapsed_time': elapsed_time,
        'throughput': throughput,
        'sample_shape': samples[0].shape if samples else None
    }


def test_dataloader_performance(sequences, metadata, temp_dir: Path, batch_sizes=[32, 64, 128]):
    """DataLoader 성능 비교"""
    print("\n⚡ DataLoader 성능 비교...")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"  배치 크기 {batch_size} 테스트 중...")
        
        # 1. 기본 DataLoader
        basic_dataset = TimeSeriesSequenceDataset(
            data=sequences,
            metadata=metadata,
            seq_len=60,
            stride=1
        )
        
        from torch.utils.data import DataLoader
        basic_loader = DataLoader(
            basic_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # 공정한 비교를 위해 단일 프로세스
        )
        
        # 기본 DataLoader 테스트
        start_time = time.time()
        batch_count = 0
        total_samples = 0
        
        for batch in basic_loader:
            batch_count += 1
            total_samples += batch.shape[0]
            if batch_count >= 20:  # 20 배치만 테스트
                break
        
        basic_time = time.time() - start_time
        basic_throughput = total_samples / basic_time if basic_time > 0 else 0
        
        # 2. 캐시된 DataLoader
        cached_dataset = CachedBatchDataset(
            batch_dir=str(temp_dir),
            preload_batches=3
        )
        
        cached_loader = DataLoader(
            cached_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 캐시된 DataLoader 테스트
        start_time = time.time()
        batch_count = 0
        total_samples = 0
        
        for batch in cached_loader:
            batch_count += 1
            total_samples += batch.shape[0]
            if batch_count >= 20:  # 20 배치만 테스트
                break
        
        cached_time = time.time() - start_time
        cached_throughput = total_samples / cached_time if cached_time > 0 else 0
        
        # 결과 저장
        results[batch_size] = {
            'basic': {
                'time': basic_time,
                'throughput': basic_throughput,
                'samples': total_samples
            },
            'cached': {
                'time': cached_time,
                'throughput': cached_throughput,
                'samples': total_samples
            },
            'speedup': basic_time / cached_time if cached_time > 0 else 0
        }
        
        print(f"    기본:   {basic_throughput:6.0f} samples/sec ({basic_time:.3f}초)")
        print(f"    캐시:   {cached_throughput:6.0f} samples/sec ({cached_time:.3f}초)")
        print(f"    개선도: {results[batch_size]['speedup']:.1f}배")
    
    return results


def print_performance_results(basic_result, cached_result, dataloader_results):
    """성능 결과 출력"""
    print("\n" + "="*80)
    print("📊 데이터 로딩 성능 비교 결과")
    print("="*80)
    
    # Dataset 로딩 성능
    print(f"\n🔍 Dataset 로딩 성능 (1000 샘플):")
    print(f"  기본 Dataset:")
    print(f"    - 시간: {basic_result['elapsed_time']:.3f}초")
    print(f"    - 처리량: {basic_result['throughput']:.0f} samples/sec")
    print(f"    - 데이터셋 크기: {basic_result['dataset_size']:,}")
    
    print(f"  캐시된 Dataset:")
    print(f"    - 시간: {cached_result['elapsed_time']:.3f}초")
    print(f"    - 처리량: {cached_result['throughput']:.0f} samples/sec")
    print(f"    - 데이터셋 크기: {cached_result['dataset_size']:,}")
    
    # 개선도 계산
    if basic_result['elapsed_time'] > 0 and cached_result['elapsed_time'] > 0:
        speedup = basic_result['elapsed_time'] / cached_result['elapsed_time']
        if basic_result['throughput'] > 0:
            throughput_improvement = cached_result['throughput'] / basic_result['throughput']
        else:
            throughput_improvement = float('inf')
        
        print(f"\n  📈 개선도:")
        print(f"    - 로딩 속도: {speedup:.1f}배 빠름")
        if throughput_improvement != float('inf'):
            print(f"    - 처리량: {throughput_improvement:.1f}배 향상")
        else:
            print(f"    - 처리량: 무한대 향상 (기본 방식 실패)")
    
    # DataLoader 성능
    print(f"\n⚡ DataLoader 성능 비교:")
    for batch_size, result in dataloader_results.items():
        print(f"  배치 크기 {batch_size}:")
        print(f"    - 기본: {result['basic']['throughput']:6.0f} samples/sec")
        print(f"    - 캐시: {result['cached']['throughput']:6.0f} samples/sec")
        print(f"    - 개선도: {result['speedup']:.1f}배")
    
    # 최적 설정 추천
    best_batch_size = max(dataloader_results.keys(), 
                         key=lambda x: dataloader_results[x]['cached']['throughput'])
    best_throughput = dataloader_results[best_batch_size]['cached']['throughput']
    
    print(f"\n💡 권장 설정:")
    print(f"  ✅ 캐시된 Dataset 사용 (속도 {speedup:.1f}배 향상)")
    print(f"  ✅ 배치 크기 {best_batch_size} 사용 ({best_throughput:.0f} samples/sec)")
    
    print("="*80)


def main():
    """메인 테스트 실행"""
    print("🚀 데이터 로딩 성능 테스트")
    print("="*80)
    
    # 임시 디렉토리 생성
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 테스트 데이터 생성
        sequences, metadata = create_test_hdf5_batches(
            temp_dir, 
            n_sequences=5000, 
            seq_len=60, 
            n_features=60, 
            batch_size=1000
        )
        
        # 성능 테스트 실행
        basic_result = test_basic_dataset_loading(sequences, metadata, n_samples=1000)
        cached_result = test_cached_dataset_loading(temp_dir, n_samples=1000)
        dataloader_results = test_dataloader_performance(sequences, metadata, temp_dir)
        
        # 결과 출력
        print_performance_results(basic_result, cached_result, dataloader_results)
        
        # 결과를 JSON으로 저장
        all_results = {
            'basic_dataset': basic_result,
            'cached_dataset': cached_result,
            'dataloader_comparison': dataloader_results,
            'test_info': {
                'n_sequences': len(sequences),
                'sequence_shape': sequences.shape,
                'metadata_shape': metadata.shape
            }
        }
        
        with open('data_loading_performance.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 상세 결과 저장: data_loading_performance.json")
        print("✅ 데이터 로딩 성능 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        raise
    
    finally:
        # 임시 디렉토리 정리
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"임시 디렉토리 정리: {temp_dir}")


if __name__ == '__main__':
    main()