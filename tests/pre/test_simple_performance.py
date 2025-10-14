"""
간단한 성능 테스트 - 실제 실행 가능한 버전

기본 데이터 로딩 vs 최적화된 데이터 로딩의 성능을 비교합니다.
"""

import time
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import logging

# 테스트용 더미 데이터 생성
def create_dummy_data(n_samples=10000, n_features=60, seq_len=60):
    """테스트용 더미 데이터 생성"""
    print(f"더미 데이터 생성 중... ({n_samples:,} 샘플, {n_features} 특징)")
    
    # 시계열 데이터 생성
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # 시퀀스 생성 (슬라이딩 윈도우)
    sequences = []
    for i in range(n_samples - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    
    sequences = np.array(sequences)
    print(f"시퀀스 생성 완료: {sequences.shape}")
    
    return sequences


def test_numpy_loading(sequences):
    """기본 NumPy 배열 로딩 테스트"""
    print("\n🔄 기본 방식 (NumPy 배열) 테스트...")
    
    # 임시 파일에 저장
    temp_file = Path(tempfile.mktemp(suffix='.npy'))
    
    try:
        # 저장 시간 측정
        start_time = time.time()
        np.save(temp_file, sequences)
        save_time = time.time() - start_time
        
        # 로딩 시간 측정
        start_time = time.time()
        loaded_data = np.load(temp_file)
        load_time = time.time() - start_time
        
        # 랜덤 액세스 테스트
        start_time = time.time()
        indices = np.random.randint(0, len(loaded_data), 1000)
        samples = [loaded_data[i] for i in indices]
        access_time = time.time() - start_time
        
        file_size = temp_file.stat().st_size / 1024 / 1024  # MB
        
        return {
            'save_time': save_time,
            'load_time': load_time,
            'access_time': access_time,
            'file_size': file_size,
            'samples_accessed': len(samples)
        }
    
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_hdf5_loading(sequences):
    """HDF5 압축 로딩 테스트"""
    print("\n🚀 최적화 방식 (HDF5 압축) 테스트...")
    
    try:
        import h5py
    except ImportError:
        print("❌ h5py가 설치되지 않음. HDF5 테스트를 건너뜁니다.")
        return None
    
    # 임시 파일에 저장
    temp_file = Path(tempfile.mktemp(suffix='.h5'))
    
    try:
        # 저장 시간 측정 (압축 적용)
        start_time = time.time()
        with h5py.File(temp_file, 'w') as f:
            f.create_dataset('sequences', data=sequences, compression='gzip', compression_opts=6)
        save_time = time.time() - start_time
        
        # 로딩 시간 측정
        start_time = time.time()
        with h5py.File(temp_file, 'r') as f:
            loaded_data = f['sequences'][:]
        load_time = time.time() - start_time
        
        # 랜덤 액세스 테스트
        start_time = time.time()
        with h5py.File(temp_file, 'r') as f:
            indices = np.random.randint(0, len(f['sequences']), 1000)
            samples = [f['sequences'][i] for i in indices]
        access_time = time.time() - start_time
        
        file_size = temp_file.stat().st_size / 1024 / 1024  # MB
        
        return {
            'save_time': save_time,
            'load_time': load_time,
            'access_time': access_time,
            'file_size': file_size,
            'samples_accessed': len(samples)
        }
    
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_memory_mapping(sequences):
    """메모리 맵핑 테스트"""
    print("\n💾 메모리 맵핑 테스트...")
    
    # 임시 파일에 저장
    temp_file = Path(tempfile.mktemp(suffix='.dat'))
    
    try:
        # 메모리 맵핑 파일 생성
        start_time = time.time()
        fp = np.memmap(temp_file, dtype='float32', mode='w+', shape=sequences.shape)
        fp[:] = sequences[:]
        del fp  # 파일 닫기
        save_time = time.time() - start_time
        
        # 메모리 맵핑으로 로딩
        start_time = time.time()
        loaded_data = np.memmap(temp_file, dtype='float32', mode='r', shape=sequences.shape)
        load_time = time.time() - start_time
        
        # 랜덤 액세스 테스트
        start_time = time.time()
        indices = np.random.randint(0, len(loaded_data), 1000)
        samples = [loaded_data[i].copy() for i in indices]  # copy() 필요
        access_time = time.time() - start_time
        
        file_size = temp_file.stat().st_size / 1024 / 1024  # MB
        
        # 메모리 맵핑 객체 명시적으로 삭제
        del loaded_data
        
        return {
            'save_time': save_time,
            'load_time': load_time,
            'access_time': access_time,
            'file_size': file_size,
            'samples_accessed': len(samples)
        }
    
    finally:
        # Windows에서 파일 삭제 재시도
        import gc
        gc.collect()  # 가비지 컬렉션 강제 실행
        
        if temp_file.exists():
            try:
                temp_file.unlink()
            except PermissionError:
                # Windows에서 파일이 사용 중일 때 잠시 대기 후 재시도
                import time as time_module
                time_module.sleep(0.1)
                try:
                    temp_file.unlink()
                except PermissionError:
                    print(f"⚠️ 임시 파일 삭제 실패: {temp_file}")
                    pass


def test_batch_processing(sequences, batch_sizes=[32, 64, 128, 256, 512]):
    """배치 처리 성능 테스트"""
    print("\n⚡ 배치 처리 성능 테스트...")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"  배치 크기 {batch_size} 테스트 중...")
        
        start_time = time.time()
        
        # 배치 단위로 처리
        num_batches = len(sequences) // batch_size
        processed_samples = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = sequences[start_idx:end_idx]
            
            # 간단한 처리 (평균 계산)
            _ = np.mean(batch, axis=(1, 2))
            processed_samples += len(batch)
            
            if i >= 100:  # 100 배치만 테스트
                break
        
        elapsed_time = time.time() - start_time
        throughput = processed_samples / elapsed_time if elapsed_time > 0 else 0
        
        results[batch_size] = {
            'elapsed_time': elapsed_time,
            'processed_samples': processed_samples,
            'throughput': throughput
        }
        
        print(f"    처리량: {throughput:.0f} samples/sec")
    
    return results


def print_comparison_results(numpy_result, hdf5_result, memmap_result, batch_results):
    """결과 비교 출력"""
    print("\n" + "="*80)
    print("📊 성능 테스트 결과 비교")
    print("="*80)
    
    # 파일 크기 비교
    print("\n💾 파일 크기 비교:")
    print(f"  NumPy (.npy):     {numpy_result['file_size']:.1f} MB")
    if hdf5_result:
        print(f"  HDF5 압축 (.h5):  {hdf5_result['file_size']:.1f} MB")
        compression_ratio = numpy_result['file_size'] / hdf5_result['file_size']
        print(f"  압축률:           {compression_ratio:.1f}배")
    print(f"  메모리 맵핑 (.dat): {memmap_result['file_size']:.1f} MB")
    
    # 저장 시간 비교
    print(f"\n💿 저장 시간 비교:")
    print(f"  NumPy:      {numpy_result['save_time']:.2f}초")
    if hdf5_result:
        print(f"  HDF5 압축:  {hdf5_result['save_time']:.2f}초")
    print(f"  메모리 맵핑: {memmap_result['save_time']:.2f}초")
    
    # 로딩 시간 비교
    print(f"\n📂 로딩 시간 비교:")
    print(f"  NumPy:      {numpy_result['load_time']:.2f}초")
    if hdf5_result:
        print(f"  HDF5 압축:  {hdf5_result['load_time']:.2f}초")
        load_speedup = numpy_result['load_time'] / hdf5_result['load_time'] if hdf5_result['load_time'] > 0 else 0
        print(f"  HDF5 개선도: {load_speedup:.1f}배")
    print(f"  메모리 맵핑: {memmap_result['load_time']:.2f}초")
    memmap_speedup = numpy_result['load_time'] / memmap_result['load_time'] if memmap_result['load_time'] > 0 else 0
    print(f"  메모리맵 개선도: {memmap_speedup:.1f}배")
    
    # 랜덤 액세스 시간 비교
    print(f"\n🎯 랜덤 액세스 시간 (1000 샘플):")
    print(f"  NumPy:      {numpy_result['access_time']:.3f}초")
    if hdf5_result:
        print(f"  HDF5 압축:  {hdf5_result['access_time']:.3f}초")
    print(f"  메모리 맵핑: {memmap_result['access_time']:.3f}초")
    
    # 배치 처리 성능
    print(f"\n⚡ 배치 처리 성능 (최적 배치 크기):")
    best_batch_size = max(batch_results.keys(), key=lambda x: batch_results[x]['throughput'])
    best_throughput = batch_results[best_batch_size]['throughput']
    print(f"  최적 배치 크기: {best_batch_size}")
    print(f"  최대 처리량:    {best_throughput:.0f} samples/sec")
    
    # 권장사항
    print(f"\n💡 권장사항:")
    if hdf5_result and hdf5_result['file_size'] < numpy_result['file_size'] * 0.7:
        print(f"  ✅ HDF5 압축 사용 권장 (파일 크기 {compression_ratio:.1f}배 절약)")
    
    if memmap_result['load_time'] < numpy_result['load_time'] * 0.5:
        print(f"  ✅ 메모리 맵핑 사용 권장 (로딩 속도 {memmap_speedup:.1f}배 개선)")
    
    print(f"  ✅ 배치 크기 {best_batch_size} 사용 권장")
    
    print("="*80)


def main():
    """메인 테스트 실행"""
    print("🚀 AutoEncoder 데이터 처리 성능 테스트")
    print("="*80)
    
    # 테스트 데이터 생성
    sequences = create_dummy_data(n_samples=5000, n_features=60, seq_len=60)
    
    try:
        # 각 방식별 테스트 실행
        numpy_result = test_numpy_loading(sequences)
        hdf5_result = test_hdf5_loading(sequences)
        memmap_result = test_memory_mapping(sequences)
        batch_results = test_batch_processing(sequences)
        
        # 결과 비교 출력
        print_comparison_results(numpy_result, hdf5_result, memmap_result, batch_results)
        
        # 결과를 JSON으로 저장
        all_results = {
            'numpy': numpy_result,
            'hdf5': hdf5_result,
            'memmap': memmap_result,
            'batch_processing': batch_results,
            'test_info': {
                'n_sequences': len(sequences),
                'sequence_shape': sequences.shape,
                'data_size_mb': sequences.nbytes / 1024 / 1024
            }
        }
        
        with open('performance_test_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 상세 결과 저장: performance_test_results.json")
        print("✅ 성능 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        raise


if __name__ == '__main__':
    main()