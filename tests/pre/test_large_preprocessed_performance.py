"""
대규모 전처리된 데이터 성능 테스트

실제 전처리된 대용량 데이터를 사용하여 성능을 측정합니다.
"""

import time
import sys
from pathlib import Path
import json

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_trader.embedding.fast_data_loader import FastAutoEncoderDataLoader
import torch


def test_large_dataset_performance(preprocessed_dirs: list):
    """대규모 데이터셋 성능 테스트"""
    print("🚀 대규모 전처리된 데이터 성능 테스트")
    print("="*80)
    
    for preprocessed_dir in preprocessed_dirs:
        print(f"\n📊 테스트 디렉토리: {Path(preprocessed_dir).name}")
        
        # 배치 정보 확인
        batch_info_file = Path(preprocessed_dir) / 'batch_info.json'
        if batch_info_file.exists():
            with open(batch_info_file, 'r') as f:
                batch_info = json.load(f)
            
            print(f"  - 총 배치: {batch_info['num_batches']:,}")
            print(f"  - 배치 크기: {batch_info['batch_size']:,}")
            print(f"  - 총 시퀀스: {batch_info['total_sequences']:,}")
            print(f"  - 시퀀스 길이: {batch_info['seq_len']}")
            print(f"  - 특징 수: {batch_info['num_features']}")
        
        # FastAutoEncoderDataLoader 생성
        try:
            fast_loader = FastAutoEncoderDataLoader(
                preprocessed_dir=preprocessed_dir,
                use_cached_batches=True,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1
            )
            
            # 데이터셋 크기 확인
            train_dataset = fast_loader.get_dataset('train')
            val_dataset = fast_loader.get_dataset('val')
            test_dataset = fast_loader.get_dataset('test')
            
            print(f"\n  📈 데이터셋 분할:")
            print(f"    - 훈련: {len(train_dataset):,} 시퀀스")
            print(f"    - 검증: {len(val_dataset):,} 시퀀스")
            print(f"    - 테스트: {len(test_dataset):,} 시퀀스")
            
            # 다양한 배치 크기로 성능 테스트
            batch_sizes = [64, 128, 256, 512, 1024]
            
            print(f"\n  ⚡ DataLoader 성능 테스트:")
            
            best_throughput = 0
            best_batch_size = 0
            
            for batch_size in batch_sizes:
                try:
                    # DataLoader 생성
                    dataloader = fast_loader.get_dataloader(
                        split='train',
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False
                    )
                    
                    # 성능 측정
                    start_time = time.time()
                    batch_count = 0
                    total_samples = 0
                    
                    for batch in dataloader:
                        batch_count += 1
                        total_samples += batch.shape[0]
                        
                        if batch_count >= 20:  # 20 배치만 테스트
                            break
                    
                    elapsed_time = time.time() - start_time
                    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_batch_size = batch_size
                    
                    print(f"    배치 {batch_size:4d}: {throughput:8.0f} samples/sec ({elapsed_time:.3f}초)")
                    
                except Exception as e:
                    print(f"    배치 {batch_size:4d}: 오류 - {e}")
            
            print(f"\n  🏆 최적 성능: 배치 크기 {best_batch_size}, {best_throughput:.0f} samples/sec")
            
            # 개별 샘플 액세스 테스트
            print(f"\n  🎯 개별 샘플 액세스 테스트:")
            
            # 순차 액세스
            start_time = time.time()
            for i in range(1000):
                if i >= len(train_dataset):
                    break
                sample = train_dataset[i]
            sequential_time = time.time() - start_time
            sequential_throughput = 1000 / sequential_time if sequential_time > 0 else 0
            
            # 랜덤 액세스
            import random
            indices = [random.randint(0, len(train_dataset)-1) for _ in range(1000)]
            
            start_time = time.time()
            for idx in indices:
                sample = train_dataset[idx]
            random_time = time.time() - start_time
            random_throughput = 1000 / random_time if random_time > 0 else 0
            
            print(f"    순차 액세스: {sequential_throughput:6.0f} samples/sec")
            print(f"    랜덤 액세스: {random_throughput:6.0f} samples/sec")
            
            # 샘플 정보
            sample = train_dataset[0]
            print(f"    샘플 형태: {sample.shape}")
            print(f"    데이터 타입: {sample.dtype}")
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
    
    print("\n" + "="*80)


def test_memory_efficiency():
    """메모리 효율성 테스트"""
    try:
        import psutil
        process = psutil.Process()
        
        print(f"\n💾 시스템 메모리 정보:")
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        print(f"  - 프로세스 메모리: {memory_info.rss / 1024**2:.1f} MB")
        print(f"  - 시스템 총 메모리: {system_memory.total / 1024**3:.1f} GB")
        print(f"  - 사용 가능 메모리: {system_memory.available / 1024**3:.1f} GB")
        print(f"  - 메모리 사용률: {system_memory.percent:.1f}%")
        
    except ImportError:
        print(f"\n💾 메모리 정보: psutil이 설치되지 않아 측정할 수 없습니다")


def main():
    # 전처리된 디렉토리들
    preprocessed_dirs = [
        "C:\\Users\\user\\Workspace\\datasets@20251005\\pre_training_data\\2024_10",
        "C:\\Users\\user\\Workspace\\datasets@20251005\\pre_training_data\\2024_10_full",
        "C:\\Users\\user\\Workspace\\datasets@20251005\\pre_training_data\\2024_11"
    ]
    
    # 존재하는 디렉토리만 필터링
    existing_dirs = []
    for dir_path in preprocessed_dirs:
        if Path(dir_path).exists():
            existing_dirs.append(dir_path)
        else:
            print(f"⚠️ 디렉토리가 존재하지 않음: {dir_path}")
    
    if not existing_dirs:
        print("❌ 테스트할 전처리된 디렉토리가 없습니다.")
        return
    
    # 메모리 정보 확인
    test_memory_efficiency()
    
    # 성능 테스트 실행
    test_large_dataset_performance(existing_dirs)
    
    print("✅ 대규모 데이터 성능 테스트 완료!")


if __name__ == '__main__':
    main()