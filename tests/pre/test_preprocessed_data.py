#!/usr/bin/env python3
"""
전처리된 데이터 로딩 및 검증 테스트
"""

import h5py
import numpy as np
import json
from pathlib import Path
import time

def test_batch_loading(data_dir, month="2024_09", num_batches=5):
    """배치 파일 로딩 테스트"""
    data_path = Path(data_dir) / month
    
    print(f"=== {month} 데이터 로딩 테스트 ===")
    
    # batch_info 로드
    with open(data_path / "batch_info.json", 'r') as f:
        batch_info = json.load(f)
    
    print(f"총 배치 수: {batch_info['num_batches']}")
    print(f"배치 크기: {batch_info['batch_size']}")
    print(f"시퀀스 길이: {batch_info['seq_len']}")
    print(f"특징 수: {batch_info['num_features']}")
    print(f"총 시퀀스 수: {batch_info['total_sequences']:,}")
    
    # 랜덤 배치들 로딩 테스트
    batch_indices = np.random.choice(batch_info['num_batches'], min(num_batches, batch_info['num_batches']), replace=False)
    
    total_load_time = 0
    total_sequences = 0
    
    for i, batch_idx in enumerate(batch_indices):
        batch_file = data_path / f"batch_{batch_idx:06d}.h5"
        
        start_time = time.time()
        with h5py.File(batch_file, 'r') as f:
            sequences = f['sequences'][:]
            metadata = f['metadata'][:]
        load_time = time.time() - start_time
        
        total_load_time += load_time
        total_sequences += len(sequences)
        
        print(f"배치 {batch_idx}: {sequences.shape}, 로딩 시간: {load_time:.3f}초")
        
        # 데이터 품질 검증
        if np.any(np.isnan(sequences)):
            print(f"  ⚠️  NaN 값 발견!")
        if np.any(np.isinf(sequences)):
            print(f"  ⚠️  Inf 값 발견!")
        
        # 통계 정보
        print(f"  평균: {np.mean(sequences):.6f}, 표준편차: {np.std(sequences):.6f}")
        print(f"  최소값: {np.min(sequences):.6f}, 최대값: {np.max(sequences):.6f}")
    
    avg_load_time = total_load_time / len(batch_indices)
    sequences_per_sec = total_sequences / total_load_time
    
    print(f"\n=== 성능 요약 ===")
    print(f"평균 로딩 시간: {avg_load_time:.3f}초/배치")
    print(f"로딩 속도: {sequences_per_sec:,.0f} 시퀀스/초")
    print(f"총 테스트 시퀀스: {total_sequences:,}개")

def test_normalization_consistency(data_dir):
    """정규화 파라미터 일관성 테스트"""
    print("\n=== 정규화 파라미터 일관성 테스트 ===")
    
    months = ["2024_09", "2024_10", "2024_11", "2025_01"]
    norm_params = {}
    
    for month in months:
        norm_file = Path(data_dir) / month / "normalization_params.json"
        if norm_file.exists():
            with open(norm_file, 'r') as f:
                norm_params[month] = json.load(f)
    
    if len(norm_params) < 2:
        print("정규화 파라미터 파일이 충분하지 않습니다.")
        return
    
    # 첫 번째 월을 기준으로 비교
    base_month = list(norm_params.keys())[0]
    base_params = norm_params[base_month]
    
    print(f"기준 월: {base_month}")
    print(f"특징 수: {len(base_params['mean'])}")
    
    for month, params in norm_params.items():
        if month == base_month:
            continue
            
        mean_diff = np.mean(np.abs(np.array(params['mean']) - np.array(base_params['mean'])))
        std_diff = np.mean(np.abs(np.array(params['std']) - np.array(base_params['std'])))
        
        print(f"{month}: 평균 차이 = {mean_diff:.6f}, 표준편차 차이 = {std_diff:.6f}")

if __name__ == "__main__":
    data_dir = r"C:\Users\user\Workspace\datasets@20251005\pre_training_data"
    
    # 여러 월 테스트
    test_months = ["2024_09", "2024_10", "2025_01"]
    
    for month in test_months:
        month_path = Path(data_dir) / month
        if month_path.exists():
            test_batch_loading(data_dir, month, num_batches=3)
            print()
    
    # 정규화 일관성 테스트
    test_normalization_consistency(data_dir)