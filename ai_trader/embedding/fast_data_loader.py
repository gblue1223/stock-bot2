"""
캐시된 배치를 사용하는 고속 데이터 로더

사전 처리된 HDF5 배치 파일을 사용하여 훈련 속도를 크게 향상시킵니다.
메모리 맵핑과 비동기 로딩을 지원합니다.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Dict
import logging
import threading
import queue
import time

logger = logging.getLogger(__name__)


class CachedBatchDataset(Dataset):
    """사전 처리된 배치 파일을 사용하는 Dataset"""
    
    def __init__(
        self,
        batch_dir: str,
        batch_indices: Optional[List[int]] = None,
        preload_batches: int = 5,
        use_memory_mapping: bool = True
    ):
        """
        Args:
            batch_dir: 배치 파일이 저장된 디렉토리
            batch_indices: 사용할 배치 인덱스 (None이면 모든 배치)
            preload_batches: 미리 로드할 배치 수
            use_memory_mapping: 메모리 맵핑 사용 여부
        """
        self.batch_dir = Path(batch_dir)
        self.preload_batches = preload_batches
        self.use_memory_mapping = use_memory_mapping
        
        # 배치 정보 로드
        with open(self.batch_dir / 'batch_info.json', 'r') as f:
            self.batch_info = json.load(f)
        
        # 사용할 배치 인덱스 결정
        if batch_indices is None:
            self.batch_indices = list(range(self.batch_info['num_batches']))
        else:
            self.batch_indices = batch_indices
        
        # 배치 파일 경로 생성
        self.batch_files = [
            self.batch_dir / f'batch_{idx:06d}.h5'
            for idx in self.batch_indices
        ]
        
        # 총 시퀀스 수 계산
        self.total_sequences = 0
        self.batch_sequence_counts = []
        
        for batch_file in self.batch_files:
            with h5py.File(batch_file, 'r') as f:
                count = f['sequences'].shape[0]
                self.batch_sequence_counts.append(count)
                self.total_sequences += count
        
        # 배치 캐시
        self.batch_cache = {}
        self.cache_lock = threading.Lock()
        
        # 비동기 로더
        self.preload_queue = queue.Queue(maxsize=preload_batches)
        self.preload_thread = None
        self.stop_preloading = threading.Event()
        
        logger.info(f"CachedBatchDataset initialized:")
        logger.info(f"  - Batches: {len(self.batch_files)}")
        logger.info(f"  - Total sequences: {self.total_sequences}")
        logger.info(f"  - Preload batches: {preload_batches}")
    
    def __len__(self):
        return self.total_sequences
    
    def _get_batch_and_local_index(self, global_index: int):
        """글로벌 인덱스를 배치 인덱스와 로컬 인덱스로 변환"""
        current_sum = 0
        for batch_idx, count in enumerate(self.batch_sequence_counts):
            if global_index < current_sum + count:
                local_index = global_index - current_sum
                return batch_idx, local_index
            current_sum += count
        
        raise IndexError(f"Index {global_index} out of range")
    
    def _load_batch(self, batch_idx: int) -> np.ndarray:
        """배치 파일 로드"""
        batch_file = self.batch_files[batch_idx]
        
        with h5py.File(batch_file, 'r') as f:
            sequences = f['sequences'][:]
        
        return sequences
    
    def _get_cached_batch(self, batch_idx: int) -> np.ndarray:
        """캐시된 배치 가져오기 (없으면 로드)"""
        with self.cache_lock:
            if batch_idx not in self.batch_cache:
                # 캐시 크기 제한
                if len(self.batch_cache) >= self.preload_batches:
                    # LRU 방식으로 오래된 배치 제거
                    oldest_key = next(iter(self.batch_cache))
                    del self.batch_cache[oldest_key]
                
                # 새 배치 로드
                self.batch_cache[batch_idx] = self._load_batch(batch_idx)
            
            return self.batch_cache[batch_idx]
    
    def __getitem__(self, index: int):
        batch_idx, local_idx = self._get_batch_and_local_index(index)
        
        # 배치 데이터 가져오기
        batch_data = self._get_cached_batch(batch_idx)
        sequence = batch_data[local_idx]
        
        return torch.from_numpy(sequence).float()


class MemoryMappedDataset(Dataset):
    """메모리 맵핑을 사용하는 Dataset"""
    
    def __init__(self, data_dir: str, filename: str = 'sequences.dat'):
        """
        Args:
            data_dir: 데이터 디렉토리
            filename: 메모리 맵핑 파일명
        """
        self.data_dir = Path(data_dir)
        self.filename = filename
        
        # 메타데이터 로드
        with open(self.data_dir / f'{filename}.meta', 'r') as f:
            self.metadata = json.load(f)
        
        # 메모리 맵핑 파일 열기
        self.data = np.memmap(
            self.data_dir / filename,
            dtype=self.metadata['dtype'],
            mode='r',
            shape=tuple(self.metadata['shape'])
        )
        
        logger.info(f"MemoryMappedDataset initialized:")
        logger.info(f"  - Shape: {self.data.shape}")
        logger.info(f"  - File: {self.data_dir / filename}")
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: int):
        sequence = self.data[index]
        return torch.from_numpy(sequence.copy()).float()


class FastAutoEncoderDataLoader:
    """고속 AutoEncoder 데이터 로더"""
    
    def __init__(
        self,
        preprocessed_dir: str,
        use_cached_batches: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ):
        """
        Args:
            preprocessed_dir: 전처리된 데이터 디렉토리
            use_cached_batches: 캐시된 배치 사용 여부
            train_ratio: 훈련 세트 비율
            val_ratio: 검증 세트 비율
            test_ratio: 테스트 세트 비율
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.use_cached_batches = use_cached_batches
        
        # 비율 검증
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 배치 정보 로드
        if use_cached_batches:
            with open(self.preprocessed_dir / 'batch_info.json', 'r') as f:
                self.batch_info = json.load(f)
            
            # 배치 분할
            num_batches = self.batch_info['num_batches']
            train_end = int(num_batches * train_ratio)
            val_end = int(num_batches * (train_ratio + val_ratio))
            
            self.train_batch_indices = list(range(0, train_end))
            self.val_batch_indices = list(range(train_end, val_end))
            self.test_batch_indices = list(range(val_end, num_batches))
            
            logger.info(f"Batch split - Train: {len(self.train_batch_indices)}, "
                       f"Val: {len(self.val_batch_indices)}, "
                       f"Test: {len(self.test_batch_indices)}")
    
    def get_dataset(self, split: str = 'train') -> Dataset:
        """Dataset 객체 반환"""
        if self.use_cached_batches:
            if split == 'train':
                batch_indices = self.train_batch_indices
            elif split == 'val':
                batch_indices = self.val_batch_indices
            elif split == 'test':
                batch_indices = self.test_batch_indices
            else:
                raise ValueError(f"Invalid split: {split}")
            
            return CachedBatchDataset(
                batch_dir=self.preprocessed_dir,
                batch_indices=batch_indices,
                preload_batches=5
            )
        else:
            # 메모리 맵핑 사용
            return MemoryMappedDataset(
                data_dir=self.preprocessed_dir,
                filename=f'{split}_sequences.dat'
            )
    
    def get_dataloader(
        self,
        split: str = 'train',
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """DataLoader 생성"""
        dataset = self.get_dataset(split)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None
        )


class AsyncBatchLoader:
    """비동기 배치 로더"""
    
    def __init__(self, batch_files: List[Path], preload_count: int = 3):
        self.batch_files = batch_files
        self.preload_count = preload_count
        self.batch_queue = queue.Queue(maxsize=preload_count)
        self.stop_event = threading.Event()
        self.loader_thread = None
    
    def start(self):
        """비동기 로딩 시작"""
        self.loader_thread = threading.Thread(target=self._load_batches)
        self.loader_thread.start()
    
    def stop(self):
        """비동기 로딩 중지"""
        self.stop_event.set()
        if self.loader_thread:
            self.loader_thread.join()
    
    def _load_batches(self):
        """배치 로딩 스레드"""
        for batch_file in self.batch_files:
            if self.stop_event.is_set():
                break
            
            try:
                with h5py.File(batch_file, 'r') as f:
                    batch_data = f['sequences'][:]
                
                # 큐에 추가 (블로킹)
                self.batch_queue.put(batch_data, timeout=10)
                
            except Exception as e:
                logger.error(f"Error loading batch {batch_file}: {e}")
    
    def get_batch(self, timeout: float = 30.0) -> Optional[np.ndarray]:
        """배치 가져오기"""
        try:
            return self.batch_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# 사용 예제
if __name__ == "__main__":
    # 전처리된 데이터로 빠른 로더 생성
    fast_loader = FastAutoEncoderDataLoader(
        preprocessed_dir="preprocessed_data",
        use_cached_batches=True
    )
    
    # 훈련 데이터 로더
    train_loader = fast_loader.get_dataloader(
        split='train',
        batch_size=256,
        shuffle=True,
        num_workers=4
    )
    
    # 속도 테스트
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 100:  # 100 배치만 테스트
            break
    
    elapsed = time.time() - start_time
    print(f"Loaded 100 batches in {elapsed:.2f} seconds")
    print(f"Throughput: {100/elapsed:.2f} batches/sec")