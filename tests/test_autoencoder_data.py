"""
AutoEncoder 데이터 로더 테스트
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from ai_trader.embedding.data import (
    AutoEncoderDataLoader, 
    TimeSeriesSequenceDataset, 
    TradingTaskDataset,
    DataLoadError
)


class TestTimeSeriesSequenceDataset:
    """TimeSeriesSequenceDataset 테스트"""
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        # 더미 데이터 생성
        n_samples = 1000
        n_features = 50
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # 메타데이터 생성 (종목코드, 날짜, 시간)
        metadata = np.array([
            ['STOCK1', '20241001', i] for i in range(n_samples)
        ])
        
        seq_len = 60
        dataset = TimeSeriesSequenceDataset(
            data=data,
            metadata=metadata,
            seq_len=seq_len,
            stride=1
        )
        
        # 데이터셋 길이 확인
        expected_length = n_samples - seq_len + 1
        assert len(dataset) == expected_length
        
        # 샘플 확인
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (seq_len, n_features)
        assert sample.dtype == torch.float32
    
    def test_stride_functionality(self):
        """스트라이드 기능 테스트"""
        n_samples = 200
        n_features = 10
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        metadata = np.array([
            ['STOCK1', '20241001', i] for i in range(n_samples)
        ])
        
        seq_len = 50
        stride = 5
        
        dataset = TimeSeriesSequenceDataset(
            data=data,
            metadata=metadata,
            seq_len=seq_len,
            stride=stride
        )
        
        # 스트라이드를 고려한 길이 확인
        expected_length = (n_samples - seq_len) // stride + 1
        assert len(dataset) == expected_length
    
    def test_metadata_return(self):
        """메타데이터 반환 테스트"""
        n_samples = 100
        n_features = 20
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        metadata = np.array([
            ['STOCK1', '20241001', i] for i in range(n_samples)
        ])
        
        seq_len = 30
        dataset = TimeSeriesSequenceDataset(
            data=data,
            metadata=metadata,
            seq_len=seq_len,
            return_metadata=True
        )
        
        sample = dataset[0]
        assert len(sample) == 4  # (sequence, stock_code, date, start_time)
        
        sequence, stock_code, date, start_time = sample
        assert isinstance(sequence, torch.Tensor)
        assert sequence.shape == (seq_len, n_features)
        assert stock_code == 'STOCK1'
        assert date == '20241001'
        assert start_time == 0.0
    
    def test_mixed_stocks_and_dates(self):
        """여러 종목과 날짜가 섞인 데이터 테스트"""
        # 3개 종목, 2개 날짜로 구성된 데이터
        data_parts = []
        metadata_parts = []
        
        # STOCK1, 20241001 (100 samples)
        data_parts.append(np.random.randn(100, 10).astype(np.float32))
        metadata_parts.append([['STOCK1', '20241001', i] for i in range(100)])
        
        # STOCK2, 20241001 (80 samples)
        data_parts.append(np.random.randn(80, 10).astype(np.float32))
        metadata_parts.append([['STOCK2', '20241001', i] for i in range(80)])
        
        # STOCK1, 20241002 (120 samples)
        data_parts.append(np.random.randn(120, 10).astype(np.float32))
        metadata_parts.append([['STOCK1', '20241002', i] for i in range(120)])
        
        # 데이터 결합
        data = np.vstack(data_parts)
        metadata = np.array([item for sublist in metadata_parts for item in sublist])
        
        seq_len = 30
        dataset = TimeSeriesSequenceDataset(
            data=data,
            metadata=metadata,
            seq_len=seq_len,
            stride=1
        )
        
        # 각 그룹에서 생성 가능한 시퀀스 수 계산
        expected_sequences = (100 - seq_len + 1) + (80 - seq_len + 1) + (120 - seq_len + 1)
        assert len(dataset) == expected_sequences
        
        # 첫 번째와 마지막 샘플 확인
        first_sample = dataset[0]
        last_sample = dataset[-1]
        
        assert first_sample.shape == (seq_len, 10)
        assert last_sample.shape == (seq_len, 10)
    
    def test_insufficient_data(self):
        """데이터가 부족한 경우 테스트"""
        n_samples = 30
        n_features = 5
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        metadata = np.array([
            ['STOCK1', '20241001', i] for i in range(n_samples)
        ])
        
        seq_len = 50  # 데이터보다 긴 시퀀스
        dataset = TimeSeriesSequenceDataset(
            data=data,
            metadata=metadata,
            seq_len=seq_len
        )
        
        # 유효한 시퀀스가 없어야 함
        assert len(dataset) == 0


class TestTradingTaskDataset:
    """TradingTaskDataset 테스트"""
    
    def test_classification_task(self):
        """분류 태스크 테스트"""
        n_samples = 100
        seq_len = 30
        n_features = 20
        n_classes = 3
        
        sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        labels = np.random.randint(0, n_classes, n_samples)
        
        dataset = TradingTaskDataset(
            sequences=sequences,
            labels=labels,
            task_type='classification'
        )
        
        assert len(dataset) == n_samples
        
        sequence, label = dataset[0]
        assert isinstance(sequence, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert sequence.shape == (seq_len, n_features)
        assert sequence.dtype == torch.float32
        assert label.dtype == torch.long  # 분류는 long 타입
    
    def test_regression_task(self):
        """회귀 태스크 테스트"""
        n_samples = 50
        seq_len = 40
        n_features = 15
        
        sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        labels = np.random.randn(n_samples).astype(np.float32)
        
        dataset = TradingTaskDataset(
            sequences=sequences,
            labels=labels,
            task_type='regression'
        )
        
        assert len(dataset) == n_samples
        
        sequence, label = dataset[0]
        assert sequence.shape == (seq_len, n_features)
        assert label.dtype == torch.float32  # 회귀는 float 타입
        assert label.dim() == 0  # 스칼라
    
    def test_multi_output_regression(self):
        """다중 출력 회귀 테스트"""
        n_samples = 30
        seq_len = 25
        n_features = 10
        n_outputs = 3
        
        sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        labels = np.random.randn(n_samples, n_outputs).astype(np.float32)
        
        dataset = TradingTaskDataset(
            sequences=sequences,
            labels=labels,
            task_type='regression'
        )
        
        sequence, label = dataset[0]
        assert sequence.shape == (seq_len, n_features)
        assert label.shape == (n_outputs,)
        assert label.dtype == torch.float32


class TestAutoEncoderDataLoader:
    """AutoEncoderDataLoader 테스트 (DuckDB 없이 기본 기능만)"""
    
    def test_initialization(self):
        """초기화 테스트"""
        loader = AutoEncoderDataLoader(
            db_path="dummy.duckdb",
            table_name="test_table",
            seq_len=60,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        assert loader.db_path == "dummy.duckdb"
        assert loader.table_name == "test_table"
        assert loader.seq_len == 60
        assert loader.train_ratio == 0.7
        assert loader.val_ratio == 0.15
        assert loader.test_ratio == 0.15
    
    def test_invalid_ratios(self):
        """잘못된 비율 테스트"""
        with pytest.raises(ValueError):
            AutoEncoderDataLoader(
                db_path="dummy.duckdb",
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # 합계가 1.1
            )
    
    def test_data_load_error_handling(self):
        """데이터 로딩 오류 처리 테스트"""
        loader = AutoEncoderDataLoader(db_path="nonexistent.duckdb")
        
        with pytest.raises(DataLoadError):
            loader.load_and_split_data()


def test_integration_sequence_to_task():
    """시퀀스 데이터셋에서 태스크 데이터셋으로 변환 통합 테스트"""
    # 1. 시계열 데이터 생성
    n_samples = 500
    n_features = 30
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    metadata = np.array([
        ['STOCK1', '20241001', i] for i in range(n_samples)
    ])
    
    seq_len = 60
    stride = 5
    
    # 2. 시퀀스 데이터셋 생성
    sequence_dataset = TimeSeriesSequenceDataset(
        data=data,
        metadata=metadata,
        seq_len=seq_len,
        stride=stride
    )
    
    # 3. 시퀀스들을 수집
    sequences = []
    for i in range(len(sequence_dataset)):
        seq = sequence_dataset[i]
        sequences.append(seq.numpy())
    
    sequences = np.array(sequences)
    
    # 4. 더미 라벨 생성 (가격 변화 방향 예측)
    labels = np.random.randint(0, 3, len(sequences))  # 0: 하락, 1: 보합, 2: 상승
    
    # 5. 트레이딩 태스크 데이터셋 생성
    task_dataset = TradingTaskDataset(
        sequences=sequences,
        labels=labels,
        task_type='classification'
    )
    
    # 6. 검증
    assert len(task_dataset) == len(sequences)
    
    seq, label = task_dataset[0]
    assert seq.shape == (seq_len, n_features)
    assert label.dtype == torch.long
    assert 0 <= label.item() <= 2


def test_dataloader_creation():
    """DataLoader 생성 테스트"""
    n_samples = 200
    n_features = 20
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    metadata = np.array([
        ['STOCK1', '20241001', i] for i in range(n_samples)
    ])
    
    dataset = TimeSeriesSequenceDataset(
        data=data,
        metadata=metadata,
        seq_len=50,
        stride=2
    )
    
    # DataLoader 생성
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0  # 테스트에서는 0으로 설정
    )
    
    # 배치 확인
    batch = next(iter(dataloader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] <= 16  # 배치 크기
    assert batch.shape[1] == 50   # 시퀀스 길이
    assert batch.shape[2] == 20   # 특징 수


if __name__ == '__main__':
    pytest.main([__file__, '-v'])