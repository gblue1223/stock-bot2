"""
임베딩 데이터 로더 테스트
"""

import pytest
import numpy as np
import tempfile
import duckdb
from ai_trader.embedding.data import EmbeddingDataLoader, ContrastiveDataset, DataLoadError


@pytest.fixture
def sample_db():
    """테스트용 샘플 데이터베이스 생성"""
    import os
    # 임시 디렉토리에 고유한 파일명 생성
    db_path = os.path.join(tempfile.gettempdir(), f'test_db_{os.getpid()}.duckdb')
    
    # 기존 파일이 있으면 삭제
    if os.path.exists(db_path):
        os.unlink(db_path)
    
    # 샘플 데이터 생성
    conn = duckdb.connect(db_path)
    
    # 테이블 생성
    conn.execute("""
        CREATE TABLE datasets (
            날짜 VARCHAR,
            종목코드 VARCHAR,
            번호 INTEGER,
            등락률 REAL,
            누적거래대금 REAL,
            거래회전율 REAL,
            체결강도 REAL
        )
    """)
    
    # 샘플 데이터 삽입 (2개 종목, 각 100개 샘플)
    np.random.seed(42)
    for stock_code in ['005930', '000660']:
        for i in range(100):
            conn.execute(f"""
                INSERT INTO datasets VALUES (
                    '20240101',
                    '{stock_code}',
                    {i},
                    {np.random.randn()},
                    {np.random.rand() * 1000000},
                    {np.random.rand() * 100},
                    {np.random.rand() * 200}
                )
            """)
    
    conn.close()
    
    yield db_path
    
    # 정리
    import os
    os.unlink(db_path)


def test_data_loader_initialization(sample_db):
    """데이터 로더 초기화 테스트"""
    loader = EmbeddingDataLoader(
        db_path=sample_db,
        table_name='datasets',
        seq_len=10
    )
    
    assert loader.db_path == sample_db
    assert loader.seq_len == 10
    assert loader.train_ratio == 0.7
    assert loader.val_ratio == 0.15
    assert loader.test_ratio == 0.15


def test_data_loader_connection(sample_db):
    """데이터베이스 연결 테스트"""
    loader = EmbeddingDataLoader(db_path=sample_db)
    loader.connect()
    
    assert loader.conn is not None
    
    loader.close()


def test_load_and_split_data(sample_db):
    """데이터 로드 및 분할 테스트"""
    loader = EmbeddingDataLoader(
        db_path=sample_db,
        seq_len=10,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    splits = loader.load_and_split_data()
    
    # 분할 확인
    assert 'train' in splits
    assert 'val' in splits
    assert 'test' in splits
    
    # 데이터 형태 확인
    assert splits['train']['data'].shape[1] == 4  # 4개 특징
    assert splits['val']['data'].shape[1] == 4
    assert splits['test']['data'].shape[1] == 4
    
    # 비율 확인 (대략적으로)
    total_samples = sum(len(splits[k]['data']) for k in ['train', 'val', 'test'])
    assert abs(len(splits['train']['data']) / total_samples - 0.7) < 0.05
    assert abs(len(splits['val']['data']) / total_samples - 0.15) < 0.05
    assert abs(len(splits['test']['data']) / total_samples - 0.15) < 0.05
    
    loader.close()


def test_contrastive_dataset():
    """대조 학습 데이터셋 테스트"""
    # 샘플 데이터 생성
    n_samples = 200
    n_features = 4
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # 메타데이터: 2개 종목, 각 100개 샘플
    metadata = np.array([
        ['005930', '20240101', i] if i < 100 else ['000660', '20240101', i-100]
        for i in range(n_samples)
    ])
    
    dataset = ContrastiveDataset(
        data=data,
        metadata=metadata,
        seq_len=10
    )
    
    # 유효한 시퀀스 개수 확인
    assert len(dataset) > 0
    
    # 샘플 가져오기
    anchor, positive, negative = dataset[0]
    
    # 형태 확인
    assert anchor.shape == (10, 4)
    assert positive.shape == (10, 4)
    assert negative.shape == (10, 4)


def test_positive_pair_generation():
    """긍정 쌍 생성 테스트"""
    n_samples = 100
    n_features = 4
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # 동일 종목, 연속된 시간
    metadata = np.array([
        ['005930', '20240101', i]
        for i in range(n_samples)
    ])
    
    dataset = ContrastiveDataset(
        data=data,
        metadata=metadata,
        seq_len=10,
        positive_time_threshold=10
    )
    
    # 긍정 쌍 찾기
    anchor_idx = dataset.valid_indices[0]
    positive_idx = dataset._find_positive_pair(anchor_idx)
    
    # 긍정 쌍이 존재해야 함
    assert positive_idx is not None
    
    # 동일 종목인지 확인
    assert metadata[anchor_idx, 0] == metadata[positive_idx, 0]
    
    # 시간 차이 확인
    time_diff = abs(int(metadata[anchor_idx, 2]) - int(metadata[positive_idx, 2]))
    assert time_diff < 10


def test_negative_pair_generation():
    """부정 쌍 생성 테스트"""
    n_samples = 200
    n_features = 4
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # 2개 종목
    metadata = np.array([
        ['005930', '20240101', i] if i < 100 else ['000660', '20240101', i-100]
        for i in range(n_samples)
    ])
    
    dataset = ContrastiveDataset(
        data=data,
        metadata=metadata,
        seq_len=10,
        negative_time_threshold=60
    )
    
    # 부정 쌍 찾기
    anchor_idx = dataset.valid_indices[0]
    negative_idx = dataset._find_negative_pair(anchor_idx)
    
    # 부정 쌍이 존재해야 함
    assert negative_idx is not None
    
    # 다른 종목이거나 먼 시간대여야 함
    if metadata[anchor_idx, 0] == metadata[negative_idx, 0]:
        # 동일 종목인 경우 시간 차이가 커야 함
        time_diff = abs(int(metadata[anchor_idx, 2]) - int(metadata[negative_idx, 2]))
        assert time_diff > 60


def test_invalid_split_ratios():
    """잘못된 분할 비율 테스트"""
    with pytest.raises(ValueError):
        EmbeddingDataLoader(
            db_path='dummy.db',
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3  # 합이 1.1
        )


def test_invalid_db_path():
    """존재하지 않는 데이터베이스 경로 테스트"""
    loader = EmbeddingDataLoader(db_path='nonexistent.db')
    
    with pytest.raises(DataLoadError):
        loader.connect()
