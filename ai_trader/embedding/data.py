"""
AutoEncoder 임베딩 모델 훈련을 위한 데이터 로더

DuckDB에서 매매 데이터를 로드하고 시계열 시퀀스를 생성합니다.
AutoEncoder 및 Fine-tuning을 위한 데이터 처리를 지원합니다.
"""

import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import duckdb
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """데이터 로딩 중 발생하는 오류"""
    pass


class AutoEncoderDataLoader:
    """
    AutoEncoder 임베딩 모델 훈련을 위한 데이터 로더
    
    DuckDB에서 데이터를 로드하고 시간 순서 기반으로 훈련/검증/테스트 세트로 분할합니다.
    시계열 시퀀스를 생성하여 AutoEncoder 훈련을 지원합니다.
    
    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명 (기본값: 'datasets')
        seq_len: 시퀀스 길이 (기본값: 60)
        train_ratio: 훈련 세트 비율 (기본값: 0.7)
        val_ratio: 검증 세트 비율 (기본값: 0.15)
        test_ratio: 테스트 세트 비율 (기본값: 0.15)
    """
    
    def __init__(
        self,
        db_path: str,
        table_name: str = 'datasets',
        seq_len: int = 60,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        start_date: str = None,
        end_date: str = None,
        stock_codes: list = None,
        max_samples: int = None,
        offset: int = 0,  # 증분 학습을 위한 오프셋
        skip_clipping: bool = False  # 클리핑 건너뛰기 옵션
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.start_date = start_date
        self.end_date = end_date
        self.stock_codes = stock_codes
        self.max_samples = max_samples
        self.offset = offset
        self.skip_clipping = skip_clipping
        
        # 비율 검증
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        self.conn = None
        self.feature_columns = None
        self.data_splits = {}
        
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DataLoadError(f"Cannot connect to database {self.db_path}: {e}")
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def _get_feature_columns(self) -> List[str]:
        """
        특징 컬럼 목록 가져오기
        특정 컬럼를 제외한 모든 컬럼을 특징으로 사용
        """
        if self.feature_columns is not None:
            return self.feature_columns
        
        try:
            query = f"DESCRIBE {self.table_name}"
            columns_df = self.conn.execute(query).fetchdf()
            all_columns = columns_df['column_name'].tolist()
            
            # 메타데이터 컬럼 제외 (문자열 컬럼 및 식별자)
            # 메타데이터 및 원본 시간 컬럼 제외
            # '시간'은 HHMMSSmmm 형식(예: 093000123 = 9천만)이라 너무 큼
            # 대신 '시간_sin', '시간_cos', '시간_scalar' 파생 피처 사용
            # 주의: '시간_scalar'는 메타데이터로 사용되므로 제외하지 않음
            exclude_columns = {'날짜', '종목코드', '번호', '종목명', '시간'}
            self.feature_columns = [col for col in all_columns if col not in exclude_columns]
            
            logger.info(f"Found {len(self.feature_columns)} feature columns")
            return self.feature_columns
            
        except Exception as e:
            logger.error(f"Failed to get feature columns: {e}")
            raise DataLoadError(f"Cannot get feature columns: {e}")

    def load_and_split_data(self) -> Dict[str, np.ndarray]:
        """
        데이터를 로드하고 시간 순서 기반으로 분할
        
        Returns:
            Dict with keys 'train', 'val', 'test', each containing:
                - 'data': numpy array of shape (n_samples, n_features)
                - 'metadata': dict with '종목코드', '날짜', '시간'
        """
        if not self.conn:
            self.connect()
        
        feature_cols = self._get_feature_columns()
        
        try:
            # WHERE 절 구성
            where_clauses = []
            
            if self.start_date:
                # 날짜 형식 변환: YYYY-MM-DD -> YYYYMMDD (문자열)
                start_date_str = self.start_date.replace('-', '')
                where_clauses.append(f"날짜 >= '{start_date_str}'")
            
            if self.end_date:
                # 날짜 형식 변환: YYYY-MM-DD -> YYYYMMDD (문자열)
                end_date_str = self.end_date.replace('-', '')
                where_clauses.append(f"날짜 <= '{end_date_str}'")
            
            if self.stock_codes:
                codes_str = "', '".join(self.stock_codes)
                where_clauses.append(f"종목코드 IN ('{codes_str}')")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # LIMIT 절만 사용 (OFFSET은 성능 저하 유발)
            limit_clause = f"LIMIT {self.max_samples}" if self.max_samples else ""
            
            # 시간 순서로 정렬하여 데이터 로드
            # 정렬 순서: 날짜 → 종목코드 → 시간
            query = f"""
                SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY 날짜, 종목코드, 시간
                {limit_clause}
            """
            
            logger.info(f"Loading data from {self.table_name}...")
            if self.start_date or self.end_date:
                logger.info(f"Date range: {self.start_date or 'start'} ~ {self.end_date or 'end'}")
            if self.stock_codes:
                logger.info(f"Stock codes: {self.stock_codes}")
            if self.max_samples:
                logger.info(f"Max samples: {self.max_samples}")
            
            df = self.conn.execute(query).fetchdf()
            logger.info(f"Loaded {len(df)} rows")
            
            if len(df) == 0:
                # 데이터가 없을 때 실제 날짜 범위 확인
                try:
                    date_range_query = f"SELECT MIN(날짜) as min_date, MAX(날짜) as max_date FROM {self.table_name}"
                    date_range = self.conn.execute(date_range_query).fetchdf()
                    if len(date_range) > 0:
                        logger.error(f"No data found for the specified filters.")
                        logger.error(f"Database date range: {date_range['min_date'].iloc[0]} to {date_range['max_date'].iloc[0]}")
                        if self.start_date or self.end_date:
                            logger.error(f"Requested date range: {self.start_date or 'start'} to {self.end_date or 'end'}")
                except Exception as e:
                    logger.warning(f"Could not retrieve date range: {e}")
                raise DataLoadError("No data found in database")
            
            # 메타데이터와 특징 분리
            metadata = df[['종목코드', '날짜', '시간']].values
            features = df[feature_cols].values.astype(np.float32)
            
            # 데이터 정제: NaN/Inf 처리
            if np.any(np.isnan(features)):
                logger.warning(f"Found {np.sum(np.isnan(features))} NaN values in features. Replacing with 0.")
                features = np.nan_to_num(features, nan=0.0)
            
            if np.any(np.isinf(features)):
                logger.warning(f"Found {np.sum(np.isinf(features))} Inf values in features. Clipping.")
                features = np.nan_to_num(features, posinf=1e10, neginf=-1e10)
            
            # 극단값 클리핑 (선택적)
            if not self.skip_clipping:
                # 각 피처별로 강력하게 클리핑
                logger.info("Applying robust clipping to features...")
                clip_info = []
                
                for i in range(features.shape[1]):
                    col = features[:, i]
                    
                    # 95 percentile 기준 (더 강력한 클리핑)
                    p95 = np.percentile(col, 95)
                    p05 = np.percentile(col, 5)
                    
                    # 절대적 상한선 설정: 10,000
                    # 거래량/거래대금 같은 큰 값도 이 범위 내로 제한
                    ABSOLUTE_MAX = 10000
                    
                    if abs(p95) > ABSOLUTE_MAX or abs(p05) > ABSOLUTE_MAX:
                        # 매우 큰 값: IQR 기반으로 클리핑
                        q75 = np.percentile(col, 75)
                        q25 = np.percentile(col, 25)
                        iqr = q75 - q25
                        median = np.median(col)
                        
                        # median ± 2*IQR로 클리핑 (3 → 2로 더 강하게)
                        p_upper = median + 2 * iqr
                        p_lower = median - 2 * iqr
                        
                        # 그래도 너무 크면 절대 상한선 적용
                        p_upper = min(p_upper, ABSOLUTE_MAX)
                        p_lower = max(p_lower, -ABSOLUTE_MAX)
                        
                        clip_info.append(f"Feature {i}: [{p_lower:.2f}, {p_upper:.2f}]")
                    else:
                        p_upper = p95
                        p_lower = p05
                    
                    features[:, i] = np.clip(col, p_lower, p_upper)
                
                if clip_info:
                    logger.info(f"Clipped {len(clip_info)} features with large values")
                    for info in clip_info[:5]:  # 처음 5개만 로깅
                        logger.info(f"  {info}")
                
                logger.info(f"Data range after clipping: [{np.min(features):.4f}, {np.max(features):.4f}]")
            else:
                logger.info("Skipping clipping (skip_clipping=True)")
                logger.info(f"Data range: [{np.min(features):.4f}, {np.max(features):.4f}]")
            
            # 시간 순서 기반 분할
            n_samples = len(features)
            train_end = int(n_samples * self.train_ratio)
            val_end = int(n_samples * (self.train_ratio + self.val_ratio))
            
            self.data_splits = {
                'train': {
                    'data': features[:train_end],
                    'metadata': metadata[:train_end]
                },
                'val': {
                    'data': features[train_end:val_end],
                    'metadata': metadata[train_end:val_end]
                },
                'test': {
                    'data': features[val_end:],
                    'metadata': metadata[val_end:]
                }
            }
            
            logger.info(f"Data split - Train: {len(self.data_splits['train']['data'])}, "
                       f"Val: {len(self.data_splits['val']['data'])}, "
                       f"Test: {len(self.data_splits['test']['data'])}")
            
            return self.data_splits
            
        except Exception as e:
            logger.error(f"Failed to load and split data: {e}")
            raise DataLoadError(f"Cannot load data: {e}")
    
    def get_dataset(
        self,
        split: str = 'train',
        stride: int = 1,
        return_metadata: bool = False
    ) -> 'TimeSeriesSequenceDataset':
        """
        특정 분할에 대한 Dataset 객체 반환
        
        Args:
            split: 'train', 'val', 또는 'test'
            stride: 시퀀스 생성 시 스트라이드 (기본값: 1)
            return_metadata: 메타데이터 반환 여부 (기본값: False)
            
        Returns:
            TimeSeriesSequenceDataset 객체
        """
        if split not in self.data_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.data_splits.keys())}")
        
        return TimeSeriesSequenceDataset(
            data=self.data_splits[split]['data'],
            metadata=self.data_splits[split]['metadata'],
            seq_len=self.seq_len,
            stride=stride,
            return_metadata=return_metadata
        )
    
    def get_dataloader(
        self,
        split: str = 'train',
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 4,
        stride: int = 1,
        return_metadata: bool = False
    ) -> DataLoader:
        """
        DataLoader 생성
        
        Args:
            split: 'train', 'val', 또는 'test'
            batch_size: 배치 크기
            shuffle: 셔플 여부
            num_workers: 워커 프로세스 수
            stride: 시퀀스 생성 시 스트라이드 (기본값: 1)
            return_metadata: 메타데이터 반환 여부 (기본값: False)
            
        Returns:
            PyTorch DataLoader
        """
        dataset = self.get_dataset(
            split,
            stride=stride,
            return_metadata=return_metadata
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            # 최적화: num_workers > 0일 때만 활성화
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None
        )



class TimeSeriesSequenceDataset(Dataset):
    """
    AutoEncoder 훈련을 위한 시계열 시퀀스 Dataset
    
    시계열 데이터에서 슬라이딩 윈도우 방식으로 시퀀스를 생성합니다.
    동일 종목, 동일 날짜 내에서만 시퀀스를 생성하여 데이터 일관성을 보장합니다.
    
    Args:
        data: 특징 데이터 (n_samples, n_features)
        metadata: 메타데이터 (n_samples, 3) - [종목코드, 날짜, 시간]
        seq_len: 시퀀스 길이 (기본값: 60)
        stride: 슬라이딩 윈도우 스트라이드 (기본값: 1)
        return_metadata: 메타데이터 반환 여부 (기본값: False)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        metadata: np.ndarray,
        seq_len: int = 60,
        stride: int = 1,
        return_metadata: bool = False
    ):
        self.data = data
        self.metadata = metadata
        self.seq_len = seq_len
        self.stride = stride
        self.return_metadata = return_metadata
        
        # 유효한 시퀀스 인덱스 계산
        self.valid_indices = self._compute_valid_indices()
        
        logger.info(f"TimeSeriesSequenceDataset initialized with {len(self.valid_indices)} valid sequences")
        logger.info(f"Sequence length: {seq_len}, Stride: {stride}")
    
    def _compute_valid_indices(self) -> List[int]:
        """
        시퀀스 생성이 가능한 유효한 인덱스 계산
        
        동일 종목, 동일 날짜 내에서만 시퀀스를 생성합니다.
        
        Returns:
            유효한 시작 인덱스 리스트
        """
        n_samples = len(self.data)
        valid_indices = []
        
        i = 0
        while i <= n_samples - self.seq_len:
            # 시퀀스 범위의 메타데이터 확인
            seq_metadata = self.metadata[i:i+self.seq_len]
            stock_codes = seq_metadata[:, 0]
            dates = seq_metadata[:, 1]
            
            # 동일 종목, 동일 날짜인지 확인
            if len(np.unique(stock_codes)) == 1 and len(np.unique(dates)) == 1:
                valid_indices.append(i)
                i += self.stride
            else:
                # 다른 종목이나 날짜가 나타나면 해당 위치로 점프
                # 다음 유효한 시작점 찾기
                next_valid = i + 1
                while (next_valid < n_samples and 
                       (self.metadata[next_valid, 0] == stock_codes[0] and 
                        self.metadata[next_valid, 1] == dates[0])):
                    next_valid += 1
                i = next_valid
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def _get_sequence(self, start_idx: int) -> np.ndarray:
        """
        주어진 시작 인덱스에서 시퀀스 추출
        
        Args:
            start_idx: 시작 인덱스
            
        Returns:
            시퀀스 데이터 (seq_len, n_features)
        """
        return self.data[start_idx:start_idx + self.seq_len]
    
    def __getitem__(self, idx: int):
        """
        배치 아이템 반환
        
        Args:
            idx: 인덱스
            
        Returns:
            return_metadata=False: 시퀀스 텐서 (seq_len, n_features)
            return_metadata=True: (시퀀스, 종목코드, 날짜, 시작시간) 튜플
        """
        start_idx = self.valid_indices[idx]
        
        # 시퀀스 추출
        sequence = self._get_sequence(start_idx)
        
        # Tensor로 변환
        sequence_tensor = torch.from_numpy(sequence).float()
        
        if self.return_metadata:
            # 메타데이터 추출 (시퀀스 시작점 기준)
            stock_code = str(self.metadata[start_idx, 0])
            date = str(self.metadata[start_idx, 1])
            start_time = float(self.metadata[start_idx, 2])
            return sequence_tensor, stock_code, date, start_time
        else:
            return sequence_tensor


class TradingTaskDataset(Dataset):
    """
    Fine-tuning을 위한 트레이딩 태스크 Dataset
    
    시계열 시퀀스와 해당하는 트레이딩 라벨(분류/회귀)을 제공합니다.
    
    Args:
        sequences: 시퀀스 데이터 (n_samples, seq_len, n_features)
        labels: 라벨 데이터 (n_samples,) 또는 (n_samples, n_classes)
        task_type: 태스크 타입 ('classification', 'regression', 'ranking')
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        task_type: str = 'classification'
    ):
        self.sequences = torch.from_numpy(sequences).float()
        self.task_type = task_type
        
        if task_type == 'classification' and labels.ndim == 1:
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = torch.from_numpy(labels).float()
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        """
        배치 아이템 반환: (시퀀스, 라벨)
        
        Args:
            idx: 인덱스
            
        Returns:
            (sequence_tensor, label_tensor) 튜플
        """
        return self.sequences[idx], self.labels[idx]


class PreprocessedDataset(Dataset):
    """전처리된 HDF5 배치 파일들을 로딩하는 데이터셋"""
    
    def __init__(self, data_dir: str, months: List[str], max_batches_per_month: Optional[int] = None):
        """
        Args:
            data_dir: 전처리된 데이터 디렉토리
            months: 사용할 월 리스트 (예: ['2024_09', '2024_10'])
            max_batches_per_month: 월별 최대 배치 수 (메모리 제한용)
        """
        self.data_dir = Path(data_dir)
        self.batch_files = []
        self.batch_info = {}
        
        # 각 월의 배치 파일들 수집
        for month in months:
            month_dir = self.data_dir / month
            if not month_dir.exists():
                logging.warning(f"Month directory not found: {month_dir}")
                continue
                
            # batch_info.json 로드
            batch_info_file = month_dir / 'batch_info.json'
            if batch_info_file.exists():
                with open(batch_info_file, 'r') as f:
                    info = json.load(f)
                    self.batch_info[month] = info
                    
                    # 배치 파일 목록 생성
                    num_batches = info['num_batches']
                    if max_batches_per_month:
                        num_batches = min(num_batches, max_batches_per_month)
                    
                    for batch_idx in range(num_batches):
                        batch_file = month_dir / f'batch_{batch_idx:06d}.h5'
                        if batch_file.exists():
                            self.batch_files.append(str(batch_file))
        
        # 배치 파일 셔플
        random.shuffle(self.batch_files)
        
        logging.info(f"Found {len(self.batch_files)} batch files from {len(months)} months")
        
        # 첫 번째 배치에서 데이터 형태 확인
        if self.batch_files:
            with h5py.File(self.batch_files[0], 'r') as f:
                sample_data = f['sequences'][:1]
                self.seq_len = sample_data.shape[1]
                self.num_features = sample_data.shape[2]
                logging.info(f"Data shape: seq_len={self.seq_len}, num_features={self.num_features}")
    
    def __len__(self):
        return len(self.batch_files)
    
    def __getitem__(self, idx):
        """배치 파일 하나를 로딩"""
        batch_file = self.batch_files[idx]
        
        with h5py.File(batch_file, 'r') as f:
            sequences = f['sequences'][:]
            # metadata = f['metadata'][:]  # 필요시 사용
        
        # numpy -> torch tensor
        sequences = torch.from_numpy(sequences).float()
        return sequences


class BatchCollator:
    """배치들을 하나의 큰 배치로 합치는 콜레이터"""
    
    def __init__(self, max_sequences_per_batch: int = 1000):
        self.max_sequences_per_batch = max_sequences_per_batch
    
    def __call__(self, batch_list):
        """
        여러 배치 파일의 시퀀스들을 하나로 합침
        
        Args:
            batch_list: List of tensors, each of shape (batch_size, seq_len, num_features)
        
        Returns:
            Combined tensor of shape (total_sequences, seq_len, num_features)
        """
        # 모든 배치를 하나로 합침
        all_sequences = torch.cat(batch_list, dim=0)
        
        # 너무 크면 샘플링
        if len(all_sequences) > self.max_sequences_per_batch:
            indices = torch.randperm(len(all_sequences))[:self.max_sequences_per_batch]
            all_sequences = all_sequences[indices]
        
        return all_sequences
