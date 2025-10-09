"""
임베딩 모델 훈련을 위한 데이터 로더

DuckDB에서 매매 데이터를 로드하고 대조 학습을 위한 긍정/부정 쌍을 생성합니다.
"""

import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import duckdb
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """데이터 로딩 중 발생하는 오류"""
    pass


class EmbeddingDataLoader:
    """
    임베딩 모델 훈련을 위한 데이터 로더
    
    DuckDB에서 데이터를 로드하고 시간 순서 기반으로 훈련/검증/테스트 세트로 분할합니다.
    대조 학습을 위한 긍정/부정 쌍을 생성합니다.
    
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
        offset: int = 0  # 증분 학습을 위한 오프셋
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
                where_clauses.append(f"날짜 >= '{self.start_date}'")
            
            if self.end_date:
                where_clauses.append(f"날짜 <= '{self.end_date}'")
            
            if self.stock_codes:
                codes_str = "', '".join(self.stock_codes)
                where_clauses.append(f"종목코드 IN ('{codes_str}')")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # OFFSET과 LIMIT 절 구성 (증분 학습 지원)
            offset_clause = f"OFFSET {self.offset}" if self.offset > 0 else ""
            limit_clause = f"LIMIT {self.max_samples}" if self.max_samples else ""
            
            # 시간 순서로 정렬하여 데이터 로드
            # 정렬 순서: 날짜 → 종목코드 → 시간
            query = f"""
                SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY 날짜, 종목코드, 시간
                {limit_clause}
                {offset_clause}
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
            
            # 극단값 클리핑 (정규화 전)
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
        return_metadata: bool = False,
        positive_time_threshold: int = 10,
        negative_time_threshold: int = 60
    ) -> 'ContrastiveDataset':
        """
        특정 분할에 대한 Dataset 객체 반환
        
        Args:
            split: 'train', 'val', 또는 'test'
            return_metadata: 메타데이터 반환 여부 (기본값: False)
            positive_time_threshold: 긍정 쌍 시간 임계값 (초, 기본값: 10)
            negative_time_threshold: 부정 쌍 시간 임계값 (초, 기본값: 60)
            
        Returns:
            ContrastiveDataset 객체
        """
        if split not in self.data_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.data_splits.keys())}")
        
        return ContrastiveDataset(
            data=self.data_splits[split]['data'],
            metadata=self.data_splits[split]['metadata'],
            seq_len=self.seq_len,
            positive_time_threshold=positive_time_threshold,
            negative_time_threshold=negative_time_threshold,
            return_metadata=return_metadata
        )
    
    def get_dataloader(
        self,
        split: str = 'train',
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 4,
        return_metadata: bool = False,
        positive_time_threshold: int = 10,
        negative_time_threshold: int = 60
    ) -> DataLoader:
        """
        DataLoader 생성
        
        Args:
            split: 'train', 'val', 또는 'test'
            batch_size: 배치 크기
            shuffle: 셔플 여부
            num_workers: 워커 프로세스 수
            return_metadata: 메타데이터 반환 여부 (기본값: False)
            positive_time_threshold: 긍정 쌍 시간 임계값 (초, 기본값: 10)
            negative_time_threshold: 부정 쌍 시간 임계값 (초, 기본값: 60)
            
        Returns:
            PyTorch DataLoader
        """
        dataset = self.get_dataset(
            split,
            return_metadata=return_metadata,
            positive_time_threshold=positive_time_threshold,
            negative_time_threshold=negative_time_threshold
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            # Colab 최적화: num_workers > 0일 때만 활성화
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None
        )



class ContrastiveDataset(Dataset):
    """
    대조 학습을 위한 Dataset
    
    긍정 쌍과 부정 쌍을 생성하여 대조 학습을 지원합니다.
    - 긍정 쌍: 동일 종목의 시간적으로 가까운 샘플 (시간 차이 < 10초)
    - 부정 쌍: 다른 종목 또는 먼 시간대의 샘플 (시간 차이 > 60초)
    
    Args:
        data: 특징 데이터 (n_samples, n_features)
        metadata: 메타데이터 (n_samples, 3) - [종목코드, 날짜, 시간]
        seq_len: 시퀀스 길이
        positive_time_threshold: 긍정 쌍 시간 임계값 (표준화된 시간 단위, 기본값: 10)
        negative_time_threshold: 부정 쌍 시간 임계값 (표준화된 시간 단위, 기본값: 60)
        return_metadata: 메타데이터 반환 여부 (기본값: False)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        metadata: np.ndarray,
        seq_len: int = 60,
        positive_time_threshold: int = 10,
        negative_time_threshold: int = 60,
        return_metadata: bool = False
    ):
        self.data = data
        self.metadata = metadata
        self.seq_len = seq_len
        self.positive_time_threshold = positive_time_threshold
        self.negative_time_threshold = negative_time_threshold
        self.return_metadata = return_metadata
        
        # 시퀀스 생성을 위한 유효한 인덱스 계산
        self.valid_indices = self._compute_valid_indices()
        
        # 종목별 인덱스 매핑 생성 (빠른 쌍 생성을 위해)
        self.stock_indices = self._build_stock_index()
        
        logger.info(f"ContrastiveDataset initialized with {len(self.valid_indices)} valid sequences")
        
        # 긍정/부정 쌍 사전 계산 (성능 최적화)
        logger.info("Precomputing positive/negative pairs...")
        self.positive_pairs_cache = {}
        self.negative_pairs_cache = {}
        self._precompute_pairs()
        logger.info("Pair precomputation complete!")
    
    def _compute_valid_indices(self) -> np.ndarray:
        """
        시퀀스 생성이 가능한 유효한 인덱스 계산
        
        시퀀스 길이만큼의 데이터가 있어야 함
        """
        n_samples = len(self.data)
        valid_indices = []
        
        for i in range(n_samples - self.seq_len + 1):
            # 동일 종목, 동일 날짜인지 확인
            stock_codes = self.metadata[i:i+self.seq_len, 0]
            dates = self.metadata[i:i+self.seq_len, 1]
            
            if len(np.unique(stock_codes)) == 1 and len(np.unique(dates)) == 1:
                valid_indices.append(i)
        
        return np.array(valid_indices)
    
    def _build_stock_index(self) -> Dict[str, List[int]]:
        """
        종목별 인덱스 매핑 생성
        
        Returns:
            Dict[종목코드, List[인덱스]]
        """
        stock_indices = {}
        for idx in self.valid_indices:
            stock_code = self.metadata[idx, 0]
            if stock_code not in stock_indices:
                stock_indices[stock_code] = []
            stock_indices[stock_code].append(idx)
        
        return stock_indices
    
    def _precompute_pairs(self):
        """
        모든 긍정/부정 쌍 후보를 사전 계산
        
        성능 최적화: __getitem__에서 매번 검색하는 대신 미리 계산
        """
        # 종목별로 시간 정렬된 인덱스 생성
        stock_time_sorted = {}
        for stock_code, indices in self.stock_indices.items():
            # 시간 순으로 정렬
            sorted_indices = sorted(indices, key=lambda i: float(self.metadata[i, 2]))
            stock_time_sorted[stock_code] = sorted_indices
        
        # 각 인덱스에 대해 긍정/부정 쌍 후보 계산
        for idx in self.valid_indices:
            stock_code = self.metadata[idx, 0]
            anchor_time = float(self.metadata[idx, 2])
            
            # 긍정 쌍 후보 찾기 (동일 종목, 가까운 시간)
            positive_candidates = []
            for candidate_idx in stock_time_sorted[stock_code]:
                if candidate_idx == idx:
                    continue
                time_diff = abs(float(self.metadata[candidate_idx, 2]) - anchor_time)
                if time_diff < self.positive_time_threshold:
                    positive_candidates.append(candidate_idx)
            
            # 긍정 쌍이 없으면 자기 자신 사용
            if not positive_candidates:
                positive_candidates = [idx]
            
            self.positive_pairs_cache[idx] = positive_candidates
            
            # 부정 쌍 후보 찾기
            negative_candidates = []
            
            # 전략 1: 다른 종목 (70%)
            other_stocks = [s for s in self.stock_indices.keys() if s != stock_code]
            if other_stocks:
                for other_stock in other_stocks[:min(5, len(other_stocks))]:
                    negative_candidates.extend(self.stock_indices[other_stock][:100])
            
            # 전략 2: 동일 종목, 먼 시간 (30%)
            for candidate_idx in stock_time_sorted[stock_code]:
                time_diff = abs(float(self.metadata[candidate_idx, 2]) - anchor_time)
                if time_diff > self.negative_time_threshold:
                    negative_candidates.append(candidate_idx)
                    if len(negative_candidates) >= 100:
                        break
            
            # 부정 쌍이 없으면 랜덤 샘플 사용
            if not negative_candidates:
                negative_candidates = list(self.valid_indices[:100])
            
            self.negative_pairs_cache[idx] = negative_candidates
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def _get_sequence(self, idx: int) -> np.ndarray:
        """
        주어진 인덱스에서 시퀀스 추출
        
        Args:
            idx: 시작 인덱스
            
        Returns:
            시퀀스 데이터 (seq_len, n_features)
        """
        return self.data[idx:idx+self.seq_len]
    
    def _find_positive_pair(self, anchor_idx: int) -> int:
        """
        긍정 쌍 찾기: 사전 계산된 후보에서 랜덤 선택
        
        Args:
            anchor_idx: 앵커 샘플 인덱스
            
        Returns:
            긍정 쌍 인덱스
        """
        candidates = self.positive_pairs_cache.get(anchor_idx, [anchor_idx])
        return np.random.choice(candidates)
    
    def _find_negative_pair(self, anchor_idx: int) -> int:
        """
        부정 쌍 찾기: 사전 계산된 후보에서 랜덤 선택
        
        Args:
            anchor_idx: 앵커 샘플 인덱스
            
        Returns:
            부정 쌍 인덱스
        """
        candidates = self.negative_pairs_cache.get(anchor_idx, list(self.valid_indices[:100]))
        return np.random.choice(candidates)
    
    def __getitem__(self, idx: int):
        """
        배치 아이템 반환: (앵커, 긍정, 부정) 또는 (앵커, 긍정, 부정, 종목코드, 타임스탬프)
        
        Args:
            idx: 인덱스
            
        Returns:
            return_metadata=False: (anchor, positive, negative) 튜플
            return_metadata=True: (anchor, positive, negative, stock_code, timestamp) 튜플
            각 텐서 shape: (seq_len, n_features)
        """
        anchor_idx = self.valid_indices[idx]
        
        # 앵커 시퀀스
        anchor = self._get_sequence(anchor_idx)
        
        # 긍정 쌍 찾기 (사전 계산된 캐시 사용)
        positive_idx = self._find_positive_pair(anchor_idx)
        positive = self._get_sequence(positive_idx)
        
        # 부정 쌍 찾기
        negative_idx = self._find_negative_pair(anchor_idx)
        negative = self._get_sequence(negative_idx)
        
        # Tensor로 변환
        anchor_tensor = torch.from_numpy(anchor).float()
        positive_tensor = torch.from_numpy(positive).float()
        negative_tensor = torch.from_numpy(negative).float()
        
        if self.return_metadata:
            # 메타데이터 추출 (앵커 기준)
            stock_code = str(self.metadata[anchor_idx, 0])
            timestamp = float(self.metadata[anchor_idx, 2])  # 번호를 타임스탬프로 사용
            return anchor_tensor, positive_tensor, negative_tensor, stock_code, timestamp
        else:
            return anchor_tensor, positive_tensor, negative_tensor
