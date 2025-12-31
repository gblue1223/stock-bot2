"""
AutoEncoder 훈련을 위한 데이터 전처리 및 캐싱 스크립트

대용량 데이터를 사전에 처리하여 훈련 속도를 크게 향상시킵니다.
- 시퀀스 생성 및 캐싱
- 정규화 파라미터 계산
- 배치별 파일 분할
- 메모리 맵핑 지원
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import duckdb
import pickle
import json
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import h5py
import mmap

from lib.normalization import get_normalization_strategy, signed_log1p, compute_stock_name_scalar_batch

logger = logging.getLogger(__name__)


class AutoEncoderPreprocessor:
    """AutoEncoder 훈련을 위한 데이터 전처리기"""
    
    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.feature_columns = None
        self.normalization_params = {}
        
    def connect(self):
        """데이터베이스 연결"""
        self.conn = duckdb.connect(self.db_path, read_only=True)
        logger.info(f"Connected to database: {self.db_path}")
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
    
    def get_feature_columns(self) -> List[str]:
        """특징 컬럼 목록 가져오기"""
        if self.feature_columns is not None:
            return self.feature_columns
        
        query = "DESCRIBE datasets"
        columns_df = self.conn.execute(query).fetchdf()
        all_columns = columns_df['column_name'].tolist()
        
        # 메타데이터 컬럼 제외
        exclude_columns = {'날짜', '종목코드', '번호', '종목명', '시간'}
        self.feature_columns = [col for col in all_columns if col not in exclude_columns]
        
        logger.info(f"Found {len(self.feature_columns)} feature columns")
        return self.feature_columns
    
    def compute_normalization_params(self, sample_size: int = 1000000):
        """
        정규화 파라미터 계산 (평균, 표준편차)
        
        참고: lib/normalization.py의 전략을 따름
        - log_std: signed_log1p 후 mean/std 계산
        - std_only: 그냥 mean/std 계산
        
        Args:
            sample_size: 샘플링할 데이터 크기
        """
        logger.info("Computing normalization parameters...")
        
        feature_cols = self.get_feature_columns()
        
        # 샘플 데이터 로드
        query = f"""
            SELECT {', '.join(feature_cols)}
            FROM datasets
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """
        
        df = self.conn.execute(query).fetchdf()
        
        # ✅ 문자열 컬럼 처리 (Scalar 변환)
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                logger.info(f"Converting string column to scalar: {col}")
                # lib/normalization.py의 함수 사용
                df[col] = compute_stock_name_scalar_batch(df[col])
        
        data = df.values.astype(np.float32)
        
        # NaN/Inf 처리
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # ✅ 정규화 전략에 따른 전처리 (Log 변환 등)
        strategies = [get_normalization_strategy(col) for col in feature_cols]
        
        for i, strategy in enumerate(strategies):
            if strategy == 'log_std':
                data[:, i] = signed_log1p(data[:, i])
        
        # 정규화 파라미터 계산
        self.normalization_params = {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'feature_columns': feature_cols,
            'strategies': strategies  # 전략 정보도 저장
        }
        
        # 저장
        with open(self.output_dir / 'normalization_params.json', 'w') as f:
            json.dump(self.normalization_params, f, indent=2)
        
        logger.info("Normalization parameters computed and saved")
    
    def create_sequence_batches(
        self,
        seq_len: int = 60,
        stride: int = 1,
        batch_size: int = 10000,
        max_samples: int = None,
        start_date: str = None,
        end_date: str = None
    ):
        """
        시퀀스 배치 파일 생성
        
        Args:
            seq_len: 시퀀스 길이
            stride: 스트라이드
            batch_size: 배치당 시퀀스 수
            max_samples: 최대 샘플 수
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
        """
        logger.info("Creating sequence batches...")
        
        feature_cols = self.get_feature_columns()
        
        # WHERE 절 구성
        where_clauses = []
        if start_date:
            start_date_str = start_date.replace('-', '')
            where_clauses.append(f"날짜 >= '{start_date_str}'")
        if end_date:
            end_date_str = end_date.replace('-', '')
            where_clauses.append(f"날짜 <= '{end_date_str}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        limit_clause = f"LIMIT {max_samples}" if max_samples else ""
        
        # 데이터 로드
        query = f"""
            SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
            FROM datasets
            WHERE {where_clause}
            ORDER BY 날짜, 종목코드, 시간
            {limit_clause}
        """
        
        logger.info("Loading data...")
        df = self.conn.execute(query).fetchdf()
        
        if len(df) == 0:
            raise ValueError("No data found")
        
        logger.info(f"Loaded {len(df)} rows")
        
        # 메타데이터와 특징 분리
        metadata = df[['종목코드', '날짜', '시간']].values
        
        # ✅ 메타데이터 컬럼 제외한 특징만 추출
        feature_df = df[feature_cols]
        
        # ✅ 문자열 컬럼 처리 (Scalar 변환) - create_sequence_batches
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object' or feature_df[col].dtype == 'string':
                # logger.debug(f"Converting string column to scalar: {col}")
                feature_df[col] = compute_stock_name_scalar_batch(feature_df[col])

        features = feature_df.values.astype(np.float32)
        
        # 데이터 정제
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # ✅ 정규화 적용
        if self.normalization_params:
            mean = np.array(self.normalization_params['mean'])
            std = np.array(self.normalization_params['std'])
            strategies = self.normalization_params.get('strategies', 
                        [get_normalization_strategy(col) for col in feature_cols])
            
            # 1. Log 변환 (필요한 경우)
            for i, strategy in enumerate(strategies):
                if strategy == 'log_std':
                    features[:, i] = signed_log1p(features[:, i])
            
            # 2. Z-Score 정규화
            std = np.where(std == 0, 1, std)  # 0으로 나누기 방지
            features = (features - mean) / std
        
        # 시퀀스 생성
        logger.info("Generating sequences...")
        sequences = []
        sequence_metadata = []
        
        i = 0
        while i <= len(features) - seq_len:
            # 시퀀스 범위의 메타데이터 확인
            seq_meta = metadata[i:i+seq_len]
            stock_codes = seq_meta[:, 0]
            dates = seq_meta[:, 1]
            
            # 동일 종목, 동일 날짜인지 확인
            if len(np.unique(stock_codes)) == 1 and len(np.unique(dates)) == 1:
                sequences.append(features[i:i+seq_len])
                sequence_metadata.append(metadata[i])  # 시작점 메타데이터
                i += stride
            else:
                # 다음 유효한 시작점으로 이동
                i += 1
        
        sequences = np.array(sequences, dtype=np.float32)
        sequence_metadata = np.array(sequence_metadata)
        
        logger.info(f"Generated {len(sequences)} sequences")
        
        # 배치별로 저장
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        batch_info = {
            'num_batches': num_batches,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_features': features.shape[1],
            'total_sequences': len(sequences)
        }
        
        logger.info(f"Saving {num_batches} batches...")
        
        for batch_idx in tqdm(range(num_batches), desc="Saving batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            
            batch_sequences = sequences[start_idx:end_idx]
            batch_metadata = sequence_metadata[start_idx:end_idx]
            
            # HDF5로 저장 (압축 및 빠른 로딩)
            batch_file = self.output_dir / f'batch_{batch_idx:06d}.h5'
            
            with h5py.File(batch_file, 'w') as f:
                f.create_dataset('sequences', data=batch_sequences, compression='gzip', compression_opts=6)
                f.create_dataset('metadata', data=batch_metadata.astype('S20'))  # 문자열을 바이트로 저장
        
        # 배치 정보 저장
        with open(self.output_dir / 'batch_info.json', 'w') as f:
            json.dump(batch_info, f, indent=2)
        
        logger.info(f"Sequence batches saved to {self.output_dir}")
        return batch_info
    
    def create_memory_mapped_file(self, sequences: np.ndarray, filename: str = 'sequences.dat'):
        """
        메모리 맵핑을 위한 바이너리 파일 생성
        
        Args:
            sequences: 시퀀스 데이터 (n_sequences, seq_len, n_features)
            filename: 파일명
        """
        filepath = self.output_dir / filename
        
        # 메모리 맵핑 파일로 저장
        fp = np.memmap(filepath, dtype='float32', mode='w+', shape=sequences.shape)
        fp[:] = sequences[:]
        del fp  # 파일 닫기
        
        # 메타데이터 저장
        metadata = {
            'shape': sequences.shape,
            'dtype': 'float32',
            'filename': filename
        }
        
        with open(self.output_dir / f'{filename}.meta', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Memory-mapped file created: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='AutoEncoder 데이터 전처리')
    
    parser.add_argument('--db', required=True, help='DuckDB 데이터베이스 경로')
    parser.add_argument('--output-dir', required=True, help='출력 디렉토리')
    parser.add_argument('--seq-len', type=int, default=60, help='시퀀스 길이')
    parser.add_argument('--stride', type=int, default=1, help='스트라이드')
    parser.add_argument('--batch-size', type=int, default=10000, help='배치당 시퀀스 수')
    parser.add_argument('--max-samples', type=int, help='최대 샘플 수')
    parser.add_argument('--start-date', help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--compute-norm-params', action='store_true', help='정규화 파라미터 계산')
    parser.add_argument('--create-batches', action='store_true', help='배치 파일 생성')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = AutoEncoderPreprocessor(args.db, args.output_dir)
    preprocessor.connect()
    
    try:
        if args.compute_norm_params:
            preprocessor.compute_normalization_params()
        
        if args.create_batches:
            preprocessor.create_sequence_batches(
                seq_len=args.seq_len,
                stride=args.stride,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                start_date=args.start_date,
                end_date=args.end_date
            )
    
    finally:
        preprocessor.close()


if __name__ == '__main__':
    main()