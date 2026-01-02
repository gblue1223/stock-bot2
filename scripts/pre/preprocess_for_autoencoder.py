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

    def load_normalization_params(self, params_path: str):
        """외부 파일에서 정규화 파라미터 로드"""
        logger.info(f"Loading normalization parameters from {params_path}")
        with open(params_path, 'r') as f:
            self.normalization_params = json.load(f)
        logger.info("Normalization parameters loaded")
    
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
        시퀀스 배치 파일 생성 (메모리 최적화: 종목별 순차 처리)
        """
        logger.info("Creating sequence batches (Iterative Processing)...")
        
        feature_cols = self.get_feature_columns()
        
        # WHERE 절 구성 (날짜)
        where_clauses = []
        if start_date:
            start_date_str = start_date.replace('-', '')
            where_clauses.append(f"날짜 >= '{start_date_str}'")
        if end_date:
            end_date_str = end_date.replace('-', '')
            where_clauses.append(f"날짜 <= '{end_date_str}'")
        
        date_where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # 1. 대상 종목 코드 조회
        logger.info("Fetching target stock codes...")
        code_query = f"""
            SELECT DISTINCT 종목코드 
            FROM datasets 
            WHERE {date_where_clause}
        """
        stock_codes = self.conn.execute(code_query).fetchdf()['종목코드'].tolist()
        logger.info(f"Found {len(stock_codes)} stocks to process.")
        
        # 정규화 파라미터 준비
        mean = np.array(self.normalization_params['mean']) if self.normalization_params else None
        std = np.array(self.normalization_params['std']) if self.normalization_params else None
        if std is not None:
             std = np.where(std == 0, 1, std)
        
        strategies = []
        if self.normalization_params:
            strategies = self.normalization_params.get('strategies', 
                        [get_normalization_strategy(col) for col in feature_cols])

        # 배치 버퍼 초기화
        sequence_buffer = []
        metadata_buffer = []
        batch_idx = 0
        total_sequences = 0
        
        # 2. 종목별 순차 처리
        for stock_code in tqdm(stock_codes, desc="Processing stocks"):
            # 종목별 데이터 로드
            query = f"""
                SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
                FROM datasets
                WHERE 종목코드 = '{stock_code}' AND {date_where_clause}
                ORDER BY 날짜, 시간
            """
            
            df = self.conn.execute(query).fetchdf()
            if len(df) <= seq_len:
                continue
                
            # 메타데이터와 특징 분리
            stock_metadata = df[['종목코드', '날짜', '시간']].values
            feature_df = df[feature_cols]
            
            # 문자열 컬럼 처리 (Scalar 변환)
            for col in feature_df.columns:
                if feature_df[col].dtype == 'object' or feature_df[col].dtype == 'string':
                    feature_df[col] = compute_stock_name_scalar_batch(feature_df[col])

            features = feature_df.values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # 정규화 적용
            if self.normalization_params:
                # 1. Log 변환
                for i, strategy in enumerate(strategies):
                    if strategy == 'log_std':
                        features[:, i] = signed_log1p(features[:, i])
                
                # 2. Z-Score
                features = (features - mean) / std
                
            # 시퀀스 생성 (Vectorized Sliding Window)
            # stride가 1일 때만 이 방식 사용 가능, 아니면 기존 루프 방식 사용
            if stride == 1:
                # numpy stride tricks could be used here, but for simplicity/readability reusing loop or optimized approach
                # 간단한 슬라이싱 루프 (메모리 효율 고려)
                num_seq = len(features) - seq_len + 1
                for i in range(0, num_seq, stride):
                    # 날짜가 연속적인지 등 체크는 여기서 생략 (단일 종목이므로 대부분 연속, 날짜 바뀌는 경계만 주의)
                    # *중요*: 날짜/시간 불연속성이 큰 경우(장 마감 후 다음날)를 구분해야 하면
                    # 메타데이터의 날짜를 확인해야 함.
                    # 여기서는 간단히 날짜가 바뀌는 지점 허용 (AutoEncoder는 패턴 학습이므로)
                    # 단, 기존에는 (동일 종목, 동일 날짜) 조건이 있었음 -> 이를 유지하려면:
                    
                    seq_dates = stock_metadata[i:i+seq_len, 1]
                    if seq_dates[0] == seq_dates[-1]: # 같은 날짜인 경우만
                        sequence_buffer.append(features[i:i+seq_len])
                        metadata_buffer.append(stock_metadata[i])
            else:
                 # Stride > 1 인 경우
                i = 0
                while i <= len(features) - seq_len:
                    seq_dates = stock_metadata[i:i+seq_len, 1]
                    if seq_dates[0] == seq_dates[-1]:
                        sequence_buffer.append(features[i:i+seq_len])
                        metadata_buffer.append(stock_metadata[i])
                        i += stride
                    else:
                        i += 1

            # 버퍼가 배치 크기를 넘으면 저장
            while len(sequence_buffer) >= batch_size:
                # 배치 추출
                batch_seqs = np.array(sequence_buffer[:batch_size], dtype=np.float32)
                batch_meta = np.array(metadata_buffer[:batch_size])
                
                # 나머지 버퍼 유지
                sequence_buffer = sequence_buffer[batch_size:]
                metadata_buffer = metadata_buffer[batch_size:]
                
                # 저장
                batch_file = self.output_dir / f'batch_{batch_idx:06d}.h5'
                with h5py.File(batch_file, 'w') as f:
                    f.create_dataset('sequences', data=batch_seqs, compression='gzip', compression_opts=6)
                    f.create_dataset('metadata', data=batch_meta.astype('S20'))
                
                batch_idx += 1
                total_sequences += batch_size
                
                # Max Samples 체크 (전체 누적 기준)
                if max_samples and total_sequences >= max_samples:
                    logger.info(f"Reached max samples limit: {max_samples}")
                    break
            
            if max_samples and total_sequences >= max_samples:
                break

        # 남은 버퍼 저장
        if sequence_buffer:
            batch_seqs = np.array(sequence_buffer, dtype=np.float32)
            batch_meta = np.array(metadata_buffer)
            
            batch_file = self.output_dir / f'batch_{batch_idx:06d}.h5'
            with h5py.File(batch_file, 'w') as f:
                f.create_dataset('sequences', data=batch_seqs, compression='gzip', compression_opts=6)
                f.create_dataset('metadata', data=batch_meta.astype('S20'))
            
            batch_idx += 1
            total_sequences += len(batch_seqs)

        # 배치 정보 저장
        batch_info = {
            'num_batches': batch_idx,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_features': len(feature_cols),
            'total_sequences': total_sequences
        }
        
        with open(self.output_dir / 'batch_info.json', 'w') as f:
            json.dump(batch_info, f, indent=2)
            
        logger.info(f"Completed! Total sequences: {total_sequences}, Batches: {batch_idx}")
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
    parser.add_argument('--load-norm-params', help='기존 정규화 파라미터 파일 경로')
    parser.add_argument('--create-batches', action='store_true', help='배치 파일 생성')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = AutoEncoderPreprocessor(args.db, args.output_dir)
    preprocessor.connect()
    
    try:
        if args.load_norm_params:
            preprocessor.load_normalization_params(args.load_norm_params)
        elif args.compute_norm_params:
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