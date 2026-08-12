#!/usr/bin/env python3
"""
xLSTM 및 E2E 학습용 재활용 가능 데이터 추출기 (Data Extractor)

DuckDB 데이터베이스에서 유효한 에피소드(종목코드, 날짜)를 추출하고
결측값 및 이상치 처리를 거쳐 초고속 학습을 위한 압축된 numpy 파일(.npz)로 저장합니다.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import duckdb
from pathlib import Path
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DataExtractor")

def get_feature_columns(conn, table_name: str, expected_features: int) -> list:
    """특징 컬럼 목록 가져오기"""
    try:
        query = f"DESCRIBE {table_name}"
        columns_df = conn.execute(query).fetchdf()
        all_columns = columns_df['column_name'].tolist()
        column_types = columns_df['column_type'].tolist()
        
        exclude_columns = {'날짜', '종목코드', '시간', '종목명', '번호'}
        
        feature_columns = []
        for col, col_type in zip(all_columns, column_types):
            if col not in exclude_columns:
                if any(numeric_type in col_type.upper() for numeric_type in ['DOUBLE', 'FLOAT', 'INTEGER', 'BIGINT', 'DECIMAL']):
                    feature_columns.append(col)
        
        if len(feature_columns) == expected_features:
            logger.info(f"✅ Feature columns match expected size: {len(feature_columns)}")
        elif len(feature_columns) > expected_features:
            logger.warning(f"Found {len(feature_columns)} features, using first {expected_features}.")
            feature_columns = feature_columns[:expected_features]
        else:
            raise RuntimeError(f"Insufficient features in DB: found {len(feature_columns)}, expected {expected_features}")
            
        return feature_columns
    except Exception as e:
        logger.error(f"Failed to retrieve feature columns: {e}")
        raise

def extract_data(db_path: str, table_name: str, output_dir: str, seq_len: int, features: int, max_steps: int, limit: int = None):
    """DuckDB에서 데이터를 추출하여 저장"""
    db_path = str(Path(db_path).resolve())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Connecting to DuckDB: {db_path}")
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return False
        
    conn = duckdb.connect(db_path, read_only=True)
    
    try:
        feature_cols = get_feature_columns(conn, table_name, features)
        return_rate_index = feature_cols.index('등락률')
        
        # 유효한 키(종목코드, 날짜) 추출
        min_required = seq_len + max_steps + 100
        logger.info(f"Searching for valid episodes with at least {min_required} steps...")
        
        query = f"""
            SELECT 종목코드, 날짜, COUNT(*) as cnt
            FROM {table_name}
            GROUP BY 종목코드, 날짜
            HAVING COUNT(*) >= ?
        """
        keys_df = conn.execute(query, [min_required]).fetchdf()
        total_keys = len(keys_df)
        logger.info(f"Found {total_keys} valid stock-date keys.")
        
        if total_keys == 0:
            logger.warning("No valid episodes found with current length requirements. Trying fallback...")
            min_required = seq_len + 100
            keys_df = conn.execute(query, [min_required]).fetchdf()
            total_keys = len(keys_df)
            logger.info(f"Fallback found {total_keys} keys.")
            
        if total_keys == 0:
            logger.error("No valid keys found. Aborting.")
            return False
            
        # 셔플을 통해 무작위 에피소드 샘플 추출 및 특정 변동성 치우침 현상 방지
        keys_df = keys_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            
        if limit and limit < total_keys:
            logger.info(f"Limiting extraction to {limit} random episodes.")
            keys_df = keys_df.iloc[:limit]
            total_keys = limit

        manifest = []
        
        # 개별 키 추출 루프
        for idx, row in enumerate(tqdm(keys_df.itertuples(), total=total_keys, desc="Extracting Episodes")):
            stock_code = str(row.종목코드)
            date = int(row.날짜)
            
            # 쿼리 및 데이터 로드
            query = f"""
                SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
                FROM {table_name}
                WHERE 종목코드 = ? AND 날짜 = ?
                ORDER BY 시간
            """
            df = conn.execute(query, [stock_code, date]).fetchdf()
            
            if len(df) < seq_len + 10:
                continue
                
            # 결측치 처리 및 분리
            df_features = df[feature_cols].fillna(0.0)
            metadata = df[['종목코드', '날짜', '시간']].values.astype(str)
            features_data = df_features.values.astype(np.float32)
            
            # 무한대나 NaN 재검증
            if not np.isfinite(features_data).all():
                continue
                
            # 비정상 등락률 검증 (수익률 35% 초과 기각)
            return_rates = features_data[:, return_rate_index] / 100.0  # 원본 데이터 기준 % 단위이므로 나누어줌
            max_abs_return = np.max(np.abs(return_rates))
            if max_abs_return > 0.35:
                continue
                
            # 파일로 저장
            filename = f"episode_{stock_code}_{date}.npz"
            file_path = output_path / filename
            
            np.savez_compressed(
                file_path,
                features=features_data,
                metadata=metadata
            )
            
            manifest.append({
                "file_path": filename,
                "stock_code": stock_code,
                "date": date,
                "length": len(features_data)
            })
            
        # Manifest 저장
        manifest_data = {
            "metadata": {
                "feature_columns": feature_cols,
                "return_rate_index": return_rate_index,
                "expected_features": features
            },
            "episodes": manifest
        }
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully extracted {len(manifest)} episodes to {output_dir}")
        logger.info(f"Manifest saved to {manifest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckDB에서 학습용 에피소드 데이터 추출")
    parser.add_argument("--db", type=str, required=True, help="DuckDB 데이터베이스 경로")
    parser.add_argument("--table", type=str, default="datasets", help="테이블 이름")
    parser.add_argument("--output_dir", type=str, default="data/extracted_episodes", help="출력 디렉토리 경로")
    parser.add_argument("--seq_len", type=int, default=3000, help="시퀀스 길이")
    parser.add_argument("--features", type=int, default=27, help="피처 개수")
    parser.add_argument("--max_steps", type=int, default=600, help="에피소드 최대 스텝 수")
    parser.add_argument("--limit", type=int, default=None, help="추출할 최대 에피소드 수 (기본: 전체)")
    
    args = parser.parse_args()
    
    extract_data(
        db_path=args.db,
        table_name=args.table,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        features=args.features,
        max_steps=args.max_steps,
        limit=args.limit
    )
