"""
학습 데이터의 정규화 통계를 계산하고 저장

Usage:
    python scripts/data/save_normalization_stats.py \
        --db 'C:\\Users\\user\\Workspace\\datasets@20251013\\datasets_norm_all.duckdb' \
        --output 'C:\\Users\\user\\Workspace\\datasets@20251013\\normalization_stats.pkl'
"""

import argparse
import pickle
import numpy as np
import duckdb
from pathlib import Path
import sys

# lib 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.normalization import FEATURE_NAMES


def compute_normalization_stats(db_path: str) -> dict:
    """데이터베이스에서 정규화 통계 계산"""
    
    print(f"Opening database: {db_path}")
    conn = duckdb.connect(db_path, read_only=True)
    
    # 특징 컬럼만 선택 (메타데이터 제외)
    feature_cols = list(FEATURE_NAMES)
    
    print(f"\nComputing normalization stats for {len(feature_cols)} features...")
    print("=" * 80)
    
    # 각 특징별 평균과 표준편차 계산
    means = []
    stds = []
    
    for i, col in enumerate(feature_cols, 1):
        try:
            query = f"""
                SELECT 
                    AVG("{col}") as mean,
                    STDDEV("{col}") as std
                FROM datasets
                WHERE "{col}" IS NOT NULL
            """
            result = conn.execute(query).fetchone()
            
            mean = float(result[0]) if result[0] is not None else 0.0
            std = float(result[1]) if result[1] is not None else 1.0
            
            # std가 0이면 1.0으로 설정 (division by zero 방지)
            if std == 0 or std < 1e-8:
                std = 1.0
            
            means.append(mean)
            stds.append(std)
            
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: mean={mean:12.4f}, std={std:12.4f}")
        
        except Exception as e:
            print(f"  [{i:2d}/{len(feature_cols)}] {col:20s}: ERROR - {e}")
            means.append(0.0)
            stds.append(1.0)
    
    conn.close()
    
    print("=" * 80)
    
    return {
        'feature_names': feature_cols,
        'mean': np.array(means, dtype=np.float32),
        'std': np.array(stds, dtype=np.float32)
    }


def main():
    parser = argparse.ArgumentParser(description='학습 데이터의 정규화 통계 계산 및 저장')
    parser.add_argument('--db', required=True, help='DuckDB database path')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not Path(args.db).exists():
        print(f"❌ Error: Database file not found: {args.db}")
        sys.exit(1)
    
    # 통계 계산
    print("\n" + "=" * 80)
    print("NORMALIZATION STATISTICS COMPUTATION")
    print("=" * 80)
    
    stats = compute_normalization_stats(args.db)
    
    # 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Normalization stats saved to: {output_path}")
    print(f"   Features: {len(stats['feature_names'])}")
    print(f"   Mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"   Std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print("=" * 80)
    
    # 주요 특징 통계 출력
    print("\nKey Features:")
    key_features = ['등락률', '누적거래대금', '거래회전율', '체결강도']
    for feat in key_features:
        if feat in stats['feature_names']:
            idx = stats['feature_names'].index(feat)
            print(f"  {feat:15s}: mean={stats['mean'][idx]:12.4f}, std={stats['std'][idx]:12.4f}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()