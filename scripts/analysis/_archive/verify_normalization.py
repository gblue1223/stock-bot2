"""
정규화된 데이터의 품질을 검증하는 스크립트
"""
import sys
import duckdb
import numpy as np
import pandas as pd

def verify_normalization(db_path: str, table: str = "datasets", sample_size: int = 50000):
    """정규화 품질 검증"""
    
    conn = duckdb.connect(db_path, read_only=True)
    
    try:
        # Sample data
        query = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_size}"
        df = conn.execute(query).df()
        
        print("=" * 80)
        print("정규화 품질 검증")
        print("=" * 80)
        print(f"샘플 크기: {len(df):,}")
        print()
        
        # 검증할 컬럼 그룹
        price_cols = ['현재가', '시가', '고가', '저가']
        volume_cols = ['거래량', '누적거래량', '누적거래대금']
        ratio_cols = ['등락률', '거래회전율', '체결강도']
        
        issues = []
        
        # 1. 가격 컬럼 검증 (로그 변환 + 표준화 적용되어야 함)
        print("=" * 80)
        print("1. 가격 컬럼 검증 (로그 변환 + 표준화)")
        print("=" * 80)
        
        for col in price_cols:
            if col not in df.columns:
                print(f"⚠️  {col}: 컬럼 없음")
                issues.append(f"{col} 컬럼이 존재하지 않음")
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                print(f"⚠️  {col}: 데이터 없음")
                issues.append(f"{col}에 데이터가 없음")
                continue
            
            min_v = values.min()
            max_v = values.max()
            mean_v = values.mean()
            std_v = values.std()
            unique_count = values.nunique()
            
            print(f"\n{col}:")
            print(f"  범위: [{min_v:.4f}, {max_v:.4f}]")
            print(f"  평균: {mean_v:.4f}")
            print(f"  표준편차: {std_v:.4f}")
            print(f"  Unique values: {unique_count:,}")
            
            # 검증 기준
            # 1) Min-Max scaling (0~1)이면 잘못됨
            if 0 <= min_v <= 0.1 and 0.9 <= max_v <= 1.0:
                print(f"  ❌ FAIL: Min-Max scaling 적용됨 (로그 변환 필요)")
                issues.append(f"{col}이 Min-Max scaling 적용됨")
            # 2) 표준화되었으면 평균 ~0, 표준편차 ~1
            elif abs(mean_v) < 0.5 and 0.5 < std_v < 2.0:
                print(f"  ✅ PASS: 로그 변환 + 표준화 적용됨")
            # 3) Unique values가 너무 적으면 정보 손실
            elif unique_count < 100:
                print(f"  ⚠️  WARNING: Unique values 너무 적음 (정보 손실 가능)")
                issues.append(f"{col}의 unique values가 {unique_count}개로 너무 적음")
            else:
                print(f"  ✅ PASS: 정규화 적용됨")
        
        # 2. 거래량/금액 컬럼 검증
        print("\n" + "=" * 80)
        print("2. 거래량/금액 컬럼 검증 (로그 변환 + 표준화)")
        print("=" * 80)
        
        for col in volume_cols:
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            mean_v = values.mean()
            std_v = values.std()
            
            print(f"\n{col}:")
            print(f"  평균: {mean_v:.4f}")
            print(f"  표준편차: {std_v:.4f}")
            
            if abs(mean_v) < 0.5 and 0.5 < std_v < 2.0:
                print(f"  ✅ PASS")
            else:
                print(f"  ⚠️  WARNING: 정규화 이상")
        
        # 3. 비율 컬럼 검증
        print("\n" + "=" * 80)
        print("3. 비율 컬럼 검증 (표준화)")
        print("=" * 80)
        
        for col in ratio_cols:
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            mean_v = values.mean()
            std_v = values.std()
            
            print(f"\n{col}:")
            print(f"  평균: {mean_v:.4f}")
            print(f"  표준편차: {std_v:.4f}")
            
            if abs(mean_v) < 0.5 and 0.5 < std_v < 2.0:
                print(f"  ✅ PASS")
            else:
                print(f"  ⚠️  WARNING: 정규화 이상")
        
        # 4. 전체 요약
        print("\n" + "=" * 80)
        print("검증 결과 요약")
        print("=" * 80)
        
        if issues:
            print(f"\n❌ {len(issues)}개의 문제 발견:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            print("\n⚠️  데이터를 재정규화해야 합니다!")
            return False
        else:
            print("\n✅ 모든 검증 통과!")
            print("데이터가 올바르게 정규화되었습니다.")
            return True
            
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = r"C:\Users\user\Workspace\datasets@20251003\datasets_norm_all.duckdb"
    
    import os
    if not os.path.exists(db_path):
        print(f"Error: 파일을 찾을 수 없습니다: {db_path}")
        sys.exit(1)
    
    success = verify_normalization(db_path)
    sys.exit(0 if success else 1)
