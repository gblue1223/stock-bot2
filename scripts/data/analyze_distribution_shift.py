"""
학습 데이터와 실시간 데이터의 분포 차이 분석
Distribution shift analysis between training and real-time data
"""
import duckdb
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def analyze_distribution_shift():
    """학습 데이터와 실시간 데이터의 분포 차이 분석"""
    
    # 경로 설정
    db_path = r"C:\Users\user\Workspace\datasets@20251016\datasets_raw_all.duckdb"
    stats_path = Path("models/grpo_scalping@2025120/raw_normalization_stats.json")
    
    print("=" * 80)
    print("분포 이동 분석 (Distribution Shift Analysis)")
    print("=" * 80)
    
    # 학습 통계 로드
    with open(stats_path, 'r', encoding='utf-8') as f:
        train_stats = json.load(f)
    
    print(f"\n학습 통계 파일: {stats_path}")
    print(f"학습 데이터 특징 수: {len(train_stats['mean'])}")
    
    # 데이터베이스 연결
    conn = duckdb.connect(db_path, read_only=True)
    
    # 시간대별 데이터 분포 확인
    print("\n" + "=" * 80)
    print("1. 시간대별 데이터 분포")
    print("=" * 80)
    
    time_dist = conn.execute("""
        SELECT 
            CAST(시간/10000 AS INTEGER) as hour,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM datasets_raw
        WHERE 시간 >= 90000 AND 시간 < 110000
        GROUP BY hour
        ORDER BY hour
    """).fetchall()
    
    print("\n시간대별 데이터 비율:")
    for hour, count, pct in time_dist[:20]:
        print(f"  {hour:02d}:00 - {count:>10,}개 ({pct:>5.2f}%)")
    
    # 특징별 통계 비교 (09:00-11:00)
    print("\n" + "=" * 80)
    print("2. 특징별 통계 비교 (학습 데이터 vs 실시간 시간대)")
    print("=" * 80)
    
    # 실시간 시간대 데이터 통계
    feature_cols = [
        '현재가', '전일대비', '등락률', '거래량', '거래대금', '시가', '고가', '저가',
        '매도호가1', '매수호가1', '매도호가_잔량1', '매수호가_잔량1',
        '매도호가_잔량_합', '매수호가_잔량_합', '매도호가_건수_합', '매수호가_건수_합',
        '호가_스프레드', '호가_불균형', '체결강도'
    ]
    
    realtime_stats = {}
    for col in feature_cols:
        result = conn.execute(f"""
            SELECT 
                AVG({col}) as mean,
                STDDEV({col}) as std,
                MIN({col}) as min,
                MAX({col}) as max,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}) as median
            FROM datasets_raw
            WHERE 시간 >= 90000 AND 시간 < 110000
            AND {col} IS NOT NULL
        """).fetchone()
        
        if result and result[0] is not None:
            realtime_stats[col] = {
                'mean': float(result[0]),
                'std': float(result[1]) if result[1] else 0.0,
                'min': float(result[2]),
                'max': float(result[3]),
                'median': float(result[4])
            }
    
    # 학습 통계와 비교
    print("\n특징별 평균값 비교:")
    print(f"{'특징':<20} {'학습 평균':>15} {'실시간 평균':>15} {'차이율':>10}")
    print("-" * 65)
    
    train_means = train_stats['mean']
    significant_diffs = []
    
    for i, col in enumerate(feature_cols):
        if col in realtime_stats and i < len(train_means):
            train_mean = train_means[i]
            realtime_mean = realtime_stats[col]['mean']
            
            if abs(train_mean) > 1e-6:
                diff_pct = ((realtime_mean - train_mean) / abs(train_mean)) * 100
            else:
                diff_pct = 0.0
            
            print(f"{col:<20} {train_mean:>15.2f} {realtime_mean:>15.2f} {diff_pct:>9.1f}%")
            
            if abs(diff_pct) > 20:  # 20% 이상 차이
                significant_diffs.append((col, diff_pct))
    
    # 표준편차 비교
    print("\n특징별 표준편차 비교:")
    print(f"{'특징':<20} {'학습 std':>15} {'실시간 std':>15} {'차이율':>10}")
    print("-" * 65)
    
    train_stds = train_stats['std']
    
    for i, col in enumerate(feature_cols):
        if col in realtime_stats and i < len(train_stds):
            train_std = train_stds[i]
            realtime_std = realtime_stats[col]['std']
            
            if abs(train_std) > 1e-6:
                diff_pct = ((realtime_std - train_std) / abs(train_std)) * 100
            else:
                diff_pct = 0.0
            
            print(f"{col:<20} {train_std:>15.2f} {realtime_std:>15.2f} {diff_pct:>9.1f}%")
    
    # 주요 차이점 요약
    print("\n" + "=" * 80)
    print("3. 주요 발견사항")
    print("=" * 80)
    
    if significant_diffs:
        print(f"\n⚠️  20% 이상 차이나는 특징 ({len(significant_diffs)}개):")
        for col, diff_pct in sorted(significant_diffs, key=lambda x: abs(x[1]), reverse=True):
            print(f"  - {col}: {diff_pct:+.1f}%")
    
    # 체결강도 분포 확인
    print("\n" + "=" * 80)
    print("4. 체결강도 분포 분석")
    print("=" * 80)
    
    strength_dist = conn.execute("""
        SELECT 
            CASE 
                WHEN 체결강도 = 0 THEN '0 (거래없음)'
                WHEN 체결강도 < 50 THEN '1-49 (매도우세)'
                WHEN 체결강도 = 50 THEN '50 (중립)'
                WHEN 체결강도 < 100 THEN '51-99 (매수우세)'
                WHEN 체결강도 >= 100 THEN '100+ (강한매수)'
                ELSE 'NULL'
            END as range,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM datasets_raw
        WHERE 시간 >= 90000 AND 시간 < 110000
        GROUP BY range
        ORDER BY range
    """).fetchall()
    
    print("\n체결강도 구간별 분포:")
    for range_name, count, pct in strength_dist:
        print(f"  {range_name:<20} {count:>10,}개 ({pct:>5.2f}%)")
    
    # 등락률 분포 확인
    print("\n" + "=" * 80)
    print("5. 등락률 분포 분석")
    print("=" * 80)
    
    rate_dist = conn.execute("""
        SELECT 
            CASE 
                WHEN 등락률 < -5 THEN '< -5% (급락)'
                WHEN 등락률 < -2 THEN '-5% ~ -2%'
                WHEN 등락률 < 0 THEN '-2% ~ 0%'
                WHEN 등락률 = 0 THEN '0% (보합)'
                WHEN 등락률 < 2 THEN '0% ~ 2%'
                WHEN 등락률 < 5 THEN '2% ~ 5%'
                WHEN 등락률 >= 5 THEN '>= 5% (급등)'
                ELSE 'NULL'
            END as range,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM datasets_raw
        WHERE 시간 >= 90000 AND 시간 < 110000
        GROUP BY range
        ORDER BY 
            CASE 
                WHEN range = '< -5% (급락)' THEN 1
                WHEN range = '-5% ~ -2%' THEN 2
                WHEN range = '-2% ~ 0%' THEN 3
                WHEN range = '0% (보합)' THEN 4
                WHEN range = '0% ~ 2%' THEN 5
                WHEN range = '2% ~ 5%' THEN 6
                WHEN range = '>= 5% (급등)' THEN 7
                ELSE 8
            END
    """).fetchall()
    
    print("\n등락률 구간별 분포:")
    for range_name, count, pct in rate_dist:
        print(f"  {range_name:<20} {count:>10,}개 ({pct:>5.2f}%)")
    
    conn.close()
    
    # 권장사항
    print("\n" + "=" * 80)
    print("6. 권장사항")
    print("=" * 80)
    print("""
1. Rolling Window 정규화 사용
   - 학습 통계 대신 최근 N개 데이터의 통계 사용
   - 시장 환경 변화에 실시간 적응
   
2. 시간대별 정규화
   - 09:00-09:30, 09:30-10:00 등 시간대별 통계 사용
   - 장 초반/중반의 특성 차이 반영
   
3. 체결강도=0 필터링 유지
   - 현재 구현된 필터링 계속 사용
   - 거래 비활발 종목 자동 제외
   
4. 모델 재학습 고려
   - 최신 데이터(11월)로 재학습
   - 또는 온라인 학습 메커니즘 도입
    """)
    
    print("\n분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    analyze_distribution_shift()
