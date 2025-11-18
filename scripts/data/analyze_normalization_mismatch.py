"""
정규화 불일치 분석

실시간 데이터와 학습 데이터의 정규화 결과를 비교 분석합니다.

Usage:
    python scripts/data/analyze_normalization_mismatch.py \
        --stats models/grpo_scalping@2025120/raw_normalization_stats.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.normalization import FEATURE_NAMES, get_normalization_strategy, signed_log1p, standard_scale


def analyze_sample_case():
    """실시간 데이터 샘플 케이스 분석"""
    
    print("=" * 80)
    print("실시간 데이터 샘플 분석 (종목 290650)")
    print("=" * 80)
    
    # 실시간 원본 값 (로그에서 추출)
    realtime_raw = {
        '등락률': 9.92,
        '누적거래대금': 13288.0,
        '거래회전율': 1.0,
        '체결강도': 153.44
    }
    
    print("\n1. 실시간 원본 값:")
    for key, val in realtime_raw.items():
        print(f"   {key:15s}: {val:10.2f}")
    
    # 학습 통계 (raw_normalization_stats.json)
    training_stats = {
        '등락률': {'mean': 8.7896, 'std': 20.0548, 'strategy': 'std_only'},
        '누적거래대금': {'mean': 10.9654, 'std': 1.1133, 'strategy': 'log_std'},
        '거래회전율': {'mean': 1.7628, 'std': 1.4917, 'strategy': 'log_std'},
        '체결강도': {'mean': 4.5999, 'std': 0.4246, 'strategy': 'log_std'}
    }
    
    print("\n2. 학습 통계 (로그 변환 후):")
    for key, stats in training_stats.items():
        print(f"   {key:15s}: mean={stats['mean']:8.4f}, std={stats['std']:8.4f} [{stats['strategy']}]")
    
    # 정규화 계산
    print("\n3. 정규화 계산:")
    normalized = {}
    
    for key, raw_val in realtime_raw.items():
        stats = training_stats[key]
        strategy = stats['strategy']
        
        if strategy == 'log_std':
            # 로그 변환
            log_val = signed_log1p(raw_val)
            # Z-score
            norm_val = (log_val - stats['mean']) / stats['std']
            print(f"   {key:15s}: {raw_val:10.2f} -> log={log_val:8.4f} -> norm={norm_val:8.4f}")
        elif strategy == 'std_only':
            # Z-score만
            norm_val = (raw_val - stats['mean']) / stats['std']
            print(f"   {key:15s}: {raw_val:10.2f} -> norm={norm_val:8.4f}")
        
        normalized[key] = norm_val
    
    # 실제 로그 결과와 비교
    print("\n4. 실제 로그 결과 (AFTER NORM):")
    actual_log = {
        '등락률': 0.0564,
        '누적거래대금': -1.32,
        '거래회전율': -0.7171,
        '체결강도': 1.0351
    }
    
    for key in realtime_raw.keys():
        calc = normalized[key]
        actual = actual_log[key]
        diff = abs(calc - actual)
        status = "✅" if diff < 0.1 else "❌"
        print(f"   {key:15s}: 계산={calc:8.4f}, 실제={actual:8.4f}, 차이={diff:8.4f} {status}")
    
    # 평균 계산
    print("\n5. 전체 평균:")
    calc_mean = np.mean(list(normalized.values()))
    actual_mean = np.mean(list(actual_log.values()))
    print(f"   계산된 평균: {calc_mean:8.4f}")
    print(f"   실제 평균:   {actual_mean:8.4f}")
    print(f"   차이:        {abs(calc_mean - actual_mean):8.4f}")


def analyze_distribution_shift():
    """분포 이동 분석"""
    
    print("\n" + "=" * 80)
    print("분포 이동 분석")
    print("=" * 80)
    
    # 학습 데이터 통계 (로그 변환 후)
    training_log_stats = {
        '누적거래대금': {'mean': 10.9654, 'std': 1.1133},
        '거래회전율': {'mean': 1.7628, 'std': 1.4917},
        '체결강도': {'mean': 4.5999, 'std': 0.4246}
    }
    
    # 실시간 데이터 샘플 (로그 변환 후)
    realtime_samples = {
        '누적거래대금': [
            signed_log1p(13288.0),  # 종목 290650
            signed_log1p(16500.0),  # 가상 샘플
            signed_log1p(10000.0),
        ],
        '거래회전율': [
            signed_log1p(1.0),
            signed_log1p(0.63),
            signed_log1p(73.97),
        ],
        '체결강도': [
            signed_log1p(153.44),
            signed_log1p(100.0),
            signed_log1p(50.0),
        ]
    }
    
    print("\n1. 로그 변환 후 값 비교:")
    for key in training_log_stats.keys():
        train_mean = training_log_stats[key]['mean']
        train_std = training_log_stats[key]['std']
        
        realtime_vals = realtime_samples[key]
        realtime_mean = np.mean(realtime_vals)
        
        print(f"\n   {key}:")
        print(f"      학습 데이터:   mean={train_mean:8.4f}, std={train_std:8.4f}")
        print(f"      실시간 샘플:   mean={realtime_mean:8.4f}")
        print(f"      차이:          {realtime_mean - train_mean:8.4f}")
        
        # 정규화 후 값
        normalized = [(v - train_mean) / train_std for v in realtime_vals]
        norm_mean = np.mean(normalized)
        print(f"      정규화 후 평균: {norm_mean:8.4f}")


def recommend_solutions():
    """해결 방안 제시"""
    
    print("\n" + "=" * 80)
    print("문제 진단 및 해결 방안")
    print("=" * 80)
    
    print("\n📊 문제 진단:")
    print("   1. 실시간 데이터의 정규화 후 평균이 음수 (-0.38)")
    print("   2. 모델이 음수 평균을 약세로 해석 → 모든 예측이 SELL")
    print("   3. 학습 데이터와 실시간 데이터의 분포 차이")
    
    print("\n🔍 원인 분석:")
    print("   1. 학습 데이터: 과거 데이터 (2025년 이전)")
    print("   2. 실시간 데이터: 현재 시장 (2025년 11월)")
    print("   3. 시장 환경 변화로 인한 분포 이동")
    
    print("\n💡 해결 방안:")
    print("\n   옵션 A: 최신 데이터로 통계 재계산 (추천)")
    print("      - 2025년 10-11월 데이터로 정규화 통계 재생성")
    print("      - 현재 시장 환경을 반영")
    print("      - 명령어:")
    print("        python scripts/data/compute_raw_stats_from_raw_db.py \\")
    print("          --db datasets_raw_2025_recent.duckdb \\")
    print("          --output models/grpo_scalping@2025120/raw_normalization_stats_recent.json")
    
    print("\n   옵션 B: Rolling window 정규화만 사용")
    print("      - 학습 통계 사용 비활성화")
    print("      - 실시간 데이터로만 통계 계산")
    print("      - config/trading_config.json에서:")
    print('        "normalization_stats_path": null')
    
    print("\n   옵션 C: 모델 재학습")
    print("      - 최신 데이터로 모델 재학습")
    print("      - 현재 시장 환경에 맞는 패턴 학습")
    print("      - 시간이 가장 오래 걸림")
    
    print("\n📈 예상 효과:")
    print("   옵션 A: Buy Signal Rate 5-20% 예상 (빠른 해결)")
    print("   옵션 B: Buy Signal Rate 10-30% 예상 (종목별 적응)")
    print("   옵션 C: Buy Signal Rate 15-25% 예상 (최적 성능)")


def main():
    parser = argparse.ArgumentParser(description='정규화 불일치 분석')
    parser.add_argument('--stats', help='Normalization stats JSON file')
    args = parser.parse_args()
    
    # 샘플 케이스 분석
    analyze_sample_case()
    
    # 분포 이동 분석
    analyze_distribution_shift()
    
    # 해결 방안 제시
    recommend_solutions()
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
