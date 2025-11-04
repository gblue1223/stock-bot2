"""
배치 파생 피처 계산 함수 테스트

lib.normalization의 배치 처리 함수를 테스트합니다.
"""

import numpy as np
import pandas as pd
from lib.normalization import (
    compute_stock_name_scalar_batch,
    compute_time_features_batch
)


def test_stock_name_scalar_batch():
    """종목명_scalar 배치 계산 테스트"""
    # 테스트 데이터
    stock_names = pd.Series(["삼성전자", "SK하이닉스", "NAVER", "카카오", "LG전자"])
    
    # 배치 계산
    scalars = compute_stock_name_scalar_batch(stock_names)
    
    # 범위 검증
    assert all(0.0 <= s <= 1.0 for s in scalars), "모든 값이 0~1 범위여야 함"
    
    # 같은 종목명은 같은 값
    stock_names2 = pd.Series(["삼성전자", "SK하이닉스", "삼성전자"])
    scalars2 = compute_stock_name_scalar_batch(stock_names2)
    assert scalars2.iloc[0] == scalars2.iloc[2], "같은 종목명은 같은 값이어야 함"
    
    # 빈 문자열 처리
    empty_series = pd.Series(["", None, "nan"])
    empty_scalars = compute_stock_name_scalar_batch(empty_series)
    assert all(s == 0.0 for s in empty_scalars), "빈 값은 0.0이어야 함"
    
    print("[PASS] compute_stock_name_scalar_batch")


def test_time_features_batch_zscore():
    """시간 파생 피처 배치 계산 테스트 (z-score 모드)"""
    # 테스트 데이터: 장 시작부터 종료까지
    time_series = pd.Series([
        "090000000",  # 09:00:00
        "093000000",  # 09:30:00
        "100000000",  # 10:00:00
        "110000000",  # 11:00:00
        "120000000",  # 12:00:00
        "130000000",  # 13:00:00
        "140000000",  # 14:00:00
        "150000000",  # 15:00:00
    ])
    
    # z-score 모드로 계산
    time_sin, time_cos, time_scalar = compute_time_features_batch(
        time_series, 
        use_zscore_for_scalar=True
    )
    
    # 범위 검증
    assert all(-1.0 <= s <= 1.0 for s in time_sin), "시간_sin은 -1~1 범위여야 함"
    assert all(-1.0 <= s <= 1.0 for s in time_cos), "시간_cos는 -1~1 범위여야 함"
    
    # z-score는 평균 0, 표준편차 1 근처여야 함
    assert abs(time_scalar.mean()) < 0.1, "z-score 평균은 0 근처여야 함"
    assert abs(time_scalar.std() - 1.0) < 0.1, "z-score 표준편차는 1 근처여야 함"
    
    # sin^2 + cos^2 = 1 검증
    for i in range(len(time_series)):
        assert abs(time_sin.iloc[i]**2 + time_cos.iloc[i]**2 - 1.0) < 0.001, \
            f"sin^2 + cos^2 = 1이어야 함 (index={i})"
    
    print("[PASS] compute_time_features_batch (z-score mode)")


def test_time_features_batch_fixed_range():
    """시간 파생 피처 배치 계산 테스트 (고정 범위 모드)"""
    # 테스트 데이터
    time_series = pd.Series([
        "090000000",  # 09:00:00 -> -1.0
        "120000000",  # 12:00:00 -> 0.0
        "150000000",  # 15:00:00 -> 1.0
    ])
    
    # 고정 범위 모드로 계산
    time_sin, time_cos, time_scalar = compute_time_features_batch(
        time_series, 
        use_zscore_for_scalar=False
    )
    
    # 범위 검증
    assert all(-1.0 <= s <= 1.0 for s in time_sin), "시간_sin은 -1~1 범위여야 함"
    assert all(-1.0 <= s <= 1.0 for s in time_cos), "시간_cos는 -1~1 범위여야 함"
    
    # 고정 범위 매핑 검증
    assert abs(time_scalar.iloc[0] - (-1.0)) < 0.01, "09:00은 -1.0 근처여야 함"
    assert abs(time_scalar.iloc[1] - 0.0) < 0.01, "12:00은 0.0 근처여야 함"
    assert abs(time_scalar.iloc[2] - 1.0) < 0.01, "15:00은 1.0 근처여야 함"
    
    print("[PASS] compute_time_features_batch (fixed range mode)")


def test_batch_consistency():
    """배치 처리와 개별 처리의 일관성 테스트"""
    from lib.normalization import compute_stock_name_scalar, compute_time_features
    
    # 종목명 일관성
    stock_names = pd.Series(["삼성전자", "SK하이닉스", "NAVER"])
    batch_scalars = compute_stock_name_scalar_batch(stock_names)
    
    for i, name in enumerate(stock_names):
        individual_scalar = compute_stock_name_scalar(name)
        # 배치와 개별 처리는 다른 알고리즘을 사용하므로 값이 다를 수 있음
        # 단지 범위만 검증
        assert 0.0 <= batch_scalars.iloc[i] <= 1.0
        assert 0.0 <= individual_scalar <= 1.0
    
    # 시간 일관성 (고정 범위 모드)
    time_series = pd.Series(["093015000", "120000000", "143000000"])
    batch_sin, batch_cos, batch_scalar = compute_time_features_batch(
        time_series, 
        use_zscore_for_scalar=False
    )
    
    for i, time_str in enumerate(time_series):
        ind_sin, ind_cos, ind_scalar = compute_time_features(time_str)
        
        # 고정 범위 모드에서는 값이 일치해야 함
        assert abs(batch_sin.iloc[i] - ind_sin) < 0.001, \
            f"시간_sin 불일치 (index={i})"
        assert abs(batch_cos.iloc[i] - ind_cos) < 0.001, \
            f"시간_cos 불일치 (index={i})"
        assert abs(batch_scalar.iloc[i] - ind_scalar) < 0.001, \
            f"시간_scalar 불일치 (index={i})"
    
    print("[PASS] batch_consistency")


def test_large_batch():
    """대용량 배치 처리 테스트"""
    # 1000개 샘플
    n = 1000
    stock_names = pd.Series([f"종목{i}" for i in range(n)])
    time_series = pd.Series([f"{9 + i % 6:02d}{(i % 60):02d}{(i % 60):02d}000" for i in range(n)])
    
    # 배치 계산
    scalars = compute_stock_name_scalar_batch(stock_names)
    time_sin, time_cos, time_scalar = compute_time_features_batch(time_series, use_zscore_for_scalar=True)
    
    # 크기 검증
    assert len(scalars) == n
    assert len(time_sin) == n
    assert len(time_cos) == n
    assert len(time_scalar) == n
    
    # 범위 검증
    assert all(0.0 <= s <= 1.0 for s in scalars)
    assert all(-1.0 <= s <= 1.0 for s in time_sin)
    assert all(-1.0 <= s <= 1.0 for s in time_cos)
    
    # NaN 없음 검증
    assert not scalars.isna().any()
    assert not time_sin.isna().any()
    assert not time_cos.isna().any()
    assert not time_scalar.isna().any()
    
    print("[PASS] large_batch")


if __name__ == "__main__":
    test_stock_name_scalar_batch()
    test_time_features_batch_zscore()
    test_time_features_batch_fixed_range()
    test_batch_consistency()
    test_large_batch()
    print("\n✅ 모든 배치 파생 피처 테스트 통과!")
