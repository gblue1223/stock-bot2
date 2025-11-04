"""
파생 피처 계산 함수 테스트

lib.normalization의 중앙화된 파생 피처 계산 함수를 테스트합니다.
"""

import numpy as np
from lib.normalization import (
    compute_stock_name_scalar,
    compute_time_features,
    compute_all_derived_features
)


def test_stock_name_scalar():
    """종목명_scalar 계산 테스트"""
    # 정상 케이스
    scalar1 = compute_stock_name_scalar("삼성전자")
    assert 0.0 <= scalar1 <= 1.0, "종목명_scalar는 0~1 범위여야 함"
    
    scalar2 = compute_stock_name_scalar("SK하이닉스")
    assert 0.0 <= scalar2 <= 1.0, "종목명_scalar는 0~1 범위여야 함"
    
    # 같은 종목명은 같은 값
    assert compute_stock_name_scalar("삼성전자") == scalar1
    
    # 빈 문자열
    assert compute_stock_name_scalar("") == 0.0
    assert compute_stock_name_scalar(None) == 0.0
    
    print("[PASS] compute_stock_name_scalar")


def test_time_features():
    """시간 파생 피처 계산 테스트"""
    # 장 시작 시간 (09:00:00)
    time_sin, time_cos, time_scalar = compute_time_features("090000000")
    assert -1.0 <= time_sin <= 1.0, "시간_sin은 -1~1 범위여야 함"
    assert -1.0 <= time_cos <= 1.0, "시간_cos는 -1~1 범위여야 함"
    assert time_scalar == -1.0, "장 시작 시간의 시간_scalar는 -1.0이어야 함"
    
    # 장 중간 시간 (12:00:00, 3시간 경과)
    time_sin, time_cos, time_scalar = compute_time_features("120000000")
    assert -1.0 <= time_sin <= 1.0
    assert -1.0 <= time_cos <= 1.0
    assert abs(time_scalar - 0.0) < 0.01, "3시간 경과 시 시간_scalar는 0.0 근처여야 함"
    
    # 장 종료 시간 (15:00:00, 6시간 경과)
    time_sin, time_cos, time_scalar = compute_time_features("150000000")
    assert -1.0 <= time_sin <= 1.0
    assert -1.0 <= time_cos <= 1.0
    assert abs(time_scalar - 1.0) < 0.01, "6시간 경과 시 시간_scalar는 1.0 근처여야 함"
    
    # 빈 문자열
    time_sin, time_cos, time_scalar = compute_time_features("")
    assert time_sin == 0.0 and time_cos == 0.0 and time_scalar == 0.0
    
    # sin^2 + cos^2 = 1 검증
    time_sin, time_cos, _ = compute_time_features("093015000")
    assert abs(time_sin**2 + time_cos**2 - 1.0) < 0.001, "sin^2 + cos^2 = 1이어야 함"
    
    print("[PASS] compute_time_features")


def test_all_derived_features():
    """모든 파생 피처 한 번에 계산 테스트"""
    stock_name_scalar, time_sin, time_cos, time_scalar = \
        compute_all_derived_features("삼성전자", "093015000")
    
    # 범위 검증
    assert 0.0 <= stock_name_scalar <= 1.0
    assert -1.0 <= time_sin <= 1.0
    assert -1.0 <= time_cos <= 1.0
    assert -1.0 <= time_scalar <= 1.0
    
    # 개별 함수와 결과 일치 검증
    expected_stock = compute_stock_name_scalar("삼성전자")
    expected_time = compute_time_features("093015000")
    
    assert stock_name_scalar == expected_stock
    assert time_sin == expected_time[0]
    assert time_cos == expected_time[1]
    assert time_scalar == expected_time[2]
    
    print("[PASS] compute_all_derived_features")


def test_realtime_compatibility():
    """실시간 데이터와의 호환성 테스트"""
    # 실제 거래 시간대 샘플
    test_cases = [
        ("삼성전자", "090000000"),  # 장 시작
        ("SK하이닉스", "093000000"),  # 9시 30분
        ("NAVER", "120000000"),  # 정오
        ("카카오", "143000000"),  # 2시 30분
        ("LG전자", "150000000"),  # 장 종료
    ]
    
    for stock_name, time_str in test_cases:
        stock_scalar, time_sin, time_cos, time_scalar = \
            compute_all_derived_features(stock_name, time_str)
        
        # 모든 값이 유효한 범위 내에 있는지 확인
        assert 0.0 <= stock_scalar <= 1.0, f"{stock_name}: 종목명_scalar 범위 오류"
        assert -1.0 <= time_sin <= 1.0, f"{stock_name}: 시간_sin 범위 오류"
        assert -1.0 <= time_cos <= 1.0, f"{stock_name}: 시간_cos 범위 오류"
        assert -2.0 <= time_scalar <= 2.0, f"{stock_name}: 시간_scalar 범위 오류"
        
        # NaN이 아닌지 확인
        assert not np.isnan(stock_scalar), f"{stock_name}: 종목명_scalar가 NaN"
        assert not np.isnan(time_sin), f"{stock_name}: 시간_sin이 NaN"
        assert not np.isnan(time_cos), f"{stock_name}: 시간_cos가 NaN"
        assert not np.isnan(time_scalar), f"{stock_name}: 시간_scalar가 NaN"
    
    print("[PASS] realtime_compatibility")


if __name__ == "__main__":
    test_stock_name_scalar()
    test_time_features()
    test_all_derived_features()
    test_realtime_compatibility()
    print("\n✅ 모든 파생 피처 테스트 통과!")
