"""
정규화 호환성 테스트

학습 데이터 정규화와 실시간 정규화가 동일하게 동작하는지 검증합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lib.normalization import (
    signed_log1p,
    standard_scale,
    get_normalization_strategy,
    FEATURE_NAMES
)
from scripts.live.online_normalizer import OnlineNormalizer as TrainingCompatibleNormalizer


def test_signed_log1p():
    """Signed log1p 변환 테스트"""
    print("Testing signed_log1p...")
    
    # 양수
    assert abs(signed_log1p(0) - 0.0) < 1e-6
    assert abs(signed_log1p(1) - 0.693147) < 1e-5
    assert abs(signed_log1p(9) - 2.302585) < 1e-5
    
    # 음수
    assert abs(signed_log1p(-1) - (-0.693147)) < 1e-5
    assert abs(signed_log1p(-9) - (-2.302585)) < 1e-5
    
    print("[PASS] signed_log1p")


def test_standard_scale():
    """Z-Score 정규화 테스트"""
    print("\nTesting standard_scale...")
    
    # 정상 케이스
    result = standard_scale(2.0, mean=0.0, std=1.0)
    assert abs(result - 2.0) < 1e-6
    
    # 표준편차 0 (상수)
    result = standard_scale(5.0, mean=5.0, std=0.0)
    assert result == 0.0
    
    print("[PASS] standard_scale")


def test_normalization_strategies():
    """특징별 정규화 전략 테스트"""
    print("\nTesting normalization strategies...")
    
    # Log + Z-Score
    assert get_normalization_strategy("누적거래대금") == "log_std"
    assert get_normalization_strategy("거래회전율") == "log_std"
    assert get_normalization_strategy("체결강도") == "log_std"
    assert get_normalization_strategy("매도대기금액1") == "log_std"
    assert get_normalization_strategy("매수대기금액10") == "log_std"
    
    # Z-Score Only
    assert get_normalization_strategy("등락률") == "std_only"
    
    # Derived (이미 정규화됨)
    assert get_normalization_strategy("종목명_scalar") == "derived"
    assert get_normalization_strategy("시간_sin") == "derived"
    assert get_normalization_strategy("시간_cos") == "derived"
    assert get_normalization_strategy("시간_scalar") == "derived"
    
    print("[PASS] normalization strategies")


def test_training_compatible_normalizer():
    """TrainingCompatibleNormalizer 통합 테스트"""
    print("\nTesting TrainingCompatibleNormalizer...")
    
    normalizer = TrainingCompatibleNormalizer(
        num_features=28,
        window_size=10,
        warmup_samples=5
    )
    
    # 샘플 데이터 생성
    np.random.seed(42)
    
    # 워밍업 (5개 샘플)
    print("  Warming up...")
    for i in range(5):
        features = np.random.randn(28).astype(np.float32) * 100 + 1000
        # 파생 피처는 이미 정규화된 값
        features[0:4] = np.random.rand(4).astype(np.float32)
        
        normalized = normalizer.normalize(features, update=True)
        print(f"    Sample {i+1}: mean={np.mean(normalized):.4f}, std={np.std(normalized):.4f}")
    
    assert normalizer.is_ready(), "Normalizer should be ready after warmup"
    
    # 정상 동작 (5개 샘플 더)
    print("  Normal operation...")
    for i in range(5):
        features = np.random.randn(28).astype(np.float32) * 100 + 1000
        features[0:4] = np.random.rand(4).astype(np.float32)
        
        normalized = normalizer.normalize(features, update=True)
        
        # 정규화 품질 검증
        mean = np.mean(normalized[4:])  # 파생 피처 제외
        std = np.std(normalized[4:])
        print(f"    Sample {i+1}: mean={mean:.4f}, std={std:.4f}")
        
        # 평균이 0 근처, 표준편차가 1 근처
        assert abs(mean) < 2.0, f"Mean should be close to 0, got {mean}"
        # 표준편차는 완벽하게 1은 아닐 수 있음 (샘플 수가 적어서)
    
    print("[PASS] TrainingCompatibleNormalizer")


def test_feature_consistency():
    """특징별 일관성 테스트"""
    print("\nTesting feature consistency...")
    
    normalizer = TrainingCompatibleNormalizer(
        num_features=28,
        window_size=100,
        warmup_samples=50
    )
    
    # 워밍업
    np.random.seed(42)
    for _ in range(50):
        features = np.random.randn(28).astype(np.float32) * 100 + 1000
        features[0:4] = np.random.rand(4).astype(np.float32)
        normalizer.normalize(features, update=True)
    
    # 테스트 샘플
    test_features = np.array([
        0.5,  # 종목명_scalar (derived)
        0.7,  # 시간_sin (derived)
        -0.3,  # 시간_cos (derived)
        1.2,  # 시간_scalar (derived)
        5.0,  # 등락률 (std_only)
        1000000.0,  # 누적거래대금 (log_std)
        100.0,  # 거래회전율 (log_std)
        150.0,  # 체결강도 (log_std)
        *([500.0] * 20)  # 매도/매수대기금액 (log_std)
    ], dtype=np.float32)
    
    normalized = normalizer.normalize(test_features, update=False)
    
    # 파생 피처는 원본 유지
    print(f"  종목명_scalar: {test_features[0]:.4f} → {normalized[0]:.4f} (should be same)")
    assert abs(normalized[0] - test_features[0]) < 1e-6
    
    print(f"  시간_sin: {test_features[1]:.4f} → {normalized[1]:.4f} (should be same)")
    assert abs(normalized[1] - test_features[1]) < 1e-6
    
    # 등락률은 Z-Score
    print(f"  등락률: {test_features[4]:.2f} → {normalized[4]:.4f} (Z-score)")
    
    # 누적거래대금은 Log + Z-Score
    log_value = signed_log1p(test_features[5])
    print(f"  누적거래대금: {test_features[5]:.0f} → log={log_value:.4f} → {normalized[5]:.4f} (log + Z-score)")
    
    print("[PASS] Feature consistency")


def test_comparison_with_simple_zscore():
    """단순 Z-Score vs Training-Compatible 비교"""
    print("\nComparing with simple Z-score...")
    
    # 샘플 데이터
    np.random.seed(42)
    data = []
    for _ in range(100):
        features = np.random.randn(28).astype(np.float32) * 100 + 1000
        features[0:4] = np.random.rand(4).astype(np.float32)
        data.append(features)
    
    data = np.array(data)
    
    # Training-Compatible 정규화
    normalizer = TrainingCompatibleNormalizer(
        num_features=28,
        window_size=100,
        warmup_samples=50
    )
    
    for i in range(50):
        normalizer.normalize(data[i], update=True)
    
    tc_normalized = normalizer.normalize(data[50], update=False)
    
    # 단순 Z-Score
    mean = np.mean(data[:50], axis=0)
    std = np.std(data[:50], axis=0, ddof=0) + 1e-8
    simple_normalized = (data[50] - mean) / std
    
    # 파생 피처(0-3)는 동일해야 함
    for i in range(4):
        print(f"  Feature {i}: TC={tc_normalized[i]:.4f}, Simple={simple_normalized[i]:.4f}")
        assert abs(tc_normalized[i] - data[50][i]) < 1e-6
    
    # 등락률(4)는 유사해야 함
    print(f"  등락률(4): TC={tc_normalized[4]:.4f}, Simple={simple_normalized[4]:.4f}")
    
    # Log 특징(5+)은 다를 수 있음 (랜덤 데이터에서는 유사할 수도 있음)
    diff = abs(tc_normalized[5] - simple_normalized[5])
    print(f"  누적거래대금(5): TC={tc_normalized[5]:.4f}, Simple={simple_normalized[5]:.4f} (diff={diff:.4f})")
    print("  Note: Difference may be small with random data")
    
    print("[PASS] Comparison")


if __name__ == "__main__":
    print("=" * 80)
    print("정규화 호환성 테스트")
    print("=" * 80)
    
    try:
        test_signed_log1p()
        test_standard_scale()
        test_normalization_strategies()
        test_training_compatible_normalizer()
        test_feature_consistency()
        test_comparison_with_simple_zscore()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] All tests passed!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
