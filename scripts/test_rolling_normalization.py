"""
Rolling Window Normalization 테스트

분포 이동 문제 해결을 위한 Rolling Window 정규화 테스트
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.rolling_normalization import RollingNormalizer
from lib.normalization import FEATURE_NAMES


def test_basic_normalization():
    """기본 정규화 테스트"""
    print("=" * 80)
    print("1. 기본 정규화 테스트")
    print("=" * 80)
    
    normalizer = RollingNormalizer(
        window_size=100,
        min_samples=10,
        feature_names=FEATURE_NAMES
    )
    
    # 테스트 데이터 생성 (평균=100, 표준편차=20)
    np.random.seed(42)
    test_data = np.random.normal(100, 20, (200, 28))
    
    print(f"\n원본 데이터:")
    print(f"  Shape: {test_data.shape}")
    print(f"  Mean: {test_data.mean():.2f}")
    print(f"  Std: {test_data.std():.2f}")
    print(f"  Range: [{test_data.min():.2f}, {test_data.max():.2f}]")
    
    # 정규화
    normalized_data = []
    for i, features in enumerate(test_data):
        normalized = normalizer.normalize(features, update=True)
        normalized_data.append(normalized)
        
        if i == 9:  # 워밍업 완료 시점
            print(f"\n워밍업 완료 (샘플 {i+1}개):")
            stats = normalizer.get_stats()
            print(f"  Mean: {stats['mean'][:5]}")
            print(f"  Std: {stats['std'][:5]}")
    
    normalized_data = np.array(normalized_data)
    
    # 워밍업 후 데이터만 분석 (첫 10개 제외)
    normalized_after_warmup = normalized_data[10:]
    
    print(f"\n정규화 후 데이터 (워밍업 후):")
    print(f"  Shape: {normalized_after_warmup.shape}")
    print(f"  Mean: {normalized_after_warmup.mean():.4f}")
    print(f"  Std: {normalized_after_warmup.std():.4f}")
    print(f"  Range: [{normalized_after_warmup.min():.4f}, {normalized_after_warmup.max():.4f}]")
    
    # 검증
    assert abs(normalized_after_warmup.mean()) < 0.5, "평균이 0에 가까워야 함"
    assert 0.5 < normalized_after_warmup.std() < 1.5, "표준편차가 1에 가까워야 함"
    
    print("\n✅ 기본 정규화 테스트 통과!")


def test_distribution_shift():
    """분포 이동 적응 테스트"""
    print("\n" + "=" * 80)
    print("2. 분포 이동 적응 테스트")
    print("=" * 80)
    
    normalizer = RollingNormalizer(
        window_size=50,
        min_samples=10,
        feature_names=FEATURE_NAMES
    )
    
    np.random.seed(42)
    
    # Phase 1: 평균=100, 표준편차=20
    phase1_data = np.random.normal(100, 20, (100, 28))
    
    # Phase 2: 평균=200, 표준편차=40 (분포 이동!)
    phase2_data = np.random.normal(200, 40, (100, 28))
    
    print(f"\nPhase 1 (원본):")
    print(f"  Mean: {phase1_data.mean():.2f}")
    print(f"  Std: {phase1_data.std():.2f}")
    
    print(f"\nPhase 2 (원본, 분포 이동):")
    print(f"  Mean: {phase2_data.mean():.2f}")
    print(f"  Std: {phase2_data.std():.2f}")
    
    # Phase 1 정규화
    normalized_phase1 = []
    for features in phase1_data:
        normalized = normalizer.normalize(features, update=True)
        normalized_phase1.append(normalized)
    
    normalized_phase1 = np.array(normalized_phase1)
    phase1_after_warmup = normalized_phase1[10:]
    
    print(f"\nPhase 1 정규화 후:")
    print(f"  Mean: {phase1_after_warmup.mean():.4f}")
    print(f"  Std: {phase1_after_warmup.std():.4f}")
    
    # Phase 2 정규화 (동일한 normalizer 사용)
    normalized_phase2 = []
    for features in phase2_data:
        normalized = normalizer.normalize(features, update=True)
        normalized_phase2.append(normalized)
    
    normalized_phase2 = np.array(normalized_phase2)
    
    # Phase 2 후반부 (rolling window가 Phase 2 데이터로 채워진 후)
    phase2_after_adaptation = normalized_phase2[-50:]
    
    print(f"\nPhase 2 정규화 후 (적응 후):")
    print(f"  Mean: {phase2_after_adaptation.mean():.4f}")
    print(f"  Std: {phase2_after_adaptation.std():.4f}")
    
    # 검증: Phase 2에서도 정규화가 잘 작동해야 함
    assert abs(phase2_after_adaptation.mean()) < 0.5, "분포 이동 후에도 평균이 0에 가까워야 함"
    assert 0.5 < phase2_after_adaptation.std() < 1.5, "분포 이동 후에도 표준편차가 1에 가까워야 함"
    
    print("\n✅ 분포 이동 적응 테스트 통과!")
    print("   Rolling window가 새로운 분포에 성공적으로 적응했습니다!")


def test_real_market_scenario():
    """실제 시장 시나리오 테스트"""
    print("\n" + "=" * 80)
    print("3. 실제 시장 시나리오 테스트")
    print("=" * 80)
    
    normalizer = RollingNormalizer(
        window_size=100,
        min_samples=20,
        feature_names=FEATURE_NAMES
    )
    
    np.random.seed(42)
    
    # 시나리오: 장 초반 (변동성 높음) → 장 중반 (안정) → 장 후반 (변동성 증가)
    
    # 장 초반: 평균=1000, 표준편차=200
    morning_data = np.random.normal(1000, 200, (50, 28))
    
    # 장 중반: 평균=1100, 표준편차=100
    midday_data = np.random.normal(1100, 100, (50, 28))
    
    # 장 후반: 평균=1050, 표준편차=150
    afternoon_data = np.random.normal(1050, 150, (50, 28))
    
    all_data = np.vstack([morning_data, midday_data, afternoon_data])
    
    print(f"\n장 초반 (원본): mean={morning_data.mean():.2f}, std={morning_data.std():.2f}")
    print(f"장 중반 (원본): mean={midday_data.mean():.2f}, std={midday_data.std():.2f}")
    print(f"장 후반 (원본): mean={afternoon_data.mean():.2f}, std={afternoon_data.std():.2f}")
    
    # 정규화
    normalized_all = []
    for i, features in enumerate(all_data):
        normalized = normalizer.normalize(features, update=True)
        normalized_all.append(normalized)
    
    normalized_all = np.array(normalized_all)
    
    # 각 시간대별 정규화 결과
    normalized_morning = normalized_all[20:50]  # 워밍업 후
    normalized_midday = normalized_all[50:100]
    normalized_afternoon = normalized_all[100:150]
    
    print(f"\n장 초반 (정규화): mean={normalized_morning.mean():.4f}, std={normalized_morning.std():.4f}")
    print(f"장 중반 (정규화): mean={normalized_midday.mean():.4f}, std={normalized_midday.std():.4f}")
    print(f"장 후반 (정규화): mean={normalized_afternoon.mean():.4f}, std={normalized_afternoon.std():.4f}")
    
    # 검증: 모든 시간대에서 정규화가 잘 작동해야 함
    for name, data in [
        ("장 초반", normalized_morning),
        ("장 중반", normalized_midday),
        ("장 후반", normalized_afternoon)
    ]:
        assert abs(data.mean()) < 0.5, f"{name}: 평균이 0에 가까워야 함"
        assert 0.5 < data.std() < 1.5, f"{name}: 표준편차가 1에 가까워야 함"
    
    print("\n✅ 실제 시장 시나리오 테스트 통과!")
    print("   시간대별 변동성 변화에 성공적으로 적응했습니다!")


def test_comparison_with_fixed_stats():
    """고정 통계 vs Rolling Window 비교"""
    print("\n" + "=" * 80)
    print("4. 고정 통계 vs Rolling Window 비교")
    print("=" * 80)
    
    np.random.seed(42)
    
    # 학습 데이터 (10월): 평균=100, 표준편차=20
    train_data = np.random.normal(100, 20, (1000, 28))
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)
    
    # 실시간 데이터 (11월): 평균=150, 표준편차=30 (분포 이동!)
    realtime_data = np.random.normal(150, 30, (100, 28))
    
    print(f"\n학습 데이터 (10월):")
    print(f"  Mean: {train_data.mean():.2f}")
    print(f"  Std: {train_data.std():.2f}")
    
    print(f"\n실시간 데이터 (11월, 분포 이동):")
    print(f"  Mean: {realtime_data.mean():.2f}")
    print(f"  Std: {realtime_data.std():.2f}")
    
    # 방법 1: 고정 통계 사용 (기존 방식)
    normalized_fixed = (realtime_data - train_mean) / train_std
    
    print(f"\n방법 1: 고정 통계 (학습 데이터 통계 사용):")
    print(f"  Mean: {normalized_fixed.mean():.4f}")
    print(f"  Std: {normalized_fixed.std():.4f}")
    print(f"  Range: [{normalized_fixed.min():.4f}, {normalized_fixed.max():.4f}]")
    
    # 방법 2: Rolling Window (새로운 방식)
    normalizer = RollingNormalizer(
        window_size=50,
        min_samples=10,
        feature_names=FEATURE_NAMES
    )
    
    normalized_rolling = []
    for features in realtime_data:
        normalized = normalizer.normalize(features, update=True)
        normalized_rolling.append(normalized)
    
    normalized_rolling = np.array(normalized_rolling)
    normalized_rolling_after_warmup = normalized_rolling[10:]
    
    print(f"\n방법 2: Rolling Window (실시간 적응):")
    print(f"  Mean: {normalized_rolling_after_warmup.mean():.4f}")
    print(f"  Std: {normalized_rolling_after_warmup.std():.4f}")
    print(f"  Range: [{normalized_rolling_after_warmup.min():.4f}, {normalized_rolling_after_warmup.max():.4f}]")
    
    print(f"\n비교:")
    print(f"  고정 통계 평균 편차: {abs(normalized_fixed.mean()):.4f}")
    print(f"  Rolling Window 평균 편차: {abs(normalized_rolling_after_warmup.mean()):.4f}")
    print(f"  → Rolling Window가 {abs(normalized_fixed.mean()) / abs(normalized_rolling_after_warmup.mean()):.1f}배 더 정확!")
    
    print("\n✅ 비교 테스트 완료!")
    print("   Rolling Window가 분포 이동 상황에서 훨씬 우수한 성능을 보입니다!")


def main():
    """모든 테스트 실행"""
    print("\n" + "=" * 80)
    print("Rolling Window Normalization 테스트")
    print("=" * 80)
    
    try:
        test_basic_normalization()
        test_distribution_shift()
        test_real_market_scenario()
        test_comparison_with_fixed_stats()
        
        print("\n" + "=" * 80)
        print("🎉 모든 테스트 통과!")
        print("=" * 80)
        print("\n✅ Rolling Window 정규화가 정상적으로 작동합니다.")
        print("✅ 분포 이동 문제를 효과적으로 해결할 수 있습니다.")
        print("✅ 실시간 거래 시스템에 적용 준비 완료!")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
