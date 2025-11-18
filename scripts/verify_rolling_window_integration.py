"""
Rolling Window 정규화 통합 검증

live_trading.py에 Rolling Window 정규화가 제대로 통합되었는지 확인
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_imports():
    """필요한 모듈 import 확인"""
    print("=" * 80)
    print("1. 모듈 Import 확인")
    print("=" * 80)
    
    try:
        from lib.rolling_normalization import RollingNormalizer, TimeBasedNormalizer
        print("✅ lib.rolling_normalization 모듈 import 성공")
    except ImportError as e:
        print(f"❌ lib.rolling_normalization 모듈 import 실패: {e}")
        return False
    
    try:
        from lib.normalization import FEATURE_NAMES
        print(f"✅ FEATURE_NAMES import 성공 (특징 수: {len(FEATURE_NAMES)})")
    except ImportError as e:
        print(f"❌ FEATURE_NAMES import 실패: {e}")
        return False
    
    return True


def verify_rolling_normalizer():
    """RollingNormalizer 기본 동작 확인"""
    print("\n" + "=" * 80)
    print("2. RollingNormalizer 기본 동작 확인")
    print("=" * 80)
    
    try:
        from lib.rolling_normalization import RollingNormalizer
        from lib.normalization import FEATURE_NAMES
        import numpy as np
        
        # Normalizer 생성
        normalizer = RollingNormalizer(
            window_size=100,
            min_samples=10,
            feature_names=FEATURE_NAMES
        )
        print("✅ RollingNormalizer 생성 성공")
        
        # 테스트 데이터
        test_features = np.random.normal(100, 20, 28)
        
        # 정규화
        normalized = normalizer.normalize(test_features, update=True)
        print(f"✅ 정규화 실행 성공 (입력: {test_features.shape}, 출력: {normalized.shape})")
        
        # 통계 확인
        stats = normalizer.get_stats()
        print(f"✅ 통계 조회 성공 (샘플 수: {stats['n_samples']})")
        
        return True
        
    except Exception as e:
        print(f"❌ RollingNormalizer 동작 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_config():
    """설정 파일 확인"""
    print("\n" + "=" * 80)
    print("3. 설정 파일 확인")
    print("=" * 80)
    
    try:
        import json
        
        config_path = Path("config/trading_config.json")
        if not config_path.exists():
            print(f"❌ 설정 파일 없음: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ 설정 파일 로드 성공: {config_path}")
        
        # 주요 설정 확인
        window_size = config.get('online_window_size', 0)
        warmup_samples = config.get('online_warmup_samples', 0)
        
        print(f"   Window size: {window_size}")
        print(f"   Warmup samples: {warmup_samples}")
        
        if window_size >= 1000 and warmup_samples >= 100:
            print("✅ Rolling Window 설정 적절함")
        else:
            print(f"⚠️  Rolling Window 설정 확인 필요 (권장: window_size>=1000, warmup>=100)")
        
        return True
        
    except Exception as e:
        print(f"❌ 설정 파일 확인 실패: {e}")
        return False


def verify_live_trading_integration():
    """live_trading.py 통합 확인"""
    print("\n" + "=" * 80)
    print("4. live_trading.py 통합 확인")
    print("=" * 80)
    
    try:
        # 파일 읽기
        live_trading_path = Path("scripts/live/live_trading.py")
        if not live_trading_path.exists():
            print(f"❌ live_trading.py 없음: {live_trading_path}")
            return False
        
        with open(live_trading_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 필수 import 확인
        checks = [
            ("RollingNormalizer import", "from lib.rolling_normalization import RollingNormalizer"),
            ("FEATURE_NAMES import", "from lib.normalization import FEATURE_NAMES"),
            ("normalizers 딕셔너리", "self.normalizers = {}"),
            ("rolling_window_size 설정", "self.rolling_window_size"),
            ("rolling_min_samples 설정", "self.rolling_min_samples"),
            ("RollingNormalizer 생성", "RollingNormalizer("),
        ]
        
        all_passed = True
        for name, pattern in checks:
            if pattern in content:
                print(f"✅ {name} 확인됨")
            else:
                print(f"❌ {name} 없음")
                all_passed = False
        
        # PerStockNormalizer 제거 확인
        if "PerStockNormalizer" in content:
            print("⚠️  PerStockNormalizer 코드가 아직 남아있음 (정상 동작하지만 정리 권장)")
        else:
            print("✅ PerStockNormalizer 제거됨")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ live_trading.py 확인 실패: {e}")
        return False


def verify_documentation():
    """문서 확인"""
    print("\n" + "=" * 80)
    print("5. 문서 확인")
    print("=" * 80)
    
    docs = [
        "docs/BUY_SIGNAL_ZERO_ANALYSIS.md",
        "docs/ROLLING_WINDOW_IMPLEMENTATION.md",
        "docs/DATA_QUALITY_VALIDATION.md",
    ]
    
    all_exist = True
    for doc_path in docs:
        path = Path(doc_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {doc_path} ({size:,} bytes)")
        else:
            print(f"❌ {doc_path} 없음")
            all_exist = False
    
    return all_exist


def main():
    """전체 검증 실행"""
    print("\n" + "=" * 80)
    print("Rolling Window 정규화 통합 검증")
    print("=" * 80)
    
    results = []
    
    # 1. Import 확인
    results.append(("모듈 Import", verify_imports()))
    
    # 2. RollingNormalizer 동작 확인
    results.append(("RollingNormalizer 동작", verify_rolling_normalizer()))
    
    # 3. 설정 파일 확인
    results.append(("설정 파일", verify_config()))
    
    # 4. live_trading.py 통합 확인
    results.append(("live_trading.py 통합", verify_live_trading_integration()))
    
    # 5. 문서 확인
    results.append(("문서", verify_documentation()))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 80)
    if passed == total:
        print(f"🎉 모든 검증 통과! ({passed}/{total})")
        print("=" * 80)
        print("\n✅ Rolling Window 정규화가 성공적으로 통합되었습니다!")
        print("✅ 실시간 거래 시스템 실행 준비 완료!")
        print("\n다음 명령어로 실행하세요:")
        print("  python scripts/live/live_trading.py --config config/trading_config.json")
        return True
    else:
        print(f"⚠️  일부 검증 실패 ({passed}/{total})")
        print("=" * 80)
        print("\n위의 실패 항목을 확인하고 수정하세요.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
