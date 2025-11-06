# Buy Signal 0% 문제 해결 방안

## 문제 요약

1. **입력 데이터 정규화 실패**: 평균 3백만, 범위 [-1.75, 1억 3천만]
2. **호가 데이터 누락**: 체결강도=0, 매도/매수대기금액=0
3. **결과**: 모델이 BUY 신호를 전혀 생성하지 않음

## 제안하는 수정 방안

### 🔥 우선순위 1: 정규화 문제 진단 및 수정

#### 1-1. 정규화 전 원본 데이터 로깅 추가

```python
# scripts/live/live_trading.py의 extract_features() 함수 수정

def extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
    """특징 추출 (정규화 전후 로깅 추가)"""
    try:
        code = market_data.get('종목코드', 'UNKNOWN')

        features = []
        feature_names = []

        for col in FINAL_COLUMNS:
            if col in ["종목코드", "종목명", "시간"]:
                continue
            if col.startswith('매도호가') or col.startswith('매수호가'):
                continue

            value = market_data.get(col, 0.0)
            features.append(float(value))
            feature_names.append(col)

        features_array = np.array(features, dtype=np.float32)

        # ✅ 정규화 전 원본 데이터 로깅 (디버그용)
        if self.stats['predictions'] % 10 == 0:  # 10번마다
            logger.debug(f"[BEFORE NORM] {code}: mean={features_array.mean():.2f}, "
                        f"std={features_array.std():.2f}, "
                        f"range=[{features_array.min():.2f}, {features_array.max():.2f}]")
            logger.debug(f"[BEFORE NORM] {code} 주요값: "
                        f"등락률={features_array[4]:.4f}, "
                        f"누적거래대금={features_array[5]:.2e}, "
                        f"거래회전율={features_array[6]:.4f}, "
                        f"체결강도={features_array[7]:.4f}")

        # 정규화 적용
        features_array = self.normalizer.normalize(code, features_array, update=True)

        # ✅ 정규화 후 데이터 로깅 (디버그용)
        if self.stats['predictions'] % 10 == 0:
            logger.debug(f"[AFTER NORM] {code}: mean={features_array.mean():.4f}, "
                        f"std={features_array.std():.4f}, "
                        f"range=[{features_array.min():.4f}, {features_array.max():.4f}]")

        return features_array
```

#### 1-2. OnlineNormalizer 검증 로직 추가

```python
# scripts/live/online_normalizer.py의 normalize() 함수 수정

def normalize(self, features: np.ndarray, update: bool = True) -> np.ndarray:
    """특징 정규화 (검증 로직 추가)"""

    # ... 기존 코드 ...

    # ✅ 정규화 결과 검증
    if self.sample_count >= self.warmup_samples:
        # 정규화 후 값이 여전히 너무 크면 경고
        if np.abs(normalized).max() > 100:
            logger.warning(
                f"[NORMALIZER] 정규화 후에도 값이 너무 큼! "
                f"max={np.abs(normalized).max():.2f}, "
                f"mean_cache={self.mean_cache[:5]}, "
                f"std_cache={self.std_cache[:5]}"
            )

    return normalized
```

### 🔥 우선순위 2: 호가 데이터 검증 및 필터링

```python
# scripts/live/live_trading.py의 process_realtime_data() 함수 수정

async def process_realtime_data(self):
    """실시간 데이터 처리 (호가 데이터 검증 추가)"""

    while self.running:
        # ... 기존 코드 ...

        for code in list(self.data_buffers.keys()):
            # ... 버퍼 체크 ...

            # ✅ 호가 데이터 검증
            latest_data = self.data_buffers[code][-1]

            # 체결강도와 대기금액이 모두 0이면 스킵
            if (latest_data.get('체결강도', 0) == 0 and
                latest_data.get('매도대기금액1', 0) == 0 and
                latest_data.get('매수대기금액1', 0) == 0):

                logger.warning(
                    f"[DATA QUALITY] {code}: 호가 데이터 누락 (체결강도=0, 대기금액=0), "
                    f"예측 스킵"
                )
                continue

            # ... 나머지 예측 로직 ...
```

### 🔥 우선순위 3: 학습 데이터 정규화 통계 로드

```python
# scripts/live/live_trading.py의 초기화 부분 수정

def __init__(self, config: TradingConfig):
    """초기화 (정규화 통계 로드 추가)"""

    # ... 기존 코드 ...

    # ✅ 학습 데이터 정규화 통계 로드 시도
    norm_stats_path = Path(config.policy_path).parent / "normalization_stats.pkl"

    if norm_stats_path.exists():
        logger.info(f"[NORMALIZER] Loading training normalization stats from {norm_stats_path}")

        import pickle
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)

        # 각 종목에 대해 사전 계산된 통계 설정
        # (실제로는 종목별이 아니라 전체 통계일 수 있음)
        logger.info(f"[NORMALIZER] Loaded stats: mean shape={norm_stats['mean'].shape}, "
                   f"std shape={norm_stats['std'].shape}")

        # 전역 정규화 통계로 사용
        self.global_norm_stats = norm_stats
    else:
        logger.warning(f"[NORMALIZER] No training normalization stats found at {norm_stats_path}")
        logger.warning("[NORMALIZER] Using online normalization only (may cause distribution mismatch)")
        self.global_norm_stats = None
```

### 🔥 우선순위 4: 정규화 통계 생성 스크립트

학습 데이터의 정규화 통계를 저장하는 스크립트 추가:

```python
# scripts/data/save_normalization_stats.py (새 파일)

"""
학습 데이터의 정규화 통계를 계산하고 저장

Usage:
    python scripts/data/save_normalization_stats.py \
        --db datasets_norm_all.duckdb \
        --output models/grpo_scalping/normalization_stats.pkl
"""

import argparse
import pickle
import numpy as np
import duckdb
from pathlib import Path
from lib.normalization import FEATURE_NAMES

def compute_normalization_stats(db_path: str) -> dict:
    """데이터베이스에서 정규화 통계 계산"""

    conn = duckdb.connect(db_path, read_only=True)

    # 특징 컬럼만 선택 (메타데이터 제외)
    feature_cols = [col for col in FEATURE_NAMES]

    print(f"Computing normalization stats for {len(feature_cols)} features...")

    # 각 특징별 평균과 표준편차 계산
    means = []
    stds = []

    for col in feature_cols:
        query = f"""
            SELECT
                AVG("{col}") as mean,
                STDDEV("{col}") as std
            FROM datasets
            WHERE "{col}" IS NOT NULL
        """
        result = conn.execute(query).fetchone()

        mean = result[0] if result[0] is not None else 0.0
        std = result[1] if result[1] is not None else 1.0

        means.append(mean)
        stds.append(std)

        print(f"  {col}: mean={mean:.4f}, std={std:.4f}")

    conn.close()

    return {
        'feature_names': feature_cols,
        'mean': np.array(means, dtype=np.float32),
        'std': np.array(stds, dtype=np.float32)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True, help='DuckDB database path')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    args = parser.parse_args()

    # 통계 계산
    stats = compute_normalization_stats(args.db)

    # 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\n✅ Normalization stats saved to {output_path}")
    print(f"   Features: {len(stats['feature_names'])}")
    print(f"   Mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"   Std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")

if __name__ == '__main__':
    main()
```

### 🔥 우선순위 5: 모델 입력 검증

```python
# ai_trader/grpo/inference/enhanced_inference.py 수정

def predict(self, sequence: np.ndarray, current_price: float = 0.0,
            stock_code: str = "") -> Tuple[int, float, Dict[str, Any]]:
    """예측 (입력 검증 추가)"""

    # ✅ 입력 데이터 검증
    if np.abs(sequence).max() > 100:
        logger.warning(
            f"[INFERENCE] {stock_code}: 입력 데이터가 정규화되지 않은 것으로 보임! "
            f"max={np.abs(sequence).max():.2f}, "
            f"mean={sequence.mean():.2f}, "
            f"std={sequence.std():.2f}"
        )
        logger.warning(
            f"[INFERENCE] {stock_code}: 예측 결과가 부정확할 수 있습니다. "
            f"정규화 통계를 확인하세요."
        )

    # ... 기존 예측 로직 ...
```

## 실행 순서

### 1단계: 정규화 통계 생성 및 저장

```bash
python scripts/data/save_normalization_stats.py \
    --db datasets_norm_all.duckdb \
    --output models/grpo_scalping/normalization_stats.pkl
```

### 2단계: 코드 수정 적용

- 위의 수정 사항들을 각 파일에 적용

### 3단계: 테스트 실행

```bash
python scripts/live/live_trading.py
```

### 4단계: 로그 확인

- `[BEFORE NORM]`과 `[AFTER NORM]` 로그 비교
- 정규화 후 값이 [-5, 5] 범위 내에 있는지 확인
- Buy Signal Rate가 정상적으로 나타나는지 확인

## 예상 결과

수정 후:

- ✅ 입력 데이터가 올바르게 정규화됨 (평균 ~0, 범위 [-5, 5])
- ✅ 모델이 정상적으로 BUY 신호 생성
- ✅ Buy Signal Rate: 0% → 30-50% (정상 범위)

## 추가 고려사항

### 만약 여전히 BUY 신호가 없다면?

1. **모델 재학습 필요**:

   - 학습 데이터의 액션 분포 확인
   - 클래스 불균형 문제 해결

2. **신뢰도 임계값 조정**:

   ```python
   # config/trading_config.json
   {
       "min_buy_confidence": 0.0,  # 0.6 → 0.0으로 낮춤
       "min_sell_confidence": 0.0
   }
   ```

3. **다른 조건 검색식 사용**:
   - 현재: "koa-분당30억"
   - 더 활발한 종목을 찾는 조건식으로 변경
