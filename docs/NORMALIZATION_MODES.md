# 정규화 모드 가이드

실시간 거래 시스템에서 두 가지 정규화 모드를 지원합니다.

## 모드 1: 학습 통계 사용 (권장)

**가정**: 실시간 데이터가 학습 데이터와 유사한 분포를 가짐

### 장점
- 학습 시와 동일한 정규화 → 모델 성능 일관성 보장
- 워밍업 기간 불필요 → 즉시 추론 가능
- 간단한 구조 → 디버깅 용이
- 종목 간 일관된 정규화

### 설정 방법

1. **정규화 통계 파일 준비**
   ```json
   {
     "mean": [0.0, 0.1, ...],  // 28개 특징의 평균
     "std": [1.0, 0.5, ...]    // 28개 특징의 표준편차
   }
   ```

2. **설정 파일에 경로 지정**
   ```json
   {
     "normalization_stats_path": "models/grpo_scalping/normalization_stats.json"
   }
   ```

3. **실행**
   ```bash
   python scripts/live/live_trading.py --config config/trading_config_with_training_stats.json
   ```

### 동작 방식
```
실시간 데이터 → GRPOInference (학습 통계로 정규화) → 추론
```

## 모드 2: 온라인 정규화

**가정**: 실시간 데이터가 학습 데이터와 다를 수 있음

### 장점
- 시장 상황 변화에 동적 적응
- 종목별 특성 반영
- 분포 변화에 강건함

### 단점
- 워밍업 기간 필요 (50-200 샘플)
- 초기 추론 품질 저하 가능
- 복잡한 구조

### 설정 방법

1. **설정 파일에서 경로 제거**
   ```json
   {
     "normalization_stats_path": null,
     "online_window_size": 200,
     "online_warmup_samples": 50
   }
   ```

2. **실행**
   ```bash
   python scripts/live/live_trading.py --config config/trading_config.json
   ```

### 동작 방식
```
실시간 데이터 → PerStockNormalizer (Rolling Window) → GRPOInference → 추론
```

## 정규화 통계 생성 방법

학습 데이터에서 정규화 통계를 추출하려면:

```python
import duckdb
import numpy as np
import json

# DuckDB에서 통계 계산
conn = duckdb.connect('datasets_norm_all.duckdb')

# 전체 데이터의 평균과 표준편차
result = conn.execute("""
    SELECT 
        AVG(feature_0) as mean_0, STDDEV(feature_0) as std_0,
        AVG(feature_1) as mean_1, STDDEV(feature_1) as std_1,
        -- ... 모든 특징에 대해 반복
    FROM features_table
""").fetchone()

# JSON으로 저장
stats = {
    'mean': [result[i*2] for i in range(28)],
    'std': [result[i*2+1] for i in range(28)]
}

with open('normalization_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

또는 학습 스크립트에서 자동 저장:

```python
# ai_trader/grpo/train_scalping.py 수정
import json

# 학습 후
normalization_stats = {
    'mean': env.mean.tolist(),
    'std': env.std.tolist()
}

stats_path = output_dir / 'normalization_stats.json'
with open(stats_path, 'w') as f:
    json.dump(normalization_stats, f, indent=2)

logger.info(f"Saved normalization stats to {stats_path}")
```

## 권장 사항

### 학습 통계 사용을 권장하는 경우
- ✅ 학습 데이터가 최근 시장 데이터를 충분히 포함
- ✅ 안정적인 시장 환경
- ✅ 빠른 시작이 중요한 경우
- ✅ 디버깅 및 재현성이 중요한 경우

### 온라인 정규화를 권장하는 경우
- ✅ 학습 데이터가 오래됨 (6개월 이상)
- ✅ 시장 변동성이 큰 경우
- ✅ 종목별 특성 차이가 큰 경우
- ✅ 워밍업 시간 여유가 있는 경우

## 성능 비교

| 항목 | 학습 통계 | 온라인 정규화 |
|------|----------|--------------|
| 추론 지연 | 2.87ms | 2.87ms |
| 워밍업 시간 | 0초 | 50-200초 |
| 메모리 사용 | 낮음 | 중간 |
| 일관성 | 높음 | 중간 |
| 적응성 | 낮음 | 높음 |

## 문제 해결

### 학습 통계 사용 시 성능 저하
→ 학습 데이터가 오래되었을 가능성. 온라인 정규화로 전환하거나 재학습 필요

### 온라인 정규화 워밍업이 너무 느림
→ `online_warmup_samples`를 줄이거나 학습 통계 사용으로 전환

### 종목별 성능 차이가 큼
→ 온라인 정규화 사용 (종목별 통계 유지)
