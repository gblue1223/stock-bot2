# Precompute Pairs Script

## 개요

`precompute_pairs.py`는 대조 학습을 위한 긍정/부정 쌍을 사전에 계산하여 파일로 저장하는 스크립트입니다.

## 왜 필요한가?

`ContrastiveDataset`의 `_precompute_pairs` 메서드는 다음 작업을 수행합니다:

1. 모든 유효한 시퀀스 인덱스 계산
2. 종목별 인덱스 매핑 생성
3. 각 샘플에 대해 긍정 쌍 후보 찾기 (동일 종목, 가까운 시간)
4. 각 샘플에 대해 부정 쌍 후보 찾기 (다른 종목 또는 먼 시간)

**문제점:** 데이터셋이 클 경우 이 과정이 10분~수 시간 소요될 수 있습니다.

**해결책:** 한 번만 계산하여 파일로 저장하고, 이후 훈련 시 빠르게 로드합니다.

## 사용법

### 기본 사용

```bash
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs
```

### 전체 옵션 (최적화된 설정)

```bash
python scripts/precompute_pairs.py \
    --db_path "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
    --table_name datasets \
    --output_dir "C:\Users\user\Workspace\datasets@20251005\precomputed_pairs" \
    --seq_len 60 \
    --positive_threshold 0.05 \
    --negative_threshold 0.5 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --start_date 2024-09-01 \
    --end_date 2025-09-30 \
    --date-chunk-days 30 \
    --skip-clipping \
    --checkpoint-interval 10000
```

**성능 최적화 옵션:**
- `--date-chunk-days 30`: 날짜 기반 청크 처리 (OFFSET 제거로 **10-50배 빠름**)
- `--skip-clipping`: 데이터 클리핑 건너뛰기 (이미 정규화된 경우)
- `--num_threads`: 자동 설정 (CPU 코어 수)

### 멀티스레드 사용

```bash
# CPU 코어 수만큼 자동 설정 (기본값)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs

# 스레드 수 직접 지정 (8개 스레드)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --num_threads 8

# 단일 스레드 모드 (디버깅용)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --num_threads 1
```

### 중단 후 재개 (Resume)

```bash
# 중단된 작업 재개 (이미 완료된 split 건너뛰기)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --resume

# 예: train_pairs.pkl만 완료된 상태에서 재개하면
# val_pairs.pkl, test_pairs.pkl만 처리
```

### 대용량 데이터 처리 (OOM 방지)

#### 방법 1: 날짜 기반 청크 (권장, 가장 빠름)

```bash
# 30일 단위로 처리 (OFFSET 없이 WHERE 절만 사용)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --date-chunk-days 30 \
    --skip-clipping

# 7일 단위 (더 작은 청크)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --date-chunk-days 7 \
    --skip-clipping
```

#### 방법 2: 행 기반 청크 (날짜 범위 없을 때)

```bash
# 자동 청크 크기 (max_samples > 5M: 2M 청크, > 2M: 1M 청크)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --max_samples 10000000

# 수동 청크 크기 지정
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --max_samples 10000000 \
    --chunk_size 1000000

# 청크 처리 + 체크포인트 (가장 안전)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --max_samples 10000000 \
    --chunk_size 1000000 \
    --num_threads 1 \
    --resume \
    --checkpoint-interval 5000
```

## 출력

스크립트는 3개의 파일을 생성합니다:

- `train_pairs.pkl`: 훈련 세트 쌍
- `val_pairs.pkl`: 검증 세트 쌍
- `test_pairs.pkl`: 테스트 세트 쌍

각 파일에는 다음이 포함됩니다:

```python
{
    'valid_indices': np.ndarray,           # 유효한 시퀀스 인덱스
    'stock_indices': Dict[str, List[int]], # 종목별 인덱스 매핑
    'positive_pairs_cache': Dict[int, List[int]], # 긍정 쌍 캐시
    'negative_pairs_cache': Dict[int, List[int]], # 부정 쌍 캐시
    'seq_len': int,
    'positive_threshold': int,
    'negative_threshold': int,
    'metadata_shape': tuple,
    'data_shape': tuple
}
```

## 훈련 시 사용

### CLI

```bash
python -m ai_trader.embedding.train_embedding \
    --db data/trading_data.duckdb \
    --table datasets \
    --out models/embedding \
    --precomputed-pairs-dir data/precomputed_pairs \
    --batch-size 128 \
    --epochs 50
```

### Python API

```python
from ai_trader.embedding.data import EmbeddingDataLoader

data_loader = EmbeddingDataLoader(db_path='data/trading_data.duckdb')
data_loader.connect()
data_loader.load_and_split_data()

train_loader = data_loader.get_dataloader(
    split='train',
    batch_size=128,
    precomputed_pairs_path='data/precomputed_pairs/train_pairs.pkl'
)
```

## 성능

### 사전 계산 시간

#### 기존 방식 (행 기반 청크, OFFSET 사용)

| 샘플 수 | 단일 스레드 | 8 스레드 | 파일 크기 |
|---------|-----------|---------|----------|
| 100만   | ~5-10분   | ~1-2분  | ~100-500MB |
| 1000만  | ~30-60분  | ~5-10분 | ~1-5GB |
| 3000만  | ~1-3시간  | ~15-30분 | ~3-15GB |
| 20억    | ~67-100시간 | ~10-15시간 | ~200GB |

#### 최적화 방식 (날짜 기반 청크, OFFSET 제거)

| 샘플 수 | 30일 청크 | 7일 청크 | 속도 향상 |
|---------|----------|---------|----------|
| 100만   | ~30초-1분 | ~20-40초 | **10-20배** |
| 1000만  | ~3-6분   | ~2-4분  | **10-20배** |
| 3000만  | ~10-20분 | ~6-12분 | **6-15배** |
| 20억    | ~6-12시간 | ~4-8시간 | **10-15배** |

**성능 향상 요인:**
- 🚀 **날짜 기반 청크**: OFFSET 제거로 DB 쿼리 **10-50배 빠름**
- ⚡ **클리핑 스킵**: percentile 계산 생략으로 **2-3배 빠름**
- 🔥 **멀티스레드**: CPU 코어 수에 따라 **4-6배 빠름**

### 훈련 시작 시간 비교

| 샘플 수 | 사전 계산 없이 | 사전 계산 사용 | 속도 향상 |
|---------|--------------|--------------|----------|
| 100만   | ~10-30분     | ~1-5초       | ~100-360x |
| 1000만  | ~1-3시간     | ~10-30초     | ~120-360x |

## 재개 기능 (Resume)

작업이 중단되었을 때 `--resume` 플래그를 사용하여 이어서 작업할 수 있습니다.

### 동작 방식

**2단계 재개 시스템:**

1. **Split 단위 재개**: 완료된 split(train/val/test) 파일 건너뛰기
2. **인덱스 단위 재개**: Split 내에서 체크포인트로 중단 지점부터 재개

### Split 단위 재개

```bash
# 1차 실행 (train만 완료하고 중단됨)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --num_threads 8
# 결과: train_pairs.pkl 생성됨

# 2차 실행 (재개)
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --num_threads 8 \
    --resume
# 결과: train은 건너뛰고 val, test만 처리
```

### 인덱스 단위 재개 (체크포인트)

```bash
# 단일 스레드 모드에서 체크포인트 사용
python scripts/precompute_pairs.py \
    --db_path data/trading_data.duckdb \
    --output_dir data/precomputed_pairs \
    --num_threads 1 \
    --resume \
    --checkpoint-interval 10000

# 중단 후 재개하면:
# - train_checkpoint.pkl 파일에서 진행 상황 로드
# - 이미 처리된 인덱스는 건너뛰고 나머지만 처리
# - 10,000개 인덱스마다 자동 저장
```

### 체크포인트 파일

- **위치**: `{output_dir}/{split_name}_checkpoint.pkl`
- **내용**: 처리된 인덱스 목록 + 긍정/부정 쌍 캐시
- **자동 삭제**: Split 완료 시 자동으로 삭제됨

### 주의사항

- ✅ **Split 단위 재개**: 모든 모드에서 지원
- ✅ **인덱스 단위 재개**: 단일 스레드 모드(`--num_threads 1`)에서만 지원
- ⚠️ **멀티스레드**: 체크포인트 미지원 (Split 단위 재개만 가능)
- 💡 **권장**: 대용량 데이터는 단일 스레드 + 체크포인트 사용

## OOM (Out of Memory) 방지

### 청크 처리 방식 비교

| 방식 | 성능 | 메모리 | 사용 시기 |
|------|------|--------|----------|
| **날짜 기반** | ⭐⭐⭐⭐⭐ | 낮음 | 날짜 범위가 있을 때 (권장) |
| **행 기반** | ⭐⭐ | 중간 | 날짜 범위가 없을 때 |
| **청크 없음** | ⭐⭐⭐⭐⭐ | 높음 | 소규모 데이터 (< 2M) |

### 날짜 기반 청크 (권장)

**장점:**
- OFFSET 없이 WHERE 절만 사용 → **10-50배 빠른 쿼리**
- 각 청크가 독립적 → 병렬 처리 가능
- 메모리 효율적

**동작 방식:**
1. 날짜 범위를 N일 단위로 분할
2. 각 청크: `WHERE 날짜 >= start AND 날짜 < end`
3. 청크별 데이터 로드 → 병합

```bash
# 예: 2024-01-01 ~ 2024-12-31을 30일씩 처리
python scripts/precompute_pairs.py \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --date-chunk-days 30
```

### 행 기반 청크 (레거시)

스크립트는 `max_samples` 값에 따라 자동으로 청크 크기를 설정합니다:

| max_samples | 자동 chunk_size | 설명 |
|-------------|----------------|------|
| < 2M | 청크 없음 | 한 번에 로드 |
| 2M - 5M | 1M | 2-5개 청크로 분할 |
| > 5M | 2M | 여러 청크로 분할 |

**동작 방식:**
1. **데이터 로드**: 청크별로 데이터베이스에서 로드 (OFFSET 사용, 느림)
2. **Split 분할**: 각 청크를 train/val/test로 분할
3. **병합**: 모든 청크의 split을 병합 (`np.vstack`)
4. **쌍 계산**: 병합된 데이터로 긍정/부정 쌍 계산

### 메모리 사용량 예측

| 샘플 수 | 피처 수 | 예상 메모리 | 권장 설정 |
|---------|---------|-----------|----------|
| 1M | 50 | ~200MB | 청크 불필요 |
| 5M | 50 | ~1GB | chunk_size=2M |
| 10M | 50 | ~2GB | chunk_size=2M |
| 30M | 50 | ~6GB | chunk_size=2M, 체크포인트 |

## 주의사항

1. **데이터 일관성**: 데이터베이스가 변경되면 쌍을 다시 계산해야 합니다.
2. **파라미터 일치**: `seq_len`, `positive_threshold`, `negative_threshold`가 일치해야 합니다.
3. **디스크 공간**: 큰 데이터셋의 경우 수 GB의 디스크 공간이 필요합니다.
4. **메모리**: 
   - 청크 처리 시: chunk_size에 비례
   - 쌍 계산 시: 전체 데이터 메모리 필요
   - 병합 시: 일시적으로 2배 메모리 사용
5. **재개 기능**: 
   - Split 단위: 모든 모드에서 지원
   - 인덱스 단위: 단일 스레드 모드에서만 지원
6. **체크포인트**: 멀티스레드 모드에서는 체크포인트가 저장되지 않습니다.
7. **청크 처리**: 
   - 날짜 기반: `--date-chunk-days` 사용 (권장)
   - 행 기반: max_samples > 2M일 때 자동 활성화
8. **클리핑**: `--skip-clipping` 사용 시 데이터가 이미 정규화되어 있어야 함

## 권장 설정

### 대용량 데이터 (> 10M 샘플)

```bash
python scripts/precompute_pairs.py \
    --db_path data.duckdb \
    --output_dir precomputed \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --date-chunk-days 30 \
    --skip-clipping \
    --resume
```

### 중간 규모 (1M-10M 샘플)

```bash
python scripts/precompute_pairs.py \
    --db_path data.duckdb \
    --output_dir precomputed \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --date-chunk-days 60 \
    --skip-clipping
```

### 소규모 (< 1M 샘플)

```bash
python scripts/precompute_pairs.py \
    --db_path data.duckdb \
    --output_dir precomputed
```

## 예제

전체 예제는 `examples/precompute_pairs_example.py`를 참조하세요.

```bash
# 예제 실행
python examples/precompute_pairs_example.py --example all
```

## 관련 문서

- [상세 가이드](../docs/PRECOMPUTE_PAIRS.md)
- [빠른 참조](../docs/PRECOMPUTE_PAIRS_QUICK_REFERENCE.md)

## FAQ

### Q: 왜 이렇게 느린가요?

**A:** 행 기반 청크 처리(`--chunk_size`)는 OFFSET을 사용하여 매우 느립니다.
- ❌ 느림: `--chunk_size 1000000` (OFFSET 사용)
- ✅ 빠름: `--date-chunk-days 30` (WHERE 절만 사용)

### Q: 날짜 기반 청크를 사용할 수 없는 경우는?

**A:** `start_date`와 `end_date`가 없으면 날짜 기반 청크를 사용할 수 없습니다.
이 경우 행 기반 청크를 사용하되, 가능하면 날짜 범위를 지정하세요.

### Q: 클리핑을 건너뛰어도 되나요?

**A:** 데이터가 이미 정규화되어 있다면 `--skip-clipping`을 사용하여 **2-3배 빠르게** 처리할 수 있습니다.
정규화되지 않은 데이터는 클리핑이 필요합니다.

### Q: 멀티스레드를 사용해야 하나요?

**A:** 기본값(CPU 코어 수)을 사용하세요. 단, 체크포인트가 필요하면 `--num_threads 1`을 사용하세요.
