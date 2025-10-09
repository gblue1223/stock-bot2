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

### 전체 옵션

```bash
python scripts/precompute_pairs.py \
    --db_path "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
    --table_name datasets \
    --output_dir "C:\Users\user\Workspace\datasets@20251005\precomputed_pairs" \
    --seq_len 60 \
    --positive_threshold 10 \
    --negative_threshold 60 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --start_date 2024-09-01 \
    --end_date 2025-09-30 \
    --max_samples 10000000 \
    --num_threads 8
```

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

| 샘플 수 | 단일 스레드 | 8 스레드 | 파일 크기 |
|---------|-----------|---------|----------|
| 100만   | ~5-10분   | ~1-2분  | ~100-500MB |
| 1000만  | ~30-60분  | ~5-10분 | ~1-5GB |
| 3000만  | ~1-3시간  | ~15-30분 | ~3-15GB |

**멀티스레드 성능 향상**: 약 **4-6배** (CPU 코어 수에 따라 다름)

### 훈련 시작 시간 비교

| 샘플 수 | 사전 계산 없이 | 사전 계산 사용 | 속도 향상 |
|---------|--------------|--------------|----------|
| 100만   | ~10-30분     | ~1-5초       | ~100-360x |
| 1000만  | ~1-3시간     | ~10-30초     | ~120-360x |

## 주의사항

1. **데이터 일관성**: 데이터베이스가 변경되면 쌍을 다시 계산해야 합니다.
2. **파라미터 일치**: `seq_len`, `positive_threshold`, `negative_threshold`가 일치해야 합니다.
3. **디스크 공간**: 큰 데이터셋의 경우 수 GB의 디스크 공간이 필요합니다.
4. **메모리**: 파일 로드 시 전체 캐시가 메모리에 로드됩니다.

## 예제

전체 예제는 `examples/precompute_pairs_example.py`를 참조하세요.

```bash
# 예제 실행
python examples/precompute_pairs_example.py --example all
```

## 관련 문서

- [상세 가이드](../docs/PRECOMPUTE_PAIRS.md)
- [빠른 참조](../docs/PRECOMPUTE_PAIRS_QUICK_REFERENCE.md)
