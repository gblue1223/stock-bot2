# AutoEncoder 전처리 최적화 스크립트

이 디렉토리는 AutoEncoder 기반 embedding 학습을 위한 사전 작업 및 성능 최적화 스크립트들을 포함합니다.

## 📁 파일 구조

### 🔧 전처리 스크립트
- `preprocess_for_autoencoder.py` - 데이터 전처리 및 캐싱
- `parallel_preprocessing.py` - 병렬 전처리 (멀티프로세싱)

### ⚡ 성능 테스트
- `test_simple_performance.py` - 기본 성능 테스트 (더미 데이터)
- `test_data_loading_performance.py` - 데이터 로딩 성능 비교
- `test_preprocessing_performance.py` - 종합 전처리 성능 테스트
- `quick_performance_test.py` - 빠른 성능 확인
- `benchmark_real_data.py` - 실제 데이터 벤치마크

### 📊 성능 보고서
- `performance_test_report.md` - 종합 성능 테스트 보고서
- `performance_test_results.json` - 파일 저장 방식 성능 결과
- `data_loading_performance.json` - 데이터 로딩 성능 결과

## 🚀 주요 성능 개선 결과

### 검증된 성능 향상
- **데이터 로딩**: 5.3배 빠름 (3,483 → 18,601 samples/sec)
- **파일 크기**: 7.2배 압축 (67.9MB → 9.5MB)
- **배치 처리**: 623,734 samples/sec 달성
- **전체 훈련**: 4-8배 시간 단축

### 실용적 효과
- **15억 데이터 학습**: 불가능 → 3-4주 내 완료
- **월별 학습**: 2-3일 → 6-8시간
- **메모리 효율성**: 일반 PC에서도 대용량 처리 가능

## 📋 사용 가이드

### 1. 데이터 전처리 (한 번만 실행)

```bash
# 단일 월 전처리
python scripts/pre/preprocess_for_autoencoder.py \
    --db "datasets_norm_all.duckdb" \
    --output-dir preprocessed_2024_10 \
    --seq-len 60 \
    --batch-size 10000 \
    --start-date 2024-10-01 \
    --end-date 2024-10-31 \
    --compute-norm-params \
    --create-batches

# 병렬 전처리 (권장)
python scripts/pre/parallel_preprocessing.py \
    --db "datasets_norm_all.duckdb" \
    --start-year 2024 --start-month 9 \
    --end-year 2024 --end-month 12 \
    --output preprocessed_data \
    --max-workers 4
```

### 2. 성능 테스트

```bash
# 빠른 성능 확인
python scripts/pre/quick_performance_test.py \
    --db "datasets_norm_all.duckdb" \
    --samples 10000

# 종합 성능 테스트
python scripts/pre/test_simple_performance.py

# 실제 데이터 벤치마크
python scripts/pre/benchmark_real_data.py \
    --db "datasets_norm_all.duckdb" \
    --quick
```

### 3. 최적화된 훈련

```bash
# 전처리된 데이터로 고속 훈련
python examples/autoencoder_training_example.py \
    --preprocessed-dir preprocessed_data_merged \
    --output-dir models/autoencoder_fast \
    --batch-size 128 \
    --use-fast-loader \
    --device cuda
```

## 🎯 최적화 기법

### 1. 파일 저장 최적화
- **HDF5 압축**: 7.2배 파일 크기 절약
- **메모리 맵핑**: 즉시 로딩 (0.000초)
- **배치 단위 저장**: 메모리 효율적 처리

### 2. 데이터 로딩 최적화
- **캐시된 Dataset**: 5.3배 빠른 로딩
- **비동기 로딩**: 백그라운드 배치 준비
- **메모리 맵핑**: 대용량 데이터 효율적 처리

### 3. 배치 처리 최적화
- **최적 배치 크기**: 128 (623,734 samples/sec)
- **병렬 처리**: 멀티프로세싱 활용
- **GPU 최적화**: 혼합 정밀도 훈련

## 📈 성능 벤치마크

### 파일 저장 방식 비교
| 방식 | 파일 크기 | 저장 시간 | 로딩 시간 | 압축률 |
|------|-----------|-----------|-----------|--------|
| NumPy | 67.9 MB | 0.045초 | 0.017초 | 1.0배 |
| **HDF5** | **9.5 MB** | 0.300초 | 0.087초 | **7.2배** |
| 메모리맵 | 67.9 MB | **0.022초** | **0.000초** | 1.0배 |

### 데이터 로딩 성능
| 방식 | 처리량 | 개선도 |
|------|--------|--------|
| 기본 Dataset | 3,483 samples/sec | 기준 |
| **캐시된 Dataset** | **18,601 samples/sec** | **5.3배** |

## 💡 권장사항

1. **전처리 우선**: 반복 훈련 전에 한 번 전처리 수행
2. **HDF5 압축**: 저장 공간이 중요한 경우
3. **메모리 맵핑**: 로딩 속도가 중요한 경우
4. **배치 크기 128**: 최적 성능
5. **병렬 처리**: 멀티코어 CPU 활용

이러한 최적화를 통해 AutoEncoder 학습이 **실용적이고 확장 가능한 솔루션**이 되었습니다!