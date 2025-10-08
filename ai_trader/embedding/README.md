# Stock Bot

## 목차

- [Features](#features)
- [데이터 정규화](#데이터-정규화)
- [임베딩 모델 (Embedding Model)](#임베딩-모델-embedding-model)
- [트러블슈팅](#트러블슈팅)

## Features

```bash
날짜 번호 종목코드 종목명 시간 등락률 누적거래대금 거래회전율 체결강도 매도대기금액1 매도대기금액2 매도대기금액3 매도대기금액4 매도대기금액5 매도대기금액6 매도대기금액7 매도대기금액8 매도대기금액9 매도대기금액10 매수대기금액1 매수대기금액2 매수대기금액3 매수대기금액4 매수대기금액5 매수대기금액6 매수대기금액7 매수대기금액8 매수대기금액9 매수대기금액10 종목명_scalar 시간_sin 시간_cos 시간_scalar
```

## 데이터 정규화

```bash
python scripts/generate_datasets.py \
  "D:\Workspace\Project\stock-bot\hoga-crawler\data" \
  -o "C:\Users\user\Workspace\datasets@raw\datasets.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --checkpoint-interval 50

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets@raw" \
  --out "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --threads 4 \
  --memory-limit 64GB

python scripts/normalize_datasets.py \
  "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  -o "C:\Users\user\Workspace\datasets\datasets_norm.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --checkpoint-interval 50 \
  --ignoring-stocks-csv scripts/ignoring_stocks.csv

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets" \
  --out "C:\Users\user\Workspace\datasets\datasets_norm_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --threads 4 \
  --memory-limit 64GB
```

## 임베딩 모델 (Embedding Model)

임베딩 모델은 60개 이상의 고차원 매매 특징을 저차원 밀집 벡터로 변환하여 GRPO 강화학습 에이전트가 효율적으로 학습할 수 있도록 합니다.

**대규모 훈련**: Google Cloud Platform A100 40GB x12 GPU를 활용한 분산 훈련 가이드는 [GCP 훈련 가이드](docs/GCP_TRAINING_GUIDE.md)를 참조하세요.

### 임베딩 모델 훈련

대조 학습(Contrastive Learning)을 사용하여 시간적으로 가까운 샘플은 유사하게, 먼 샘플은 다르게 임베딩합니다.

```bash
# 9월 데이터로 초기 훈련
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding_2024_09 \
  --seq-len 60 \
  --embedding-dim 128 \
  --batch-size 256 \
  --epochs 30 \
  --lr 1e-4 \
  --device cuda \
  --positive-time-threshold 10 \
  --negative-time-threshold 60 \
  --temperature 0.07 \
  --num-heads 4 \
  --val-every 3 \
  --checkpoint-interval 5 \
  --start-date 20240901 \
  --end-date 20240930 \
  --num-workers 8 \
  --max-samples 5000000

# 10월 데이터로 fine-tuning
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding_2024_10 \
  --seq-len 60 \
  --embedding-dim 128 \
  --batch-size 256 \
  --epochs 15 \
  --lr 5e-5 \
  --device cuda \
  --positive-time-threshold 10 \
  --negative-time-threshold 60 \
  --temperature 0.07 \
  --num-heads 4 \
  --val-every 3 \
  --checkpoint-interval 5 \
  --start-date 20241001 \
  --end-date 20241031 \
  --num-workers 8 \
  --max-samples 5000000 \
  --resume models/embedding_2024_09/checkpoint_epoch30.pt

# 변경 사항:
  --epochs 15 (30→15, fine-tuning이므로 절반)
  --lr 5e-5 (1e-4→5e-5, 학습률 절반)
  --resume 이전 달 모델 로드

# 월별 스크립트
python scripts/train_embedding_monthly.py \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --start-year 2024 --start-month 9 \
  --end-year 2024 --end-month 12 \
  --output models/embedding \
  --max-samples 5000000

python scripts/train_embedding_monthly.py \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --quarterly \
  --year 2024 --quarter 4 \
  --output models/embedding \
  --max-samples 5000000
```

#### CLI 인수 설명

**필수 인수:**

- `--db`: DuckDB 데이터베이스 경로
- `--table`: 데이터베이스 테이블 이름
- `--out`: 모델 체크포인트 저장 디렉토리

**모델 설정:**

- `--seq-len`: 입력 시퀀스 길이 (기본값: 60)
- `--embedding-dim`: 임베딩 벡터 차원 (기본값: 128, 권장: 128-256)
- `--num-heads`: Multi-head Attention 헤드 수 (기본값: 4)

**훈련 설정:**

- `--batch-size`: 배치 크기 (기본값: 128)
- `--epochs`: 훈련 에포크 수 (기본값: 50)
- `--lr`: 학습률 (기본값: 1e-4)
- `--device`: 디바이스 (cuda/cpu, 기본값: cuda)
- `--num-workers`: 데이터 로더 워커 프로세스 수 (기본값: 4)

**대조 학습 설정:**

- `--temperature`: InfoNCE 손실 온도 파라미터 (기본값: 0.07, 권장: 0.05-0.1)
- `--positive-time-threshold`: 긍정 쌍 시간 임계값 (초, 기본값: 10)
- `--negative-time-threshold`: 부정 쌍 시간 임계값 (초, 기본값: 60)

**체크포인트 및 로깅:**

- `--checkpoint-interval`: 체크포인트 저장 간격 (에포크, 기본값: 5)
- `--val-every`: 검증 주기 (에포크, 기본값: 1 = 매 에포크마다 검증)
- `--resume`: 재개할 체크포인트 경로 (선택사항)

**데이터 필터링 (메모리 절약):**

- `--start-date`: 시작 날짜 (YYYYMMDD 형식, 예: 20240901)
- `--end-date`: 종료 날짜 (YYYYMMDD 형식, 예: 20240930)
- `--stock-codes`: 훈련할 종목 코드 리스트 (예: 005930 000660)
- `--max-samples`: 최대 샘플 수 (메모리 제한 시 사용, 예: 1000000)

#### 대용량 데이터 훈련 전략 (12개월 5억 개 데이터)

**권장 설정**:

- 월별 최대 샘플: 500만 개 (`--max-samples 5000000`)
- 배치 크기: 256 (`--batch-size 256`)
- 워커 수: 8 (`--num-workers 8`)
- 검증 주기: 3 에포크마다 (`--val-every 3`)

**월별 순차 훈련 (자동화)**:

```bash
# 9월부터 12월까지 자동으로 순차 훈련
python scripts/train_embedding_monthly.py \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --start-year 2024 --start-month 9 \
  --end-year 2024 --end-month 12 \
  --output models/embedding \
  --max-samples 5000000
```

**분기별 재훈련**:

```bash
# Q4 (10-12월) 데이터로 전체 재훈련
python scripts/train_embedding_monthly.py \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --quarterly \
  --year 2024 --quarter 4 \
  --output models/embedding \
  --max-samples 5000000
```

**수동 월별 훈련**:

```bash
# 9월 초기 훈련
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding_2024_09 \
  --start-date 20240901 \
  --end-date 20240930 \
  --seq-len 60 \
  --embedding-dim 128 \
  --batch-size 256 \
  --epochs 30 \
  --lr 1e-4 \
  --device cuda \
  --num-workers 8 \
  --max-samples 5000000

# 10월 Fine-tuning
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding_2024_10 \
  --start-date 20241001 \
  --end-date 20241031 \
  --seq-len 60 \
  --embedding-dim 128 \
  --batch-size 256 \
  --epochs 15 \
  --lr 5e-5 \
  --device cuda \
  --num-workers 8 \
  --max-samples 5000000 \
  --resume models/embedding_2024_09/checkpoint_epoch30.pt
```

#### 메모리 절약 훈련 예제

메모리가 부족한 경우 다음 옵션을 사용하세요:

```bash
# 특정 기간만 훈련 (1개월)
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding \
  --start-date 20240901 \
  --end-date 20240930 \
  --epochs 50 \
  --device cuda

# 특정 종목만 훈련
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding \
  --stock-codes 005930 000660 035420 \
  --epochs 50 \
  --device cuda

# 최대 샘플 수 제한
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding \
  --max-samples 1000000 \
  --epochs 50 \
  --device cuda
```

### 임베딩 모델 평가

```bash
python -m ai_trader.embedding.evaluate_embedding \
  --model models/embedding/checkpoint_epoch50.pt \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --device cuda
```

평가 메트릭:

- **Silhouette Score**: 종목별 클러스터링 품질 (높을수록 좋음)
- **Temporal Coherence**: 시간적 일관성 (높을수록 좋음)

## 트러블슈팅

### 임베딩 모델

- **손실이 감소하지 않는 경우**: temperature 파라미터를 조정하세요 (0.05-0.1 범위).
- **메모리 부족**: batch-size를 줄이거나 embedding-dim을 낮추세요 (128 → 64).
- **훈련 속도 느림**: positive/negative 쌍 생성 비율을 조정하세요.

## 예제 스크립트

`examples/` 디렉토리에서 다양한 사용 예제를 확인할 수 있습니다:

- `embedding_evaluation_example.py`: 임베딩 모델 평가

자세한 사용법은 각 예제 파일을 참조하세요.
