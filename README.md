# Stock Bot

## 목차

- [Features](#features)
- [데이터 정규화](#데이터-정규화)
- [임베딩 모델 (Embedding Model)](#임베딩-모델-embedding-model)
- [GRPO 강화학습 (Reinforcement Learning)](#grpo-강화학습-reinforcement-learning)
- [지도학습 훈련](#지도학습-훈련)
- [트러블슈팅](#트러블슈팅)

## Features

```bash
날짜, 등락률, 누적거래대금, 거래회전율, 체결강도, 매도호가수량1, 매도호가수량2, 매도호가수량3, 매도호가수량4, 매도호가수량5, 매도호가수량6, 매도호가수량7, 매도호가수량8, 매도호가수량9, 매도호가수량10, 매수호가수량1, 매수호가수량2, 매수호가수량3, 매수호가수량4, 매수호가수량5, 매수호가수량6, 매수호가수량7, 매수호가수량8, 매수호가수량9, 매수호가수량10, 매도호가총잔량, 매수호가총잔량, 매도거래원수량1, 매도거래원수량2, 매도거래원수량3, 매도거래원수량4, 매도거래원수량5, 매도거래원별증감1, 매도거래원별증감2, 매도거래원별증감3, 매도거래원별증감4, 매도거래원별증감5, 매수거래원수량1, 매수거래원수량2, 매수거래원수량3, 매수거래원수량4, 매수거래원수량5, 매수거래원별증감1, 매수거래원별증감2, 매수거래원별증감3, 매수거래원별증감4, 매수거래원별증감5, 매도거래원1_scalar, 매도거래원2_scalar, 매도거래원3_scalar, 매도거래원4_scalar, 매도거래원5_scalar, 매수거래원1_scalar, 매수거래원2_scalar, 매수거래원3_scalar, 매수거래원4_scalar, 매수거래원5_scalar, 종목명_scalar, 시간_scalar, 시간_sin, 시간_cos
```

```bash
날짜, 등락률, 누적거래대금, 거래회전율, 체결강도, 매도호가수량1, 매도호가수량2, 매도호가수량3, 매도호가수량4, 매도호가수량5, 매도호가수량6, 매도호가수량7, 매도호가수량8, 매도호가수량9, 매도호가수량10, 매도호가직전대비1, 매도호가직전대비2, 매도호가직전대비3, 매도호가직전대비4, 매도호가직전대비5, 매도호가직전대비6, 매도호가직전대비7, 매도호가직전대비8, 매도호가직전대비9, 매도호가직전대비10, 매수호가수량1, 매수호가수량2, 매수호가수량3, 매수호가수량4, 매수호가수량5, 매수호가수량6, 매수호가수량7, 매수호가수량8, 매수호가수량9, 매수호가수량10, 매수호가직전대비1, 매수호가직전대비2, 매수호가직전대비3, 매수호가직전대비4, 매수호가직전대비5, 매수호가직전대비6, 매수호가직전대비7, 매수호가직전대비8, 매수호가직전대비9, 매수호가직전대비10, 매도호가총잔량, 매도호가총잔량직전대비, 매수호가총잔량, 매수호가총잔량직전대비, 매도거래원1, 매도거래원2, 매도거래원3, 매도거래원4, 매도거래원5, 매도거래원수량1, 매도거래원수량2, 매도거래원수량3, 매도거래원수량4, 매도거래원수량5, 매도거래원별증감1, 매도거래원별증감2, 매도거래원별증감3, 매도거래원별증감4, 매도거래원별증감5, 매수거래원1, 매수거래원2, 매수거래원3, 매수거래원4, 매수거래원5, 매수거래원수량1, 매수거래원수량2, 매수거래원수량3, 매수거래원수량4, 매수거래원수량5, 매수거래원별증감1, 매수거래원별증감2, 매수거래원별증감3, 매수거래원별증감4, 매수거래원별증감5, 외국계매수추정합변동, 외국계매도추정합변동, 외국계매수추정합, 외국계매도추정합, 매도거래원1_scalar, 매도거래원2_scalar, 매도거래원3_scalar, 매도거래원4_scalar, 매도거래원5_scalar, 매수거래원1_scalar, 매수거래원2_scalar, 매수거래원3_scalar, 매수거래원4_scalar, 매수거래원5_scalar, 종목명_scalar, 시간_scalar, 시간_sin, 시간_cos
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

### 임베딩 모델 훈련

대조 학습(Contrastive Learning)을 사용하여 시간적으로 가까운 샘플은 유사하게, 먼 샘플은 다르게 임베딩합니다.

```bash
python -m ai_trader.embedding.train_embedding \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/embedding \
  --seq-len 60 \
  --embedding-dim 128 \
  --batch-size 128 \
  --epochs 50 \
  --lr 1e-4 \
  --device cuda \
  --positive-time-threshold 10 \
  --negative-time-threshold 60 \
  --temperature 0.07 \
  --num-heads 4 \
  --val-every 5 \
  --checkpoint-interval 10 \
  --num-workers 4
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

## GRPO 강화학습 (Reinforcement Learning)

GRPO(Group Relative Policy Optimization)는 시장 상황별로 에피소드를 그룹화하여 상대적 성능을 기반으로 정책을 최적화합니다.

### GRPO 에이전트 훈련

```bash
python -m ai_trader.grpo.train_grpo \
  --embedding-model models/embedding/checkpoint_epoch50.pt \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/grpo_scalping \
  --seq-len 60 \
  --episodes-per-group 12 \
  --num-groups 4 \
  --total-timesteps 1000000 \
  --lr 3e-4 \
  --gamma 0.99 \
  --clip-epsilon 0.2 \
  --device cuda \
  --quick-exit-threshold 1.5 \
  --quick-exit-penalty 0.01 \
  --transaction-cost 0.00215 \
  --ckpt-every 50000
```

#### CLI 인수 설명

**필수 인수:**

- `--embedding-model`: 훈련된 임베딩 모델 체크포인트 경로
- `--db`: DuckDB 데이터베이스 경로
- `--table`: 데이터베이스 테이블 이름
- `--out`: 정책 체크포인트 저장 디렉토리

**환경 설정:**

- `--seq-len`: 입력 시퀀스 길이 (기본값: 60, 임베딩 모델과 동일해야 함)
- `--transaction-cost`: 거래 비용 비율 (기본값: 0.00215, 즉 0.215%)
- `--max-holding-time`: 최대 보유 시간 (초, 기본값: 60)

**빠른 손절 룰:**

- `--quick-exit-threshold`: 빠른 손절 시간 임계값 (초, 기본값: 1.5)
- `--quick-exit-penalty`: 빠른 손절 룰 위반 페널티 (기본값: 0.01)

**GRPO 알고리즘 설정:**

- `--episodes-per-group`: 그룹당 에피소드 수 (기본값: 12, 권장: 8-16)
- `--num-groups`: 시장 상황 그룹 수 (기본값: 4)
- `--total-timesteps`: 총 훈련 타임스텝 (기본값: 1000000)

**정책 네트워크 설정:**

- `--hidden-dim`: 정책 네트워크 은닉층 차원 (기본값: 256)
- `--lr`: 학습률 (기본값: 3e-4)
- `--gamma`: 할인 계수 (기본값: 0.99)
- `--clip-epsilon`: PPO 클리핑 파라미터 (기본값: 0.2)
- `--kl-target`: KL 발산 목표값 (기본값: 0.01)

**체크포인트 및 로깅:**

- `--ckpt-every`: 체크포인트 저장 주기 (타임스텝, 기본값: 50000)
- `--resume-from`: 체크포인트에서 재개
- `--device`: 디바이스 (cuda/cpu, 기본값: cuda)

### GRPO 에이전트 평가 및 백테스팅

```bash
python -m ai_trader.grpo.evaluate_grpo \
  --embedding-model models/embedding/checkpoint_epoch50.pt \
  --policy models/grpo_scalping/checkpoint_step1000000.pt \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --device cuda \
  --num-episodes 1000
```

평가 메트릭:

- **승률 (Win Rate)**: 수익 거래 비율 (목표: > 50%)
- **거래당 평균 수익**: 거래당 평균 수익률
- **샤프 비율 (Sharpe Ratio)**: 위험 대비 수익 (목표: > 1.0)
- **최대 낙폭 (Max Drawdown)**: 최대 손실 폭 (목표: < 10%)
- **평균 보유 시간**: 평균 포지션 보유 시간 (목표: < 30초)
- **빠른 손절 룰 위반 횟수**: 1.5초 룰 위반 횟수

### 실시간 추론

```bash
python -m ai_trader.inference.infer_grpo \
  --embedding-model models/embedding/checkpoint_epoch50.pt \
  --policy models/grpo_scalping/checkpoint_step1000000.pt \
  --device cuda \
  --compile
```

추론 최적화:

- `--compile`: TorchScript 컴파일로 추론 속도 향상
- 임베딩 캐시로 중복 계산 방지
- 목표 지연 시간: < 10ms per sample

## 지도학습 훈련

- 러닝레이트 2e-4 ~ 5e-4
- 훈련 방식
  - “상승/하락 분류” 같은 분류 문제라면 → CrossEntropyLoss가 무조건 더 좋습니다.
    - direction3-threshold → 0.015~0.02
  - “가격 수치 예측” 같은 회귀 문제라면 → Huber가 더 낫습니다.

##### 분류(Focal, CrossEntropy)방식

```bash
python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251003\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/supervised_sample_test \
  --seq-len 60 \
  --horizon 20 \
  --chunk-size 500 \
  --device cuda \
  --batch-size 128 \
  --epochs 3 \
  --lr 2e-5 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --target-col 현재가 \
  \
  --aux-task direction3 \
  --loss focal \
  --focal-gamma 1.0 \
  --use-weighted-sampler \
  --direction3-threshold 0.015 \
  \
  --progress-every 10000 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised_sample_test/checkpoints \
  --auto-diagnosis warn \
  --diag-warmup-chunks 30 \
  --code 000100 \
  --start-date 20240902 \
  --end-date 20240930

python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251003\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 20 \
  --chunk-size 50000 \
  --device cuda \
  --batch-size 128 \
  --epochs 10 \
  --lr 5e-5 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --target-col 현재가 \
  \
  --aux-task direction3 \
  --loss focal \
  --focal-gamma 1.0 \
  --use-weighted-sampler \
  --direction3-threshold 0.015 \
  \
  --progress-every 10000 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised/checkpoints \
  --auto-diagnosis warn \
  --diag-warmup-chunks 30

python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251003\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 20 \
  --chunk-size 50000 \
  --device cuda \
  --batch-size 128 \
  --epochs 10 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --progress-every 100 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised/checkpoints \
  --target-col 현재가 \
  \
  --aux-task direction3 \
  --loss ce \
  --label-smoothing 0.05 \
  --use-weighted-sampler \
  --direction3-threshold 0.015 \
  \
  --auto-diagnosis abort \
  --diag-warmup-chunks 50 \
  --diag-warmup-epochs 1 \
  --diag-min-val-samples 4096 \
  --diag-require-consecutive 2
```

##### 회기방식

```bash
python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251003" \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 10 \
  --chunk-size 50000 \
  --device cuda \
  --batch-size 128 \
  --epochs 10 \
  --lr 1e-4 \
  --progress-every 100 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised/checkpoints \
  --target-col 등락률 \
  --aux-task regression \
  --loss huber \
  --huber-delta 1.0
```

## 로스 보기

```bash
$ tensorboard --logdir models/supervised_sample_test/tensorboard_logs/

$ python scripts/analysis/check_tensorboard.py \
  --logdir "D:/Workspace/Project/stock-bot/stock-bot2/models/supervised/tensorboard_logs" \
  --watch --interval 10 --idle-seconds 180 --verbose \
  --plot-last 400 \
  --save-png "D:/Workspace/Project/stock-bot/stock-bot2/models/supervised/plots" \
  --out-html ./train_result-2025-10-03.html
```

## 트러블슈팅

### 일반

- GPU OOM 발생 시: batch-size를 단계적으로 낮추세요(256 → 128 → 64).
- CPU/RAM 압박 시: chunk-size를 10~20k 단위로 줄이세요.
- 학습 속도: chunk-size를 키우면 I/O 오버헤드가 줄고 학습이 다소 빨라질 수 있으나, RAM 여유를 꼭 확인하세요.

### 임베딩 모델

- **손실이 감소하지 않는 경우**: temperature 파라미터를 조정하세요 (0.05-0.1 범위).
- **메모리 부족**: batch-size를 줄이거나 embedding-dim을 낮추세요 (128 → 64).
- **훈련 속도 느림**: positive/negative 쌍 생성 비율을 조정하세요.

### GRPO 훈련

- **정책이 수렴하지 않는 경우**:
  - episodes-per-group을 늘리세요 (12 → 16).
  - 학습률을 낮추세요 (3e-4 → 1e-4).
  - clip-epsilon을 조정하세요 (0.2 → 0.1).
- **빠른 손절 룰 위반이 많은 경우**:
  - quick-exit-threshold를 늘리세요 (1.5초 → 3초).
  - quick-exit-penalty를 높여 페널티를 강화하세요.
- **승률이 낮은 경우**:
  - 임베딩 모델을 더 오래 훈련하세요.
  - transaction-cost를 확인하고 보상 구조를 검토하세요.

## 예제 스크립트

`examples/` 디렉토리에서 다양한 사용 예제를 확인할 수 있습니다:

- `embedding_evaluation_example.py`: 임베딩 모델 평가
- `grpo_backtest_example.py`: GRPO 에이전트 백테스팅
- `grpo_evaluation_example.py`: GRPO 에이전트 평가
- `grpo_inference_example.py`: 실시간 추론
- `grpo_predict_example.py`: 단일 예측
- `grpo_live_trading_example.py`: 실시간 거래 (Kiwoom API 연동)
- `generate_html_report_example.py`: HTML 보고서 생성
- `grpo_alert_example.py`: 성능 알림 시스템

자세한 사용법은 각 예제 파일을 참조하세요.
