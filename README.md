# Stock Bot 🚀

**AI-Powered Korean Stock Market Scalping System**

한국 주식 시장에서 초단위 스캘핑을 위한 AI 트레이딩 시스템입니다. AutoEncoder 기반 임베딩과 GRPO 강화학습을 결합하여 실시간 매매 결정을 내립니다.

## ✨ 주요 특징

- **🚀 초고속 추론**: 2.87ms 실시간 매매 결정 (목표 10ms 대비 3배 빠름)
- **⚡ 혁신적 학습 속도**: 10-100배 빠른 AutoEncoder 기반 자기지도 학습
- **🎯 스캘핑 특화**: 30초 이내 초단위 거래에 최적화
- **🧠 GRPO 강화학습**: 그룹 상대 정책 최적화로 안정적 학습
- **📊 대용량 데이터**: 15억개 데이터도 몇 주 안에 학습 가능

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    DuckDB 데이터베이스                        │
│         (datasets_norm_all.duckdb / table: datasets)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────────┬─────────────────────┐
                     │                  │                     │
                     ▼                  ▼                     ▼
          ┌──────────────────┐  ┌──────────────┐   ┌─────────────────┐
          │  AutoEncoder     │  │  GRPO 훈련   │   │  실시간 추론     │
          │  사전 훈련        │  │  (RL Agent)  │   │  (Live Trading) │
          └──────────┬───────┘  └──────┬───────┘   └────────┬────────┘
                     │                  │                     │
                     ▼                  │                     │
          ┌──────────────────┐         │                     │
          │  Fine-tuning     │◄────────┴─────────────────────┘
          │  (트레이딩 특화)   │
          └──────────────────┘
```

## 📋 목차

- [빠른 시작](#빠른-시작)
- [AutoEncoder 임베딩](#autoencoder-임베딩)
- [GRPO 강화학습](#grpo-강화학습)
- [성능 벤치마크](#성능-벤치마크)
- [데이터 정규화](#데이터-정규화)
- [설치 및 설정](#설치-및-설정)

## 🚀 빠른 시작

## 0단계: 데이터 정규화

```bash
python scripts/data/generate_datasets.py \
  "D:\Workspace\Project\stock-bot\hoga-crawler\data" \
  -o "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --checkpoint-interval 50 \
  --single-output \
  --start-date 20250922 --end-date 20250930

python scripts/data/merge_datasets.py "C:\Users\user\Workspace\datasets@raw" \
  --out "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --threads 4 \
  --memory-limit 64GB

python scripts/data/normalize_datasets.py \
  "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  -o "C:\Users\user\Workspace\datasets\datasets_norm.duckdb" \
  --tmp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --workers 12 \
  --time-start 90000000 \
  --time-end 110000000 \
  --checkpoint-interval 50 \
  --ignoring-stocks-csv scripts/data/ignoring_stocks.csv \
  --qualifying-minutes 1

python scripts/data/merge_datasets.py "C:\Users\user\Workspace\datasets" \
  --out "C:\Users\user\Workspace\datasets\datasets_norm_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --threads 4 \
  --memory-limit 64GB

# OR

python scripts/data/export_datasets.py \
  --input-db "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  --input-table datasets \
  --output-db "C:\Users\user\Workspace\datasets\datasets_raw.duckdb" \
  --output-table datasets \
  --tmp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --workers 12 \
  --time-start 90000000 \
  --time-end 110000000 \
  --checkpoint-interval 50 \
  --qualifying-minutes 1

python scripts/data/merge_datasets.py "C:\Users\user\Workspace\datasets" \
  --out "C:\Users\user\Workspace\datasets\datasets_raw_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --table datasets \
  --threads 4 \
  --memory-limit 64GB
```

### 1단계: AutoEncoder 훈련

```bash
# 사전 훈련 데이터 생성
$ python scripts/pre/parallel_preprocessing.py \
    --db "C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb" \
    --seq-len 120 \
    --batch-size 5000 \
    --start-year 2024 --start-month 9 \
    --end-year 2025 --end-month 9 \
    --output "C:\Users\user\Workspace\datasets@20260117\pre_training_data" \
    --max-workers 4

# autoencoder_training_complete.ipynb 실행
```

### 2단계: 트레이딩 특화 Fine-tuning

```python
from ai_trader.embedding.fine_tuning import fine_tune_for_trading_task

# 사전 훈련된 모델을 트레이딩 태스크에 맞게 Fine-tuning
ft_model, trainer, history = fine_tune_for_trading_task(
    pretrained_model_path='models/autoencoder/best_model.pt',
    train_sequences=sequences,
    train_labels=labels,
    task_type='classification',
    num_classes=3,  # 매수/보유/매도
    config={
        'batch_size': 128,
        'max_epochs': 30,
        'encoder_lr': 1e-4,
        'head_lr': 1e-3
    }
)
```

### 3단계: GRPO 강화학습 훈련

```bash
# 속도를 위한 전처리 작업.
python scripts/data/generate_embeddings_v3_parallel.py \
  --db_path "C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb" \
  --model_path "C:\Users\user\Workspace\datasets@20260117\autoencoder_seq120\best_model.pt" \
  --output_dir "C:\Users\user\Workspace\datasets@20260117\embeddings_v2" \
  --state_file "C:\Users\user\Workspace\datasets@20260117\embeddings_v2\processed_stocks.txt" \
  --table_name "datasets" \
  --seq_len 120 \
  --batch_size 4096 \
  --num_workers 4 \
  --device cpu
```

```bash
# GRPO로 스캘핑 전략 학습 (RollingNormalizer 사용 - 권장, entropy_coef: 0.05(모험적))
python ai_trader/grpo/train_scalping.py \
    --db "C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb" \
    --embedding_model "C:\Users\user\Workspace\datasets@20260117\autoencoder_seq120\best_model.pt" \
    --seq_len 120 \
    --features 28 \
    --total_timesteps 200000 \
    --use_raw_data true \
    --output_dir "models/grpo_scalping_v7_holding_rule" \
    --load_policy "models/grpo_scalping_v7_holding_rule_soft/checkpoints/checkpoint_iter500.pt" \
    --num_workers 2 \
    --min_holding 2 \
    --max_holding 300 \
    --entropy_coef 0.05 \
    --transaction_cost 0.0 \
    --lr 0.0001

# Fine-tuning (기존 모델 로드)
python ai_trader/grpo/train_scalping.py \
    --db "C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb" \
    --embedding_model "C:\Users\user\Workspace\datasets@20260117\autoencoder_seq120\best_model.pt" \
    --seq_len 120 \
    --features 28 \
    --total_timesteps 200000 \
    --use_raw_data true \
    --output_dir "models/grpo_scalping_v7_holding_rule_soft" \
    --load_policy "models/grpo_scalping_v7_holding_rule_soft/checkpoints/checkpoint_iter500.pt" \
    --num_workers 2 \
    --min_holding 2 \
    --max_holding 300 \
    --entropy_coef 0.01 \
    --transaction_cost 0.00215 \
    --lr 0.0001
```

### 4단계: 실시간 추론 (2.87ms)

```python
from ai_trader.grpo.inference.grpo_infer import GRPOInference

# 초고속 실시간 매매 결정
inference = GRPOInference(
    policy_path='models/grpo_scalping/policy_final.pt',
    embedding_model_path='models/autoencoder/best_model.pt',
    device='cuda'
)

# 실시간 데이터로 매매 결정
action, confidence = inference.predict(live_market_data)  # action: 0=보유, 1=매수, 2=매도
```

## 🧠 AutoEncoder 임베딩

### 핵심 혁신: AutoEncoder 기반 임베딩

AutoEncoder + Fine-tuning 방식으로 효율적인 자기지도 학습을 제공합니다.

| 방식            | 학습 속도 | 추론 속도  | 구현 복잡도 | 확장성   | 15억 데이터 학습 |
| --------------- | --------- | ---------- | ----------- | -------- | ---------------- |
| **AutoEncoder** | **몇 주** | **2.87ms** | **낮음**    | **우수** | **가능**         |

### 주요 장점

- ✅ **10-100배 빠른 학습**: 복잡한 positive/negative 쌍 생성 불필요
- ✅ **3배 빠른 추론**: 2.87ms vs 목표 10ms
- ✅ **간단한 구현**: 재구성 손실만으로 학습
- ✅ **확장성**: 대용량 데이터에서도 선형적 시간 복잡도

### Masked AutoEncoder

```python
from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder

# 마스킹 기법으로 강건한 표현 학습
model = MaskedAutoEncoder(
    input_dim=60,
    embedding_dim=128,
    seq_len=120,
    mask_ratio=0.15  # 15% 마스킹
)

# 자기지도 학습으로 빠른 사전 훈련
reconstruction, embedding, mask = model(input_sequence)
```

### Fine-tuning for Trading

```python
from ai_trader.embedding.fine_tuning import FineTunedEmbedding, TradingTaskHead

# 트레이딩 특화 헤드 추가
task_head = TradingTaskHead(
    embedding_dim=128,
    task_type='classification',  # 또는 'regression', 'ranking'
    num_classes=3
)

# 사전 훈련된 인코더 + 트레이딩 헤드
finetuned_model = FineTunedEmbedding(
    base_model=pretrained_autoencoder,
    task_head=task_head,
    freeze_encoder=False  # 인코더도 함께 학습
)
```

## 🎮 GRPO 강화학습

### 그룹 상대 정책 최적화

#### 훈련-추론 일관성 (중요!)

훈련 환경과 실시간 거래가 **동일한 정규화 전략**을 사용합니다:
- **RollingNormalizer**: 최근 1000개 샘플의 mean/std로 적응형 정규화
- **분포 이동 문제 해결**: 훈련과 추론의 데이터 분포 일치
- **실시간 성능 향상**: 일관된 전처리 파이프라인

```bash
# 원본 데이터 + RollingNormalizer 사용 (권장)
python ai_trader/grpo/train_scalping.py \
    --db "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
    --embedding_model models/autoencoder/best_model.pt \
    --total_timesteps 100000 \
    --use_raw_data true \
    --rolling_window_size 1000 \
    --output_dir models/grpo_scalping

# JSON 설정 파일 사용
python ai_trader/grpo/train_scalping.py --config config/training_config.json
```

### 실시간 추론 파이프라인

```python
from ai_trader.grpo.inference.grpo_infer import GRPOInference

# 초고속 실시간 매매 결정 (2.87ms)
inference = GRPOInference(
    policy_path='models/grpo_scalping/policy_final.pt',
    embedding_model_path='models/autoencoder/best_model.pt',
    device='cuda',
    use_torchscript=True  # TorchScript 컴파일로 속도 향상
)

# 실시간 데이터로 매매 결정
action, confidence = inference.predict(live_market_data)
# action: 0=보유, 1=매수, 2=매도
# confidence: 예측 신뢰도 (0.0 ~ 1.0)
```

### 스캘핑 특화 보상 구조

- **거래 비용**: 0.215% (수수료 + 세금)
- **빠른 손절 룰**: 1.5초 내 미상승 시 자동 매도
- **장기 보유 페널티**: 60초 초과 시 페널티
- **그룹 상대 어드밴티지**: 시장 상황별 상대 성능 비교

## 📊 성능 벤치마크

### 🎯 성능 목표 vs 실제 결과

| 메트릭            | 목표   | 실제 결과  | 달성도          |
| ----------------- | ------ | ---------- | --------------- |
| **추론 속도**     | < 10ms | **2.87ms** | ✅ **3배 개선** |
| **승률**          | > 50%  | TBD        | 🔄 백테스팅 중  |
| **샤프 비율**     | > 1.0  | TBD        | 🔄 백테스팅 중  |
| **최대 낙폭**     | < 10%  | TBD        | 🔄 백테스팅 중  |
| **평균 보유시간** | < 30초 | TBD        | 🔄 백테스팅 중  |

### 추론 속도 (목표 10ms 대비 3배 개선)

```
단일 샘플: 2.87ms (목표 대비 71% 개선)
배치 16:   880 samples/sec
배치 64:   1,052 samples/sec
메모리:    GPU 사용량 < 2GB
```

### 학습 속도 (혁신적 성능)

```
1,000 샘플:  281 samples/sec
5,000 샘플:  434 samples/sec
확장성:      선형적 시간 복잡도
15억 데이터: 몇 주 내 학습 가능 (기존: 몇 년)
```

### 벤치마크 실행

```bash
# 성능 벤치마크 테스트
python -m pytest tests/test_performance_benchmark.py -v -s

# AutoEncoder 성능 벤치마크
python scripts/benchmark_autoencoder.py \
    --model-path models/autoencoder/best_model.pt \
    --batch-sizes 1,16,64 \
    --num-samples 1000 \
    --device cuda

# 간단한 추론 속도 테스트
python -c "
import torch, time
from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding

model = AutoEncoderEmbedding(input_dim=50, embedding_dim=128, seq_len=120)
model.eval()

test_input = torch.randn(1, 60, 50)
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        reconstruction, embedding = model(test_input)

avg_time_ms = (time.time() - start_time) / 100 * 1000
print(f'추론 시간: {avg_time_ms:.2f}ms')
"
```

## 🧪 테스트

### 전체 테스트 실행

```bash
# 모든 단위 테스트 (38개)
python -m pytest tests/test_embedding_model.py tests/test_embedding_losses.py tests/test_autoencoder_data.py -v

# 성능 벤치마크
python -m pytest tests/test_performance_benchmark.py -v -s

# 통합 테스트
python -m pytest tests/test_autoencoder_integration.py -v
```

### 테스트 결과

```
✅ 단위 테스트: 38개 모두 통과
✅ 성능 테스트: 목표 대비 3배 빠른 추론 속도
✅ 통합 테스트: End-to-End 파이프라인 검증
```

## 🏗️ 프로젝트 구조

```
ai_trader/
├── embedding/              # AutoEncoder 임베딩 모델
│   ├── autoencoder_model.py    # AutoEncoder & MaskedAutoEncoder
│   ├── autoencoder_trainer.py  # 훈련 파이프라인
│   ├── fine_tuning.py          # Fine-tuning 시스템
│   └── data.py                 # 데이터 로더
├── grpo/                   # GRPO 강화학습
│   ├── environments/
│   │   └── scalping_env.py     # 스캘핑 환경 (RollingNormalizer 통합)
│   ├── policies/
│   │   └── scalping_policy.py  # GRPO 정책 네트워크
│   ├── grpo.py                 # GRPO 알고리즘
│   ├── train_scalping.py       # 훈련 스크립트 (Enhanced)
│   └── inference/
│       ├── grpo_infer.py       # 기본 추론 엔진
│       └── enhanced_grpo_infer.py  # 향상된 추론 엔진
└── reporting/              # 보고서 생성
    └── html_report.py          # HTML 리포트

lib/                        # 공통 라이브러리
├── rolling_normalization.py    # RollingNormalizer
└── normalization.py            # 정규화 전략

scripts/                    # 데이터 처리
├── data/
│   ├── normalize_datasets.py   # 데이터 정규화
│   └── merge_datasets.py       # 데이터 병합
└── live/
    └── live_trading.py         # 실시간 거래 시스템

tests/                      # 테스트 스위트
├── test_autoencoder_*.py
├── test_performance_benchmark.py
└── test_autoencoder_integration.py
```

## 📁 데이터 스키마

```bash
# 60개 이상의 특징을 가진 시계열 데이터
날짜 번호 종목코드 종목명 시간 등락률 누적거래대금 거래회전율 체결강도
매도대기금액1~10 매수대기금액1~10 종목명_scalar 시간_sin 시간_cos 시간_scalar
```

## ⚙️ 설치 및 설정

### 시스템 요구사항

- Python 3.8+
- PyTorch 2.2+
- CUDA 지원 GPU (권장)
- 16GB+ RAM (32GB 권장)

### 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# GPU 지원 (CUDA 사용 시)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 개발 환경 설정
pip install -e .
```

### 환경 변수

```bash
# .env 파일 생성
DB_PATH="C:\Users\user\Workspace\datasets@20260109\datasets_norm_all.duckdb"
MODEL_DIR="models"
DEVICE="cuda"  # 또는 "cpu"
```
