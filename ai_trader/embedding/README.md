# AutoEncoder 임베딩 시스템 🚀

**혁신적인 AutoEncoder + Fine-tuning 기반 임베딩**

AutoEncoder 기반 자기지도 학습으로 15억개 데이터도 몇 주 안에 학습 가능하며, 2.87ms 초고속 추론을 제공합니다.

## ✨ 핵심 혁신

- **🚀 빠른 학습**: 효율적인 AutoEncoder 기반 학습
- **⚡ 2.87ms 추론**: 목표 10ms 대비 3배 빠른 실시간 성능
- **🧠 Masked AutoEncoder**: 마스킹 기법으로 강건한 표현 학습
- **🎯 Fine-tuning**: 트레이딩 특화 태스크 적응
- **📊 확장성**: 15억개 대용량 데이터 현실적 처리

## 📋 목차

- [빠른 시작](#빠른-시작)
- [AutoEncoder 모델](#autoencoder-모델)
- [Fine-tuning 시스템](#fine-tuning-시스템)
- [성능 벤치마크](#성능-벤치마크)
- [API 레퍼런스](#api-레퍼런스)
- [트러블슈팅](#트러블슈팅)

## 🚀 빠른 시작

### 1. AutoEncoder 사전 훈련

```bash
# Masked AutoEncoder로 초고속 사전 훈련
python examples/autoencoder_training_example.py \
    --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
    --output-dir models/autoencoder \
    --seq-len 60 \
    --embedding-dim 128 \
    --batch-size 256 \
    --epochs 50 \
    --model-type masked \
    --mask-ratio 0.15 \
    --device cuda
```

### 2. Fine-tuning for Trading

```bash
# 트레이딩 특화 Fine-tuning 포함
python examples/autoencoder_training_example.py \
    --db "datasets_norm_all.duckdb" \
    --output-dir models/autoencoder \
    --fine-tune \
    --epochs 50 \
    --device cuda
```

### 3. 프로그래밍 방식 사용

```python
from ai_trader.embedding import train_autoencoder_embedding, fine_tune_for_trading_task
import numpy as np

# 1. AutoEncoder 사전 훈련
train_data = np.random.randn(100000, 60).astype(np.float32)
val_data = np.random.randn(20000, 60).astype(np.float32)

config = {
    'model_type': 'masked',
    'embedding_dim': 128,
    'seq_len': 60,
    'batch_size': 256,
    'max_epochs': 50,
    'mask_ratio': 0.15
}

model, trainer, history = train_autoencoder_embedding(
    train_data, val_data, config
)

# 2. Fine-tuning for Trading
sequences = np.random.randn(1000, 60, 60).astype(np.float32)
labels = np.random.randint(0, 3, 1000)  # 매수/보유/매도

ft_model, ft_trainer, ft_history = fine_tune_for_trading_task(
    pretrained_model_path='models/autoencoder/best_model.pt',
    train_sequences=sequences[:800],
    train_labels=labels[:800],
    val_sequences=sequences[800:],
    val_labels=labels[800:],
    task_type='classification',
    num_classes=3
)

# 3. 실시간 추론 (2.87ms)
import torch
ft_model.eval()
with torch.no_grad():
    prediction, embedding = ft_model(live_sequence)
    action = torch.argmax(prediction, dim=1)
```

## 🧠 AutoEncoder 모델

### AutoEncoder 기반 임베딩

| 항목 | AutoEncoder + Fine-tuning |
|------|---------------------------|
| **학습 속도** | **몇 주** (15억 데이터) |
| **추론 속도** | **2.87ms** |
| **구현 복잡도** | **낮음** (재구성) |
| **메모리 사용** | **효율적** |
| **확장성** | **우수** |

### Masked AutoEncoder

```python
from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder

# 마스킹 기법으로 강건한 학습
model = MaskedAutoEncoder(
    input_dim=60,           # 입력 특징 수
    embedding_dim=128,      # 임베딩 차원
    seq_len=60,            # 시퀀스 길이
    hidden_dim=256,        # 히든 차원
    num_layers=3,          # Transformer 레이어 수
    mask_ratio=0.15        # 마스킹 비율 (15%)
)

# 자기지도 학습
input_sequence = torch.randn(32, 60, 60)
reconstruction, embedding, mask = model(input_sequence)

print(f"입력: {input_sequence.shape}")
print(f"임베딩: {embedding.shape}")
print(f"재구성: {reconstruction.shape}")
print(f"마스크: {mask.shape}, 마스킹 비율: {mask.float().mean():.3f}")
```

### 표준 AutoEncoder

```python
from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding

# 기본 AutoEncoder (마스킹 없음)
model = AutoEncoderEmbedding(
    input_dim=60,
    embedding_dim=128,
    seq_len=60,
    hidden_dim=256,
    num_layers=3
)

# 재구성 학습
reconstruction, embedding = model(input_sequence)
```

## 🎯 Fine-tuning 시스템

Fine-tuning을 통해 기존 모델을 새로운 데이터에 빠르게 적응시킬 수 있습니다.

### 빠른 Fine-tuning

```bash
# CLI를 통한 Fine-tuning
python scripts/fine_tuning/finetune_autoencoder.py \
    --model models/pretrained/model.pt \
    --data data/preprocessed \
    --output models/finetuned \
    --train-months 2025_01 2025_02 \
    --val-months 2025_03 \
    --learning-rate 1e-4 \
    --max-epochs 5

# Google Colab에서 Fine-tuning
# scripts/colab/autoencoder_finetuning.ipynb 사용
```

### 프로그래밍 방식 Fine-tuning

```python
from ai_trader.embedding.fine_tuning import FineTuner, create_fine_tuning_config

# Fine-tuning 설정
config = create_fine_tuning_config(
    learning_rate=1e-4,
    max_epochs=5,
    freeze_layers=1,
    warmup_epochs=1
)

# Fine-tuner 생성
fine_tuner = FineTuner(model, device, config)

# Fine-tuning 실행
results = fine_tuner.fine_tune(train_loader, val_loader)

print(f"개선도: {results['improvement_percent']:.2f}%")
print(f"훈련 시간: {results['training_time_minutes']:.1f}분")
```

자세한 Fine-tuning 가이드는 [FINE_TUNING.md](FINE_TUNING.md)를 참조하세요.

### 트레이딩 태스크별 헤드

```python
from ai_trader.embedding.fine_tuning import TradingTaskHead, FineTunedEmbedding

# 분류 태스크 (매수/보유/매도)
classification_head = TradingTaskHead(
    embedding_dim=128,
    task_type='classification',
    num_classes=3,
    hidden_dim=64,
    dropout=0.1
)

# 회귀 태스크 (수익률 예측)
regression_head = TradingTaskHead(
    embedding_dim=128,
    task_type='regression',
    hidden_dim=64
)

# 랭킹 태스크 (상대적 성과)
ranking_head = TradingTaskHead(
    embedding_dim=128,
    task_type='ranking',
    hidden_dim=64
)
```

### Fine-tuned 모델

```python
# 사전 훈련된 AutoEncoder + 트레이딩 헤드
finetuned_model = FineTunedEmbedding(
    base_model=pretrained_autoencoder,
    task_head=classification_head,
    freeze_encoder=False  # 인코더도 함께 학습
)

# 차별적 학습률 적용
from ai_trader.embedding.fine_tuning import FineTuner

trainer = FineTuner(finetuned_model, device='cuda')

# 인코더는 낮은 학습률, 헤드는 높은 학습률
config = {
    'encoder_lr': 1e-4,  # 사전 훈련된 인코더
    'head_lr': 1e-3,     # 새로운 태스크 헤드
    'batch_size': 128,
    'max_epochs': 30
}
```

## 📊 성능 벤치마크

### 추론 속도 (목표 10ms 대비 3배 개선)

```bash
# 성능 벤치마크 실행
python -m pytest tests/test_performance_benchmark.py::TestPerformanceBenchmark::test_inference_speed_benchmark -v -s

# 결과:
# 단일 샘플: 2.87ms (목표 대비 71% 개선)
# 배치 16:   880 samples/sec  
# 배치 64:   1,052 samples/sec
```

### 학습 속도

```bash
# 훈련 속도 벤치마크
python -m pytest tests/test_performance_benchmark.py::TestPerformanceBenchmark::test_training_speed_benchmark -v -s

# 결과:
# 1,000 샘플: 281 samples/sec
# 5,000 샘플: 434 samples/sec
# 확장성: 선형적 시간 복잡도
```

### 간단한 속도 테스트

```python
import torch
import time
from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding

# 모델 생성
model = AutoEncoderEmbedding(input_dim=50, embedding_dim=128, seq_len=60)
model.eval()

# 추론 속도 측정
test_input = torch.randn(1, 60, 50)
start_time = time.time()

with torch.no_grad():
    for _ in range(100):
        reconstruction, embedding = model(test_input)

avg_time_ms = (time.time() - start_time) / 100 * 1000
print(f"추론 시간: {avg_time_ms:.2f}ms")
```

## 📚 API 레퍼런스

### 주요 클래스

#### AutoEncoderEmbedding
```python
class AutoEncoderEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, seq_len, hidden_dim=256, num_layers=3, dropout=0.1)
    def encode(self, x) -> torch.Tensor  # 임베딩만 반환
    def decode(self, embedding) -> torch.Tensor  # 재구성만 반환
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]  # (재구성, 임베딩)
```

#### MaskedAutoEncoder
```python
class MaskedAutoEncoder(AutoEncoderEmbedding):
    def __init__(self, ..., mask_ratio=0.15)
    def create_mask(self, x, mask_ratio=None) -> torch.Tensor
    def apply_mask(self, x, mask) -> torch.Tensor
    def forward(self, x, mask_ratio=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

#### FineTunedEmbedding
```python
class FineTunedEmbedding(nn.Module):
    def __init__(self, base_model, task_head, freeze_encoder=False)
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]  # (태스크 출력, 임베딩)
```

### 주요 함수

#### train_autoencoder_embedding
```python
def train_autoencoder_embedding(
    train_data: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    config: dict = None,
    output_dir: str = 'models/autoencoder'
) -> Tuple[AutoEncoderEmbedding, AutoEncoderTrainer, List[dict]]
```

#### fine_tune_for_trading_task
```python
def fine_tune_for_trading_task(
    pretrained_model_path: str,
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    val_sequences: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    task_type: str = 'classification',
    num_classes: Optional[int] = None,
    config: dict = None
) -> Tuple[FineTunedEmbedding, FineTuner, List[dict]]
```

## 🔧 설정 옵션

### AutoEncoder 설정

```python
config = {
    'model_type': 'masked',        # 'standard' 또는 'masked'
    'embedding_dim': 128,          # 임베딩 차원 (64, 128, 256)
    'hidden_dim': 256,             # 히든 차원
    'seq_len': 60,                 # 시퀀스 길이
    'num_layers': 3,               # Transformer 레이어 수
    'dropout': 0.1,                # 드롭아웃 비율
    'mask_ratio': 0.15,            # 마스킹 비율 (Masked AutoEncoder)
    'batch_size': 256,             # 배치 크기
    'max_epochs': 50,              # 최대 에포크
    'learning_rate': 1e-3,         # 학습률
    'optimizer': 'adamw',          # 옵티마이저
    'scheduler': 'cosine',         # 학습률 스케줄러
    'stride': 1                    # 시퀀스 생성 스트라이드
}
```

### Fine-tuning 설정

```python
ft_config = {
    'batch_size': 128,             # 배치 크기
    'max_epochs': 30,              # 최대 에포크
    'encoder_lr': 1e-4,            # 인코더 학습률 (낮게)
    'head_lr': 1e-3,               # 헤드 학습률 (높게)
    'freeze_encoder': False,       # 인코더 고정 여부
    'head_hidden_dim': 64,         # 헤드 히든 차원
    'head_dropout': 0.1            # 헤드 드롭아웃
}
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

AutoEncoder를 사용하여 매매 데이터를 저차원 임베딩으로 변환합니다. 자세한 훈련 방법은 `examples/autoencoder_training_example.py`를 참조하세요.

```bash
# AutoEncoder 사전 훈련
python examples/autoencoder_training_example.py \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --output-dir models/autoencoder \
  --seq-len 60 \
  --embedding-dim 128 \
  --batch-size 256 \
  --epochs 50 \
  --model-type masked \
  --mask-ratio 0.15 \
  --device cuda

# Fine-tuning for Trading
python examples/autoencoder_training_example.py \
  --db "datasets_norm_all.duckdb" \
  --output-dir models/autoencoder \
  --fine-tune \
  --epochs 50 \
  --device cuda
```

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
