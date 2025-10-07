# 대용량 데이터셋 훈련 가이드 (월별 3천만 개)

## 📊 데이터 규모

- **전체 데이터**: 4억 개
- **월별 데이터**: 약 3천만 개
- **현재 문제**: 50만 개만 사용 → **과적합 발생**

## 🔍 과적합 원인

### 1. **샘플 수 부족**
```python
MAX_SAMPLES = 500,000  # 3천만의 1.7%만 사용
```
→ 모델이 데이터를 쉽게 암기

### 2. **Temperature 너무 낮음**
```python
TEMPERATURE = 0.07  # 너무 엄격한 학습
```
→ 쉬운 샘플에 과도하게 최적화

### 3. **Regularization 부족**
- Dropout 없음
- Weight Decay 없음
- Early Stopping 없음

### 4. **배치 크기 작음**
```python
BATCH_SIZE = 256  # 데이터 양 대비 너무 작음
```
→ 각 배치가 전체 분포를 대표하지 못함

## 🎯 최적화된 하이퍼파라미터

### 📈 단계별 접근

#### Phase 1: 중간 규모 (메모리 안전)
```python
MAX_SAMPLES = 5000000      # 500만 (3천만의 16.7%)
BATCH_SIZE = 512           # 256 → 512
TEMPERATURE = 0.10         # 0.07 → 0.10
INITIAL_LR = 3e-4          # 1e-4 → 3e-4
DROPOUT = 0.2              # 추가
WEIGHT_DECAY = 1e-4        # 추가
INITIAL_EPOCHS = 20        # 30 → 20
```

**예상 결과**:
- Train Loss: 0.01 ~ 0.02
- Val Loss: 0.015 ~ 0.025
- Gap: 작음 (과적합 감소)

#### Phase 2: 대규모 (메모리 충분 시)
```python
MAX_SAMPLES = 10000000     # 1천만 (3천만의 33%)
BATCH_SIZE = 1024          # V100: 512, A100: 1024
TEMPERATURE = 0.12         # 더 부드러운 학습
INITIAL_LR = 5e-4          # 더 빠른 학습
USE_AMP = True             # Mixed Precision
```

**예상 결과**:
- Train Loss: 0.02 ~ 0.03
- Val Loss: 0.025 ~ 0.035
- 일반화 능력 향상

#### Phase 3: 최대 규모 (Colab Pro+)
```python
MAX_SAMPLES = 20000000     # 2천만 (3천만의 66%)
BATCH_SIZE = 1024
USE_GRADIENT_ACCUMULATION = True
ACCUMULATION_STEPS = 2     # 실질적 배치: 2048
```

## 🛠️ 구현 방법

### 1. 모델 수정 (Dropout 추가)

`ai_trader/embedding/models.py` 수정:

```python
class TradingEmbeddingModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 60,
        embedding_dim: int = 128,
        seq_len: int = 60,
        num_heads: int = 4,
        conv_channels: Optional[list] = None,
        dropout: float = 0.2  # 0.1 → 0.2
    ):
        super().__init__()
        # ... (기존 코드)
```

### 2. 훈련 스크립트 수정

`ai_trader/embedding/train_embedding.py` 수정:

```python
# Optimizer에 weight decay 추가
optimizer = optim.AdamW(  # Adam → AdamW
    model.parameters(), 
    lr=args.lr,
    weight_decay=1e-4  # 추가
)

# Learning Rate Scheduler 추가
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=args.lr * 0.01
)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Training loop에서
with torch.cuda.amp.autocast():
    loss = criterion(anchor_emb, positive_emb, negative_emb)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

### 3. Early Stopping 구현

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# 사용
early_stopping = EarlyStopping(patience=5)
if early_stopping(val_loss):
    print("Early stopping triggered!")
    break
```

## 📊 메모리 관리

### Colab Pro (25GB RAM)

| 샘플 수 | 배치 크기 | 예상 메모리 | 가능 여부 |
|---------|-----------|-------------|-----------|
| 500만 | 256 | ~8GB | ✅ 안전 |
| 500만 | 512 | ~10GB | ✅ 안전 |
| 1000만 | 512 | ~15GB | ✅ 가능 |
| 1000만 | 1024 | ~20GB | ⚠️ 위험 |
| 2000만 | 512 | ~25GB | ⚠️ 위험 |

### Colab Pro+ (52GB RAM)

| 샘플 수 | 배치 크기 | 예상 메모리 | 가능 여부 |
|---------|-----------|-------------|-----------|
| 1000만 | 1024 | ~20GB | ✅ 안전 |
| 2000만 | 1024 | ~35GB | ✅ 안전 |
| 3000만 | 1024 | ~50GB | ⚠️ 위험 |

## 🚀 실행 방법

### 방법 1: 노트북에서 직접 수정

```python
# 노트북의 설정 셀에서
MAX_SAMPLES = 5000000  # 500만
BATCH_SIZE = 512
TEMPERATURE = 0.10
INITIAL_LR = 3e-4
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
INITIAL_EPOCHS = 20
VAL_EVERY = 1  # 매 에포크마다 검증
```

### 방법 2: 설정 파일 사용

```python
# 노트북에서
import sys
sys.path.insert(0, '/content/stock-bot2')

from scripts.colab.config_large_dataset import *

# 설정 확인
print_config()
```

## 📈 예상 훈련 시간

### V100 GPU 기준

| 샘플 수 | 배치 크기 | 에포크당 시간 | 20 에포크 총 시간 |
|---------|-----------|---------------|-------------------|
| 500만 | 512 | ~30분 | ~10시간 |
| 1000만 | 512 | ~60분 | ~20시간 |
| 1000만 | 1024 | ~45분 | ~15시간 |

### A100 GPU 기준 (약 40% 빠름)

| 샘플 수 | 배치 크기 | 에포크당 시간 | 20 에포크 총 시간 |
|---------|-----------|---------------|-------------------|
| 500만 | 512 | ~20분 | ~7시간 |
| 1000만 | 1024 | ~30분 | ~10시간 |
| 2000만 | 1024 | ~60분 | ~20시간 |

## 🎯 권장 전략

### 단계 1: 빠른 검증 (1-2시간)
```python
MAX_SAMPLES = 1000000  # 100만
INITIAL_EPOCHS = 5
```
→ 설정이 제대로 작동하는지 확인

### 단계 2: 중간 규모 (10시간)
```python
MAX_SAMPLES = 5000000  # 500만
INITIAL_EPOCHS = 20
```
→ 과적합 없이 학습되는지 확인

### 단계 3: 최종 훈련 (20시간)
```python
MAX_SAMPLES = 10000000  # 1000만
INITIAL_EPOCHS = 20
```
→ 최고 성능 모델 생성

## 🔍 모니터링 체크리스트

### 매 에포크마다 확인

- [ ] Train Loss 감소 중
- [ ] Val Loss 감소 중
- [ ] **Val Loss - Train Loss < 0.01** (과적합 없음)
- [ ] Silhouette Score > 0.3
- [ ] Temporal Coherence > 0.6

### 경고 신호

⚠️ **Val Loss > Train Loss * 1.5** → 과적합 시작
⚠️ **Val Loss 증가** → Early Stopping 고려
⚠️ **Train Loss < 0.001** → Temperature 높이기

## 💡 추가 최적화

### 1. Data Augmentation
```python
# data.py에 추가
def add_noise(sequence, noise_level=0.01):
    noise = np.random.randn(*sequence.shape) * noise_level
    return sequence + noise
```

### 2. Hard Negative Mining
```python
# 어려운 negative 샘플 우선 선택
def find_hard_negative(anchor_emb, all_embeddings):
    # 앵커와 유사하지만 다른 종목인 샘플
    similarities = cosine_similarity(anchor_emb, all_embeddings)
    # 중간 유사도 샘플 선택 (너무 쉽지도 어렵지도 않게)
    hard_indices = np.argsort(similarities)[len(similarities)//3:2*len(similarities)//3]
    return np.random.choice(hard_indices)
```

### 3. Curriculum Learning
```python
# 쉬운 샘플부터 어려운 샘플로
epoch_1_5: easy samples
epoch_6_10: medium samples  
epoch_11_20: all samples
```

## 🆘 문제 해결

### Q: 여전히 과적합 발생
**A**: Temperature를 0.15로 높이고, Dropout을 0.3으로 증가

### Q: 메모리 부족
**A**: Gradient Accumulation 사용 또는 샘플 수 감소

### Q: 훈련 너무 느림
**A**: Mixed Precision (AMP) 활성화, NUM_WORKERS 증가

### Q: Val Loss가 Train Loss보다 낮음
**A**: 정상 (Dropout 때문), 계속 진행

---

**다음 단계**: `config_large_dataset.py` 설정으로 재훈련 시작!
