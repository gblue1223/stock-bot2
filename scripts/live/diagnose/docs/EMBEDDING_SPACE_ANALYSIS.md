# 임베딩 공간 분석: 왜 랜덤은 성공하고 실제 데이터는 실패하는가?

## 🎯 핵심 질문

**왜 랜덤 임베딩은 다양한 예측을 하는데, 실제 데이터는 HOLD만 예측하는가?**

## 📊 테스트 결과 비교

### 테스트 1: 랜덤 임베딩 (성공)

```python
# 128차원 랜덤 벡터 생성
test_embeddings = torch.randn(100, 128)

# 예측 결과
HOLD:  56 (56.0%)
BUY:    6 (6.0%)
SELL:  38 (38.0%)
```

**특징:**
- 표준 정규분포 N(0, 1)에서 샘플링
- 128차원 공간 전체에 고르게 분포
- 각 차원이 독립적으로 랜덤

### 테스트 2: 합성 데이터 → 임베딩 (실패)

```python
# 실제 시장 데이터 모방 → AutoEncoder → 임베딩
synthetic_data → normalize → AutoEncoder.encode() → embedding

# 예측 결과
HOLD: 100 (100.0%)
BUY:    0 (0.0%)
SELL:   0 (0.0%)
```

**특징:**
- 실제 시장 데이터 패턴 반영
- AutoEncoder가 학습한 매니폴드에 제약됨
- 특정 영역에만 임베딩 생성

## 🔬 근본 원인 분석

### 원인 1: 임베딩 공간의 제약

**랜덤 임베딩:**
```
128차원 공간 전체를 자유롭게 탐색
┌─────────────────────────────┐
│  ●  ●    ●   ●    ●   ●    │  ← 고르게 분포
│    ●   ●   ●    ●   ●   ●  │
│  ●   ●    ●   ●    ●    ●  │
│    ●   ●    ●   ●   ●   ●  │
└─────────────────────────────┘
```

**실제 임베딩:**
```
학습된 매니폴드에만 존재
┌─────────────────────────────┐
│                             │
│         ●●●●●●●             │  ← 특정 영역에 집중
│         ●●●●●●●             │
│                             │
└─────────────────────────────┘
```

### 원인 2: AutoEncoder의 학습 편향

**AutoEncoder가 학습한 것:**
```python
# 학습 데이터의 대부분이 HOLD 상태
HOLD 샘플: 90%+  → 임베딩 공간의 대부분 차지
BUY 샘플:  <5%   → 임베딩 공간의 작은 영역
SELL 샘플: <5%   → 임베딩 공간의 작은 영역
```

**결과:**
- AutoEncoder는 HOLD 패턴을 재구성하는데 최적화됨
- BUY/SELL 패턴은 제대로 학습되지 않음
- 모든 입력이 "HOLD 같은" 임베딩으로 변환됨

### 원인 3: 정책 네트워크의 결정 경계

**정책 네트워크가 학습한 결정 경계:**

```
임베딩 공간을 3개 영역으로 분할:

┌─────────────────────────────┐
│ BUY 영역 (작음)             │
│   ┌─┐                       │
│   └─┘                       │
│                             │
│     ┌───────────────┐       │
│     │ HOLD 영역     │       │  ← 대부분의 공간
│     │  (매우 큼)    │       │
│     └───────────────┘       │
│                             │
│                   ┌─┐       │
│                   └─┘       │
│                SELL 영역    │
└─────────────────────────────┘

실제 임베딩은 모두 HOLD 영역에 위치
```

## 🧪 실험적 증거

### 실험 1: 임베딩 통계 비교

**랜덤 임베딩:**
```python
mean = [0.0, 0.0, ..., 0.0]  # 128개 모두 0 근처
std  = [1.0, 1.0, ..., 1.0]  # 128개 모두 1 근처
range = [-3.0, +3.0]          # 넓은 범위
```

**실제 임베딩 (추정):**
```python
mean = [0.5, -0.2, ..., 0.1]  # 특정 값으로 편향
std  = [0.1, 0.05, ..., 0.08] # 매우 작은 분산
range = [0.3, 0.7]            # 좁은 범위
```

### 실험 2: 정규화의 영향

**합성 데이터 생성:**
```python
# 상승 추세
prices = [10000, 10100, 10200, ..., 11000]
등락률 = [+1%, +0.99%, +0.98%, ...]
```

**정규화 후:**
```python
# 학습 통계로 정규화
normalized = (raw - training_mean) / training_std

# 결과: 모든 시나리오가 유사한 값으로 변환
상승 추세 → [0.5, 0.6, 0.5, ...]
하락 추세 → [0.5, 0.6, 0.5, ...]
횡보      → [0.5, 0.6, 0.5, ...]
```

**문제점:**
- 학습 통계가 실제 데이터 분포를 반영하지 못함
- 정규화 후 모든 데이터가 비슷해짐
- AutoEncoder가 구별할 수 없음

## 💡 왜 이런 일이 발생했는가?

### 1. 학습 데이터의 클래스 불균형

```python
# 학습 데이터 분포 (추정)
HOLD: 1,000,000 샘플 (95%)
BUY:     25,000 샘플 (2.5%)
SELL:    25,000 샘플 (2.5%)
```

**영향:**
- AutoEncoder는 HOLD 패턴 재구성에 집중
- 정책 네트워크는 HOLD 예측에 집중
- BUY/SELL 패턴은 "노이즈"로 취급

### 2. 보상 함수의 보수성

```python
# GRPO 학습 시 보상
HOLD → 작은 손실/이익 → 안정적
BUY  → 큰 손실/이익 → 위험
SELL → 큰 손실/이익 → 위험
```

**결과:**
- 모델이 "안전한" HOLD를 선호하도록 학습
- 위험을 회피하는 정책 학습

### 3. 임베딩 공간의 붕괴 (Collapse)

```python
# 이상적인 임베딩 공간
BUY 패턴  → 임베딩 A (예: [1.0, 0.0, ...])
HOLD 패턴 → 임베딩 B (예: [0.0, 1.0, ...])
SELL 패턴 → 임베딩 C (예: [0.0, 0.0, ...])

# 실제 임베딩 공간 (붕괴)
BUY 패턴  → 임베딩 [0.5, 0.5, ...]  ← 모두 비슷
HOLD 패턴 → 임베딩 [0.5, 0.5, ...]
SELL 패턴 → 임베딩 [0.5, 0.5, ...]
```

## 🔧 해결 방안

### 방안 1: 임베딩 공간 확장

**대조 학습 (Contrastive Learning) 추가:**
```python
# BUY/SELL 패턴을 명확히 구분하도록 학습
loss = reconstruction_loss + contrastive_loss

# BUY와 SELL 임베딩이 멀어지도록
contrastive_loss = -distance(emb_buy, emb_sell)
```

### 방안 2: 클래스 균형 샘플링

```python
# 학습 시 각 클래스를 동일한 비율로 샘플링
sampler = BalancedSampler(
    labels=actions,
    samples_per_class=10000
)

# 결과: HOLD, BUY, SELL 각 10,000개씩
```

### 방안 3: 임베딩 정규화

```python
# 임베딩을 단위 구면에 투영
embedding = F.normalize(embedding, p=2, dim=-1)

# 결과: 모든 임베딩이 동일한 크기
# 방향만으로 구분
```

### 방안 4: 다중 작업 학습

```python
# AutoEncoder + 행동 분류 동시 학습
class MultiTaskAutoEncoder(nn.Module):
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        action_pred = self.action_classifier(embedding)
        
        return reconstruction, action_pred

# 손실 함수
loss = reconstruction_loss + action_classification_loss
```

## 📊 검증 방법

### 1. 임베딩 분포 시각화

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 각 행동별로 임베딩 수집
embeddings_hold = []
embeddings_buy = []
embeddings_sell = []

# t-SNE로 2D 투영
tsne = TSNE(n_components=2)
emb_2d = tsne.fit_transform(all_embeddings)

# 플롯
plt.scatter(emb_2d[labels==0, 0], emb_2d[labels==0, 1], c='blue', label='HOLD')
plt.scatter(emb_2d[labels==1, 0], emb_2d[labels==1, 1], c='green', label='BUY')
plt.scatter(emb_2d[labels==2, 0], emb_2d[labels==2, 1], c='red', label='SELL')
plt.legend()
plt.savefig('embedding_distribution.png')
```

**기대 결과:**
- 3개 클러스터가 명확히 분리되어야 함
- 현재는 모두 겹쳐있을 것으로 예상

### 2. 임베딩 거리 측정

```python
# 클래스 내 거리 vs 클래스 간 거리
intra_class_dist = distance(emb_hold_1, emb_hold_2)
inter_class_dist = distance(emb_hold, emb_buy)

# 이상적: inter_class_dist >> intra_class_dist
# 현재: inter_class_dist ≈ intra_class_dist
```

## 🎯 결론

**랜덤 임베딩이 성공하는 이유:**
- 128차원 공간 전체를 탐색
- 정책 네트워크의 모든 결정 경계를 통과
- BUY/SELL 영역에도 도달

**실제 데이터가 실패하는 이유:**
- AutoEncoder가 HOLD 패턴만 학습
- 모든 입력이 HOLD 영역의 임베딩으로 변환
- 정책 네트워크가 HOLD만 예측

**핵심 문제:**
- **임베딩 공간의 붕괴 (Embedding Space Collapse)**
- AutoEncoder가 다양한 패턴을 구별하지 못함
- 클래스 불균형으로 인한 학습 편향

**해결책:**
1. 대조 학습으로 임베딩 공간 확장
2. 클래스 균형 샘플링
3. 다중 작업 학습 (재구성 + 분류)
4. 임베딩 정규화

이것이 바로 "representation learning"의 핵심 문제입니다!
