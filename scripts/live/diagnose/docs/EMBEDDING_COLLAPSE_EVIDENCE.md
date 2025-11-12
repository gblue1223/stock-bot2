# 임베딩 공간 붕괴 증거

## 🔬 실험 결과

### 통계 비교

**랜덤 임베딩 (정상):**
```
Mean:  -0.0042
Std:    1.0040
Range: [-3.87, +3.63]
Action: HOLD=48, BUY=6, SELL=46
```

**실제 임베딩 (붕괴):**
```
Mean:   0.0127
Std:    0.3098  ← 랜덤의 1/3
Range: [-0.71, +0.73]  ← 랜덤의 1/5
Action: HOLD=100, BUY=0, SELL=0
```

### 차원별 분산 비교

**처음 10개 차원:**
```
Dim  0: Random=1.05, Real=0.00, Ratio=0.00
Dim  1: Random=1.28, Real=0.00, Ratio=0.00
Dim  2: Random=0.95, Real=0.00, Ratio=0.00
Dim  3: Random=1.21, Real=0.00, Ratio=0.00
Dim  4: Random=0.90, Real=0.00, Ratio=0.00
Dim  5: Random=0.80, Real=0.00, Ratio=0.00
Dim  6: Random=1.07, Real=0.00, Ratio=0.00
Dim  7: Random=0.95, Real=0.00, Ratio=0.00
Dim  8: Random=1.40, Real=0.00, Ratio=0.00
Dim  9: Random=1.11, Real=0.00, Ratio=0.00
```

**결론: 실제 임베딩의 분산이 0에 가까움 → 모든 샘플이 동일한 값**

## 💡 핵심 발견

### 1. 임베딩 공간의 극단적 붕괴

**랜덤 임베딩:**
- 128차원 공간 전체를 활용
- 각 차원의 분산 ≈ 1.0
- 넓은 범위 [-3.87, +3.63]

**실제 임베딩:**
- 128차원 중 대부분의 차원에서 분산 = 0
- 모든 샘플이 거의 동일한 임베딩
- 좁은 범위 [-0.71, +0.73]

### 2. 시각화 결과 (PCA & t-SNE)

**예상되는 패턴:**

```
랜덤 임베딩 (PCA):
  ●  ●    ●   ●    ●   ●     ← 넓게 분포
    ●   ●   ●    ●   ●   ●
  ●   ●    ●   ●    ●    ●

실제 임베딩 (PCA):
              ●●●●●●●         ← 한 점에 집중
              ●●●●●●●
```

### 3. 행동 예측 결과

**랜덤 임베딩:**
- HOLD: 48% (정상)
- BUY: 6% (정상)
- SELL: 46% (정상)

**실제 임베딩:**
- HOLD: 100% (비정상)
- BUY: 0% (비정상)
- SELL: 0% (비정상)

## 🎯 왜 이런 일이 발생했는가?

### 원인 1: AutoEncoder의 과도한 압축

```python
# AutoEncoder 구조
Input: (60, 28) = 1,680 차원
  ↓
Encoder: Transformer (3 layers)
  ↓
Bottleneck: 128 차원  ← 1/13로 압축
  ↓
Decoder: Transformer (3 layers)
  ↓
Output: (60, 28) = 1,680 차원
```

**문제:**
- 1,680 → 128 차원으로 압축 (92% 정보 손실)
- 재구성 손실만 최적화
- 다양성 보존 메커니즘 없음

### 원인 2: 학습 데이터의 단조로움

```python
# 학습 데이터 (추정)
HOLD 패턴: 95%  ← 대부분이 비슷한 패턴
BUY 패턴:  2.5%
SELL 패턴: 2.5%
```

**결과:**
- AutoEncoder가 HOLD 패턴만 학습
- BUY/SELL 패턴은 "노이즈"로 취급
- 모든 입력을 HOLD 패턴으로 재구성

### 원인 3: 정규화의 영향

```python
# 정규화 후 모든 데이터가 유사해짐
상승 추세 → [0.5, 0.6, 0.5, ...]
하락 추세 → [0.5, 0.6, 0.5, ...]
횡보      → [0.5, 0.6, 0.5, ...]
급등      → [0.5, 0.6, 0.5, ...]
급락      → [0.5, 0.6, 0.5, ...]
```

**문제:**
- 학습 통계가 실제 데이터 분포를 반영하지 못함
- 정규화 후 구별 불가능

## 🔧 해결 방안

### 방안 1: 대조 학습 (Contrastive Learning)

```python
class ContrastiveAutoEncoder(nn.Module):
    def forward(self, x_anchor, x_positive, x_negative):
        # 임베딩 생성
        emb_anchor = self.encoder(x_anchor)
        emb_positive = self.encoder(x_positive)
        emb_negative = self.encoder(x_negative)
        
        # 재구성 손실
        recon_loss = F.mse_loss(self.decoder(emb_anchor), x_anchor)
        
        # 대조 손실 (같은 클래스는 가깝게, 다른 클래스는 멀게)
        pos_sim = F.cosine_similarity(emb_anchor, emb_positive)
        neg_sim = F.cosine_similarity(emb_anchor, emb_negative)
        contrastive_loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        
        return recon_loss + 0.1 * contrastive_loss
```

### 방안 2: 변분 오토인코더 (VAE)

```python
class VariationalAutoEncoder(nn.Module):
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # 다양성 보장
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        
        # KL divergence로 분포 정규화
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = F.mse_loss(recon, x)
        
        return recon_loss + 0.001 * kl_loss
```

### 방안 3: 다중 작업 학습

```python
class MultiTaskAutoEncoder(nn.Module):
    def forward(self, x, action_label):
        # 임베딩
        embedding = self.encoder(x)
        
        # 재구성
        reconstruction = self.decoder(embedding)
        recon_loss = F.mse_loss(reconstruction, x)
        
        # 행동 분류
        action_pred = self.action_classifier(embedding)
        action_loss = F.cross_entropy(action_pred, action_label)
        
        # 결합 손실
        return recon_loss + 0.5 * action_loss
```

### 방안 4: 임베딩 정규화

```python
def encode(self, x):
    embedding = self.encoder(x)
    
    # L2 정규화 (단위 구면에 투영)
    embedding = F.normalize(embedding, p=2, dim=-1)
    
    return embedding
```

## 📊 검증 방법

### 1. 임베딩 분산 측정

```python
# 목표: 각 차원의 분산 > 0.1
embedding_var = embeddings.var(dim=0)
print(f"Dimensions with var > 0.1: {(embedding_var > 0.1).sum()} / 128")
```

### 2. 클래스 간 거리 측정

```python
# 목표: 클래스 간 거리 > 클래스 내 거리
intra_class_dist = distance(emb_hold_1, emb_hold_2)
inter_class_dist = distance(emb_hold, emb_buy)

ratio = inter_class_dist / intra_class_dist
print(f"Distance ratio: {ratio:.2f}")  # 목표: > 2.0
```

### 3. 시각화 검증

```python
# PCA로 2D 투영
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

# 목표: 3개 클러스터가 명확히 분리
plt.scatter(emb_2d[labels==0, 0], emb_2d[labels==0, 1], label='HOLD')
plt.scatter(emb_2d[labels==1, 0], emb_2d[labels==1, 1], label='BUY')
plt.scatter(emb_2d[labels==2, 0], emb_2d[labels==2, 1], label='SELL')
```

## 🎯 결론

**명확한 증거:**
1. 실제 임베딩의 표준편차 = 0.31 (랜덤의 1/3)
2. 실제 임베딩의 범위 = [-0.71, +0.73] (랜덤의 1/5)
3. 차원별 분산 ≈ 0 (대부분의 차원이 사용되지 않음)
4. 모든 샘플이 HOLD로 예측 (100%)

**근본 원인:**
- **임베딩 공간의 극단적 붕괴**
- AutoEncoder가 모든 입력을 동일한 임베딩으로 변환
- 정책 네트워크가 구별할 수 없음

**해결책:**
- 대조 학습으로 임베딩 공간 확장
- VAE로 다양성 보장
- 다중 작업 학습으로 구별 능력 향상

이것이 바로 "representation collapse" 문제입니다!
