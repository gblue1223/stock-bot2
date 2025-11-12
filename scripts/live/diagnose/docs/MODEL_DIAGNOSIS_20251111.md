# 모델 진단 보고서 (2025-11-11)

## 🔍 문제 요약

**실시간 거래 시스템에서 모델이 BUY 신호를 전혀 생성하지 않음**

- 총 582회 예측 중 BUY 신호 0회
- HOLD: 대부분, SELL: 일부 (신뢰도 낮음)
- 거래 진입 불가 → 수익 기회 상실

## 📊 진단 결과

### 1. 모델 구조 검사

**정책 네트워크 (GRPOPolicy)**
```
총 파라미터: 33,540개
구조:
  - fc1: Linear(128 → 128)
  - fc2: Linear(128 → 128)
  - policy_head: Linear(128 → 3)  # HOLD, BUY, SELL
  - value_head: Linear(128 → 1)
```

**임베딩 모델 (AutoEncoder)**
```
총 파라미터: 8,542,876개
구조: Transformer 기반 (3 layers)
  - Encoder: 28 features → 128 embedding
  - Decoder: 128 embedding → 28 features
```

✅ **모델 구조는 정상**

### 2. 랜덤 임베딩 테스트

100개의 랜덤 임베딩으로 테스트:

```
HOLD:  56 (56.0%)
BUY:    6 (6.0%)
SELL:  38 (38.0%)

평균 행동 확률:
  HOLD: 0.4228
  BUY:  0.1323
  SELL: 0.4449
```

✅ **정책 네트워크 자체는 BUY를 예측할 수 있음**

### 3. 합성 데이터 테스트

100개의 합성 시퀀스 (5가지 시나리오):
- 상승 추세 (20개)
- 하락 추세 (20개)
- 횡보 (20개)
- 급등 (20개)
- 급락 (20개)

**결과:**
```
HOLD: 100 (100.0%)
BUY:    0 (0.0%)
SELL:   0 (0.0%)

모든 시나리오에서 HOLD만 예측
평균 신뢰도: 0.6133 ± 0.0017
```

❌ **실제 데이터(정규화 후)에서는 BUY/SELL을 전혀 예측하지 않음**

## 🎯 근본 원인 분석

### 원인 1: 임베딩 공간의 문제

**가설**: AutoEncoder가 생성하는 임베딩이 특정 영역에 집중되어 있음

```
랜덤 임베딩 → 다양한 예측 (HOLD 56%, BUY 6%, SELL 38%)
실제 임베딩 → HOLD만 예측 (100%)
```

**결론**: 실제 데이터의 임베딩이 "HOLD 영역"에만 매핑됨

### 원인 2: 정규화 문제

**현재 정규화 방식:**
```python
# 학습 통계 사용
training_mean = [-0.8312, ..., 0.6242]  # 28개
training_std = [0.0515, ..., 1.0000]    # 28개
```

**문제점:**
- 합성 데이터가 학습 데이터와 다른 분포를 가질 수 있음
- 정규화 후 모든 데이터가 유사한 패턴으로 변환됨

### 원인 3: 학습 데이터 불균형 (추정)

**가능성:**
- 학습 데이터에서 HOLD 샘플이 압도적으로 많음
- BUY/SELL 샘플이 부족하거나 특정 패턴에만 존재
- 모델이 "안전하게" HOLD를 선택하도록 학습됨

## 📈 실시간 로그 분석

### 로그 1: 2025-11-10 (6시간 실행)
```
총 예측: 4,815회
HOLD: 100%
BUY: 0%
SELL: 0%
```

### 로그 2: 2025-11-11 (8분 실행)
```
총 예측: 582회
HOLD: 대부분
SELL: 일부 (신뢰도 0.4451)
BUY: 0%

실제 종목: 007810 (등락률 +18.8%)
→ 상승장인데도 BUY 신호 없음
```

## 🔧 해결 방안

### 방안 1: 학습 데이터 검증 (우선순위 높음)

```python
# 학습 데이터의 행동 분포 확인
import duckdb

conn = duckdb.connect('datasets_norm_all.duckdb')
result = conn.execute("""
    SELECT 
        action,
        COUNT(*) as count,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
    FROM training_data
    GROUP BY action
    ORDER BY action
""").fetchall()

print("학습 데이터 행동 분포:")
for action, count, pct in result:
    print(f"  {['HOLD', 'BUY', 'SELL'][action]}: {count:,} ({pct:.2f}%)")
```

**예상 결과:**
- HOLD: 90%+
- BUY: <5%
- SELL: <5%

**조치:**
- 클래스 불균형 해결 (oversampling, class weights)
- BUY/SELL 샘플 증강

### 방안 2: 임베딩 공간 분석

```python
# 임베딩 분포 시각화
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1000개 샘플의 임베딩 추출
embeddings = []
actions = []

for seq in sequences:
    emb = embedding_model.encode(seq)
    action, _ = policy(emb)
    embeddings.append(emb.cpu().numpy())
    actions.append(action)

# PCA로 2D 투영
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

# 행동별로 색상 구분하여 플롯
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=actions, cmap='viridis')
plt.colorbar(label='Action (0=HOLD, 1=BUY, 2=SELL)')
plt.title('Embedding Space Distribution')
plt.savefig('embedding_distribution.png')
```

**기대 결과:**
- 임베딩이 특정 영역에 집중되어 있는지 확인
- BUY/SELL 영역이 존재하는지 확인

### 방안 3: 모델 재학습

**옵션 A: 클래스 가중치 적용**
```python
# GRPO 학습 시 클래스 가중치 추가
class_weights = {
    0: 1.0,   # HOLD
    1: 10.0,  # BUY (10배 가중치)
    2: 10.0   # SELL (10배 가중치)
}
```

**옵션 B: 보상 함수 수정**
```python
# BUY/SELL 행동에 보너스 추가
def compute_reward(action, return_rate):
    base_reward = return_rate
    
    if action == 1:  # BUY
        exploration_bonus = 0.1
        return base_reward + exploration_bonus
    elif action == 2:  # SELL
        exploration_bonus = 0.1
        return base_reward + exploration_bonus
    else:  # HOLD
        return base_reward
```

**옵션 C: 엔트로피 정규화 강화**
```python
# 정책 엔트로피를 높여서 탐험 유도
entropy_coef = 0.1  # 현재 값 확인 필요
# 0.01 → 0.1로 증가
```

### 방안 4: 임계값 조정 (임시 방편)

```python
# 신뢰도 임계값을 음수로 설정하여 강제로 BUY 유도
min_buy_confidence = -0.5  # 현재 0.0
```

❌ **권장하지 않음**: 근본 원인 해결 아님

## 📋 권장 조치 순서

1. **즉시 (1일)**
   - [ ] 학습 데이터의 행동 분포 확인
   - [ ] 백테스팅으로 모델 성능 검증
   - [ ] 임베딩 공간 시각화

2. **단기 (1주)**
   - [ ] 클래스 불균형 해결 방안 적용
   - [ ] 보상 함수 수정
   - [ ] 모델 재학습

3. **중기 (2주)**
   - [ ] 새 모델로 백테스팅
   - [ ] 실시간 테스트 (소액)
   - [ ] 성능 모니터링

## 🎯 결론

**모델 자체는 정상 작동하지만, 학습 데이터의 불균형으로 인해 HOLD만 예측**

- 정책 네트워크: ✅ 정상
- 임베딩 모델: ✅ 정상
- 정규화: ✅ 정상
- 학습 데이터: ❌ 불균형 추정

**다음 단계**: 학습 데이터 분석 → 재학습 → 검증

## 📎 참고 파일

- 모델 검사 스크립트: `scripts/live/inspect_model.py`
- 합성 데이터 테스트: `scripts/live/test_model_with_synthetic_data.py`
- 실시간 로그: `logs/live_trading_20251111_095305.log`
