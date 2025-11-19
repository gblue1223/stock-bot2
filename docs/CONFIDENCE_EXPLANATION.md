# Confidence (신뢰도) 설명

## 📊 Confidence란?

**Confidence**는 모델이 예측한 행동에 대한 **확률값**입니다.

### 범위
```
0.0 ≤ confidence ≤ 1.0
```

### 의미
- **1.0에 가까울수록**: 모델이 해당 행동을 확신함
- **0.5 근처**: 모델이 불확실함
- **0.0에 가까울수록**: 모델이 해당 행동을 확신하지 못함

---

## 🔍 계산 방법

### 1. 정책 네트워크 출력
```python
# ai_trader/grpo/inference/infer_grpo.py
with torch.no_grad():
    action_logits, _ = self.policy(policy_input)
    # action_logits: 각 행동에 대한 로짓 값 (raw scores)
    # 예: [-1.2, 0.5, -0.3] (HOLD, BUY, SELL)
```

### 2. Softmax 변환
```python
action_probs = torch.softmax(action_logits, dim=-1)
# Softmax: 로짓을 확률로 변환 (합이 1.0)
# 예: [0.15, 0.60, 0.25] (HOLD, BUY, SELL)
```

### 3. Confidence 추출
```python
if deterministic:
    # 최대 확률 행동 선택
    action = torch.argmax(action_probs, dim=-1)  # 1 (BUY)
    confidence = action_probs[0, action].item()  # 0.60
```

---

## 📈 실제 예시

### 예시 1: 높은 신뢰도 (BUY)
```python
action_logits = [-2.0, 1.5, -1.0]  # HOLD, BUY, SELL
action_probs = [0.05, 0.85, 0.10]  # Softmax 후
action = 1  # BUY (최대 확률)
confidence = 0.85  # 85% 확신
```
→ **모델이 BUY를 강하게 확신**

### 예시 2: 낮은 신뢰도 (BUY)
```python
action_logits = [0.0, 0.1, -0.1]  # HOLD, BUY, SELL
action_probs = [0.33, 0.37, 0.30]  # Softmax 후
action = 1  # BUY (최대 확률)
confidence = 0.37  # 37% 확신
```
→ **모델이 불확실하지만 BUY가 약간 우세**

### 예시 3: 중간 신뢰도 (SELL)
```python
action_logits = [-0.5, -0.3, 0.8]  # HOLD, BUY, SELL
action_probs = [0.20, 0.25, 0.55]  # Softmax 후
action = 2  # SELL (최대 확률)
confidence = 0.55  # 55% 확신
```
→ **모델이 SELL을 선호하지만 확신은 중간**

---

## 📊 로그 분석 (실제 데이터)

### 로그에서 확인된 Confidence 값

```
confidence=0.5928  # HOLD
confidence=0.5606  # SELL
confidence=0.5675  # SELL
confidence=0.6164  # SELL
confidence=0.5654  # SELL
confidence=0.4454  # HOLD
confidence=0.4207  # HOLD
confidence=0.5729  # SELL
confidence=0.5629  # SELL
confidence=0.4926  # SELL
```

### 통계
- **최소값**: 약 0.42
- **최대값**: 약 0.62
- **평균**: 약 0.50 ~ 0.55
- **범위**: 모두 0.0 ~ 1.0 사이 ✅

### 특징
- 대부분 **0.4 ~ 0.6 범위**에 분포
- 매우 높은 신뢰도(0.9+)나 매우 낮은 신뢰도(0.1-)는 드물음
- 모델이 **불확실한 상황**에서 예측하고 있음을 의미

---

## 🎯 Confidence 활용

### 1. 신호 필터링
```python
# ai_trader/grpo/inference/enhanced_inference.py
if raw_action == Action.BUY.value:
    if raw_confidence < self.min_buy_confidence:  # 기본값: 0.0
        # 신뢰도가 너무 낮으면 무시
        return Action.HOLD.value, raw_confidence, info
```

**현재 설정**:
```json
{
  "min_buy_confidence": 0.0,   // 모든 BUY 신호 허용
  "min_sell_confidence": 0.0   // 모든 SELL 신호 허용
}
```

### 2. 신뢰도 기반 필터링 예시

#### 보수적 전략 (높은 신뢰도만)
```json
{
  "min_buy_confidence": 0.6,   // 60% 이상만 매수
  "min_sell_confidence": 0.6   // 60% 이상만 매도
}
```
→ **장점**: 확실한 신호만 거래  
→ **단점**: 거래 기회 감소

#### 공격적 전략 (낮은 신뢰도도 허용)
```json
{
  "min_buy_confidence": 0.3,   // 30% 이상 매수
  "min_sell_confidence": 0.3   // 30% 이상 매도
}
```
→ **장점**: 더 많은 거래 기회  
→ **단점**: 잘못된 신호 증가

---

## 🔬 Confidence 분포 분석

### 이상적인 분포
```
High confidence (0.7-1.0): 확실한 신호
Medium confidence (0.5-0.7): 보통 신호
Low confidence (0.0-0.5): 불확실한 신호
```

### 현재 모델의 분포
```
High confidence (0.7-1.0): 거의 없음 ❌
Medium confidence (0.5-0.7): 대부분 ✅
Low confidence (0.0-0.5): 일부 존재
```

### 문제점
- 모델이 **확신 있는 예측을 하지 못함**
- 대부분의 예측이 **0.5 근처** (불확실)
- 높은 신뢰도(0.7+) 신호가 거의 없음

### 개선 방안
1. **모델 재학습**: 더 명확한 패턴 학습
2. **데이터 품질 개선**: 더 좋은 특징 추출
3. **정규화 개선**: 입력 데이터 품질 향상
4. **보상 함수 조정**: 확실한 행동에 더 높은 보상

---

## 💡 Confidence 해석 가이드

### BUY 신호
```python
confidence = 0.85  # 매우 확신 → 강력한 매수 신호
confidence = 0.65  # 확신 → 좋은 매수 신호
confidence = 0.55  # 약간 확신 → 보통 매수 신호
confidence = 0.45  # 불확실 → 약한 매수 신호
confidence = 0.35  # 매우 불확실 → 무시 권장
```

### SELL 신호
```python
confidence = 0.85  # 매우 확신 → 강력한 매도 신호
confidence = 0.65  # 확신 → 좋은 매도 신호
confidence = 0.55  # 약간 확신 → 보통 매도 신호
confidence = 0.45  # 불확실 → 약한 매도 신호
confidence = 0.35  # 매우 불확실 → 무시 권장
```

---

## 📝 요약

### Confidence 특성
- ✅ **범위**: 0.0 ~ 1.0
- ✅ **의미**: 모델의 예측 확신도
- ✅ **계산**: Softmax(action_logits)
- ✅ **활용**: 신호 필터링

### 현재 상태
- **평균 신뢰도**: 0.50 ~ 0.55
- **분포**: 대부분 0.4 ~ 0.6
- **문제**: 높은 신뢰도 신호 부족
- **개선**: 모델 재학습 필요

### 권장 설정
```json
{
  "min_buy_confidence": 0.5,   // 50% 이상만 매수
  "min_sell_confidence": 0.5   // 50% 이상만 매도
}
```

---

**작성일**: 2025-11-19  
**상태**: ✅ 분석 완료
