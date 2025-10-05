# Task 7.5 구현 요약: 정책 업데이트 구현

## 개요

Task 7.5에서는 GRPO (Group Relative Policy Optimization) 알고리즘의 핵심인 정책 업데이트 로직을 구현했습니다. 이 구현은 PPO 스타일의 클리핑된 목적 함수와 KL 발산 제약을 사용하여 안정적이고 효율적인 정책 학습을 가능하게 합니다.

## 구현된 컴포넌트

### 1. GRPOPolicy 클래스 (`ai_trader/grpo/policy.py`)

정책 네트워크를 구현하여 임베딩 벡터를 입력받아 행동 확률 분포를 출력합니다.

**주요 기능:**
- `forward()`: 정책 네트워크 forward pass (행동 로짓 + 상태 가치)
- `get_action()`: 행동 샘플링 (deterministic/stochastic 모드)
- `evaluate_actions()`: 주어진 상태-행동 쌍에 대한 로그 확률, 엔트로피, 가치 계산

**아키텍처:**
```
입력 (embedding_dim) 
  → FC1 (hidden_dim) + ReLU
  → FC2 (hidden_dim) + ReLU
  → 정책 헤드 (action_dim) - 행동 확률
  → 가치 헤드 (1) - 상태 가치
```

### 2. update_policy() 메서드 (`ai_trader/grpo/grpo.py`)

PPO 스타일의 클리핑된 목적 함수와 KL 발산 제약을 사용하여 정책을 업데이트합니다.

**구현 세부사항:**

#### 2.1 PPO 클리핑 목적 함수 (요구사항 4.4)

```python
# 확률 비율 계산
ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log_prob - old_log_prob)

# 클리핑되지 않은 목적 함수
surr1 = ratio * advantage

# 클리핑된 목적 함수
ratio_clipped = clamp(ratio, 1-ε, 1+ε)
surr2 = ratio_clipped * advantage

# 최소값 선택 (보수적 업데이트)
policy_loss = -min(surr1, surr2)
```

**클리핑의 효과:**
- 정책이 너무 급격하게 변하는 것을 방지
- 안정적인 학습 보장
- 과도한 업데이트로 인한 성능 저하 방지

#### 2.2 KL 발산 제약 (요구사항 4.5)

```python
# 참조 정책 유지
reference_policy = copy(current_policy)

# KL 발산 계산
kl_divergence = KL(π_old || π_new)

# 조기 종료 조건
if kl_divergence > kl_target * 1.5:
    break  # 정책이 너무 많이 변했으므로 업데이트 중단
```

**KL 발산 제약의 효과:**
- 정책이 참조 정책에서 너무 멀어지는 것을 방지
- 학습 안정성 향상
- 성능 붕괴(performance collapse) 방지

#### 2.3 추가 손실 항목

**가치 손실 (Value Loss):**
```python
value_loss = MSE(predicted_value, actual_return)
```
- 상태 가치 함수 학습
- 어드밴티지 추정 개선

**엔트로피 보너스 (Entropy Bonus):**
```python
entropy_loss = -entropy(policy)
```
- 탐험(exploration) 장려
- 조기 수렴 방지
- 다양한 행동 시도

**총 손실:**
```python
total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

#### 2.4 그래디언트 클리핑

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
```
- 그래디언트 폭발 방지
- 학습 안정성 향상

### 3. _compute_returns() 메서드

할인된 누적 보상(discounted cumulative reward)을 계산합니다.

```python
R_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
```

**구현:**
- 역순으로 계산하여 효율성 향상
- 할인 계수(gamma) 적용

## 알고리즘 흐름

```
1. 데이터 준비
   ├─ 에피소드 데이터 수집 (states, actions, rewards, log_probs)
   ├─ 리턴 계산 (할인된 누적 보상)
   ├─ 어드밴티지 정규화
   └─ 텐서 변환

2. 참조 정책 저장
   ├─ 첫 업데이트 시 참조 정책 초기화
   └─ 현재 정책을 참조 정책으로 복사

3. 여러 에포크 동안 업데이트 (PPO의 multiple epochs)
   └─ 각 에포크마다:
       ├─ 데이터 셔플
       └─ 미니배치 단위로:
           ├─ 정책 평가 (log_probs, entropy, values)
           ├─ PPO 클리핑 목적 함수 계산
           ├─ 가치 손실 계산
           ├─ 엔트로피 보너스 계산
           ├─ 총 손실 계산
           ├─ 그래디언트 업데이트
           ├─ 그래디언트 클리핑
           └─ KL 발산 체크 (조기 종료)

4. 메트릭 반환
   ├─ policy_loss
   ├─ value_loss
   ├─ entropy
   ├─ kl_divergence
   └─ clip_fraction
```

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `clip_epsilon` | 0.2 | PPO 클리핑 파라미터 |
| `kl_target` | 0.01 | KL 발산 목표값 |
| `entropy_coef` | 0.01 | 엔트로피 계수 |
| `value_coef` | 0.5 | 가치 손실 계수 |
| `max_grad_norm` | 0.5 | 그래디언트 클리핑 최대값 |
| `gamma` | 0.99 | 할인 계수 |
| `learning_rate` | 3e-4 | 학습률 |
| `num_epochs` | 4 | PPO 업데이트 에포크 수 |
| `batch_size` | 64 | 미니배치 크기 |

## 테스트

### 구현된 테스트 (`tests/test_grpo_policy_update.py`)

1. **test_policy_forward_pass**: 정책 네트워크 forward pass 검증
2. **test_policy_get_action**: 행동 샘플링 검증
3. **test_policy_evaluate_actions**: 행동 평가 검증
4. **test_compute_returns**: 리턴 계산 검증
5. **test_update_policy_basic**: 기본 정책 업데이트 검증
6. **test_update_policy_ppo_clipping**: PPO 클리핑 동작 검증
7. **test_update_policy_kl_divergence**: KL 발산 계산 검증
8. **test_update_policy_entropy_bonus**: 엔트로피 보너스 검증
9. **test_update_policy_value_loss**: 가치 손실 계산 검증
10. **test_update_policy_gradient_clipping**: 그래디언트 클리핑 검증
11. **test_reference_policy_initialization**: 참조 정책 초기화 검증
12. **test_multiple_updates**: 여러 번의 정책 업데이트 검증

### 테스트 결과

```
tests/test_grpo_policy_update.py ............                           [100%]
===================================== 12 passed in 2.65s =====================================
```

모든 테스트가 성공적으로 통과했습니다.

## 요구사항 검증

### ✅ 요구사항 4.4
**WHEN 정책이 업데이트되면 THEN 시스템은 베이스라인 차감 어드밴티지 대신 그룹 상대 어드밴티지를 사용하는 PPO 스타일의 클리핑된 목적 함수를 사용해야 합니다**

- PPO 클리핑 목적 함수 구현 완료
- 그룹 상대 어드밴티지를 파라미터로 받아 사용
- 확률 비율 클리핑으로 안정적인 업데이트 보장

### ✅ 요구사항 4.5
**IF 정책이 참조 정책에서 크게 벗어나면 THEN 시스템은 안정적인 학습을 보장하기 위해 KL 발산 제약을 적용해야 합니다**

- 참조 정책 유지 및 관리
- KL 발산 계산 및 모니터링
- KL 발산이 임계값 초과 시 조기 종료

## 주요 특징

1. **안정적인 학습**: PPO 클리핑과 KL 발산 제약으로 안정적인 정책 업데이트
2. **효율적인 학습**: Multiple epochs와 미니배치 업데이트로 샘플 효율성 향상
3. **탐험 장려**: 엔트로피 보너스로 다양한 행동 시도
4. **가치 함수 학습**: Actor-Critic 구조로 정책과 가치 함수 동시 학습
5. **그래디언트 안정성**: 그래디언트 클리핑으로 학습 안정성 향상

## 성능 최적화

1. **배치 처리**: 미니배치 단위로 효율적인 GPU 활용
2. **데이터 셔플**: 각 에포크마다 데이터 셔플로 과적합 방지
3. **조기 종료**: KL 발산 체크로 불필요한 업데이트 방지
4. **어드밴티지 정규화**: 안정적인 학습을 위한 어드밴티지 정규화

## 다음 단계

Task 7.5가 완료되었으므로 다음 작업을 진행할 수 있습니다:

- **Task 7.6**: TensorBoard 로깅 구현 (일부 이미 구현됨)
- **Task 8.1**: GRPOPolicy 클래스 구현 (이미 완료됨)
- **Task 9**: GRPO 훈련 스크립트 구현

## 파일 변경 사항

### 새로 생성된 파일
- `ai_trader/grpo/policy.py`: GRPOPolicy 클래스 구현
- `tests/test_grpo_policy_update.py`: 정책 업데이트 테스트

### 수정된 파일
- `ai_trader/grpo/grpo.py`: 
  - `update_policy()` 메서드 구현
  - `_compute_returns()` 메서드 추가
  - `torch.nn.functional` import 추가

## 결론

Task 7.5의 구현으로 GRPO 알고리즘의 핵심인 정책 업데이트 로직이 완성되었습니다. PPO 스타일의 클리핑된 목적 함수와 KL 발산 제약을 통해 안정적이고 효율적인 강화학습이 가능해졌습니다. 모든 테스트가 통과했으며, 요구사항 4.4와 4.5를 완전히 충족합니다.
