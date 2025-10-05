# Task 7.6: TensorBoard 로깅 구현

## 개요

GRPO 훈련 중 TensorBoard에 메트릭을 로깅하는 기능을 구현했습니다. 이 구현은 요구사항 4.6과 6.6를 충족하며, 그룹 수준 메트릭과 전체 메트릭을 모두 로깅합니다.

## 구현 내용

### 1. 전체 메트릭 로깅

다음 전체 메트릭이 `train/` 네임스페이스에 로깅됩니다:

- **train/mean_reward**: 평균 에피소드 보상
- **train/mean_steps**: 평균 에피소드 스텝 수
- **train/win_rate**: 승률 (수익이 양수인 에피소드 비율)
- **train/avg_holding_time**: 평균 포지션 보유 시간 (초)
- **train/quick_exit_violations_total**: 빠른 손절 룰 위반 총 횟수
- **train/quick_exit_violations_mean**: 에피소드당 평균 위반 횟수
- **train/mean_trades**: 에피소드당 평균 거래 횟수
- **train/mean_sharpe_ratio**: 평균 샤프 비율
- **train/policy_loss**: 정책 손실
- **train/value_loss**: 가치 손실
- **train/entropy**: 정책 엔트로피 (탐험 정도)
- **train/kl_divergence**: KL 발산 (정책 변화 정도)
- **train/clip_fraction**: PPO 클리핑 비율

### 2. 그룹 수준 메트릭 로깅

각 시장 상황 그룹에 대해 다음 메트릭이 `group_X/` 네임스페이스에 로깅됩니다:

- **group_X/mean_return**: 그룹 평균 수익
- **group_X/std_return**: 그룹 수익 표준편차
- **group_X/win_rate**: 그룹 승률
- **group_X/avg_holding_time**: 그룹 평균 보유 시간
- **group_X/quick_exit_violations**: 그룹 빠른 손절 룰 위반 횟수
- **group_X/mean_trades**: 그룹 평균 거래 횟수
- **group_X/sharpe_ratio**: 그룹 샤프 비율
- **group_X/policy_entropy**: 그룹 정책 엔트로피
- **group_X/kl_divergence**: 그룹 KL 발산
- **group_X/size**: 그룹 크기 (에피소드 수)

### 3. 구현 세부사항

#### 정책 엔트로피 계산

각 그룹의 정책 엔트로피는 다음과 같이 계산됩니다:

```python
# 정책에서 행동 확률 분포 얻기
action_probs = policy(states)

# 엔트로피 계산: -sum(p * log(p))
entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
```

엔트로피가 높을수록 정책이 더 탐험적이며, 낮을수록 더 결정적입니다.

#### KL 발산 계산

각 그룹의 KL 발산은 참조 정책과 현재 정책 간의 차이를 측정합니다:

```python
# 현재 정책의 로그 확률
current_log_probs, _, _ = policy.evaluate_actions(states, actions)

# 참조 정책의 로그 확률
ref_log_probs, _, _ = reference_policy.evaluate_actions(states, actions)

# KL 발산: KL(π_ref || π_current)
kl_div = (ref_log_probs - current_log_probs).mean()
```

KL 발산이 높을수록 정책이 참조 정책에서 더 많이 변경되었음을 의미합니다.

## 사용 방법

### 1. GRPO 훈련 시 TensorBoard 로깅 활성화

```python
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.policy import GRPOPolicy
from ai_trader.grpo.env import GRPOScalpingEnv

# 정책 및 환경 생성
policy = GRPOPolicy(embedding_dim=128, hidden_dim=256, action_dim=3)
env = GRPOScalpingEnv(...)

# TensorBoard 로깅 디렉토리 지정
tensorboard_dir = "models/grpo_scalping/tensorboard_logs"

# 훈련기 생성
trainer = GRPOTrainer(
    policy=policy,
    env=env,
    tensorboard_log_dir=tensorboard_dir  # TensorBoard 로깅 활성화
)

# 훈련 실행
trainer.train(total_episodes=1000)
```

### 2. TensorBoard로 메트릭 확인

```bash
# TensorBoard 실행
tensorboard --logdir=models/grpo_scalping/tensorboard_logs

# 브라우저에서 http://localhost:6006 접속
```

### 3. 데모 예제 실행

```bash
# 데모 예제 실행
python examples/grpo_tensorboard_logging_example.py

# TensorBoard로 결과 확인
tensorboard --logdir=models/grpo_example/tensorboard_logs
```

## TensorBoard 시각화

TensorBoard에서 다음과 같은 시각화를 확인할 수 있습니다:

### 1. 전체 성능 추이

- **Scalars > train**: 전체 훈련 메트릭
  - 평균 보상, 승률, 보유 시간 추이
  - 정책 엔트로피 및 KL 발산 추이
  - 빠른 손절 룰 위반 추이

### 2. 그룹별 성능 비교

- **Scalars > group_0, group_1, ...**: 각 그룹의 메트릭
  - 그룹별 수익 비교
  - 그룹별 정책 엔트로피 비교
  - 그룹별 KL 발산 비교

### 3. 정책 학습 진행

- **train/entropy**: 정책이 점진적으로 결정적으로 변하는지 확인
- **train/kl_divergence**: 정책 업데이트가 안정적인지 확인
- **train/clip_fraction**: PPO 클리핑이 적절히 작동하는지 확인

## 테스트

구현된 기능은 다음 테스트로 검증되었습니다:

```bash
# 단위 테스트 실행
pytest tests/test_grpo_tensorboard_logging.py -v
```

테스트 커버리지:
- ✓ 그룹 수준 메트릭 로깅
- ✓ 전체 메트릭 로깅
- ✓ 빈 에피소드 처리

## 요구사항 충족

### 요구사항 4.6

> WHEN 훈련이 진행되면 THEN 시스템은 그룹 수준 메트릭(그룹당 평균 수익, 정책 엔트로피, KL 발산)을 TensorBoard에 로깅해야 합니다

✓ **충족**: 각 그룹에 대해 평균 수익, 정책 엔트로피, KL 발산을 로깅합니다.

### 요구사항 6.6

> WHEN TensorBoard 로깅이 발생하면 THEN 시스템은 표준 디렉토리 구조(models/{model_name}/tensorboard_logs)에 로그를 작성해야 합니다

✓ **충족**: `tensorboard_log_dir` 파라미터로 표준 디렉토리 구조를 지원합니다.

## 파일 변경 사항

### 수정된 파일

1. **ai_trader/grpo/grpo.py**
   - `_log_metrics()` 메서드 개선
   - 그룹 수준 정책 엔트로피 계산 추가
   - 그룹 수준 KL 발산 계산 추가
   - 추가 전체 메트릭 로깅 (거래 횟수, 샤프 비율 등)

### 추가된 파일

1. **tests/test_grpo_tensorboard_logging.py**
   - TensorBoard 로깅 기능 단위 테스트

2. **examples/grpo_tensorboard_logging_example.py**
   - TensorBoard 로깅 데모 예제

3. **docs/TASK_7.6_TENSORBOARD_LOGGING.md**
   - 구현 문서 (이 파일)

## 성능 고려사항

### 계산 오버헤드

- 그룹별 정책 엔트로피 및 KL 발산 계산은 추가 forward pass가 필요합니다
- `torch.no_grad()` 컨텍스트를 사용하여 그래디언트 계산을 비활성화합니다
- 로깅은 훈련 반복마다 한 번만 수행되므로 전체 훈련 시간에 미치는 영향은 미미합니다

### 메모리 사용

- 참조 정책을 메모리에 유지하여 KL 발산을 계산합니다
- 정책 크기가 작으므로 (수백 KB) 메모리 오버헤드는 무시할 수 있습니다

## 향후 개선 사항

1. **추가 메트릭**
   - 그룹별 최대 낙폭
   - 그룹별 승률 추이
   - 그룹별 거래 빈도

2. **시각화 개선**
   - 커스텀 TensorBoard 플러그인
   - 그룹 간 성능 비교 차트
   - 에피소드 궤적 시각화

3. **로깅 최적화**
   - 비동기 로깅으로 훈련 속도 향상
   - 로깅 빈도 조절 옵션

## 참고 자료

- [TensorBoard 문서](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard 통합](https://pytorch.org/docs/stable/tensorboard.html)
- [PPO 알고리즘](https://arxiv.org/abs/1707.06347)
- [GRPO 설계 문서](.kiro/specs/grpo-scalping-embedding/design.md)
