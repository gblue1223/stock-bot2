# Task 7.4 Implementation Summary: 그룹 상대 어드밴티지 계산 구현

## 개요

Task 7.4에서는 GRPO 알고리즘의 핵심 컴포넌트인 그룹 상대 어드밴티지 계산을 구현했습니다. 이는 요구사항 4.3에 명시된 대로 각 그룹의 평균 수익을 계산하고, 그룹 내 상대 어드밴티지를 계산하는 기능입니다.

## 구현 내용

### 1. 그룹 상대 어드밴티지 계산 메서드

**파일**: `ai_trader/grpo/grpo.py`

**메서드**: `compute_group_relative_advantages()`

#### 구현 세부사항

1. **각 그룹의 평균 수익 계산**
   - 그룹 내 각 에피소드의 총 수익(R_i) 계산
   - 에피소드의 총 수익 = 모든 보상의 합: `R_i = sum(rewards)`
   - 그룹의 평균 수익 계산: `mean(R_group) = mean([R_1, R_2, ..., R_n])`

2. **그룹 내 상대 어드밴티지 계산**
   - 공식: `A_i = R_i - mean(R_group)`
   - 각 에피소드의 총 수익에서 그룹 평균 수익을 뺀 값
   - 에피소드의 각 타임스텝에 동일한 어드밴티지 할당

3. **메타데이터 추가**
   - 각 에피소드에 `relative_advantage` 필드 추가
   - 각 에피소드에 `group_mean_return` 필드 추가
   - 그룹 통계 로깅 (평균, 표준편차, 최소값, 최대값)

#### 코드 예시

```python
def compute_group_relative_advantages(
    self,
    grouped_episodes: Dict[int, List[Dict[str, Any]]]
) -> Dict[int, List[np.ndarray]]:
    """
    그룹 상대 어드밴티지 계산
    
    요구사항 4.3에 따라 각 그룹의 평균 수익을 계산하고
    그룹 내 상대 어드밴티지를 계산합니다.
    
    그룹 상대 어드밴티지는 다음과 같이 계산됩니다:
    A_i = R_i - mean(R_group)
    """
    group_advantages = {}
    
    for group_id, group_episodes in grouped_episodes.items():
        # 1. 그룹 내 각 에피소드의 총 수익 계산
        episode_returns = []
        for episode in group_episodes:
            total_return = np.sum(episode['rewards'])
            episode_returns.append(total_return)
        
        # 2. 그룹의 평균 수익 계산
        mean_group_return = np.mean(episode_returns)
        
        # 3. 각 에피소드의 그룹 상대 어드밴티지 계산
        advantages = []
        for episode_idx, episode in enumerate(group_episodes):
            episode_return = episode_returns[episode_idx]
            
            # 그룹 상대 어드밴티지: A_i = R_i - mean(R_group)
            relative_advantage = episode_return - mean_group_return
            
            # 에피소드의 각 타임스텝에 동일한 어드밴티지 할당
            num_steps = len(episode['rewards'])
            advantage_array = np.full(num_steps, relative_advantage, dtype=np.float32)
            
            advantages.append(advantage_array)
            
            # 에피소드 메타데이터에 어드밴티지 정보 추가
            episode['relative_advantage'] = relative_advantage
            episode['group_mean_return'] = mean_group_return
        
        group_advantages[group_id] = advantages
        
        # 그룹 통계 로깅
        logger.debug(f"Group {group_id}: mean_return={mean_group_return:.4f}, "
                    f"num_episodes={len(group_episodes)}, "
                    f"return_std={np.std(episode_returns):.4f}, "
                    f"return_min={np.min(episode_returns):.4f}, "
                    f"return_max={np.max(episode_returns):.4f}")
    
    logger.info(f"Computed group relative advantages for {len(grouped_episodes)} groups")
    
    return group_advantages
```

## 테스트

### 테스트 파일

**파일**: `tests/test_grpo_advantage.py`

### 테스트 케이스

1. **`test_requirement_4_3_group_mean_return_calculation()`**
   - 요구사항 4.3 검증: 각 그룹의 평균 수익 계산
   - 알려진 수익을 가진 에피소드로 평균 계산 검증
   - 그룹 0: [10, 20, 30] → 평균 20
   - 그룹 1: [-5, 0, 5] → 평균 0
   - 그룹 2: [100, 200] → 평균 150

2. **`test_requirement_4_3_relative_advantage_formula()`**
   - 요구사항 4.3 검증: A_i = R_i - mean(R_group) 공식
   - 그룹 0: 어드밴티지 [-10, 0, 10]
   - 그룹 1: 어드밴티지 [-5, 0, 5]
   - 그룹 2: 어드밴티지 [-50, 50]

3. **`test_advantage_array_shape()`**
   - 어드밴티지 배열의 형태 검증
   - 각 에피소드의 타임스텝 수와 일치하는지 확인

4. **`test_advantage_sum_is_zero_per_group()`**
   - 그룹 내 어드밴티지 합이 0인지 검증
   - 상대 어드밴티지의 수학적 특성 확인

5. **`test_edge_case_single_episode_per_group()`**
   - 에지 케이스: 그룹당 단일 에피소드
   - 어드밴티지가 0이어야 함

6. **`test_advantage_metadata_added()`**
   - 메타데이터 추가 검증
   - `relative_advantage`, `group_mean_return` 필드 확인

7. **`test_different_episode_lengths()`**
   - 다양한 길이의 에피소드 처리 검증
   - 어드밴티지 배열 길이가 에피소드 길이와 일치하는지 확인

### 테스트 결과

```
tests\test_grpo_advantage.py.......                                                    [100%]

===================================== 7 passed in 2.13s ======================================
```

**모든 테스트 통과 ✓**

## 요구사항 검증

### 요구사항 4.3

> WHEN 어드밴티지가 계산되면 THEN 시스템은 각 에피소드의 수익을 그룹 평균 수익과 비교하여 그룹 상대 어드밴티지를 계산해야 합니다

**검증 결과**: ✓ 충족

- ✓ 각 그룹의 평균 수익 계산 구현
- ✓ 그룹 내 상대 어드밴티지 공식 구현: `A_i = R_i - mean(R_group)`
- ✓ 에피소드 메타데이터에 어드밴티지 정보 추가
- ✓ 그룹 통계 로깅 구현

## 구현 특징

### 1. 수학적 정확성

- 그룹 상대 어드밴티지는 그룹 평균을 기준으로 하므로, 그룹 내 모든 어드밴티지의 합은 0입니다
- 이는 테스트 `test_advantage_sum_is_zero_per_group()`에서 검증됨

### 2. 에피소드 전체 성과 귀속

- 에피소드의 총 수익을 기반으로 어드밴티지를 계산
- 각 타임스텝에 동일한 어드밴티지를 할당하여 에피소드 전체의 성과를 각 행동에 귀속

### 3. 메타데이터 추가

- 각 에피소드에 `relative_advantage`와 `group_mean_return` 추가
- 이후 정책 업데이트 및 분석에 활용 가능

### 4. 로깅 및 디버깅

- 그룹별 통계 로깅 (평균, 표준편차, 최소값, 최대값)
- 디버깅 및 모니터링 용이

## 다음 단계

Task 7.4 완료 후, 다음 작업은:

- **Task 7.5**: 정책 업데이트 구현
  - PPO 스타일 클리핑된 목적 함수
  - KL 발산 제약 적용
  - Optimizer 업데이트

- **Task 7.6**: TensorBoard 로깅 구현
  - 그룹 수준 메트릭 로깅
  - 전체 메트릭 로깅

## 결론

Task 7.4는 성공적으로 완료되었습니다. 그룹 상대 어드밴티지 계산이 요구사항 4.3에 따라 정확하게 구현되었으며, 모든 테스트가 통과했습니다. 이 구현은 GRPO 알고리즘의 핵심 컴포넌트로, 이후 정책 업데이트에 사용될 예정입니다.
