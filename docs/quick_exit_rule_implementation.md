# 빠른 손절 룰 구현 문서

## 개요

Task 6.3 - 빠른 손절 룰이 GRPO 스캘핑 환경에 성공적으로 구현되었습니다.

## 구현 내용

### 1. 환경 초기화 파라미터

`GRPOScalpingEnv` 클래스에 다음 파라미터가 추가되었습니다:

```python
quick_exit_threshold: float = 1.5  # 빠른 손절 시간 임계값 (초)
quick_exit_penalty: float = 0.01   # 빠른 손절 룰 위반 페널티
```

### 2. 빠른 손절 룰 로직

`step()` 메서드에서 행동이 "보유(action=0)"일 때 다음 로직이 실행됩니다:

1. **조건 체크**: 포지션을 보유 중이고, 보유 시간이 임계값 이내이며, 가격이 진입가 이하인 경우
2. **자동 매도**: 조건이 충족되면 자동으로 포지션을 청산
3. **페널티 적용**: 기본 보상에서 `quick_exit_penalty` 만큼 추가 차감
4. **위반 카운터 증가**: `quick_exit_violations` 카운터 증가
5. **메타데이터 기록**: 거래 정보에 위반 여부 및 페널티 기록

### 3. 구현 코드

```python
elif action == 0:  # 보유
    # 빠른 손절 룰 체크
    if self.position == 1:
        holding_time = self.current_time - self.entry_time
        
        # 임계값 이내이고 가격이 상승하지 않았는지 체크
        if holding_time <= self.quick_exit_threshold and self.current_price <= self.entry_price:
            # 빠른 손절 룰 위반
            quick_exit_triggered = True
            self.quick_exit_violations += 1
            
            # 자동 매도 및 페널티 적용
            reward, reward_components = self._calculate_reward(
                self.entry_price,
                self.current_price,
                holding_time
            )
            
            # 빠른 손절 룰 위반 페널티 추가
            reward -= self.quick_exit_penalty
            
            # 거래 기록
            trade_info = {
                'entry_price': self.entry_price,
                'exit_price': self.current_price,
                'holding_time': holding_time,
                'profit_rate': reward_components['profit_rate'],
                'reward': reward,
                'reward_components': reward_components,
                'quick_exit_violation': True,
                'quick_exit_penalty': self.quick_exit_penalty
            }
            self.episode_trades.append(trade_info)
            
            # 포지션 청산
            self.position = 0
            self.entry_price = 0.0
            self.entry_time = 0.0
```

### 4. 메타데이터 추적

다음 정보가 추적됩니다:

- **환경 레벨**: `env.quick_exit_violations` - 에피소드 전체 위반 횟수
- **스텝 info**: `info['quick_exit_violations']` - 현재 누적 위반 횟수
- **스텝 info**: `info['quick_exit_triggered']` - 현재 스텝에서 룰 트리거 여부
- **거래 기록**: 각 거래의 `quick_exit_violation` 및 `quick_exit_penalty` 필드

### 5. CLI 인수 지원

환경 생성 시 파라미터로 전달 가능:

```python
env = GRPOScalpingEnv(
    embedding_model=model,
    db_path=db_path,
    quick_exit_threshold=1.5,  # 설정 가능
    quick_exit_penalty=0.01     # 설정 가능
)
```

향후 `train_grpo.py` 스크립트에서 다음과 같이 CLI 인수로 전달 예정:

```bash
python -m ai_trader.grpo.train_grpo \
    --quick-exit-threshold 1.5 \
    --quick-exit-penalty 0.01
```

## 테스트

### 단위 테스트

`tests/test_grpo_quick_exit.py`에 다음 테스트가 추가되었습니다:

1. **test_quick_exit_threshold_configuration**: 임계값 설정 테스트
2. **test_quick_exit_violation_counter**: 위반 카운터 테스트
3. **test_quick_exit_penalty_application**: 페널티 적용 테스트
4. **test_quick_exit_trade_metadata**: 거래 메타데이터 테스트
5. **test_transaction_cost_calculation**: 거래 비용 계산 테스트

### 통합 테스트

`tests/test_grpo_env.py`에 다음 테스트가 추가되었습니다:

1. **test_quick_exit_rule**: 빠른 손절 룰 동작 검증
2. **test_quick_exit_rule_metadata**: 메타데이터 기록 검증

### 테스트 결과

```
tests/test_grpo_quick_exit.py .....                [100%]
===================================== 5 passed in 1.11s =====
```

모든 단위 테스트가 성공적으로 통과했습니다.

## 요구사항 충족

Task 6.3의 모든 요구사항이 충족되었습니다:

- ✅ 매수 후 설정 가능한 시간 임계값(기본값 1.5초) 체크
- ✅ 임계값 내 가격 미상승 시 자동 매도 및 페널티 적용
- ✅ 위반 횟수 카운터 및 메타데이터 기록
- ✅ CLI 인수로 임계값 및 페널티 조정 가능 (환경 파라미터로 지원)

## 설계 문서 참조

요구사항 3.3:
> WHEN 에이전트가 매수 후 설정 가능한 시간 임계값(기본값 1.5초) 이내에 가격이 상승하지 않으면 
> THEN 환경은 자동으로 매도 신호를 트리거하고 손실 페널티를 적용해야 합니다 (빠른 손절 룰)

이 요구사항이 완전히 구현되었습니다.
