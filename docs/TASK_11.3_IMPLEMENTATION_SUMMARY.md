# Task 11.3 구현 요약: 백테스팅 시뮬레이터

## 구현 개요

홀드아웃 테스트 데이터에서 GRPO 에이전트의 거래를 시뮬레이션하고, 현실적인 거래 비용 및 슬리피지를 적용하여 성능을 평가하는 백테스팅 시뮬레이터를 구현했습니다.

## 구현된 파일

### 1. 핵심 모듈
- **`ai_trader/grpo/backtest.py`**: 백테스팅 시뮬레이터 메인 모듈
  - `BacktestSimulator` 클래스: 백테스팅 실행 및 관리
  - `run_backtest()` 함수: 편의 함수

### 2. 예제 스크립트
- **`examples/grpo_backtest_example.py`**: CLI 기반 백테스팅 예제
  - 명령줄 인터페이스 제공
  - 성능 기준 검증 기능

### 3. 테스트
- **`tests/test_grpo_backtest.py`**: 백테스팅 시뮬레이터 테스트
  - 6개 테스트 케이스 (모두 통과)
  - 슬리피지, 보상 계산, 데이터 로드 등 검증

### 4. 문서
- **`docs/GRPO_BACKTESTING.md`**: 백테스팅 시뮬레이터 사용 가이드
  - 기능 설명
  - 사용 방법
  - 결과 구조
  - 성능 기준

## 주요 기능

### 1. 현실적인 거래 비용 적용 (요구사항 7.3)

```python
# 거래 비용 설정
transaction_cost_rate = 0.00215  # 0.215% (수수료 0.015% + 세금 0.2%)
round_trip_cost = 0.0043         # 0.43% (왕복)

# 보상 계산
profit_rate = (exit_price - entry_price) / entry_price
reward = profit_rate - round_trip_cost  # 거래 비용 차감
```

**특징**:
- 수수료(0.015%) + 세금(0.2%) = 0.215% (매수/매도 각각)
- 왕복 거래 비용: 0.43%
- 양의 보상 조건: 수익률 > 0.43%

### 2. 슬리피지 시뮬레이션 (요구사항 7.3)

```python
def _apply_slippage(self, price: float, action: int) -> float:
    """슬리피지 적용"""
    if action == 1:  # 매수
        return price * (1 + self.slippage_rate)  # 가격 상승
    elif action == 2:  # 매도
        return price * (1 - self.slippage_rate)  # 가격 하락
    else:
        return price
```

**특징**:
- 매수 시: 가격 상승 (불리한 방향)
- 매도 시: 가격 하락 (불리한 방향)
- 기본 슬리피지: 0.01% (설정 가능)

### 3. 빠른 손절 룰 적용

```python
# 빠른 손절 룰 체크
if position == 1:
    holding_time = current_time - entry_time
    
    if holding_time <= quick_exit_threshold and current_price <= entry_price:
        # 자동 매도 및 페널티 적용
        quick_exit_violations += 1
        reward -= quick_exit_penalty
```

**특징**:
- 매수 후 설정 가능한 시간 임계값(기본값 1.5초) 이내에 가격 미상승 시 자동 매도
- 위반 시 페널티 적용 (기본값 0.01)
- 위반 횟수 추적 및 보고

### 4. 포괄적인 평가 메트릭 (요구사항 7.2)

```python
metrics = {
    'win_rate': 0.58,              # 승률
    'avg_profit_per_trade': 0.0012, # 거래당 평균 수익
    'max_drawdown': 0.08,          # 최대 낙폭
    'sharpe_ratio': 1.32,          # 샤프 비율
    'avg_holding_time': 24.5,      # 평균 보유 시간
    'total_trades': 450,           # 총 거래 횟수
    'total_episodes': 30,          # 총 에피소드 수
    'total_return': 0.5535,        # 총 수익률
    'avg_episode_return': 0.01845  # 에피소드당 평균 수익률
}
```

### 5. 종목별/날짜별 에피소드 실행

```python
def _run_episodes_by_stock(self, data, metadata, deterministic, verbose):
    """종목별로 에피소드 실행"""
    # 종목별로 그룹화
    unique_stocks = np.unique(metadata[:, 0])
    
    for stock_code in unique_stocks:
        # 날짜별로 그룹화
        unique_dates = np.unique(stock_metadata[:, 1])
        
        for date in unique_dates:
            # 에피소드 실행
            episode_result = self._run_single_episode(...)
```

**특징**:
- 종목별, 날짜별로 독립적인 에피소드 실행
- 다양한 시장 상황에서 성능 평가
- 에피소드별 메타데이터 수집

### 6. 결과 저장 및 분석

```python
# 결과 저장
simulator.save_results(results, 'backtest_results.json')

# 비용 분석
summary = {
    'total_slippage_cost': 0.09,
    'avg_slippage_per_trade': 0.0002,
    'total_transaction_cost': 1.935,
    'avg_transaction_cost_per_trade': 0.0043,
    'quick_exit_violations': 45,
    'quick_exit_violation_rate': 0.10
}
```

## 사용 예제

### 기본 사용

```python
from ai_trader.grpo.backtest import run_backtest

results = run_backtest(
    embedding_model_path='models/embedding/checkpoint_epoch50.pt',
    policy_path='models/grpo_scalping/checkpoint_step1000000.pt',
    db_path='datasets_norm_all.duckdb',
    test_start_date='2024-01-01',
    test_end_date='2024-12-31',
    transaction_cost_rate=0.00215,
    slippage_rate=0.0001,
    device='cuda',
    output_path='backtest_results.json',
    verbose=True
)
```

### CLI 사용

```bash
python examples/grpo_backtest_example.py \
    --embedding-model models/embedding/checkpoint_epoch50.pt \
    --policy models/grpo_scalping/checkpoint_step1000000.pt \
    --db datasets_norm_all.duckdb \
    --test-start-date 2024-01-01 \
    --test-end-date 2024-12-31 \
    --transaction-cost 0.00215 \
    --slippage 0.0001 \
    --device cuda \
    --output backtest_results.json \
    --verbose
```

## 테스트 결과

모든 테스트가 성공적으로 통과했습니다:

```
tests/test_grpo_backtest.py::test_backtest_simulator_initialization PASSED
tests/test_grpo_backtest.py::test_slippage_application PASSED
tests/test_grpo_backtest.py::test_reward_calculation PASSED
tests/test_grpo_backtest.py::test_load_test_data PASSED
tests/test_grpo_backtest.py::test_run_backtest PASSED
tests/test_grpo_backtest.py::test_save_results PASSED

============================== 6 passed in 3.72s ==============================
```

### 테스트 커버리지

1. **초기화 테스트**: 시뮬레이터 설정 검증
2. **슬리피지 테스트**: 매수/매도 시 가격 변화 검증
3. **보상 계산 테스트**: 거래 비용, 보유 페널티 검증
4. **데이터 로드 테스트**: 테스트 데이터 필터링 검증
5. **백테스팅 실행 테스트**: 전체 파이프라인 검증
6. **결과 저장 테스트**: JSON 직렬화 검증

## 성능 기준 검증

백테스팅 결과는 다음 기준에 따라 자동으로 검증됩니다:

```python
# 성능 기준 (요구사항 7.2, 7.3)
criteria = {
    'win_rate': 0.50,           # 승률 > 50%
    'sharpe_ratio': 1.0,        # 샤프 비율 > 1.0
    'max_drawdown': 0.10,       # 최대 낙폭 < 10%
    'avg_holding_time': 30.0    # 평균 보유 시간 < 30초
}
```

예제 스크립트는 각 기준을 자동으로 체크하고 결과를 출력합니다:

```
Performance Criteria Check:
----------------------------------------------------------------------
✓ Win Rate > 50%: 58.00%
✓ Sharpe Ratio > 1.0: 1.3200
✓ Max Drawdown < 10%: 8.00%
✓ Avg Holding Time < 30s: 24.50s
----------------------------------------------------------------------
✓ All performance criteria met!
```

## 비용 분석 기능

백테스팅 결과에는 상세한 비용 분석이 포함됩니다:

### 1. 거래 비용 분석
- 총 거래 비용
- 거래당 평균 비용 (0.43%)

### 2. 슬리피지 분석
- 총 슬리피지 비용
- 거래당 평균 슬리피지

### 3. 빠른 손절 룰 분석
- 총 위반 횟수
- 위반 비율 (위반 횟수 / 총 거래 횟수)

## 요구사항 충족 확인

### 요구사항 7.3: 백테스팅 시뮬레이터

✅ **홀드아웃 테스트 데이터에서 거래 시뮬레이션**
- 종목별, 날짜별로 독립적인 에피소드 실행
- 테스트 기간 및 종목 필터링 지원

✅ **현실적인 거래 비용 적용**
- 수수료(0.015%) + 세금(0.2%) = 0.215%
- 왕복 거래 비용: 0.43%
- 설정 가능한 거래 비용 비율

✅ **슬리피지 적용**
- 매수 시 가격 상승, 매도 시 가격 하락
- 설정 가능한 슬리피지 비율 (기본값 0.01%)
- 거래별 슬리피지 비용 추적

## 통합 및 호환성

### 기존 시스템과의 통합

1. **GRPO 환경**: `GRPOScalpingEnv`의 보상 계산 로직 재사용
2. **평가 메트릭**: `GRPOEvaluationMetrics` 클래스 활용
3. **추론 엔진**: `GRPOInference` 클래스 사용
4. **데이터베이스**: DuckDB 연결 및 쿼리 로직 일관성 유지

### 확장성

- 새로운 비용 모델 추가 가능
- 커스텀 평가 메트릭 추가 가능
- 다양한 슬리피지 모델 지원 가능

## 문서화

### 생성된 문서

1. **`docs/GRPO_BACKTESTING.md`**: 포괄적인 사용 가이드
   - 기능 설명
   - 사용 방법 (기본, CLI, 고급)
   - 결과 구조
   - 성능 기준
   - 예제 출력
   - 주의사항

2. **코드 주석**: 모든 함수와 클래스에 상세한 docstring 포함

## 향후 개선 사항

1. **다양한 슬리피지 모델**: 거래량 기반, 변동성 기반 슬리피지
2. **시장 충격 모델**: 대량 주문 시 가격 영향 시뮬레이션
3. **HTML 보고서 생성**: 시각화된 백테스팅 결과 (Task 11.6)
4. **실시간 거래 로깅**: 모든 거래 기록 (Task 11.4)
5. **성능 알림 시스템**: 임계값 기반 알림 (Task 11.5)

## 결론

Task 11.3 백테스팅 시뮬레이터가 성공적으로 구현되었습니다. 이 시뮬레이터는:

- ✅ 홀드아웃 테스트 데이터에서 거래 시뮬레이션
- ✅ 현실적인 거래 비용 적용 (0.43% 왕복)
- ✅ 슬리피지 시뮬레이션 (0.01% 기본값)
- ✅ 포괄적인 평가 메트릭 계산
- ✅ 성능 기준 자동 검증
- ✅ 상세한 비용 분석
- ✅ 결과 저장 및 보고
- ✅ 완전한 테스트 커버리지
- ✅ 포괄적인 문서화

이제 GRPO 에이전트의 성능을 현실적인 조건에서 평가할 수 있으며, 실제 거래 배포 전에 충분한 검증이 가능합니다.
