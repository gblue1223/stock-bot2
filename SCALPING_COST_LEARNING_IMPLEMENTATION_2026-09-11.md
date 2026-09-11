# 비용 관측·시간 기준 판단·학습 비교 실험

run03에서는 5·10·15회차 검증이 모두 무거래였으며, 평균 매수 확률은 42.21% → 26.74% → 14.88%로 감소했다. 학습 손실 감소가 거래 감소와 함께 나타났다. 이번 수정은 실제 거래비용과 청산 손익을 모델에 직접 제공하고, 짧은 보유·청산의 원인을 기록하며, 시간 주기와 학습 신호를 개별 비교하도록 만든다. 실제 A100 재학습 수익률은 아직 측정하지 않았다.

## Colab 실행

`ai_trader/grpo/colab_train_xlstm_return_priority.ipynb`의 설정 셀에서 다음 항목을 지정한다. 코드와 노트북을 함께 갱신한 후 런타임을 재시작하고 처음부터 실행한다.

```python
MODE = 'new'
RUN_NAME = 'scalping_v5_execution_cost_20260911_run01'
LOAD_POLICY = ''
EXPERIMENT = 'cost_observations'
SEED = 42
```

| EXPERIMENT | 기준 실험에서 달라지는 값 |
|---|---|
| `cost_observations` | 새 63차원 비용 관측, 나머지 학습·시간 설정은 기준값 |
| `timed_decisions` | `decision_interval_seconds=1.0`, `episode_duration_seconds=300.0` |
| `gae_only` | `group_advantage_coef=0.0` |
| `long_credit` | `lambda_gae=0.99` |
| `monte_carlo_credit` | `lambda_gae=1.0` |

기준값은 판단 간격 0(매 이벤트), 시간 상한 0(비활성), 그룹 계수 1.0, lambda 0.95이다. 후보값은 최적값으로 확정한 수치가 아니다. 각 비교 실험에는 새 실행 폴더를 사용한다. 설정은 `colab_config.json`에 기록되고 기존 `--config` 경로로 학습기에 전달된다. 직접 CLI에서는 `--execution_observations`, `--decision_interval_seconds`, `--episode_duration_seconds`, `--group_advantage_coef`, `--lambda_gae`, `--training_seed`를 사용할 수 있다.

새 학습은 Python·NumPy·PyTorch와 각 학습 환경에 seed를 지정한다. 같은 데이터·자원 설정의 새 실행끼리 비교한다. 기존 resume는 optimizer·진행도와 저장된 설정을 복원하지만 중단 시점의 모든 난수 생성기 상태를 복원하는 기능은 이번 변경에 포함하지 않는다.

## 관측 스키마 v4

시장 27 + 계좌 11 + 실행 상태 10 + 단계 상태 15 = **63차원**이다. 마지막 15개 단계 필드 위치를 보존한다. v2(42차원)·v3(53차원) 체크포인트는 원래 관측과 시간 설정으로 재개·백테스트할 수 있다. 새 비용 관측으로 전환하려면 새 학습이 필요하다.

| 실행 필드 | 정의 |
|---|---|
| `spread_bps` | 최우선 매수·매도호가 차이 / 중간호가 × 10,000 |
| `quoted_round_trip_cost_bps` | 현재 최우선호가·슬리피지·수수료 기준 즉시 왕복 손실 비율 × 10,000 |
| `liquidation_return` | 잔여 포지션의 예상 순청산 대금 / 잔여 원가(배분 진입 수수료 포함) − 1 |
| `breakeven_return` | 잔여 원가 / 예상 순청산 대금 − 1 |
| `buy_depth_ratio` | 현재 사용 가능한 매도호가 금액의 진입 예산 대비 비율, 0~1 |
| `sell_depth_ratio` | 현재 사용 가능한 매수호가 수량의 보유 수량 대비 비율, 0~1 |
| `quote_age_fraction` | 호가 경과시간 / 허용 경과시간. 오래된 호가는 1을 초과할 수 있음 |
| `quote_timestamp_known` | 호가 시각 제공 여부, 0 또는 1 |
| `episode_elapsed_fraction` | 경과시간 / 설정된 에피소드 기간. 기간 비활성 시 보유시간 제한을 분모로 사용 |
| `remaining_time_fraction` | 활성화된 시간 상한까지 남은 비율. 시간 상한 비활성 시 0 |

수익률 필드는 소수 비율(0.01=1%), bps 필드는 bp 숫자 그대로다. 호가 시각이 없으면 age=0, known=0으로 구분한다. 허용 호가 나이가 0일 때 age fraction은 실제 나이가 양수인지의 지표다. 예상 청산 대금은 최우선 매도 실행가 기준 평가이며, 수량 전체의 체결을 보장하지 않는다. 깊이와 미청산 수량을 함께 확인한다.

추론에는 명시적인 `account_state`와 `execution_state`가 필요하다. 미래 호가를 사용하거나 누락된 상태를 평탄한 계좌로 채우지 않는다. 계좌·실행 이력이 없는 기존 원시 데이터 기반 BC는 v4에서 명확하게 거절한다.

## 시간과 체결

`episode_steps`는 정책 판단 횟수 상한이다. 시간 간격이 0이면 기존처럼 매 이벤트마다 판단한다. 양수이면 현재 판단 시각에 간격을 더한 시각 이상의 첫 시장 이벤트에서 다음 관측을 반환한다. 판단 사이 모든 원시 이벤트를 처리하며 주문·취소 지연, TTL, 부분 체결, 손절, 보유 기한을 유지한다. 그 구간의 NAV 변화를 하나의 정책 보상에 합산한다.

정책 횟수 상한, 시간 상한, 데이터 끝 중 먼저 도달한 조건으로 정책 구간이 끝난다. 이어서 실제 후속 이벤트를 `liquidation_max_steps`까지 처리한다. 불규칙한 이벤트로 판단 시점이 늦어지는 경우까지 고려해 샘플의 청산 여유를 확보한다. 하루 전체 데이터가 부족하면 데이터 끝에서 종료하고 실제 경과시간·잔여 수량·청산 종료 사유를 기록한다. 시간 간격과 시간 상한은 평가 서명에 포함되어, 다른 조건의 best 점수를 그대로 계승하지 않는다.

시간 실험에서는 더 많은 시장 행을 읽을 수 있다. Colab은 manifest의 최대 파일 길이를 사용해 추가 호스트 메모리를 보수적으로 추정하고, 실제 관측을 만드는 사전 점검을 수행한다. 기간과 판단 상한이 모두 실제 시간을 정확히 채운다는 뜻은 아니므로 `policy_duration_seconds`를 확인한다.

## 새 진단

최초 청산 원인을 `signal`, `stop_loss`, `max_holding`, `episode_end`로 보존한다. 재시도 주문이 최초 원인을 덮어쓰지 않는다. 원인별 완료 왕복 횟수·수량·순손익·수량 가중 보유시간, 1초 이하 청산 수량 비율과 왕복 비율을 기록한다.

실현 순손익은 다음 식으로 대조한다.

```text
순실현손익 = 보유 중 중간호가 변화 손익
           - 스프레드 비용 - 호가 깊이 비용 - 슬리피지/틱 반올림 비용
           - 매도 수량에 배분된 진입·청산 수수료와 세금
```

`pnl_attribution_residual`은 식의 잔차다. 부분 체결에는 평균 원가 비례 배분을 적용한다. 양쪽 호가가 없으면 중간호가 대신 해당 이벤트 현재가를 기준으로 분해하며 `attribution_reference_kind`로 기준을 명시한다. 잔여 포지션이 있으면 이미 실현한 손익의 매칭 비용과 전체 지불 비용은 다를 수 있다.

TensorBoard의 `train/learning_signal/...`에는 HOLD·BUY·SELL 및 flat·holding별 raw GAE, 그룹 항, 정규화 전후 advantage의 평균·표준편차·양수 비율과 critic 오차를 기록한다. 캐시된 배열을 사용해 추가 정책 추론을 하지 않는다. `train/episode_diagnostics/.../mean`은 이름에 total이 있더라도 에피소드 평균이며 `available_episodes`로 자료가 있는 에피소드 수를 구분한다.

검증 JSON과 로그의 `evaluation_outcome`은 `no_trade`, `profitable`, `nonprofitable`, `incomplete_liquidation`, `unknown_execution`을 구분한다. `profitable_with_trades`는 순수익이 양수이고 실제 체결이 있으며 모든 에피소드가 청산됐는지 표시한다. 기존 `selection eligible`은 수익성 판정과 별개다.

## 검증과 해석

최종 전체 회귀 테스트: **386 passed in 59.14s**. `git diff --check` 통과. 실행 명령은 `.venv64/Scripts/python.exe -B -m pytest tests -o addopts='' -q`이며, 임시 결과와 pytest 캐시는 `.test_artifacts` 아래에 분리했다. 로컬 CUDA 검증 장치는 RTX 2080 Ti, PyTorch 2.8.0+cu126이다.

관측·스키마 호환성, 실시간 상태 누락, 거래비용 분해, 시간 간격 안의 손절·주문 만료, 불규칙 틱의 청산 여유, 검증 집계, Colab의 5가지 실험·CLI·구버전 복원에 회귀 테스트를 추가했다. 비용을 넘는 지연 수익과 손실 신호에 대해 한 번의 정책 업데이트가 매수 확률을 각각 올리고 내리는 작은 합성 검증도 포함한다.

실제 63차원 NPZ 환경에서 시간 기준 수집 → 캐시된 likelihood·value 대조 → 정책 업데이트 → 직렬·배치 검증을 CPU와 로컬 CUDA에서 실행했다. 이는 회계·학습 경로의 동작 확인이며 A100 전체 학습의 속도나 실제 시장 수익성을 입증하지 않는다.

새 실행의 판단 기준은 동일한 날짜·종목·시작 시점·평가 기간·비용에서 거래를 수행하면서 검증 순수익률이 무거래 기준 0%를 넘는지다. 시간 설정을 바꿨다면 기존 모델도 같은 평가 조건에서 재평가해야 한다. 평가 날짜를 기준으로 조정하고 테스트 날짜는 최종 평가에만 사용한다.
