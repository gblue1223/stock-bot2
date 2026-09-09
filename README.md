# Stock Bot — xLSTM 스캘핑 연구·리플레이

원본 틱 데이터로 매수·보유·매도 정책을 학습하고, 비용과 체결 제약을 포함한 시뮬레이터에서 평가하는 프로젝트입니다. 현재 실행 경로는 xLSTM + GRPO입니다. 이 저장소는 주문 의사결정과 리플레이를 제공하며 브로커에 실주문을 전송하지 않습니다.

## 실행

Python 3.10 이상을 사용합니다. 현재 회귀 검사는 Python 3.12 / PyTorch 2.8 CPU에서 실행했습니다. 아래 명령은 프로젝트 루트에서 해당 가상환경을 활성화한 후 실행합니다.

```powershell
python -m pip install -r requirements.txt
python -m pytest --basetemp .test_artifacts/pytest-manual
python -m ai_trader.grpo.train_xlstm --config config/scalping_v3.example.json
```

예제 설정은 비교적 작은 모델과 120틱 관측으로 시작합니다. 최적 수익을 보장하는 설정은 아닙니다. `seq_len`과 `episode_steps`는 이벤트 개수이고, `max_holding_seconds`는 실제 초입니다. 기존 기본값처럼 3000틱의 겹치는 관측창을 많이 수집하면 RAM 사용량이 커지므로 모델·배치·rollout 크기를 함께 조정해야 합니다.

훈련은 거래일을 시간순으로 train/validation/test에 분리합니다(기본 60/20/20, 최소 3일). 설정의 `train_end_date`, `validation_end_date`로 경계를 지정하거나 `embargo_dates`로 경계 뒤 거래일을 제외할 수 있습니다. 한 에피소드는 하나의 종목·거래일 안에 있으므로 관측창과 보유 구간이 다른 분할로 넘어가지 않습니다.

업데이트된 가중치를 고정 검증 시드로 평가한 뒤 `checkpoint_best.pt`를 선택합니다. 마지막에 그 가중치로만 test를 평가합니다. 결과에는 합성 호가 사용 여부와 미청산 수량도 포함됩니다.

출력 파일:

- `training_config.json`, `date_splits.json`: 실제 설정과 날짜 분할
- `checkpoints/checkpoint_best.pt`: validation 순수익으로 선택한 가중치
- `evaluation_report.json`: 학습·검증·테스트 결과
- `scalping_xlstm_model.pt`: 선택된 best와 동일한 최종 모델

`--load_policy`만 지정하면 호환 가중치로 새 fine-tuning을 시작하고 optimizer와 진행 카운터는 초기화합니다. `--load_policy ... --resume`를 함께 지정하면 optimizer·iteration·누적 timestep을 복원합니다. 재개할 때 `total_timesteps`는 추가량이 아니라 누적 목표이며, 저장된 진행량보다 커야 합니다. 두 경로 모두 관측 스키마와 기존 학습 날짜가 새 holdout을 오염시키는지 검사합니다.

## Colab A100 학습

`ai_trader/grpo/colab_train_xlstm.ipynb`를 사용합니다. 수정된 프로젝트 코드가 포함된 Git revision 또는 업로드한 소스 폴더가 필요합니다. 노트북만 교체하고 이전 코드를 clone하면 수정된 학습기를 사용할 수 없습니다.

관측 1024틱, 에피소드 300틱, CNN/mLSTM/FC 64/128/256으로 시작합니다. A100 40GB의 배치 상한은 64, 80GB는 128이며 실제 역전파 사전 점검에서 메모리가 부족하면 줄입니다. 회당 rollout은 16개이고 여유 RAM 16GiB 미만에서는 8개로 축소합니다. worker는 CPU와 RAM에 맞춰 최대 4/8개, 캐시는 worker당 128MiB로 제한합니다. GAE 및 TensorBoard 추론도 `batch_size` 이하로 나누어 GPU에 전송합니다.

`MODE='new'`와 새 `RUN_NAME`이 기본입니다. 재개는 `MODE='resume'`와 `LOAD_POLICY`를 명시하며 관측·체결·학습 설정을 복원합니다. 초기 목표는 100,000틱이고 재개할 때는 누적 목표를 늘립니다. `ENABLE_TF32`는 사전 점검과 실제 학습 프로세스에 함께 적용하며 AMP/BF16은 사용하지 않습니다.

Drive에 `colab_config.json`, 소스·manifest 해시와 GPU 점검 결과가 담긴 `colab_run.json`, 학습 로그와 회당 체크포인트를 저장합니다. 끝의 평가 셀에서 순수익·실현손익·합성 체결·미청산 수량을 확인할 수 있고, 선택적인 체결 스트레스 검사는 validation에만 적용합니다. 실제 A100의 최대 메모리와 학습 속도는 Colab 사전 점검 및 실제 학습에서 확인해야 합니다.

## 무거래 원인 진단

검증 수익률과 거래 수가 모두 0이면 기존 체크포인트를 같은 검증 경로에서 다시 실행합니다. 재학습은 필요하지 않습니다. 진단 명령은 체크포인트의 관측 규격·날짜 분할·비용·체결 설정을 복원하며, 기본 에피소드 수와 시드도 저장된 값을 사용합니다.

```powershell
python -m ai_trader.grpo.diagnose_xlstm --checkpoint models/scalping_v3/checkpoints/checkpoint_best.pt --split validation --device auto --output models/scalping_v3/validation_diagnostics.json
```

Colab에서는 수정된 프로젝트 소스를 반영한 뒤 프로젝트 루트에서 실행합니다. 예를 들어 이 실행의 선택 모델은 다음 셀로 진단할 수 있습니다.

```python
import json
import subprocess
import sys
from pathlib import Path

run_dir = Path('/content/drive/MyDrive/ColabData/stockbot/models/scalping_v3_a100_astral_depth_run01')
diagnostic_path = run_dir / 'validation_diagnostics.json'
subprocess.run([
    sys.executable, '-m', 'ai_trader.grpo.diagnose_xlstm',
    '--checkpoint', str(run_dir / 'checkpoints' / 'checkpoint_best.pt'),
    '--split', 'validation', '--episodes', '8', '--seed', '42', '--device', 'cuda',
    '--output', str(diagnostic_path),
], check=True)
report = json.loads(diagnostic_path.read_text(encoding='utf-8'))
diagnostics = report['metrics']['diagnostics']
print(json.dumps({key: value for key, value in diagnostics.items() if key != 'episodes'},
                 ensure_ascii=False, indent=2))
```

런타임 재시작으로 추출 데이터 경로가 바뀌었다면 같은 데이터의 새 위치를 `--extracted-dir /content/현재_추출_폴더`로 지정합니다. DB 기반 데이터는 `--db-path`를 사용합니다. 동일한 경로 재현에는 원래 데이터 내용과 에피소드 목록 순서도 같아야 합니다. 새 출력 파일명을 사용하며, 기존 파일이나 `evaluation_report.json`은 덮어쓰지 않습니다.

`metrics.diagnostics`를 다음 순서로 읽습니다. 일반 학습의 새 `evaluation_report.json`에도 검증·테스트 각각 `diagnostics`가 추가됩니다.

| 확인할 값 | 해석 |
| --- | --- |
| `action_counts.buy == 0` | 평가 경로에서 정책이 매수를 선택하지 않음. `hold`, `sell` 횟수와 함께 확인 |
| `action_counts.buy > 0`, `submitted_orders == 0` | 매수는 선택했지만 주문 생성 조건에서 차단됨. `buy_action_outcomes` 확인 |
| `submitted_orders > 0`, `filled_quantity == 0` | 주문을 제출했으나 체결되지 않음. `expired_orders`, `cancelled_orders`, `execution_blocked_checks` 확인 |
| `filled_quantity > 0`, `mean_num_trades == 0` | 체결은 있었지만 매도 청산 기록이 없음. `metrics.max_open_quantity`, 미청산 에피소드 확인 |

`mean_action_probabilities`와 `max_action_probabilities`는 행동 마스크 적용 후 확률입니다. xLSTM은 실제 행동 선택과 같은 한 번의 forward에서 이를 수집합니다. 예를 들어 관망 확률이 항상 매수 확률보다 조금만 높아도 deterministic 평가에서는 매수가 0회일 수 있습니다. 이 확률은 수익을 낼 확률이 아닙니다. 확률 수집을 지원하지 않는 정책이나 통계가 없는 환경은 해당 값을 `null`로 표시합니다.

`buy_action_outcomes`는 매수 선택마다 한 가지 결과를 기록합니다. `insufficient_budget_for_one_share`는 가용 현금과 단계별 배정 한도로 한 주도 주문할 수 없는 경우, `risk_exit_active`는 손절·보유시간·에피소드 종료 청산 중인 경우입니다. `max_stages`, `max_trades`, `pending_buy`는 각각 진입 단계 한도·거래 수 한도·기존 매수 주문에 의한 차단입니다. 기존 조건 검사 순서를 따르므로 미체결 매수 주문이 한도를 채우면 `max_stages`로 집계될 수 있습니다.

`execution_blocked_checks`는 오래된 호가(`stale_quote`), 주문 지연(`order_latency`), 호가 부족(`no_displayed_depth`, `displayed_depth_exhausted`), 현금/재고 부족 등의 검사 횟수입니다. 주문별·이벤트별 횟수이므로 같은 주문이 반복 집계될 수 있으며, 만료·취소가 먼저 처리된 이벤트는 차단 검사에 포함되지 않습니다. `fill_ratio`는 전체 체결 수량 / 전체 제출 수량이고, 매수와 매도를 모두 포함합니다.

`diagnostics.episodes`에는 에피소드별 시드, 종목·날짜·시작 인덱스, 행동·주문 통계가 남습니다. `checkpoint_iteration`과 `checkpoint_total_timesteps`로 실제 진단한 모델 시점도 확인합니다. 무거래의 원인을 좁힐 때는 우선 저장된 validation 경로를 사용합니다.

## 체결 시뮬레이터

`ai_trader/grpo/environments/execution.py`가 주문과 체결을 담당하고, 환경이 현금·수량·FIFO 포지션·수수료를 정산합니다.

| 설정 | 기본값 | 의미 |
| --- | ---: | --- |
| initial_cash | 1,000,000 | 초기 원화 자금 |
| max_holding_seconds | 300 | 모든 포지션의 청산 주문을 요청하는 보유시간 |
| stop_loss_pct | 2 | 손절 기준 퍼센트 |
| order_latency_ms | 100 | 주문이 시장에 도착하기까지의 지연 |
| cancel_latency_ms | 50 | 취소 요청이 반영되기까지의 지연 |
| order_ttl_seconds | 2 | 잔여 주문의 만료시간 |
| slippage_bps | 2 | 표시 호가에 추가 적용하는 불리한 가격 조정 |
| spread_bps | 10 | 실제 호가가 없을 때만 가정하는 왕복 스프레드 |
| fallback_depth | 100 | 합성 호가 사용 시 이벤트마다 가정하는 주식 수 |
| require_order_book | false | true이면 실제 호가가 없는 데이터 거부 |
| max_quote_age_seconds | 1 | 호가 시각이 있는 데이터에서 허용하는 호가 나이 |
| tick_size | 0 | 0이면 추가 반올림 없음; 양수면 명시한 틱으로 불리하게 반올림 |

체결 설정은 JSON의 `execution_config` 안에 둡니다. 위 값은 시뮬레이션 가정이며, 실제 증권사의 지연·세금·수수료를 대신하지 않습니다. 매수/매도 수수료와 세금은 `transaction_cost_rate`, `buy_tax_rate`, `sell_tax_rate`로 분리합니다.

체결기는 다음을 보장합니다.

- 주문 제출 시 이미 본 이벤트로 체결하지 않습니다. 지연 이후의 이벤트에서 매수는 ask, 매도는 bid를 사용합니다.
- 여러 호가를 가격순으로 소진하고 실제 가능한 수량만 부분체결합니다. 같은 계좌가 소비한 잔량은 반복 관측했다고 복원하지 않습니다.
- 잔여 주문은 대기·취소·만료될 수 있습니다. 취소가 도착하기 전에 체결될 수 있는 경합도 처리합니다.
- 정수 주식 수, 가용 현금과 재고를 적용해 음수 현금·공매도를 막습니다.
- 지정가는 반대편 호가와 실제로 맞는 경우만 체결합니다. 최근 체결가가 지정가에 닿았다는 이유로 수동 대기열 체결을 추정하지 않습니다.
- 호가 시각이 보존된 데이터에서는 오래된 호가로 체결하지 않습니다.
- 보유시간 만료는 틱 사이의 타이머로 주문을 예약합니다. 시장 이벤트가 없거나 호가 잔량이 부족하면 정해진 초에 체결을 보장할 수 없으며, 실제 경과시간과 미청산 수량을 그대로 보고합니다.

에피소드 종료 때도 마지막 이벤트의 가격과 비용을 반영합니다. 지연·잔량 때문에 청산할 수 없으면 가상의 체결을 만들지 않습니다. `liquidation_complete=false`, `open_quantity`, `realized_net_pnl`, `unrealized_net_pnl`을 함께 확인해야 합니다.

보상은 매 스텝 청산 평가액(NAV)의 변화 / 초기 자금 × 100입니다. 현금과 보유 주식을 매도 호가·매도 비용 기준으로 평가하며, 보상 합과 보고 `net_return`이 일치합니다. 이전의 상승 보너스·승리 보너스·미거래 페널티는 무시되고 경고를 남깁니다. 거래하지 않으면 순수익과 보상은 0입니다.

## 별도 백테스트와 비용 스트레스

새 학습 체크포인트에는 관측 스키마, 비용·체결 설정과 날짜 분할이 저장됩니다. 별도 백테스트가 이를 그대로 읽습니다.

```powershell
python -m ai_trader.grpo.backtest --policy models/scalping_v3/checkpoints/checkpoint_best.pt --output models/scalping_v3/replay_test.json
python -m ai_trader.grpo.backtest --policy models/scalping_v3/checkpoints/checkpoint_best.pt --partition validation --order-latency-ms 300 --slippage-bps 5 --output models/scalping_v3/replay_stress.json
```

`--extracted-dir`로 데이터 경로를 이동할 수 있고, `--episodes`, `--seed`, `--spread-bps`, `--require-order-book`을 지원합니다. 모델·조건을 고르는 비교에는 validation을 사용하고 최종 test를 반복적인 모델 선택에 사용하지 않습니다.

지표의 단위는 다음과 같습니다.

- `mean_net_return`: 초기 자금 대비 에피소드별 비용 차감 NAV 수익률의 평균(%)
- `max_drawdown`: 에피소드별 NAV 최대낙폭의 최댓값(%); 전체 기간을 이어 붙인 계좌 낙폭은 아님
- `mean_realized_net_pnl`: 실현손익 평균(원)
- `synthetic_execution_episodes`, `execution_models`: 합성 호가 사용 여부
- `incomplete_liquidation_episodes`, `max_open_quantity`: 종료 시 잔여 재고
- 환경의 `sharpe_ratio`: 연율화하지 않은 이벤트별 NAV 변화 기준; 일별 계좌 수익률 Sharpe와 다름

현재 시뮬레이터는 공개된 호가 잔량을 사용하는 리플레이입니다. 자신의 주문이 이후 시장 경로를 바꾸는 시장충격이나 수동 지정가 대기열 순위는 추정하지 않습니다. 보유 주식의 평가액도 전체 수량이 즉시 그 가격에 매도된다는 증거는 아닙니다.

## 데이터 생성과 기존 캐시

새 추출은 숫자 시간·동일 시각의 이벤트 순번을 보존하며, 모델 특징과 원화 가격/주식 수 실행 배열을 분리합니다.

```powershell
python -m ai_trader.grpo.data_extractor --db "D:/path/to/datasets_raw.duckdb" --output_dir data/extracted_episodes_v2 --seq_len 120 --features 27 --max_steps 300 --price-unit krw
```

분할 매수 단계 수는 학습 CLI의 `--max_stages`(또는 `--max-stages`)로 설정합니다. 허용 범위는 1~5이며 기본값은 **1**입니다. 1이면 초기 자금과 가용 현금 한도에서 한 번 매수하고 매도 신호에 보유 전량을 주문합니다. 5이면 매수 주문당 초기 자금의 최대 1/5을 사용하고, 매도 신호에 가장 오래된 단계부터 정리합니다. 부분 체결·수수료·호가 수량에 따라 실제 투자 비중과 청산 시점은 달라집니다.

```powershell
python -m ai_trader.grpo.train_xlstm --extracted_dir data/extracted_episodes_v2 --max_stages 1 --output_dir models/scalping_single
python -m ai_trader.grpo.train_xlstm --extracted_dir data/extracted_episodes_v2 --max_stages 5 --output_dir models/scalping_five
```

CLI 값은 JSON 설정의 `max_stages`보다 우선합니다. `--max_trades_per_episode`는 별도의 거래 횟수 제한입니다. 관측값은 5개 슬롯을 예약하고 미사용 슬롯을 0으로 채우므로 27개 시장 특징 기준 42차원을 유지합니다. 분할 수는 체크포인트 관측 스키마와 학습 설정에 기록되며, 추론·백테스트는 저장된 값을 복원합니다. 기존 v2 5분할 체크포인트는 그대로 추론할 수 있지만, CLI로 이어서 학습하려면 `--max_stages 5`를 명시해야 합니다. 다른 분할 수의 체크포인트 로딩은 거부되므로 변경 시 새 학습을 시작하세요. 원시 에피소드는 다시 추출할 필요가 없습니다.

Colab 노트북의 새 학습도 기본 1단계이며 설정 셀의 `MAX_STAGES`(기존 노트북은 `max_stages`)로 변경합니다. A100 노트북의 `resume`/`finetune`은 체크포인트의 분할 수를 보존합니다.

호가를 보존한 원본 DB에서 A100 노트북용 오전 9~11시 데이터를 직접 만들 수도 있습니다. 대기금액 컬럼이 없으면 원본 호가×수량으로 계산하고, 실제 호가·수량은 별도 실행 배열로 저장합니다. 중간 DB 전체를 복제할 필요가 없습니다.

```powershell
python -m ai_trader.grpo.data_extractor --db "D:/path/to/raw_with_orderbook.duckdb" --output_dir data/extracted_episodes_v2 --seq_len 1024 --features 27 --max_steps 300 --price-unit krw --time-start 90000000 --time-end 110000000 --workers 4 --threads 2 --memory-limit 3GB --compression-level 1
```

`--workers`는 병렬 에피소드 처리 수입니다. `--executor process`는 Python 문자열 처리도 여러 CPU 프로세스로 나눕니다. 이때 DuckDB의 `--memory-limit`은 프로세스마다 적용되므로, 예를 들어 `--workers 4 --executor process --memory-limit 1GB`는 DB 버퍼만 최대 약 4GB이며 worker별 Pandas/NumPy 배열 메모리가 추가로 필요합니다. 기본 thread 모드에서는 DB 버퍼를 공유합니다. `--compression-level 1`은 압축 시간을 줄이며 NPZ 형식은 같습니다. `seq_len`은 추출 대상의 최소 길이를 결정하고, `max_steps`는 실행 설정 기록용이며 파일에는 해당 시간 구간 전체를 저장합니다.

중단 시 같은 명령에 `--resume`을 붙이면 `completed.jsonl`에 기록된 파일부터 이어갑니다. 원본 경로·크기·수정 시각과 추출 설정이 같아야 하며, 완성된 `manifest.json`이 있는 출력은 덮어쓰지 않습니다. `progress.json`에서 진행량을 확인할 수 있습니다. 원본에 호가 갱신 시각이 없으면 이를 임의로 만들어 저장하지 않습니다.

소스 DB는 명시적 특징 스키마의 컬럼을 포함해야 합니다. `scripts/data/generate_datasets.py`로 원본 CSV의 체결·호가·거래원 이벤트를 합칠 수 있습니다. `--price-unit`은 원본 가격 단위를 반드시 지정합니다. 현재 `extract_260811.py`가 만드는 데이터는 KRW이며, 과거에 만들어진 백만원 단위 DB에는 `million_krw`가 필요합니다. 서로 다른 가격 단위를 섞은 DB는 먼저 단위를 통일해야 합니다.

추출기는 이미 있는 manifest나 에피소드 파일을 덮어쓰지 않습니다. 새 출력 디렉터리를 사용합니다. `scripts/data/normalize_datasets.py`는 이제 결정적인 파생 피처를 생성하고 원본 숫자를 보존합니다. 하루 전체 mean/std 정규화는 하지 않습니다.

NPZ 계약:

- `features`: manifest에 적힌 순서의 원본 모델 피처
- `metadata`: 종목코드·날짜·HHMMSSmmm 시각, shape `(N, 3)`
- `execution_last_price`: 원화 가격, shape `(N,)`
- `execution_bid_prices / execution_ask_prices`: 원화 호가, shape `(N, levels)`
- `execution_bid_sizes / execution_ask_sizes`: 주식 수, 같은 shape
- `execution_quote_timestamp`: 보존된 호가 갱신 시각(자정 이후 초), 있을 때만 사용
- manifest metadata: `schema_version=2`, `feature_columns`, `price_unit`, `execution_price_unit=krw`, `feature_transform=raw`

호가가 없는 경우 이를 대기금액만으로 복원하지 않습니다. 합성 스프레드/잔량 모드를 쓰고 결과를 표시합니다. 실제 호가 검증에는 원본 호가 가격·수량·시각을 보존한 새 추출 데이터가 필요합니다.

기존 `data/extracted_episodes`는 원본 파일 변경 없이 로드 시 안정 정렬합니다. 알려진 기존 27개 컬럼 순서는 명시적 legacy adapter로 백만원 가격을 원화로 환산합니다. 기존 파일에 없던 호가나 수집 당시 이름 인코딩 정보는 복원하지 않습니다. 실제 운영용 데이터를 준비할 때에는 현재 파생 피처 규칙과 원본 호가를 사용해 재추출하는 것이 필요합니다.

## 관측과 추론

`lib/observations.py`의 `ObservationBuilder`를 학습·사전학습·추론에서 공유합니다. 현재까지의 관측창에서만 log 변환 및 rolling 통계를 계산합니다. 미래 데이터를 넣어 통계를 만들지 않습니다.

체크포인트 스키마가 특징 순서·가격 단위·관측 길이·정규화·최대 보유시간·15개 포지션 피처를 고정합니다. 구모델에 이 정보가 없거나 규격이 다르면 로드를 거부합니다. 기존 가중치를 곧바로 실전에 재사용하지 말고 수정된 파이프라인으로 재학습해야 합니다.

```python
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import (
    GRPOInferenceE2EXLSTM, EnhancedGRPOInferenceXLSTM, Position,
)

base = GRPOInferenceE2EXLSTM("models/scalping_v3/scalping_xlstm_model.pt", device="cpu")
engine = EnhancedGRPOInferenceXLSTM(base)
# raw_window: 현재까지 도착한 원본 피처; checkpoint의 순서와 단위를 따릅니다.
# position의 가격은 원화, entry_time/current_time_seconds는 자정 이후 초입니다.
action, probability, info = engine.predict(
    raw_window,
    current_position=None,
    current_price=latest_price_krw,
    current_time_seconds=latest_seconds,
    feature_columns=base.observation_schema["feature_columns"],
)
```

`probability`는 행동 선택 확률이며 거래 수익 확률로 보정된 값은 아닙니다. 실시간 래퍼는 최대 보유시간, 손절, 익절, 침체 청산, 활성 상태를 유지하는 트레일링 스탑을 적용합니다. 리플레이 기본 리스크는 최대 보유시간과 손절을 사용합니다. 실시간 추가 규칙을 사용할 때에는 해당 규칙까지 포함한 별도 검증이 필요합니다.

반환값은 주문 의도입니다. `risk_exit_all=true`이면 전체 포지션 청산 의도이며, 실제 주문·체결 확인 없이 포지션을 삭제해서는 안 됩니다. 브로커 접수/부분체결/취소/잔고 대사는 외부 실행 계층에서 연결해야 합니다.

## 사전학습

BC는 호환되는 스키마와 학습 날짜 이력이 있는 GRU teacher만 받습니다. 현재 5분할 포지션 규격은 시장 피처 + 15채널입니다. 데이터의 train 날짜만 사용하고 teacher의 학습·선택 날짜가 holdout에 섞이면 거부합니다. fast 경로도 동일한 forward/관측을 사용하며 지원되지 않던 TorchScript 단계를 제거했습니다.

```powershell
python -m ai_trader.grpo.pretrain_behavior_cloning --teacher_policy models/teacher.pt --extracted_dir data/extracted_episodes_v2 --output_path models/pretrain/xlstm.pt
```

스키마나 날짜 이력을 모르는 구 teacher는 사용할 수 없습니다. BC 포지션 예제는 보이는 과거 가격에서 만든 가상 상태이며, 실제 체결 데이터로 만든 매매 성과가 아닙니다.

## 검증 범위

회귀 검사는 시간 역행·미래 변경 불변성·원본 가격·마지막 틱 정산·보상/NAV 일치·지연/잔량/취소/만료·리스크 청산·관측 호환성·정책 확률 재현·날짜 분할·best 저장·BC 역전파·작은 CPU 학습/검증/백테스트를 포함합니다.

기존 NPZ 3개에서도 정렬된 관측값과 원화 환산을 확인했습니다. 전체 기존 데이터 재생성, 장시간 모델 재학습, 실거래 수익성 또는 현재 xLSTM의 2.87ms 추론은 이번 변경으로 검증한 항목이 아닙니다.
