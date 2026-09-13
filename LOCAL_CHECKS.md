# 로컬 자동 검사

회귀 테스트, 동일 rollout 학습률 비교, 원본과 업데이트한 후보들의 validation 수익 비교를 한 명령으로 순차 실행한다. 실제 test 날짜 평가는 실행하지 않으며, 장기 학습이나 모델 배포도 포함하지 않는다.

## 현재 로컬 환경

2026-09-13 확인: Ryzen 9 9950X(16코어/32스레드), RAM 125.6 GiB, RTX 2080 Ti 11 GiB, Python 3.12.10, PyTorch 2.8.0+cu126. 확인 당시 사용 가능한 RAM은 약 13.5 GiB, VRAM은 약 7.7 GiB였다. 총 용량과 실제 여유 용량은 구분해야 한다.

Colab 데이터와 `data/extracted_episodes_v2/manifest.json`의 SHA256은 일치했다. 12,899개 에피소드가 기록돼 있다. 실행 시 manifest 해시와 각 NPZ의 경로·존재·기록된 크기를 확인하며, 체크포인트의 관측 스키마와 날짜 분할을 유지한다. NPZ 내용 전체의 암호학적 해시를 다시 계산하는 검사는 아니다.

로컬 FP32 진단은 가능하다. RTX 2080 Ti의 Turing 하드웨어는 A100의 TF32 연산을 재현하지 못하며, PyTorch 버전 차이도 있으므로 Colab과 숫자가 같다고 가정하지 않는다. [NVIDIA Ampere 안내](https://docs.nvidia.com/cuda/ampere-tuning-guide/)

## 실행

저장소 루트에서 64비트 가상환경을 사용한다.

```powershell
.\.venv64\Scripts\python.exe -X utf8 -u -m ai_trader.grpo.local_checks `
  --checkpoint 'G:\내 드라이브\ColabData\stockbot\models\scalping_v7_update_checks_20260912_run01\checkpoints\checkpoint_iter19.pt' `
  --extracted-dir '.\data\extracted_episodes_v2'
```

또는 PowerShell 실행기를 사용한다.

```powershell
.\scripts\run_local_checks.ps1 -Checkpoint 'G:\내 드라이브\ColabData\stockbot\models\scalping_v7_update_checks_20260912_run01\checkpoints\checkpoint_iter19.pt'
```

기본 설정은 worker 2개, 모델당 validation 8회(원본과 세 후보 총 32회), LR `3e-5 / 1e-5 / 3e-6`, 단계당 시간 제한 7,200초다. 평가 시드는 원본에 저장된 `evaluation_seed`를 공통으로 사용한다. 자식 프로세스의 OMP/MKL/OpenBLAS 연산 스레드를 1개로 설정하고 GPU 작업은 순차 실행한다. 모델 크기·관측 길이·업데이트 배치·보상·비용·KL 기준은 메모리에 맞추기 위해 자동으로 축소하거나 완화하지 않는다.

실행 전 및 단계 사이에 여유 RAM 8 GiB, 여유 VRAM 4 GiB를 확인한다. 직전 단계의 메모리 반환이 지연되면 최대 60초 기다린 뒤 재확인한다. 기준을 낮추거나 다른 응용 프로그램을 종료하지 않는다. 이 기준은 실제 최대 메모리 사용량의 보장이 아니다. 대기 후에도 자원이 부족하거나 실제 OOM·검사 실패가 발생하면 실패 로그를 남기고 후속 단계를 중단한다.

## 실행 순서와 결과

| 단계 | 작업 | 결과 파일 |
|---|---|---|
| 입력 검사 | 저장된 정책/Adam, manifest, 파일 크기, 관측·날짜 분할, 실행 자원 확인 | `capture_config.json`, `summary.json` |
| regression | 전체 pytest 회귀 검사 | `regression.log`, `pytest.xml` |
| capture | FP32로 다음 실제 rollout 묶음을 한 번 수집·검사 | `capture.log`, `fixed_rollout.pt` |
| entry_credit | CPU에서 BUY 결정·실제 주문 손익·학습 신호 연결 | `entry_credit.log`, `entry_credit.json` |
| compare | 같은 정책·Adam·RNG·rollout으로 세 LR을 독립 실행하고 실제 업데이트 결과 저장 | `compare.log`, `lr_comparison.json`, `candidates/*.pt` |
| validation | **원본과 저장된 세 후보**를 같은 validation 날짜·시드·에피소드·비용으로 평가 | `validation.log`, `validation_comparison.json`, `validation_comparison_evaluations/*.json` |

결과는 기본적으로 `.test_artifacts/local_checks/<날짜_고유번호>/` 아래에 저장한다. 입력 파일과 기존 결과 폴더를 덮어쓰지 않는다. `summary.json`에는 사용 장치, 입력 변경 내역, 실행 명령, 단계별 성공/실패·시간, KL과 validation 요약이 기록된다.

`validation_comparison.json`에는 원본과 각 후보의 순수익률·거래 수·수수료·낙폭·미청산 여부와 원본 대비 수익률 차이를 기록한다. 실제 선택된 에피소드의 종목·날짜·시작 위치·시드까지 일치하는지 확인한다. 모두 무거래이면 수익 개선으로 판정하지 않는다. 모델당 validation 8회는 짧은 비교이며 기존 64회 검증을 대체하지 않는다.

후보는 같은 rollout 묶음에 한 번의 GRPO 업데이트를 적용한 결과다. 배치별 optimizer step 190회는 새로운 rollout으로 학습을 190회 반복했다는 뜻이 아니다. 저장된 후보는 평가 전용으로, 원본 설정·날짜 분할과 실제 업데이트한 가중치를 보관하며 재개용 Adam 상태나 원본의 best 성적을 포함하지 않는다.

`diagnose_update_lr`를 직접 호출할 때는 `--source-checkpoint`와 `--export-dir`를 함께 지정하면 후보를 저장한다. 기존 Colab 진단 명령처럼 두 옵션을 생략하면 KL 비교만 수행한다. 로컬 자동 실행기는 두 옵션을 자동으로 전달하므로 기존 PowerShell 명령을 그대로 사용할 수 있다.

후보가 저장된 후 평가 횟수만 늘릴 때는 `compare_validation`의 `--comparison-report`에 기존 `lr_comparison.json`을 지정한다. `--checkpoint`는 같은 원본, `--extracted-dir`는 같은 데이터로 유지하고 `--episodes 64 --seed 42 --device cuda`와 새로운 `--output` 경로를 지정하면 업데이트를 다시 실행하지 않고 평가를 반복할 수 있다. 시드 42는 위 예시 원본의 저장된 평가 시드다.

LR의 KL 중단은 진단 결과로 기록한다. 예외·비정상 종료·시간 초과·likelihood 불일치 등은 다음 단계 실행을 차단한다. 중단 또는 시간 초과 시 이 실행기가 시작한 자식과 worker만 정리한다.

## 옵션

- `--plan-only`: 입력을 확인하고 계획을 저장하며 회귀·캡처·GPU 업데이트·validation을 실행하지 않는다.
- `--validation-episodes 64`: 같은 validation 날짜에서 원래 규모로 평가한다.
- `--num-workers 2`: 캡처 worker 수. Colab과 worker 수가 다르면 새로 수집한 rollout도 달라질 수 있다.
- `--bundle <기존 fixed_rollout.pt>`: 캡처를 생략한다. 원본 체크포인트의 정책·Adam·스키마·학습 및 환경 설정과 일치하는 번들만 허용한다. 로컬 경로·장치·worker·캐시 차이와 명시적인 정책 검사 활성화는 허용하고 기록한다.
- `--skip-regression`: 이미 통과한 코드에서 비교만 재시도할 때 회귀 단계를 명시적으로 생략한다.
- `--output-dir <새 폴더>`: 결과 폴더를 지정한다. 이미 존재하면 거부한다.
- `--stage-timeout 7200`: 출력이 없는 worker 정지를 포함해 단계별 시간을 제한한다.
- `--resource-wait-seconds 60`: 단계 사이 자원 회복을 기다리는 최대 시간. `0`이면 즉시 판정한다.

번들을 재사용할 때도 새 출력 폴더를 사용한다. Colab에서 `/content/stockbot_update_diagnostics/`에만 저장한 번들은 이 로컬 PC에 자동으로 나타나지 않는다. 그 파일을 같은 입력으로 재사용하려면 로컬에서 접근할 수 있는 경로에 있어야 한다. 번들을 지정하지 않으면 이 PC에서 새로운 rollout을 수집한다.

현재 자동화는 Windows에서 검증했으며, GPU별 성능이나 전체 데이터의 최적 수익을 보장하지 않는다. 로컬에서 선택한 후보를 Colab의 동일 조건 및 충분한 validation 규모에서 다시 확인할 수 있다.

## BUY 손익과 학습 신호 진단

새 코드로 수집한 rollout은 모든 BUY 결정의 햣제출 여부와 주문 ID를 보존한다. 실제로 청산이 끝난 주문의 순손익을 그 BUY 결정의 GAE·정규화 advantage와 연결하여, 수익 거래도 음수 학습 신호를 받는지 확인한다. 차단·미체결·부분 체결·미청산을 구분하며, 미청산 주문은 손익이 확정된 거래로 분류하지 않는다. 비용별 손익은 매도된 수량에 대한 중간 호가 변화·스프레드·호가 깊이·슬리피지·매칭된 수수료로 분해한다.

`entry_credit.json`의 `report.outcome_summary.profitable`에서 `sample_count`, `raw_gae_negative_fraction`, `normalized_advantage_negative_fraction`을 함께 확인한다. `report.action_summary.buy`는 즉시 보상·critic 값 변화·이후 TD 오차를 분리한다. `report.entries`와 `report.buy_decisions`는 개별 주문과 결정의 근거다. 돈 단위 PnL과 초기자산 대비 백분율 보상은 서로 다른 단위다. 정규화 advantage가 음수라는 이유만으로 보상 오류라고 판정할 수는 없다.

로컬 자동 실행에는 이 CPU 진단이 포함된다. 이미 저장한 번들만 분석하려면 다음을 실행한다. 정책 추론·optimizer 갱신·validation·test 평가는 실행하지 않는다.

```powershell
.\.venv64\Scripts\python.exe -X utf8 -m ai_trader.grpo.diagnose_entry_credit `
  --bundle '.test_artifacts/local_checks/<실행 폴더>/fixed_rollout.pt' `
  --output '.test_artifacts/entry_credit/<새 보고서 이름>.json'
```

이전 버전 번들은 주문 연결 정보가 없으므로 `entry_attribution_available=0`, `missing_episode_count>0`을 명시한다. 행동별 학습 신호는 분석하지만 거래별 손익을 추정해 채우지는 않는다.

Colab은 수정 코드를 반영한 뒤 기존 **7-2번 학습률 비교 셀**을 실행하면 `_entry_credit.json`도 자동 생성한다. **8번 정상 학습**에서는 매 반복마다 `checkpoints/diagnostics/entry_credit/iteration_*.json`과 TensorBoard `train/entry_credit/`에 기록한다. 기존 설정으로 resume할 수 있으며 새 하이퍼파라미터는 필요 없다. 이번 변경은 관측·체결·보상·GAE·정책 선택 규칙을 유지하며, 관측된 거래의 원인 분석을 추가한다.

### 실제 BUY 추적 결과: 2026-09-13

`.test_artifacts/entry_credit/20260913_run01`에서 원본 19회차 체크포인트로 학습용 rollout 16개를 다시 수집했다(313.5초). 3,028개 표본의 행동·보상·종료·log probability·critic 값·마스크가 이전 고정 번들과 모두 정확히 일치했다. 진단 기록 추가에 따른 행동/보상 변화 없이 기존 학습 신호에 주문별 근거를 연결했다. 전체 선택 likelihood 오차는 최대 `1.28746e-5`로 기존 허용값 `0.001` 이내였다. 가중치 업데이트와 validation/test는 실행하지 않았다.

- BUY 선택·제출 240건, 미체결 28건, 청산 완료 212건, 미청산 0건.
- 청산 완료 중 수익 3건·손실 209건. 수익 3건은 raw GAE와 정규화 advantage가 모두 양수였다. 이 표본에서 수익 거래를 음수 신호로 억제한 사례는 발견되지 않았지만, 3건으로 일반화할 수는 없다.
- 중간 호가 변화 손익은 양수 41건·음수 44건·변화 없음 127건. 보유 시간은 주문별 수량 가중값의 중앙값 4.794초, 평균 8.632초다. 정책의 SELL 신호로 청산된 주문은 207건, 에피소드 종료 청산은 5건이었다.

아래는 **16개 에피소드의 완료 거래 212건 합계**이며 초기자산은 에피소드마다 100만 원이다.

| 항목 | 합계(원) |
|---|---:|
| 중간 호가 변화 손익 | +1,320.00 |
| 스프레드 비용 | 268,095.00 |
| 호가 깊이 비용 | 22,570.00 |
| 슬리피지 비용 | 78,365.70 |
| 매칭된 수수료·세금 | 411,089.04 |
| 실현 순손익 | −778,799.74 |

가격 변화로 얻은 이익이 비용을 회수하지 못한 것이 이번 손실의 주된 원인이다. 주문별 손익 분해 잔차는 최대 `1.17e-10`원 이내였고, 에피소드 평균 순수익 보상은 `−4.8675%`였다. BUY advantage의 음수 비율 73.75%와 일치하는 손실 근거를 확인했다. 짧은 보유가 손실을 일으켰다거나 보유 시간을 늘리면 수익이 난다는 인과관계는 이 관측만으로 확정하지 않는다.

근거 파일: `entry_credit.json`(결정·주문별 상세), `findings.json`(집계 및 이전 번들과 일치 검사), `capture_result.json`(실행 상태), `fixed_rollout.pt`(동결 입력). 수정본 전체 회귀 **747개가 통과**했다(57.727초, 실패·skip 0개). 결과는 `.test_artifacts/entry_credit/full_tests.xml`에 저장했다.

## 원본·후보 수익 비교 결과: 2026-09-13

`20260913_profit_run01`에서 같은 원본 19회차 체크포인트와 앞서 저장한 3,028개 관측 번들을 사용했다. 전체 회귀 테스트 **708개가 통과(57.46초)**했고, 업데이트 후보 3개를 실제로 저장한 뒤 원본을 포함한 4개 모델을 각각 같은 8개 validation 에피소드에서 평가했다. 전체 실행은 약 41분이었다(후보 생성·KL 비교 1,577.1초, validation 815.7초).

| 모델 | 완료한 optimizer step | 평균 KL | 평균 순수익률 | 원본 대비 | 제출 주문 | 평균 BUY 확률 |
|---|---:|---:|---:|---:|---:|---:|
| 원본 | — | — | 0% | — | 0 | 12.79% |
| LR `3e-5` | 37 / 190 | 0.007790 | 0% | 0%p | 0 | 9.67% |
| LR `1e-5` | 190 / 190 | 0.001682 | 0% | 0%p | 0 | 11.78% |
| LR `3e-6` | 190 / 190 | 0.001089 | 0% | 0%p | 0 | 12.18% |

모든 모델에서 1,167개 결정이 전부 HOLD였고, 왕복 거래·수수료·낙폭·미청산 포지션은 모두 0이었다. 종목·날짜·시작 위치·시드가 같은 에피소드끼리 비교한 수익률 차이 8개도 후보마다 전부 0%p였다. **이번 한 번의 GRPO 업데이트와 8개 validation 구간에서는 수익 개선이 없었다.** 세 후보 모두 평균 BUY 확률이 낮아졌으며, 가중치 업데이트가 실제 매수 행동으로 이어지지 않았다.

부가 분석에서 이 rollout의 정규화 advantage 평균은 HOLD `+0.0819`, BUY `-0.6239`, SELL `-0.2973`이었다. `group_advantage_coef=0`이므로 실제 학습에는 GAE가 반영됐고, 16개 episode의 보상 합은 전부 음수였다. 이는 이 buffer의 actor 항이 집계상 HOLD를 강화하는 방향임을 시사한다. 공유 신경망·critic·entropy·Adam까지 포함한 인과 분석이나 다른 구간의 수익 예측을 대신하지 않는다.

같은 시작 RNG와 rollout을 사용했지만, `1e-5`는 이전 실행의 46 step 조기 중단과 달리 이번에는 190 step을 완료했다. 작은 CPU fixture에서 후보 저장 기능을 켜고 끈 비교는 초기 상태·배치 순서·최종 가중치·Adam·KL이 완전히 일치했다. 대형 GPU 실행의 차이 원인은 이번 작업에서 분리하지 않았으며, KL 한 번의 결과만으로 후보의 우위를 확정하지 않는다.

현재 결과 파일:

- `.test_artifacts/local_checks/20260913_profit_run01/summary.json`: 자동 실행 요약.
- `.test_artifacts/local_checks/20260913_profit_run01/lr_comparison.json`: 업데이트 및 후보 경로·SHA256.
- `.test_artifacts/local_checks/20260913_profit_run01/validation_comparison.json`: 원본과 후보의 같은 에피소드 수익 비교.
- `.test_artifacts/local_checks/20260913_profit_run01/validation_comparison_evaluations/`: 원본·후보별 상세 평가 JSON.
- `.test_artifacts/local_checks/20260913_profit_run01/candidates/`: `lr_00_3e-05.pt`, `lr_01_1e-05.pt`, `lr_02_3e-06.pt` 평가 전용 가중치.
- `.test_artifacts/local_checks/20260913_profit_run01/pytest.xml`: 회귀 테스트 708개 결과.
- `.test_artifacts/local_checks/20260913_profit_run01/rollout_signal_analysis.py` 및 `.json`: 관측 전체를 복사하지 않는 CPU 학습 신호 분석과 결과.
- `.test_artifacts/export_option_ab_cpu_c32119dd/comparison.json`: 후보 저장 기능의 CPU 비교 결과.

## 수익 비교 추가 전 실행 결과: 2026-09-13

위 예시의 원본 `checkpoint_iter19.pt`로 실행했다. 당시 코드의 전체 회귀 테스트는 **680개 통과(53.83초)**했다. PowerShell 실행기도 `-PlanOnly`로 실제 체크포인트·번들·데이터 입력 검사를 통과했다.

`20260913_run02`에서 worker 2개로 16개 rollout, 3,028개 관측을 수집하고 번들을 저장했다(328.6초). 단계 종료 직후 여유 RAM이 5.7 GiB로 내려가 자원 기준에 따라 중단했으며, RAM 회복 후 `20260913_run03`에서 같은 번들을 재사용했다. 이어진 LR 비교는 1,103.9초, 원본 validation 8회는 201.1초에 성공했다. 최종 코드에는 메모리 반환 지연을 처리하는 최대 60초 재확인을 포함했다.

| LR | 수락한 업데이트 / 예정 | 중단 사유 | 전체 평균 KL | 관측별 KL > 0.015 비율 |
|---|---:|---|---:|---:|
| `3e-5` | 37 / 190 | 업데이트 전 sampled KL 제한 | 0.007890 | 10.04% |
| `1e-5` | 46 / 190 | 업데이트 전 sampled KL 제한 | 0.009781 | 24.27% |
| `3e-6` | 190 / 190 | 두 epoch 완료 | 0.001538 | 1.35% |

이번 입력에서는 `3e-6`이 두 epoch를 모두 완료하고 평균 KL도 가장 작았다. 개별 관측의 최대 KL은 0.136423으로, 모든 관측의 변화량이 제한 이내였다는 뜻은 아니다. 이전 Colab의 2,807개 관측과는 다른 로컬 rollout이므로 각 실험 내부의 비교로 해석한다. 수익률이 가장 높은 LR이라는 결론은 아직 내릴 수 없다.

원본 체크포인트의 validation은 8회 모두 무거래였다. 1,167개 결정에서 HOLD만 선택했고 제출 주문·체결·왕복 거래는 모두 0건, 순수익률 0%, `evaluation_outcome=no_trade`였다. 평균 BUY 확률은 12.79%, 최대는 45.81%였다. 이 결과는 원본 모델의 무거래 상태를 확인하며, LR 비교의 임시 업데이트 모델을 평가한 결과는 아니다.

로컬 결과 파일:

- `.test_artifacts/local_checks/20260913_run03/summary.json`: 실제 순차 실행 결과(실행 당시 회귀 658개).
- `.test_artifacts/local_checks/20260913_run03/lr_comparison.json`: 세 LR의 업데이트·KL 전체 결과.
- `.test_artifacts/local_checks/20260913_run03/validation.json`: 원본 체크포인트 평가 및 에피소드별 진단.
- `.test_artifacts/local_checks/20260913_run03/bundle_verification.json`: 보강된 학습·환경 설정 일치 검사.
- `.test_artifacts/local_checks/20260913_run03/pytest_final.xml`: 추가 검사 반영 후 최종 회귀 680개 결과.
- `.test_artifacts/local_checks/20260913_launcher_plan04/summary.json`: 최종 PowerShell 실행기 입력 검사.

이 파일들은 `.test_artifacts` 아래의 로컬 산출물이며 Git에는 포함하지 않는다.

## KL 사전 검사 교정: 2026-09-13

`policy_update_checks=True`에서는 업데이트 전에도 전체 행동 분포의 정확한 `KL(원정책 || 현재 정책)` 평균으로 중단 여부를 판단한다. 기존 `kl_target * 1.5` 한도, 업데이트 후 고정 표본·미니배치 검사, 모델·Adam 복원을 유지한다. 매 배치의 기존 학습 forward를 재사용하며 새 CLI나 하이퍼파라미터는 필요 없다. 코드 갱신 후 기존 `policy_update_checks=True` 설정에 적용되고, 로그의 `pre_guard=exact`로 확인할 수 있다. 검사를 끈 기존 정책은 sampled KL 경로를 유지한다.

원본 19회차 정책·Adam·초기 RNG와 동결된 16개 에피소드(3,028개 스텝), LR `3e-5`, batch 32, 2 epoch를 사용해 변경 전후를 각각 재실행했다. 입력 학습 신호 지표 1,169개와 학습 설정이 모두 일치했다.

| 구현 | 반영된 Adam 단계 / 최대 | 중단 배치 sampled KL | 중단 배치 exact KL | 최종 전체 평균 KL |
|---|---:|---:|---:|---:|
| 변경 전 | 45 / 190 | 0.017807 | 0.014120 | 0.008680 |
| 변경 후 | 129 / 190 | 0.025077 | 0.015135 | 0.011594 |

변경 전에는 정확한 KL이 한도 `0.015` 미만인데도 sampled 값으로 중단됐다. 변경 후에는 sampled만 한도를 넘은 배치 4개에서 계속 학습하고 정확한 평균 KL이 한도를 넘으면 중단했다. 최종 전체 평균은 한도 이내지만, 개별 상태는 729/3,028개가 `0.015`를 넘었다. 현재 제한은 각 상태의 최대값을 제한하는 규칙이 아니다. CUDA 결정적 알고리즘은 비활성이므로 두 실행의 최적화 경로가 비트 단위로 같다고 주장하지 않는다. 최적화 횟수 증가 자체는 수익 개선의 근거가 아니다.

별도로 이전 3개 시드에서 수익으로 청산된 진입 49건을 분석했다. raw GAE 음수 18건, 정규화 advantage 음수 11건이며, 정규화가 양수 raw GAE를 음수로 바꾼 사례는 없었다. 음수 11건의 평균 신호는 즉시 보상 `-0.386453` + critic 보정 `+0.052469` + 이후 TD 기여 `+0.007544` = raw GAE `-0.326440`이었다. 산술 불일치의 증거는 발견하지 못했다. 개별 거래의 수익과 이후 행동까지 반영한 advantage는 다르므로 GAE·보상·정규화는 변경하지 않았다. 전체 보상/가치 시퀀스가 남지 않은 이 JSON만으로 λ 변경의 효과나 청산 후 TD 기여를 정확히 재계산할 수는 없다.

전체 회귀 **755개 통과, 실패·skip 0개**(57.795초). sampled KL의 과대·과소 추정, 분포 API가 없는 기존 정책, 정확한 KL 중단 사유와 과거 보고서 호환, 기존 사후 검사와 Adam 복원을 확인했다.

근거 파일:

- `.test_artifacts/kl_guard/20260913_run01/before.json`, `after.json`: 실제 업데이트와 최종 후보 해시.
- `.test_artifacts/kl_guard/20260913_run01/update_comparison.json`: 입력·학습 신호·코드 해시·회귀 결과 비교.
- `.test_artifacts/kl_guard/20260913_run01/pytest.xml`: 전체 회귀 결과.
- `.test_artifacts/entry_credit_review/20260913_run01/RESULTS.md`: 수익 진입의 음수 학습 신호 분석.

### KL 교정 후 동일 경로 validation

원본과 수정 전후의 실제 최종 후보를 동일한 8개 validation 경로(평가 시드 42)에서 비교했다. 각 모델의 1,167개 결정은 모두 HOLD였고, 주문·체결·수수료·미청산과 순수익은 0이었다. 원본 대비 경로별 수익률 차이도 모든 후보에서 전부 0이었다.

| 모델 | 평균 BUY 확률 | 최대 BUY 확률 | 주문 | 평균 순수익률 |
|---|---:|---:|---:|---:|
| 원본 | 12.790% | 45.813% | 0 | 0% |
| 수정 전 업데이트 | 9.438% | 26.189% | 0 | 0% |
| 수정 후 업데이트 | 9.636% | 30.627% | 0 | 0% |

이번 수정으로 sampled KL만 한도를 초과한 배치에서 학습을 이어가는 동작은 확인했지만, 추가 최적화가 검증 거래나 수익 개선으로 이어지지는 않았다. `test` 분할은 사용하지 않았다. `.test_artifacts/kl_guard/20260913_run01/validation.json`에 동일 조건·경로 검사와 정밀도/RNG 복원 결과를, 같은 폴더의 `RESULTS.md`에 전체 결과를 저장했다. 평가에 사용한 두 후보는 evaluation-only 체크포인트다.
