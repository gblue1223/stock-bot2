# 재개 학습률 변경과 동일 rollout 비교

FP32 재개 실행에서는 rollout log probability 최대 오차가 `2.8610229e-6`로 줄었지만, 4개의 optimizer step 이후 다음 미니배치의 sampled KL이 `0.020018`, 같은 배치의 exact KL이 `0.016239`로 중단 기준 `0.015`를 넘었다. 검증은 64개 에피소드 모두 무거래였고 연속 무거래 4회 상한으로 종료했다. 정밀도 오차 감소와 수익성 개선은 별도로 판단해야 한다.

이번 변경은 동일한 입력과 시작 상태에서 학습률의 영향을 비교한다. 보상, 거래 비용, KL 기준과 무거래 상한은 변경하지 않는다.

## Colab에서 먼저 실행할 진단

최신 `ai_trader/grpo/colab_train_xlstm_return_priority.ipynb`와 `dev-tf-rl` 소스를 사용한다. 이전 모듈을 이미 import했다면 런타임을 다시 시작하고 준비 셀을 실행한다.

5번 설정에서 `ENABLE_TF32=False`로 두고, **7-2. 같은 rollout과 Adam 상태로 학습률 비교 (선택)** 셀의 `UPDATE_DIAGNOSTIC_CHECKPOINT`에 비교를 시작할 체크포인트 경로를 넣는다. 이번 문제를 비교하려면 기존 19회차 체크포인트를 사용한다. 8번 학습 셀 실행은 필요하지 않다.

이 셀은 다음 작업을 수행한다.

1. 체크포인트의 정책과 Adam을 복원하고 FP32로 다음 실제 rollout 묶음(기본 16개 에피소드)을 한 번 수집한다.
2. 실제 학습과 같은 그룹 정렬 및 advantage 계산 후, 전체 rollout·가중치·Adam 상태·난수 상태를 bundle에 저장한다.
3. 저장 자료로 `3e-5`, `1e-5`, `3e-6`을 각각 같은 상태에서 업데이트한다. 각 조건 사이에는 정책과 Adam, 난수 상태를 다시 복원한다.
4. 기존 KL 중단 및 rollback 결과와 업데이트 후 전체 rollout의 exact KL을 JSON으로 기록한다.

`UPDATE_BUNDLE_PATH`와 `UPDATE_LR_REPORT`를 비우면 고유한 새 이름을 사용한다. bundle은 Colab 로컬 `/content/stockbot_update_diagnostics/`에, JSON은 체크포인트 실행 폴더의 `diagnostics/`에 저장한다. 로컬 bundle은 런타임 삭제 시 사라지므로 다시 비교할 자료가 필요하면 직접 보관한다. 관측이 `2,807 × 1,024 × 63`, float32라면 관측만 약 724 MB(691 MiB)이며 모델과 Adam 등의 공간이 추가된다.

자료 수집과 세 번의 업데이트는 시간이 걸리며 셀에 진행 로그를 출력한다. 캡처는 검증·테스트·optimizer step을 실행하지 않는다. 비교 과정은 bundle을 읽고 임시 정책만 업데이트하며 학습 체크포인트나 최종 모델을 내보내지 않는다. 기존 bundle과 결과 보고서 파일은 덮어쓰지 않는다.

## CLI에서 분리 실행

저장된 학습 설정과 체크포인트를 사용해 한 번 캡처한다. 설정 JSON의 실행 요청 필드는 자동 재실행하지 않으므로 `--capture_update_bundle`을 직접 지정한다.

```bash
python -m ai_trader.grpo.train_xlstm \
  --config /content/drive/MyDrive/ColabData/stockbot/models/RUN/training_config.json \
  --resume --load_policy /content/drive/MyDrive/ColabData/stockbot/models/RUN/checkpoints/checkpoint_iter19.pt \
  --capture_update_bundle /content/stockbot_update_diagnostics/iteration19.pt \
  --policy_update_checks

python -m ai_trader.grpo.diagnose_update_lr \
  --bundle /content/stockbot_update_diagnostics/iteration19.pt \
  --output /content/drive/MyDrive/ColabData/stockbot/models/RUN/diagnostics/lr_comparison.json \
  --learning-rates 3e-5 1e-5 3e-6 --device cuda
```

캡처와 비교 모두 TF32를 끈 FP32를 적용한다. 각 조건은 같은 minibatch 난수열을 사용하지만 KL 중단 시점이 달라 실행한 step 수는 달라질 수 있다. 전체 rollout KL은 모든 행동의 `old || new` 분포를 측정한다. 특정 미니배치 또는 고정 표본의 KL만으로 전체 정책 변화량을 대신하지 않는다.

비교 중 관측은 CPU에 두고 미니배치만 GPU로 옮긴다. full-rollout 기준 분포는 작은 `N × action_dim` 표로 유지한다. 에피소드 경계에서 배치를 임의로 나누지 않고 실제 `update_policy`의 전체 연결 순서와 배치 경계를 따른다.

JSON의 주요 필드는 다음과 같다.

| 필드 | 의미 |
|---|---|
| `variants[].metrics.optimizer_accepted_steps / optimizer_attempted_steps / optimizer_rejected_steps` | 허용·시도·rollback된 optimizer step 수 |
| `variants[].guard_reason` | 전체 epoch 완료, 사전 sampled KL 중단, 사후 exact KL rollback, 실행 오류 구분 |
| `variants[].full_rollout_kl.mean_kl / max_kl / quantiles` | 업데이트 후 전체 관측의 exact KL 평균·최대·분위수 |
| `variants[].full_rollout_kl.fraction_over_threshold` | 개별 관측 KL이 기준을 넘은 비율 |
| `capture_to_replay_reference_kl` | 캡처 당시와 비교 환경의 시작 분포 차이 |
| `precision_restored / rng_restored` | 진단 종료 후 호출자 정밀도·난수 상태 복원 여부 |

`status=ok`는 진단 계산이 완료됐다는 뜻이다. KL 중단이 없거나 수익성 있는 모델이라는 뜻은 아니다. 전체 평균 KL이 기준을 넘었는지도 `mean_over_threshold`로 따로 확인한다. 개별 관측의 초과 비율은 진단 통계이며 기존 평균 KL 중단 기준과 동일한 판정이 아니다.

## 진단 후 학습 재개

비교 결과로 다음 실험의 LR을 정한 뒤 5번 설정 셀에서 명시한다. 아래 `1e-5`는 사용법 예시이며 최적값 판정이 아니다.

```python
MODE = 'resume'
RUN_NAME = 'scalping_v7_lr1e5_comparison_run01'  # 새 결과 폴더
LOAD_POLICY = '/content/drive/MyDrive/ColabData/stockbot/models/RUN/checkpoints/checkpoint_iter19.pt'
ENABLE_TF32 = False
RESUME_LR = 1e-5
```

그런 다음 6→7→8번을 실행한다. `TOTAL_TIMESTEPS`는 저장된 누적 스텝보다 커야 한다.

- `RESUME_LR=0.0`: 저장된 Adam LR 유지.
- `RESUME_LR>0`: resume에서만 해당 LR 적용. CLI 이름은 `--resume_lr`다.
- `LEARNING_RATE`: new/finetune용. resume의 Adam LR 변경에는 `RESUME_LR`을 사용한다.

학습률 변경은 Adam의 모멘트와 step 카운터를 초기화하지 않는다. 적용된 LR은 실행 로그, 학습 설정과 다음 체크포인트에 반영하며, 검증 best로 되돌아갈 때도 현재 LR을 유지한다. 나중에 `RESUME_LR=0`으로 재개하면 그 체크포인트에 실제 저장된 LR을 사용한다. 과거의 `resume_lr` 요청값을 자동으로 다시 적용하지 않는다.

이전 실행에서 가져와 보존한 best 후보는 후보 자신의 Adam 상태와 LR을 저장한다. 해당 파일의 설정 기록도 그 LR에 맞춘다. 현재 학습에 그 best를 복원할 때는 현재 실험의 LR을 다시 적용한다.

학습률 변경만으로 무거래 검증 이력을 초기화하지 않는다. 이미 무거래 상한에 도달한 체크포인트를 재개하면 추가 검증에서도 중단될 수 있다. 동일 자료의 LR 비교는 이 제한을 늘리지 않고 업데이트 자체를 조사하는 경로다.

## 결과 해석의 범위

작은 LR에서 KL이 줄고 허용된 step이 늘어나는지는 이번 진단으로 확인할 수 있다. 이 결과만으로 수익이 높아졌거나 최적 LR을 찾았다고 판단할 수는 없다. 후보를 정한 후 같은 비용과 검증 조건에서 거래가 발생하는지, 순수익이 양수인지 확인해야 한다.

저장하는 자료는 이번에 새로 수집한 rollout이다. 기존 20회차의 전체 자료가 없으므로 과거 실행의 입력을 복구하지는 않는다. 일반 resume의 환경 상태나 난수 복원 범위는 확장하지 않는다. 같은 bundle의 비교에서는 저장된 정책·Adam·난수 상태를 고정한다.

## 로컬 통합 검사

RTX 2080 Ti / PyTorch 2.8에서 실제 저장 모델(2,514,696 파라미터), Adam 상태와 32개의 `1024 × 63` 관측으로 bundle 저장부터 비교 CLI까지 실행했다. 이 검사의 보상과 종료 표시는 합성 자료이며 Colab의 원래 전체 rollout은 아니다. 에피소드를 17개·15개 관측으로 나눠 에피소드 경계를 넘는 32행 배치도 확인했다. bundle 크기는 38,507,355바이트였다.

세 LR과 반복한 `3e-5` 조건 모두 진단에 성공했고, 사후 KL 초과 단계의 정책·Adam 복원과 정상 epoch 완료 경로를 확인했다. 호출자의 정밀도 및 난수 상태도 복원됐다. 이 결과는 기능 검증이며 최적 LR이나 수익률에 대한 근거로 사용하지 않는다.

전체 회귀 테스트는 **625개 통과**했다(61.99초). 실제 CPU/CUDA xLSTM의 초기화된 Adam 상태에서 중복 LR을 사이에 다른 LR을 넣어 재실행해 정책·Adam 결과와 shuffle 일치를 확인했다. 재개 LR·무거래 이력 보존, best 복원과 LR 기록, 캡처의 검증·학습 출력 분리, 기존 파일 및 동시 저장 충돌 보호, 실패 시 상태 복원을 포함한다.

```powershell
.\.venv64\Scripts\python.exe -m pytest --basetemp=.test_artifacts/pytest_lr_diagnostics_full_01
```
