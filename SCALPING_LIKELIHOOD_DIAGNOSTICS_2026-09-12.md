# 확률 불일치 재현 진단과 TF32 설정

`scalping_v7_update_checks_20260912_run01`의 최초 20회차는 log probability 오차 `0.0010656714 > 0.001`로 optimizer 실행 전에 실패했다. 재개 실행은 다른 2,807개 관측에서 `0.00057923794`로 통과했지만, 실제 `colab_run.json`의 `enable_tf32`는 여전히 `true`였다. 이 결과는 TF32 비활성화로 문제를 해결한 증거가 아니다.

재개 후 검증은 무거래 4회 상한으로 정상 종료했다. 매수 확률이 15회차 5.85%에서 20회차 9.77%로 반등하면서 기존 확률 기반 카운터는 초기화됐지만 실제 거래가 없어 새 카운터는 4회를 유지했다. 이 변경은 확률 오류의 원인을 구분하기 위한 진단이며, 수익률·보상·KL 한계·무거래 상한을 바꾸지 않는다.

## Colab의 TF32 선택

남은 주 노트북 `ai_trader/grpo/colab_train_xlstm_return_priority.ipynb`를 최신 파일로 갱신한다. 이미 실행한 Python 모듈이 있으면 런타임을 재시작하고 처음부터 준비한다.

5번 설정 셀의 `ENABLE_TF32`가 new/resume/finetune 모두에서 우선한다. 재개 실행의 이전 TF32 값은 비교 안내에만 사용하고 현재 선택을 덮어쓰지 않는다. 비활성화 진단은 다음과 같이 설정한다.

```python
ENABLE_TF32 = False
```

이제 셀 맨 끝에 같은 값을 다시 대입할 필요가 없다. 기본값은 기존대로 `True`이며, 재개 실험에서는 의도한 값을 직접 선택한다. 이전 실행 자료를 보존하려면 새 `RUN_NAME`을 사용한다.

GPU 사전 검사와 학습 자식 프로세스는 공통 정밀도 설정 함수를 사용한다. PyTorch의 새 `fp32_precision` API 또는 구버전 `allow_tf32` API 중 지원되는 한 계열만 사용하며, 실제 backend 설정을 학습 시작 로그와 실행 기록에 남긴다. GPU 점검 후 설정을 바꾸면 점검을 다시 실행해야 한다.

## 실패 자료 자동 저장

학습의 확률 일치 검사가 허용값을 초과하면 다음 경로를 로그에 출력한다.

```text
<output_dir>/diagnostics/likelihood_update_000020_<고유번호>.pt
```

파일에는 다음 자료를 저장한다.

- 처음 실패한 학습 미니배치의 관측, 고정 행동, 실행 마스크, 수집 당시와 재계산한 log probability
- 해당 시점의 정책 가중치, 모델 재구성 설정, 관측 스키마
- 회차와 누적 스텝, 수집·학습 배치 크기, 오차 허용값과 관측 위치
- 실제 정밀도 설정과 PyTorch·CUDA·GPU 정보

관측은 독립적인 CPU 메모리로 복사한다. 전체 rollout을 저장하지 않으며, 저장 파일이 원래 큰 배열의 공유 메모리를 따라 복제하지 않도록 한다. optimizer 실행 전에 실패하므로 해당 회차의 업데이트는 진행하지 않는다. 저장에 실패해도 원래 확률 불일치 오류를 유지한다. 출력 경로가 없는 직접 생성 `GRPOTrainer`는 저장하지 않고 경고한다.

## 같은 입력으로 TF32 ON/OFF 비교

Colab의 **7-1. 실패한 likelihood 배치 재현 (선택)** 셀에서 `LIKELIHOOD_FAILURE_BUNDLE`에 로그에 출력된 파일 경로를 넣는다. `DIAGNOSTIC_OUTPUT`을 비우면 같은 폴더에 고유한 JSON 보고서를 만들며, `DIAGNOSTIC_DEVICE`로 장치를 선택한다. 7번 GPU 점검을 통과하지 못해도 이 셀은 별도로 실행할 수 있다. 또는 프로젝트 루트에서 다음 명령을 실행한다.

```bash
python -m ai_trader.grpo.diagnose_likelihood \
  --bundle /content/drive/MyDrive/ColabData/stockbot/models/RUN/diagnostics/likelihood_update_000020_ID.pt \
  --output /content/drive/MyDrive/ColabData/stockbot/models/RUN/diagnostics/likelihood_report.json \
  --device cuda
```

배치 크기는 저장 자료에서 읽는다. 별도로 비교하려면 `--rollout-batch-size 8 --update-batch-size 32`를 지정한다. 저장된 동일 가중치·관측·행동·마스크를 고정하고 다음 네 조건을 비교하며 학습하거나 행동을 다시 샘플링하지 않는다.

| TF32 | 실행 경로 |
|---|---|
| ON | 수집 배치 크기의 eval / no_grad |
| ON | 학습 배치 크기의 train / gradient checkpointing |
| OFF | 수집 배치 크기의 eval / no_grad |
| OFF | 학습 배치 크기의 train / gradient checkpointing |

이 파일은 실패한 **학습 미니배치**를 보존한다. 원래 수집 순간에 함께 계산된 다른 worker의 관측과 순서는 보존하지 않으므로, 수집 경로는 저장 관측을 다시 배치로 묶은 진단이다. 보고서의 `original_rollout_batch_reconstructed=False`가 이 한계를 표시한다. 같은 실패 입력에서 TF32와 두 실행 경로의 차이를 비교할 수 있지만, 이 비교만으로 원래 전체 rollout 계산을 완전히 재현했다고 주장하지 않는다.

기존에 실패 자료를 저장하지 않은 실행은 과거의 해당 입력을 복구할 수 없다. 19회차 체크포인트만 재개해 수집한 새 관측으로는 이전 오류를 같은 조건에서 비교할 수 없다. 기존 `resume`의 난수·환경 상태 복원 범위는 이번 변경에서 확장하지 않는다.

CPU나 TF32를 지원하지 않는 GPU의 결과는 A100의 TF32 효과를 검증하지 못한다. 보고서에서 실행 장치와 실제 정밀도를 함께 확인한다. 명령 종료 시 변경한 정밀도 설정을 복원한다.

## 사전 KL 중단의 추가 진단

기존 선택 행동 기반 KL이 한계를 넘으면, 그 판단에 사용한 **동일 forward·동일 미니배치**의 전체 행동 KL도 기록한다. 추가 모델 forward 없이 이미 계산한 분포를 사용한다. 이전 단계의 고정 표본 KL과 이번 중단 배치의 KL을 혼동하지 않도록 로그에 `same_batch exact KL`로 표시한다.

TensorBoard에는 다음 수치를 추가한다.

- `pre_update_kl_early_stopped`: optimizer 실행 전 KL 중단 여부
- `pre_update_exact_kl_checked`: 해당 배치의 전체 행동 KL을 계산했는지 여부
- `pre_update_minibatch_exact_kl`, `pre_update_minibatch_max_state_kl`, `pre_update_minibatch_samples`

`pre_update_exact_kl_checked=0`일 때 다른 사전 진단 값의 0은 측정 결과가 아니다. 기존 sampled KL에 의한 중단과 사후 exact KL 초과 단계 복원은 그대로 유지한다. 이번 로그의 진단만으로 허용 오차나 KL 상한을 자동 완화하지 않는다.

## 통합 검사 범위

로컬 RTX 2080 Ti / PyTorch 2.8에서 실제 저장된 2,514,696개 파라미터 모델과 32개의 `1024 × 63` 관측을 사용했다. 저장 log probability 하나에 의도적으로 `0.02`를 더한 검사 자료로 실패 저장부터 진단 CLI까지 실행했다. 번들은 18,339,792바이트였고 네 조건 모두 해당 오염을 검출했으며 정밀도 설정 복원도 확인했다.

이 검사는 새 저장·재검사 경로의 통합 검사다. 원래 Colab 실패 입력은 저장돼 있지 않으므로 그 오류를 재현한 결과가 아니다. RTX 2080 Ti는 TF32를 지원하지 않으며 보고서에도 `tf32_hardware_effect_testable=False`로 표시한다. 실제 TF32 영향은 A100에서 확인해야 한다.

전체 회귀 테스트는 **573개 통과**했다(55.29초). 실패 미니배치만 직렬화되는지, 저장 오류가 원래 오류를 가리지 않는지, 입력 번들 덮어쓰기 방지, 동일 forward의 사전 exact KL과 기존 중단 동작, Colab 재개 TF32 우선순위를 포함한다. 정밀도 API 분기는 테스트 대역으로 구·신 API를 확인했고 실제 실행 검사는 PyTorch 2.8 CPU/CUDA에서 수행했다.

```powershell
.\.venv64\Scripts\python.exe -m pytest --basetemp .test_artifacts/pytest_likelihood_diagnostics_full_01
```
