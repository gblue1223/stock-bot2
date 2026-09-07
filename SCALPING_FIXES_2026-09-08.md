스캘핑 리뷰 후속 수정 — 2026-09-08

리뷰의 코드 오류를 수정하고 주문·체결·회계를 분리한 리플레이를 추가했다. 기존 NPZ·모델 파일은 변경하지 않았다. 아래 번호는 SCALPING_REVIEW_2026-09-08.md의 발견 번호다.

| 리뷰 | 수정 | 주 검증 |
| --- | --- | --- |
| 1. 시간 역행 | 숫자 시간 정렬, 동일 시각 이벤트 순서 보존, 기존 NPZ 로드 시 stable sort | 데이터 추출 테스트와 기존 NPZ 3개 재생 |
| 2. 미래 정규화 누수 | 관측 시점까지의 창만 사용하는 공통 ObservationBuilder | 미래 꼬리 변경 시 과거 관측 불변 |
| 3. 가격 왜곡 | 원본 가격/원화 실행 배열과 모델 피처 분리, 가격 단위 schema | 원본 가격·단위·관측 테스트 |
| 4. 손실 거래의 양의 보상 | 비용 차감 NAV 변화로 보상 통일, 보너스/미거래 페널티 제거 | 보상 합/NAV/실현손익 대사 |
| 5. 결과 기반 GRPO 그룹 | 행동 이전 시장 특징으로 그룹화 | 보상 변경에 독립적인 그룹 검사 |
| 6. 정책 확률 불일치 | BN 통계 고정·Dropout 비활성화, 행동 마스크 공유 | 같은 가중치/배치 변경의 log-prob 일치 |
| 7. best 평가/가중치 불일치 | 업데이트 후 동일 가중치의 validation 평가, 날짜 train/validation/test 분리 | best 가중치와 점수, CPU 전체 학습 경로 |
| 8. 관측 규격 불일치 | BC/RL/추론의 공통 builder·15개 포지션 피처·단위/순서 strict schema | 관측 동등성·strict load·증류 재로드 |
| 9. 보유시간/추론 리스크 | 실제 경과시간, 최대 보유 타이머, 현재 가격 손익, 지속되는 trailing 활성 상태, 침체 청산 | 시간 공백/트레일링/자동청산 검사 |
| 10. 마지막 틱 정산 | 이벤트 갱신 후 체결·NAV·종료 처리, 불가능한 청산은 미청산으로 표시 | 마지막 가격 하락과 잔여 재고 검사 |
| 11. 즉시 전량 체결 | 지연·bid/ask·호가 깊이·부분체결·잔량 보존·취소/만료·현금/재고 제한 | 전용 체결 시뮬레이터 회귀 검사 |
| 12. CSV 합병 손상 | coalesce 결과 유지, raw 호가/수량/호가 시각 보존, ffill만 사용 | 서로 다른 시각의 CSV 병합 |
| 13. BC 연결/JIT 실패 | 현재 관측 규격·같은 forward 사용, JIT 제거, teacher 날짜 이력 검사 | 작은 CPU 증류와 역전파 |
| 14. 정규화 실행 실패 | 시간 피처 반환 개수·중복 컬럼 수정, 파생 피처만 결정적으로 생성 | single/batch 시간 피처와 prefix 불변성 |
| 15. 테스트/문서 불일치 | 현재 실행 명령·정의·제약으로 README 교체, pytest 테스트 디렉터리 설정 | 전체 회귀 검사 |

추가로 기본 `--load_policy`는 새 fine-tuning, 명시적 `--resume`는 optimizer·iteration·누적 timestep 복원으로 구분했다. 사전학습과 재개 모델의 날짜 이력도 새 holdout과 비교한다. 사용하지 않던 next_states 저장을 제거했고, 데이터 캐시는 프로세스별 256 MiB 한도로 제한했다.

**체결 동작**

`ai_trader/grpo/environments/execution.py`는 주문이 시장에 도착한 뒤의 이벤트에서 반대편 호가를 소진한다. 지정가의 수동 대기열 체결은 추정하지 않으며, 반대편 표시 호가와 실제로 맞는 경우만 처리한다. 기록된 잔량이 줄어들 때 자체 소비 잔량을 복원하지 않는 회귀 검사도 추가했다.

지연이나 잔량 때문에 종료 시 청산하지 못하면 가상의 체결을 만들지 않는다. 보고 NAV는 매도 호가와 비용으로 평가한 값이고, `liquidation_complete`, `open_quantity`, 실현/미실현 손익을 별도로 제공한다. 호가 시각이 있을 때 오래된 호가를 거부한다. 없는 과거 정보는 추정한 실제 데이터처럼 취급하지 않는다.

기존 NPZ에는 실제 호가 가격/수량이 없어 `synthetic_spread` 모드를 사용한다. 가정한 스프레드·잔량·지연은 설정과 결과에 남는다. 실제 호가 평가에는 원본 데이터에서 실행 배열을 포함해 새 디렉터리로 추출하고 `require_order_book=true`를 사용해야 한다.

**실행**

```powershell
python -m ai_trader.grpo.train_xlstm --config config/scalping_v3.example.json
python -m ai_trader.grpo.backtest --policy models/scalping_v3/checkpoints/checkpoint_best.pt --output models/scalping_v3/replay_test.json
```

새 예제는 작은 모델·120틱 관측으로 시작하는 연구 설정이다. 수익 최적화 결과가 아니다. 구모델에 현재 관측 스키마·날짜 이력이 없으면 로드하지 않으며, 수정된 환경으로 재학습해야 한다. 자세한 비용·체결 옵션과 데이터 계약은 README.md에 있다.

**검증 결과**

- `python -m pytest tests --basetemp .test_artifacts/pytest_final -o addopts='' -q`: **50개 통과**, 36.57초.
- Windows 단일/2개 worker 학습, holdout 검증, best 저장, 최종 테스트와 fine-tuning/resume를 작은 합성 데이터로 확인했다.
- 기존 NPZ 첫·중간·마지막 3개에서 `(120, 42)` 관측이 모두 유한했고, 시간 역행 0건, 가격 단위 `million_krw`, 보상/NAV 일치를 확인했다. 앞선 검사에서는 `(3000, 42)` 관측도 확인했다.
- Python 파일 48개의 메모리 내 문법 컴파일과 `git diff --check`를 통과했다.

전체 기존 캐시를 재생성하거나 장시간 모델을 학습시키지는 않았다. 실제 수익성, 전체 기간 계좌 성과, 실주문과 현재 xLSTM의 실전 지연 성능은 별도 검증이 필요하다. 겹치는 긴 관측창의 메모리 압축과 브로커 주문/잔고 대사 계층은 이번 수정 범위에 포함하지 않았다.

**Colab A100 후속 조정**

- `colab_train_xlstm.ipynb`의 깨진 Python/명령문과 이전 보상 설정을 제거하고 현재 관측·체결·날짜 검증 설정으로 교체했다.
- A100 40GB/80GB에 배치 상한 64/128을 사용하며, 실제 forward/backward/Adam 사전 점검에서 메모리 부족 시 배치를 줄인다. 모델은 두 프로필에서 CNN/mLSTM/FC 64/128/256, 관측 1024틱, 에피소드 300틱을 유지한다.
- 여유 RAM과 CPU에 맞춰 worker·rollout을 제한하고, GAE와 TensorBoard의 전체 에피소드 GPU 전송도 미니배치 단위로 변경했다. 지표의 에피소드별 평균 정의는 유지했다.
- 신규 학습/재개/미세조정을 분리하고 저장된 스키마·거래 설정을 복원한다. 구형 체크포인트, 누적 목표 초과, 다른 실험 출력 덮어쓰기를 사전에 확인한다.
- 재개 출발 가중치와 호환되는 기존 best를 현재 validation으로 다시 평가한다. 재개 후 성적이 악화되면 이전 best를 보존하며, 실제 진행률은 iteration 체크포인트와 최종 보고서에 별도로 남긴다.
- TF32를 실제 학습 프로세스에 적용하고, Drive 로그·자원 정보·소스/manifest 해시를 기록한다. Git 없는 업로드 소스도 지원하며 중단 시 worker를 포함한 프로세스 그룹을 정리한다.
- 체결 스트레스 검사는 validation만 사용한다. A100 GPU의 실제 속도·최대 메모리는 로컬 CPU 환경에서 측정하지 않았으며 Colab 사전 점검에서 확인하도록 했다.

후속 검증: `python -m pytest tests --basetemp .test_artifacts/pytest_a100_final -o addopts='' -q` — **75개 통과**, 39.20초. 노트북 nbformat 검증, 11개 실행 셀 문법 검사 및 실제 학습 subprocess의 CLI 로딩도 통과했다.
