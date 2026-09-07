스캘핑 AI 프로젝트 리뷰 — 2026-09-08

후속 상태: 이 문서는 `363a811` 시점의 리뷰를 보관한 기록이며 줄 번호도 당시 기준입니다. 이후 코드 수정, 체결 시뮬레이터와 실행·검증 방법은 README.md 및 SCALPING_FIXES_2026-09-08.md에 정리했습니다.

검토 기준: 현재 체크아웃 `363a811`, 초~분 단위 한국 주식 스캘핑. 소스 37개와 설정, 캐시 manifest, 기존 NPZ 3개를 읽고 작은 합성 입력으로 주요 결함을 재현했다. 실제 학습·주문·전체 백테스트는 실행하지 않았다. 소스 코드는 변경하지 않았다.

**판단**

현재 구현에는 수익성 평가를 무효화할 수 있는 시간 역행, 미래 정보 누수, 가격 계산 오류가 있다. 기존 학습 수익이나 best 체크포인트만으로 실전 수익성을 판단할 수 없다. 모델 용량이나 학습량을 늘리기 전에 데이터 → 손익·체결 → 학습 → 검증의 순서로 수정해야 한다.

현재 주 실행 경로는 `train_xlstm.py → GRPOScalpingEnvXLSTM → GRPOTrainer`이며, 기본 캐시 경로는 `data/extracted_episodes`다. 이 manifest에는 27개 시장 피처와 12,827개 종목·일자 파일이 등록되어 있다. 환경은 15개 분할 포지션 피처를 추가하므로 현재 관측은 총 42채널이다. README의 AutoEncoder, 일부 실행 명령 및 테스트 설명은 현재 파일 구성과 맞지 않는다.

P1은 성과 신뢰성 또는 핵심 학습·추론을 우선 수정해야 하는 문제, P2는 다음 순서의 정확성·연결·운영 문제로 표시했다.

**우선 수정할 발견**

1. **[P1] 실제 캐시의 시간이 역행한다.**

   [data_extractor.py](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/data_extractor.py:118)는 `ORDER BY 시간`을 사용한다. [ensure_table_duckdb](D:/Workspace/Project/stock-bot/stock-bot2/scripts/data/normalize_datasets.py:634)는 기존 테이블의 시간을 VARCHAR로 강제한다. 8자리 09시와 9자리 10시를 문자열로 정렬하면 10시가 먼저 온다.

   기존 manifest의 첫·중간·마지막 NPZ 3개 모두 역행 1회를 확인했다.

   | 캐시 파일 | 확인한 역행 |
   | --- | --- |
   | episode_035720_20250512.npz | 10:59:57.916 → 09:00:23.133 |
   | episode_042670_20250702.npz | 10:59:56.685 → 09:00:21.181 |
   | episode_064350_20250530.npz | 10:59:58.366 → 09:00:03.542 |

   전체 12,827개를 검사한 결과는 아니다. 그러나 현재 학습에 쓰이는 실데이터에서 이미 재현되므로 우선순위가 높다. 시간은 거래일과 결합한 timestamp 또는 장 시작 이후 정수 마이크로초/밀리초로 저장하고, 동일 시각의 이벤트 순서도 보존해야 한다. 추출 시 단조 증가 검증을 강제하고 기존 캐시를 재생성해야 한다.

2. **[P1] 하루의 미래 데이터가 과거 관측에 들어간다.**

   [scalping_env_xlstm.py](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/environments/scalping_env_xlstm.py:164)는 NPZ 전체의 mean/std로 정규화한 다음 192줄에서 에피소드 시간창을 자른다. 장중 판단 시점에 아직 관측하지 못한 나머지 구간이 정규화에 사용된다.

   합성 재현: 과거 등락률 `[0, 0.1]`을 그대로 두고 미래 값만 `0.2 → 20`으로 바꾸면 과거 관측이 `[-1.2247, 0] → [-0.7124, -0.7018]`로 달라졌다.

   학습 기간에서만 추정한 고정 통계 또는 현재 시점까지의 rolling 통계를 사용해야 한다. 미래 꼬리를 바꾸어도 과거 관측이 같다는 불변성 검사가 필요하다.

3. **[P1] 정규화한 등락률을 실제 등락률로 취급해 손익을 계산한다.**

   캐시 환경이 반환한 z-score 배열이 그대로 `episode_data`가 되고, [_compute_prices](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/environments/scalping_env_e2e.py:609)가 그 값을 100으로 나누어 기준가격에 곱한다. 원본 등락률과 z-score는 같은 단위가 아니다.

   합성 재현: 원본 등락률 `[0, 0.1, 0.2]%`에서 실제 첫 구간 +0.10%가 환경에서는 +1.23993%로 계산된다.

   원본 현재가·bid/ask·수량·시간은 모델 입력과 별도 배열로 보관해야 한다. 정규화는 관측에만 적용한다. 기존 NPZ의 현재가에는 별도 스케일도 있으므로 가격 단위를 manifest에 명시해야 한다.

4. **[P1] 손실 거래에 양의 보상을 주며, GRPO가 그 보상을 최적화한다.**

   [패턴 보너스](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/environments/scalping_env_e2e.py:958)는 1~5초 사이 조금이라도 상승하면 단계 비중과 무관하게 +0.5를 지급한다. [advantage 계산](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/grpo.py:490)은 이런 보너스를 포함한 step reward 합을 사용한다. 보고되는 거래 순손익은 별도로 계산된다.

   합성 재현: 코드 기본 비용은 매수 수수료 0.015%, 매도 수수료 0.015%와 매도세금 설정 0.18%, 합계 0.21%다. 가격 100 → 100.01에서 20% 비중으로 buy → hold → sell하면 순손익은 -0.04%인데 학습 보상은 +0.46002다. 이 비용 수치는 코드 설정을 설명한 것이며 실제 적용 세율·증권사 수수료를 검증한 값은 아니다.

   보상을 비용 차감 계좌 평가액 변화로 통일하고, 진입 보너스·승리 보너스·미거래 페널티는 별도 실험으로 효과를 검증해야 한다. 거래하지 않아 순손익 0인 선택을 기본 비교 대상으로 둔다. 보상, 실현손익, 미실현손익을 로그에서 구분해야 한다.

5. **[P1] GRPO 그룹을 시장 상태 대신 보상 결과로 나눈다.**

   [group_episodes](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/grpo.py:324)는 정책 보상의 평균·표준편차·범위로 클러스터링한다. 좋은 결과끼리, 나쁜 결과끼리 그룹이 만들어지면 비교하려던 수익 차이가 그룹 평균을 빼는 과정에서 사라진다.

   실제 두 메서드를 사용한 합성 재현: 10스텝 보상이 각각 -10/-1/+1/+10인 에피소드 16개씩, 총 64개를 넣으면 총보상 -100/-10/+10/+100의 차이가 있어도 결과별 4그룹으로 나뉘어 모든 advantage가 0이 됐다. 항상 0이 된다는 뜻은 아니며, 현재 설계가 유효한 수익 신호를 지울 수 있다는 반례다.

   같은 종목·거래일·시작 시점·초기 포지션의 시장 경로에서 여러 행동 경로를 비교하거나, 행동 이전의 시장 특징으로 그룹을 고정해야 한다. 동일한 체결 환경에서 PPO+GAE 기준 모델과 비교하는 것도 권한다.

6. **[P1] 같은 정책의 행동확률을 다시 계산해도 값이 달라진다.**

   [PPO 확률비](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/grpo.py:704) 계산 전후 주 정책에 train/eval 모드 관리가 없다. 정책의 [BatchNorm·Dropout](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/policies/scalping_policy_xlstm.py:250) 때문에 가중치가 같아도 난수와 minibatch 구성에 따라 행동확률이 변한다. 실전 추론은 eval 모드다.

   소규모 CPU 재현에서 동일 입력·가중치의 log-prob 차이는 최대 0.05138이었다. Dropout을 끈 뒤에도 batch 4 → 1 변경으로 0.03485 차이가 났다. `no_grad()`만으로 BatchNorm 통계 갱신이나 Dropout이 멈추지는 않는다.

   정책 확률 경로의 Dropout을 제거하고 BatchNorm을 고정하거나 표본별 정규화로 바꾸는 방안을 검토해야 한다. 학습 업데이트 전 동일 관측·행동의 likelihood ratio가 1인지 검사한다. BatchNorm 모드 동작은 [PyTorch 공식 문서](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html)와도 일치한다.

7. **[P1] best 체크포인트와 평가 점수가 서로 다른 가중치에 해당한다.**

   [학습 루프](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/grpo.py:962)는 rollout 수집 → 978줄 가중치 업데이트 → 994줄 이전 rollout 점수 계산 → 1012줄 이후 업데이트된 가중치 저장 순서다. 저장된 best 모델 자체를 그 점수로 평가한 적이 없다.

   또한 현재 학습 경로에는 날짜를 분리한 검증 환경·최종 테스트가 없고 전체 manifest에서 임의 샘플링한다. 업데이트 후 정책을 고정된 검증 구간에서 평가한 결과로 best를 고르고, 최종 테스트 구간은 모델 선택에 사용하지 않아야 한다.

8. **[P2] 학습과 추론의 관측·정규화 규격이 일치하지 않는다.**

   [추론 정규화](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/inference/enhanced_grpo_infer_xlstm.py:107)는 통계가 없으면 입력을 그대로 보내고, 있어도 단순 z-score만 적용한다. 학습 캐시 경로는 일부 특징의 signed_log1p 후 z-score다. [체크포인트 저장](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/grpo.py:1332)에는 특징 순서, 시간/가격 단위, 정규화 상태와 버전이 저장되지 않는다.

   현재 추론 클래스는 완성된 관측을 외부에서 받아 처리하는 구조이므로, 호출자가 정확히 가공하면 일부 불일치는 피할 수 있다. 하지만 저장소 내에서 이를 보장하는 공통 생성기는 확인되지 않았다.

   하나의 ObservationBuilder와 명시적인 feature schema를 사전학습·RL·리플레이·실시간 추론이 공유해야 한다. 같은 입력 경로의 관측과 action logits가 일치하는지 검사한다. checkpoint 로드는 구조·schema 불일치를 즉시 실패시켜야 한다.

9. **[P2] 초~분 보유시간 제한과 추론 리스크 처리가 완성되지 않았다.**

   [환경 청산 조건](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/environments/scalping_env_e2e.py:961)은 5초 이후에도 가격상승률이 0 이하인 경우만 자동 청산한다. 수익 포지션 전체에 적용되는 최대 보유시간은 없다. [경과시간 계산](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/environments/scalping_env_e2e.py:825)은 기본 300초에서 시간을 잘라 장기 보유시간을 실제보다 짧게 보고한다.

   [추론 트레일링 스탑](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/inference/enhanced_grpo_infer_xlstm.py:222)은 현재 수익률이 활성화 기준 이상일 때만 검사한다. 실제 클래스로 수익률 +4% → +2.9%, activation 3%, callback 1%를 입력하면 1.1%p 하락했는데 HOLD가 반환됐다. 한 번 활성화되면 청산까지 상태를 유지해야 한다.

   `stagnation_exit_seconds`는 저장만 되고 predict에서 사용되지 않는다. `holding_period`도 제한 검사에 쓰이지 않는다. `Position.profit_rate`는 현재가로 계산하지 않고 외부 `cumulative_return`에 의존하므로 외부 갱신이 누락되면 가격 -3%에서도 0%로 보고할 수 있다.

   실제 경과시간은 그대로 보존하고 `max_holding_seconds`, 장 종료, 데이터 지연, 손실 한도 등은 별도 리스크 엔진에서 일관되게 처리해야 한다. 최저 보유시간을 도입하더라도 손절·비상청산을 막지 않도록 한다.

10. **[P2] 에피소드 마지막 틱을 반영하기 전에 강제 청산한다.**

    [종료 처리](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/environments/scalping_env_e2e.py:995)는 인덱스를 증가시킨 뒤 이전 current_price로 청산하고, 종료 시 가격·시간 갱신을 건너뛴다.

    합성 재현: 가격 [100, 101, 90]에서 buy → hold하면 마지막 시장가격 90 대신 101로 청산하여 비중 반영 순손익 +0.158%를 보고한다.

    이벤트 시점, 행동, 체결, 가격 갱신, 종료 평가 순서를 명시해야 한다. 내부 `episode_rewards`에도 종료 청산 보상이 추가되기 전에 값이 기록되어 반환 보상과 어긋나는 문제가 있다.

11. **[P2] 관측 현재가 즉시 체결 가정은 스캘핑 수익성을 검증하기에 부족하다.**

    [매수 체결](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/environments/scalping_env_e2e.py:883)와 902줄의 매도는 현재가로 고정 20% 비중이 즉시 체결됐다고 가정한다. 호가 스프레드·지연·부분체결·호가 잔량·주문 대기·취소 경합이 없다.

    먼저 시장가 중심의 보수적인 리플레이를 만들고, 의사결정 및 주문 지연 이후의 ask로 매수하고 bid로 매도하도록 해야 한다. 주문량에 따른 호가 소진을 반영한다. 지정가는 가격 도달만으로 전량 체결을 인정하지 말고 대기열과 체결량 가정을 둬야 한다. KRX의 가격·시간 우선 체결 설명은 [공식 거래 안내](https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf)를 참고했다.

    현재 저장소에서는 주문 ID·접수/부분체결/완전체결/취소 상태·재접속 잔고 대사까지 연결된 브로커 실행 계층이 확인되지 않았다. 다른 저장소나 외부 서비스에 있을 수 있으므로 시스템 전체에 없다고 단정하지 않는다.

**별도 경로에서 확인한 연결 문제**

12. **[P2] 원본 CSV 생성 경로가 시간과 호가 정보를 잃는다.**

    [_coalesce_into_base](D:/Workspace/Project/stock-bot/stock-bot2/scripts/data/generate_datasets.py:131)가 base만 수정한 뒤 163~165줄에서 이전 temp로 덮어써 병합된 시간 값이 사라진다. 합성 이벤트 09:00/09:00:01/09:00:02/09:00:03을 합치면 09:00/09:00/09:00:02/09:00:02로 변했다.

    171~180줄은 호가 가격×수량으로 대기금액을 계산하기 전에 최종 컬럼만 선택하여, 원본에 대기금액이 없으면 20개 대기금액 피처가 0으로 채워지고 원본 호가가 버려진다. [결측 보간](D:/Workspace/Project/stock-bot/stock-bot2/scripts/data/normalize_datasets.py:298)의 bfill은 처음 결측 구간에 미래 값을 넣는다.

    이 문제를 기존 캐시 전체에 일반화해서는 안 된다. 점검한 현재 NPZ 3개에서는 20개 호가 피처가 모두 0인 행이 없었다. 현재 캐시와 CSV 생성 경로의 계보를 분리하여 추적해야 한다. merge는 이벤트 시점에 이미 도착한 값만 사용하고 초기 결측은 유효성 마스크 또는 준비 완료 전 제외로 처리한다.

13. **[P2] BC 사전학습이 현재 RL 모델 규격과 연결되지 않는다.**

    [BC 관측 생성](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/pretrain_behavior_cloning_fast.py:74)와 일반 BC 버전은 포지션 피처 3개만 붙인다. 현재 환경은 15개다. 시장 피처 27개 기준 BC는 30채널, RL은 42채널이다. teacher 또는 이후 checkpoint 로드에서 conv1 채널 불일치가 발생한다. 원본 입력과 log+z-score 입력 차이도 있다.

    fast 버전 [JIT 실행](D:/Workspace/Project/stock-bot/stock-bot2/ai_trader/grpo/pretrain_behavior_cloning_fast.py:225)은 현재 mLSTM forward의 중첩 함수 때문에 Torch 2.8.0에서 `UnsupportedNodeError`로 실패하는 것을 소규모 재현했다. 관측 공통화 후 한 batch의 전방·역전파와 checkpoint 연결을 확인해야 한다.

14. **[P2] 기존 정규화 스크립트가 시간 피처 반환 개수 불일치로 실패한다.**

    [시간 피처 함수](D:/Workspace/Project/stock-bot/stock-bot2/lib/normalization.py:459)는 2개 값을 반환하지만 [호출부](D:/Workspace/Project/stock-bot/stock-bot2/scripts/data/normalize_datasets.py:566)는 3개를 기대한다. 정상 합성 데이터로 `ValueError`를 재현했다. 반환 개수를 보완한 조건에서는 기존 파생 컬럼과 새 컬럼이 중복되어 뒤에서 TypeError가 발생하는 문제도 확인했다. 이 발견은 해당 스크립트 경로에 한정한다.

15. **[P2] 테스트와 성능 문서를 현재 구현에 맞게 재구성해야 한다.**

    Python 소스 37개는 메모리 내 문법 컴파일을 통과했다. 이는 런타임이나 매매 정확성이 검증됐다는 뜻이 아니다. `pytest ai_trader scripts lib --collect-only -p no:cacheprovider -o addopts='' -v` 결과는 0개 수집, 종료코드 5였다. 루트 전체 수집은 기존 .pytest_cache의 접근권한 오류로 실패해 실제 소스 디렉터리를 대상으로 다시 확인했다.

    README의 38개 테스트 통과, AutoEncoder 실행 경로 및 2.87ms 수치는 현재 xLSTM 시스템의 재현 가능한 증거로 확인되지 않았다. 추론 지연은 데이터 수신 → 특징 생성 → 모델 → 주문 접수 → 체결까지 구간별 p50/p95/p99로 다시 측정해야 한다.

**개선 순서와 완료 기준**

1. 데이터와 회계 수정: 시간 단조 증가, 원본 가격 보존, 미래 꼬리 변경에 대한 관측 불변성, feature schema/단위 검증. 캐시의 생성 코드·원본 버전·변환 버전을 기록하고 재생성한다.
2. 시뮬레이터 수정: bid/ask 및 지연 이후 체결, 비용·수량·부분체결, FIFO 포지션 원장, 마지막 틱 청산, 계좌 평가액과 보상 대사, 실제 초 단위 최대 보유시간.
3. 학습 입력 및 목적 통일: 관측 공통 생성기, 순손익 중심 보상, 행동 이전 기준의 GRPO 비교, 재현 가능한 행동확률, BC/RL/추론 규격 일치.
4. 수익성 검증: 날짜 순서의 train/validation/test와 rolling walk-forward. 경계에서 입력창과 미래 보유·라벨 구간이 겹치는 샘플을 제거한다. 모델은 검증 구간에서만 선택한다. 종목별·거래일별·시간대별 결과와 미거래 기준을 함께 보고한다.
5. 실행 검증: 주문 없이 실시간 신호를 기록하는 shadow 운영, 브로커 모의체결과 시뮬레이터 체결 차이 비교, 주문 상태 및 잔고 대사 확인.

수익성 실험에는 비용 차감 거래당 기대수익, 일별 계좌수익, 최대낙폭, 거래 수, 노출시간, 회전율, 체결률, 보유시간 p50/p95, 지연·비용 악화 시 성과를 포함한다. 현재 Sharpe는 거래/보상 기반으로 계산되어 일정 시간 간격의 계좌수익 Sharpe와 다르므로 정의를 명확히 해야 한다.

모델 구조는 데이터와 평가를 고친 뒤 비교한다. 작은 지도학습 기준 모델로 여러 미래 시간 구간의 비용 차감 수익 또는 목표/손절/시간 만료 결과를 예측하고, 같은 체결·리스크 조건에서 작은 GRU/TCN 및 현재 xLSTM·GRPO와 비교하는 실험을 권한다. `seq_len`은 틱 수이므로 고정 초 단위 관측 길이와 같지 않다. 실제 timestamp로 계산한 `dt`와 남은 최대 보유시간을 입력에 포함하는 것이 목적에 맞는다.

이번 리뷰는 수익률을 산출하거나 보장한 결과가 아니다. 우선 데이터·회계 결함을 해소한 뒤, 분리된 기간과 현실적인 체결 조건에서 비용 차감 우위가 남는지 측정해야 한다.
