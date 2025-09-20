# ai_trader

SQLite의 REAL(숫자) 타입 피처 컬럼들로 구성된 데이터셋을 기반으로 한 ML/RL 학습·추론 유틸리티입니다.

- 지도학습: CNN + LSTM + Multi-Head Attention (PyTorch)
- 강화학습: Gymnasium + Stable-Baselines3 기반 PPO / A2C

## 디렉터리 구조
- `ai_trader/ml/`: 지도학습 모듈
  - `data.py`: SQLite 로더(REAL/숫자 컬럼만 사용), 슬라이딩 윈도우 생성
  - `models.py`: `CNNLSTMAttn` 모델
  - `train_supervised.py`: 학습 CLI
  - `infer_supervised.py`: 추론 CLI
- `ai_trader/rl/`: 강화학습 모듈
  - `env.py`: `TradingEnv` (SQLite REAL 컬럼 사용)
  - `policies.py`: 2D 윈도우용 `TimeSeriesCNNExtractor`
  - `train_rl.py`: PPO/A2C 학습 CLI
  - `infer_rl.py`: PPO/A2C 추론 CLI

## 사전 준비
- Python 3.10+
- 프로젝트 루트에서 의존성 설치:
```bash
pip install -r requirements.txt
```

Windows + CUDA 미사용 환경에서는 PyTorch CPU 전용 휠을 사용하는 것을 권장합니다(설치 방법은 PyTorch 공식 문서 참고).

## 데이터 전제
- 기본적으로 하나의 SQLite DB에 `datasets` 테이블이 존재한다고 가정합니다. 구성 예:
  - 키 컬럼(선택): `날짜`, `종목코드`, `번호`
  - 다수의 피처 컬럼: REAL(또는 FLOAT/DOUBLE/NUMERIC) 타입. 비숫자 컬럼은 모델 입력에서 제외됩니다.
- `scripts/normalize_datasets.py`를 통해 이러한 DB를 생성할 수 있습니다.

## 지도학습(Supervised Learning)
`seq_len` 길이의 최근 시퀀스로부터 `horizon` 스텝 이후의 `target_col` 값을 회귀 예측합니다.

```bash
python -m ai_trader.ml.train_supervised \
  --db models/datasets.db \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 1 \
  --target-col 현재가
```

- 선택 필터:
  - `--code 005930` (특정 종목 필터)
  - `--date 20240501` (특정 일자 필터)
- 출력물:
  - `models/supervised/model.pt`: 체크포인트(state_dict, config, feature_names 등 포함)
  - `models/supervised/config.json`: 학습 메타데이터

최근 윈도우에 대한 추론:
```bash
python -m ai_trader.ml.infer_supervised \
  --db models/datasets.db \
  --ckpt models/supervised \
  --top-k 10
```

## 강화학습(Reinforcement Learning)
`TradingEnv`는 `seq_len` 길이의 피처 윈도우를 관측으로 제공하며, `target_col` 변화 기반 보상을 계산합니다.

### 보상 커스터마이징
- `--reward-mode`: `delta`, `return`, `log_return` 중 선택
- `--tc-bps`: 거래 비용(기준점(bp) 단위, 예: `5` = 0.05%) — 포지션 오픈/청산 시 적용
- `--hc-bps`: 보유 비용(포지션 유지 중 매 스텝마다 bp 단위 비용)

### 관측 포맷 및 CNN 익스트랙터
- `--obs-format flat`(기본): 관측이 `(T*F,)` 형태로 평탄화
- `--obs-format matrix`: `(T, F)` 형태의 2D 관측
- `--policy cnn`: `(T, F)` 윈도우를 처리하는 `TimeSeriesCNNExtractor` 사용 (SB3 `MlpPolicy`의 커스텀 features extractor로 동작)

### 지도학습 인코더(CNN+LSTM+MHA)와 PPO 결합
`env.py`는 `--encoder-ckpt`를 통해 지도학습된 `CNNLSTMAttn` 인코더를 불러와, 관측 윈도우를 상태 벡터(`state_dim = 2*lstm_hidden`)로 변환하여 RL 정책에 제공합니다. 이렇게 하면 지도학습 표현과 PPO를 결합한 파이프라인이 됩니다.

예시:
```bash
# 1) 지도학습 인코더 학습
python -m ai_trader.ml.train_supervised \
  --db models/datasets.db --table datasets \
  --out models/supervised --seq-len 60 --horizon 1 --target-col 현재가

# 2) PPO 학습 시 인코더 결합
python -m ai_trader.rl.train_rl \
  --algo ppo \
  --db models/datasets.db \
  --out models/rl_ppo_enc \
  --seq-len 60 --target-col 현재가 \
  --encoder-ckpt models/supervised \
  --total-timesteps 200000 --device cpu
```

### 스캘핑 휴리스틱과 3초 손절(초단위 스텝 가정)
`env.py`에는 초단위 스텝 스캘핑에 유용한 다음 옵션들이 추가되었습니다. 기본값은 비활성화(0.0 또는 보수적)이며, 필요 시 활성화하세요.

- `--w-fast`: 빠른 체결/활동성(최근 절대수익의 합) 가중치. 높을수록 활발한 테이프에서 매수 우선순위↑
- `--w-sticky`: 가격 점착성(최근 하락 빈도 낮음) 가중치. 높을수록 쉽게 떨어지지 않는 종목 매수 우선순위↑
- `--priority-bonus`: 위 휴리스틱의 가중합 점수에 곱해, 롱 진입 시 보상에 추가되는 보너스 스케일
- `--activity-window`: 휴리스틱 계산에 쓰는 최근 관측 창 길이(스텝)
- `--stoploss-wait`: 매수 후 n초(스텝) 내 가격이 진입가 이상으로 오르지 않으면 자동 손절(포지션 강제 청산)

예시(휴리스틱 활성화 + 3초 손절):
```bash
python -m ai_trader.rl.train_rl \
  --algo ppo \
  --db models/datasets.db \
  --out models/rl_ppo_scalp \
  --seq-len 60 --target-col 현재가 \
  --policy cnn --obs-format matrix \
  --reward-mode return --tc-bps 30 --hc-bps 1 \
  --w-fast 0.5 --w-sticky 0.5 \
  --priority-bonus 0.1 --activity-window 5 \
  --stoploss-wait 3 \
  --device cpu
```

### PPO 학습 예시
CPU + MLP 정책:
```bash
python -m ai_trader.rl.train_rl \
  --algo ppo \
  --db models/datasets.db \
  --out models/rl_ppo \
  --seq-len 60 \
  --target-col 현재가 \
  --total-timesteps 200000 \
  --device cpu
```

CNN 익스트랙터 + 수익률 기반 보상 + 비용 적용:
```bash
python -m ai_trader.rl.train_rl \
  --algo ppo \
  --db models/datasets.db \
  --out models/rl_ppo_cnn \
  --seq-len 60 \
  --target-col 현재가 \
  --reward-mode return \
  --tc-bps 5 --hc-bps 1 \
  --policy cnn --obs-format matrix \
  --device cpu
```

### RL 추론
```bash
python -m ai_trader.rl.infer_rl \
  --algo ppo \
  --db models/datasets.db \
  --model models/rl_ppo/ppo_model \
  --seq-len 60 \
  --target-col 현재가 \
  --w-fast 0.5 --w-sticky 0.5 \
  --priority-bonus 0.1 --activity-window 5 \
  --stoploss-wait 3
```
- 한 에피소드에 대한 누적 보상(대략적 PnL)을 출력합니다.
- 로더는 기본적으로 REAL/숫자 컬럼만 사용합니다. `target_col`이 숫자형인지 확인하세요.

## 옵션 상세

- **reward_mode**: 보상 계산 방식 선택
  - `delta`: px_next - px_t (가격 차이)
  - `return`: (px_next / px_t - 1)
  - `log_return`: log(px_next) - log(px_t)
  - 스케일이 달라 비용/보너스의 상대적 영향이 달라지므로 주의

- **transaction_cost_bps**: 거래 비용(bp)
  - 1bp = 0.01%, 5bp = 0.05%
  - 진입/청산 시 각각 1회 차감. 3초 강제 손절 시에도 1회 추가 차감
  - 과도한 매매를 억제하고 스프레드/수수료를 근사

- **holding_cost_bps**: 보유 비용(bp, per-step)
  - 포지션 보유 중인 매 스텝(초)마다 가격에 비례해 차감
  - 오래 들고 가는 전략을 억제하여 초단기 스캘핑을 유도

- **seq_len / target_col / scale_obs**
  - `seq_len`: 관측 윈도우 길이(T). 인코더 체크포인트와 반드시 동일해야 함
  - `target_col`: 보상 계산의 기준이 되는 가격 유사 컬럼(예: 현재가)
  - `scale_obs`: 관측 피처 표준화 사용 여부

- **policy / obs_format / encoder_ckpt**
  - `policy`: `mlp` 또는 `cnn`(커스텀 CNN 익스트랙터)
  - `obs_format`: `flat`(T*F) 또는 `matrix`(T,F). `policy=cnn`이면 `matrix` 권장
  - `encoder_ckpt`: CNN+LSTM+MHA 지도학습 인코더를 불러 관측을 상태벡터로 변환

- **w_fast / w_sticky / priority_bonus / activity_window / stoploss_wait**
  - `w_fast`: 최근 절대수익 합(활동성) 가중치. 활발한 테이프에서 매수 우선순위↑
  - `w_sticky`: 하락 빈도 낮음(점착성) 가중치. 쉽게 안 떨어지는 종목 우선순위↑
  - `priority_bonus`: 위 휴리스틱 가중합 점수에 곱해 롱 진입 시 보상 보너스
  - `activity_window`: 휴리스틱 계산에 쓰는 최근 창 길이(스텝)
  - `stoploss_wait`: 매수 후 n스텝 내 가격이 진입가 이상으로 오르지 않으면 강제 손절

- **max_steps / total_timesteps / device**
  - `max_steps`: 에피소드 당 최대 스텝(데이터 절단 또는 학습 안정화용)
  - `total_timesteps`: SB3 학습 스텝 수
  - `device`: `cpu` 또는 `cuda`

### 튜닝 가이드(예시)

- 거래 과다 → `transaction_cost_bps`를 올려 빈도 억제
- 보유 시간 과다 → `holding_cost_bps`를 소폭 올려 단타 유도
- 진입 타이밍 미흡 → `w_fast`, `w_sticky`와 `priority_bonus`를 소량 가중(0.05~0.2)
- 3초 룰 강화/완화 → `stoploss_wait` 조정(1~5 사이 실험)

### 옵션 예시 조합

- 학습(train):
```bash
python -m ai_trader.rl.train_rl \
  --algo ppo \
  --db models/datasets.db \
  --out models/rl_ppo_scalp \
  --seq-len 60 --target-col 현재가 \
  --policy cnn --obs-format matrix \
  --reward-mode return --tc-bps 30 --hc-bps 1 \
  --w-fast 0.5 --w-sticky 0.5 \
  --priority-bonus 0.1 --activity-window 5 \
  --stoploss-wait 3 \
  --total-timesteps 200000 \
  --device cpu
```

- 추론(infer):
```bash
python -m ai_trader.rl.infer_rl \
  --algo ppo \
  --db models/datasets.db \
  --model models/rl_ppo_scalp/ppo_model \
  --seq-len 60 --target-col 현재가 \
  --w-fast 0.5 --w-sticky 0.5 \
  --priority-bonus 0.1 --activity-window 5 \
  --stoploss-wait 3
```

## 문제 해결(트러블슈팅)
- `no such table: datasets` 오류:
  - `--db`에 절대경로를 사용해 실행해보세요.
  - 다음으로 테이블 목록을 확인하세요:
    ```python
    import sqlite3
    con = sqlite3.connect('models/datasets.db')
    print([r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()])
    ```
  - 테이블에 REAL 타입의 숫자 피처 컬럼이 존재하는지 확인하세요.
- Windows에서 PyTorch 임포트 오류(`fbgemm.dll`)가 날 경우:
  - Python 버전에 맞는 PyTorch CPU 전용 휠을 설치하고, MSVC 재배포 패키지가 설치되어 있는지 확인하세요.
- PPO + MLP는 GPU 효율이 낮습니다. CNN 또는 대규모 네트워크가 아니면 `--device cpu`를 권장합니다.

## 확장 아이디어
- `ai_trader/rl/env.py`에 슬리피지, 포지션 사이징, 숏 포지션 등을 추가
- `TimeSeriesCNNExtractor`를 1D Temporal CNN 또는 Transformer 기반 익스트랙터로 대체
- SB3 학습에 TensorBoard 콜백을 추가하여 지표 로깅
