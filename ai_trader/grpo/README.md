# Stock Bot

## 목차

- [Features](#features)
- [데이터 정규화](#데이터-정규화)
- [GRPO 강화학습 (Reinforcement Learning)](#grpo-강화학습-reinforcement-learning)
- [트러블슈팅](#트러블슈팅)

## Features

```bash
날짜 번호 종목코드 종목명 시간 등락률 누적거래대금 거래회전율 체결강도 매도대기금액1 매도대기금액2 매도대기금액3 매도대기금액4 매도대기금액5 매도대기금액6 매도대기금액7 매도대기금액8 매도대기금액9 매도대기금액10 매수대기금액1 매수대기금액2 매수대기금액3 매수대기금액4 매수대기금액5 매수대기금액6 매수대기금액7 매수대기금액8 매수대기금액9 매수대기금액10 종목명_scalar 시간_sin 시간_cos 시간_scalar
```

## 데이터 정규화

```bash
python scripts/generate_datasets.py \
  "D:\Workspace\Project\stock-bot\hoga-crawler\data" \
  -o "C:\Users\user\Workspace\datasets@raw\datasets.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --checkpoint-interval 50

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets@raw" \
  --out "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --threads 4 \
  --memory-limit 64GB

python scripts/normalize_datasets.py \
  "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  -o "C:\Users\user\Workspace\datasets\datasets_norm.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --checkpoint-interval 50 \
  --ignoring-stocks-csv scripts/ignoring_stocks.csv

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets" \
  --out "C:\Users\user\Workspace\datasets\datasets_norm_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --threads 4 \
  --memory-limit 64GB
```

## GRPO 강화학습 (Reinforcement Learning)

GRPO(Group Relative Policy Optimization)는 시장 상황별로 에피소드를 그룹화하여 상대적 성능을 기반으로 정책을 최적화합니다.

### GRPO 에이전트 훈련

```bash
python -m ai_trader.grpo.train_grpo \
  --embedding-model models/embedding/checkpoint_epoch50.pt \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/grpo_scalping \
  --seq-len 60 \
  --episodes-per-group 12 \
  --num-groups 4 \
  --total-timesteps 1000000 \
  --lr 3e-4 \
  --gamma 0.99 \
  --clip-epsilon 0.2 \
  --device cuda \
  --quick-exit-threshold 1.5 \
  --quick-exit-penalty 0.01 \
  --transaction-cost 0.00215 \
  --ckpt-every 50000
```

#### CLI 인수 설명

**필수 인수:**

- `--embedding-model`: 훈련된 임베딩 모델 체크포인트 경로
- `--db`: DuckDB 데이터베이스 경로
- `--table`: 데이터베이스 테이블 이름
- `--out`: 정책 체크포인트 저장 디렉토리

**환경 설정:**

- `--seq-len`: 입력 시퀀스 길이 (기본값: 60, 임베딩 모델과 동일해야 함)
- `--transaction-cost`: 거래 비용 비율 (기본값: 0.00215, 즉 0.215%)
- `--max-holding-time`: 최대 보유 시간 (초, 기본값: 60)

**빠른 손절 룰:**

- `--quick-exit-threshold`: 빠른 손절 시간 임계값 (초, 기본값: 1.5)
- `--quick-exit-penalty`: 빠른 손절 룰 위반 페널티 (기본값: 0.01)

**GRPO 알고리즘 설정:**

- `--episodes-per-group`: 그룹당 에피소드 수 (기본값: 12, 권장: 8-16)
- `--num-groups`: 시장 상황 그룹 수 (기본값: 4)
- `--total-timesteps`: 총 훈련 타임스텝 (기본값: 1000000)

**정책 네트워크 설정:**

- `--hidden-dim`: 정책 네트워크 은닉층 차원 (기본값: 256)
- `--lr`: 학습률 (기본값: 3e-4)
- `--gamma`: 할인 계수 (기본값: 0.99)
- `--clip-epsilon`: PPO 클리핑 파라미터 (기본값: 0.2)
- `--kl-target`: KL 발산 목표값 (기본값: 0.01)

**체크포인트 및 로깅:**

- `--ckpt-every`: 체크포인트 저장 주기 (타임스텝, 기본값: 50000)
- `--resume-from`: 체크포인트에서 재개
- `--device`: 디바이스 (cuda/cpu, 기본값: cuda)

### GRPO 에이전트 평가 및 백테스팅

```bash
python -m ai_trader.grpo.evaluate_grpo \
  --embedding-model models/embedding/checkpoint_epoch50.pt \
  --policy models/grpo_scalping/checkpoint_step1000000.pt \
  --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
  --table datasets \
  --device cuda \
  --num-episodes 1000
```

평가 메트릭:

- **승률 (Win Rate)**: 수익 거래 비율 (목표: > 50%)
- **거래당 평균 수익**: 거래당 평균 수익률
- **샤프 비율 (Sharpe Ratio)**: 위험 대비 수익 (목표: > 1.0)
- **최대 낙폭 (Max Drawdown)**: 최대 손실 폭 (목표: < 10%)
- **평균 보유 시간**: 평균 포지션 보유 시간 (목표: < 30초)
- **빠른 손절 룰 위반 횟수**: 1.5초 룰 위반 횟수

### 실시간 추론

```bash
python -m ai_trader.inference.grpo_infer \
  --embedding-model models/embedding/checkpoint_epoch50.pt \
  --policy models/grpo_scalping/checkpoint_step1000000.pt \
  --device cuda \
  --compile
```

추론 최적화:

- `--compile`: TorchScript 컴파일로 추론 속도 향상
- 임베딩 캐시로 중복 계산 방지
- 목표 지연 시간: < 10ms per sample

## 지도학습 훈련

- 러닝레이트 2e-4 ~ 5e-4
- 훈련 방식
  - “상승/하락 분류” 같은 분류 문제라면 → CrossEntropyLoss가 무조건 더 좋습니다.
    - direction3-threshold → 0.015~0.02
  - “가격 수치 예측” 같은 회귀 문제라면 → Huber가 더 낫습니다.

## 트러블슈팅

### GRPO 훈련

- **정책이 수렴하지 않는 경우**:
  - episodes-per-group을 늘리세요 (12 → 16).
  - 학습률을 낮추세요 (3e-4 → 1e-4).
  - clip-epsilon을 조정하세요 (0.2 → 0.1).
- **빠른 손절 룰 위반이 많은 경우**:
  - quick-exit-threshold를 늘리세요 (1.5초 → 3초).
  - quick-exit-penalty를 높여 페널티를 강화하세요.
- **승률이 낮은 경우**:
  - 임베딩 모델을 더 오래 훈련하세요.
  - transaction-cost를 확인하고 보상 구조를 검토하세요.

## 예제 스크립트

`examples/` 디렉토리에서 다양한 사용 예제를 확인할 수 있습니다:

- `grpo_backtest_example.py`: GRPO 에이전트 백테스팅
- `grpo_evaluation_example.py`: GRPO 에이전트 평가
- `grpo_inference_example.py`: 실시간 추론
- `grpo_predict_example.py`: 단일 예측
- `grpo_live_trading_example.py`: 실시간 거래 (Kiwoom API 연동)
- `generate_html_report_example.py`: HTML 보고서 생성
- `grpo_alert_example.py`: 성능 알림 시스템

자세한 사용법은 각 예제 파일을 참조하세요.
