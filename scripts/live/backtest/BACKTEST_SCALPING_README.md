# GRPOScalpingEnv 백테스팅 가이드

## 📖 개요

`backtest_scalping.py`는 **GRPOScalpingEnv**와 **AutoEncoder 임베딩**을 사용하는 스캘핑 전략을 검증하는 백테스팅 스크립트입니다.

### 주요 특징

- ✅ **AutoEncoder 임베딩 사용**: 원본 특징을 128차원 임베딩으로 압축
- ✅ **스캘핑 특화 환경**: 빠른 손절, 장기 보유 페널티 등
- ✅ **상세한 거래 분석**: 수익률, 보유 시간, 승률 등
- ✅ **Quick Exit 룰**: 빠른 손절 위반 추적

---

## 🚀 사용법

### 기본 실행

```bash
python scripts/live/backtest/backtest_scalping.py \
    --policy models/grpo_scalping@2025120/model.pt \
    --embedding models/autoencoder@20251013/model.pt \
    --db "C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb" \
    --episodes 100
```

### 파라미터 조정

```bash
python scripts/live/backtest/backtest_scalping.py \
    --policy models/grpo_scalping@2025120/model.pt \
    --embedding models/autoencoder@20251013/model.pt \
    --db datasets_raw_all.duckdb \
    --episodes 100 \
    --transaction-cost 0.00215 \
    --quick-exit-threshold 1.5 \
    --quick-exit-penalty 0.01 \
    --max-holding-time 10.0 \
    --holding-penalty-rate 0.001 \
    --quick-exit-mode penalty_only
```

---

## 📋 파라미터 설명

### 필수 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--policy` | 정책 모델 경로 (.pt) | `models/grpo_scalping@2025120/policy.pt` |
| `--embedding` | AutoEncoder 임베딩 모델 경로 | `models/autoencoder@20251013/model.pt` |
| `--db` | DuckDB 데이터베이스 경로 | `datasets_raw_all.duckdb` |

### 백테스팅 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--episodes` | 실행할 에피소드 수 | 100 |
| `--device` | 디바이스 ('cuda' 또는 'cpu') | 'cuda' |

### 거래 비용 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--transaction-cost` | 편도 거래 비용 비율 | 0.00215 (0.215%) |

**왕복 거래 비용**: `0.00215 × 2 = 0.0043 (0.43%)`

### 스캘핑 규칙 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--quick-exit-threshold` | 빠른 손절 시간 임계값 (초) | 1.5 |
| `--quick-exit-penalty` | 빠른 손절 위반 페널티 | 0.01 |
| `--max-holding-time` | 최대 보유 시간 (초) | 10.0 |
| `--holding-penalty-rate` | 장기 보유 페널티 비율 | 0.001 |
| `--quick-exit-mode` | 빠른 손절 모드 | 'penalty_only' |

### Quick Exit Mode

**1. penalty_only (권장)**
- 빠른 손절 위반 시 페널티만 부여
- 정책이 학습하도록 유도
- 과도한 거래 방지

**2. force_close**
- 빠른 손절 위반 시 강제 청산
- 페널티 부여 + 자동 매도
- 과도한 거래 유발 가능

---

## 📊 출력 예시

```
================================================================================
📊 Scalping Backtest Results
================================================================================

📈 Performance:
Episodes: 100
Mean Reward: 156.32 ± 245.67
Min/Max Reward: -89.45 / 892.34
Total Return: 15632.10
Mean Return per Episode: 156.32

💰 Trading:
Total Trades: 3245
Trades per Episode: 32.45
Winning Trades: 2601 (80.15%)
Losing Trades: 644
Forced Liquidations: 12

📊 Profit Analysis:
Mean Profit Rate: 1.23% ± 2.45%
Median Profit Rate: 0.87%
Min/Max Profit Rate: -5.67% / 8.92%

💚 Winning Trades:
Mean Profit: 1.85%
Max Profit: 8.92%

💔 Losing Trades:
Mean Loss: -1.12%
Max Loss: -5.67%

⏱️ Holding Time Analysis:
Mean: 5.67s
Median: 4.23s
Min/Max: 0.50s / 15.89s

📈 Risk Metrics:
Mean Episode Win Rate: 80.15%
Mean Sharpe Ratio: 1.45

⚠️ Quick Exit Violations:
Total: 245
Average per Episode: 2.45
================================================================================
```

---

## 🔍 DirectFeatureEnv와의 차이

| 특징 | DirectFeatureEnv | GRPOScalpingEnv |
|------|------------------|-----------------|
| **입력** | 원본 특징 (24차원) | 임베딩 벡터 (128차원) |
| **임베딩** | ❌ 없음 | ✅ AutoEncoder |
| **스캘핑 규칙** | ❌ 없음 | ✅ Quick exit, 보유 페널티 |
| **거래 비용** | ✅ 0.215% | ✅ 0.215% |
| **보상 구조** | 단순 | 복잡 (스캘핑 특화) |

---

## ⚙️ 환경 구성

### GRPOScalpingEnv 파라미터

```python
GRPOScalpingEnv(
    embedding_model=autoencoder,          # AutoEncoder 모델
    db_path='datasets.duckdb',            # 데이터베이스
    table_name='datasets',                # 테이블명
    seq_len=60,                           # 시퀀스 길이
    embedding_dim=128,                    # 임베딩 차원
    expected_features=28,                 # 입력 특징 수
    transaction_cost_rate=0.00215,        # 거래 비용
    quick_exit_threshold=1.5,             # 빠른 손절 임계값
    quick_exit_penalty=0.01,              # 빠른 손절 페널티
    max_holding_time=10.0,                # 최대 보유 시간
    holding_penalty_rate=0.001,           # 보유 페널티 비율
    max_episode_steps=None,               # 최대 스텝 (None=무제한)
    quick_exit_mode='penalty_only',       # Quick exit 모드
    device='cuda'
)
```

---

## 📈 성능 지표 설명

### 기본 지표

- **Mean Reward**: 에피소드당 평균 보상
- **Total Return**: 전체 누적 보상
- **Win Rate**: 수익 거래 비율

### 거래 지표

- **Total Trades**: 전체 거래 횟수
- **Trades per Episode**: 에피소드당 평균 거래 횟수
- **Forced Liquidations**: 강제 청산 횟수 (에피소드 종료 시)

### 수익률 지표

- **Mean Profit Rate**: 평균 수익률 (%)
- **Median Profit Rate**: 중앙값 수익률
- **Mean Winning Profit**: 수익 거래 평균 수익률
- **Mean Losing Profit**: 손실 거래 평균 손실률

### 시간 지표

- **Mean Holding Time**: 평균 보유 시간 (초)
- **Median Holding Time**: 중앙값 보유 시간

### 리스크 지표

- **Sharpe Ratio**: 샤프 비율 (리스크 대비 수익)
- **Quick Exit Violations**: 빠른 손절 규칙 위반 횟수

---

## 🧪 테스트 시나리오

### 1. 기본 설정 테스트

```bash
python scripts/live/backtest_scalping.py \
    --episodes 100
```

**목적**: 기본 설정으로 모델 성능 확인

### 2. 보수적 설정 테스트

```bash
python scripts/live/backtest_scalping.py \
    --episodes 100 \
    --quick-exit-threshold 1.0 \
    --max-holding-time 5.0 \
    --holding-penalty-rate 0.002
```

**목적**: 더 빠른 손절과 짧은 보유 시간 강제

### 3. 공격적 설정 테스트

```bash
python scripts/live/backtest_scalping.py \
    --episodes 100 \
    --quick-exit-threshold 3.0 \
    --max-holding-time 20.0 \
    --holding-penalty-rate 0.0005
```

**목적**: 더 긴 보유 시간 허용

### 4. Force Close 모드 테스트

```bash
python scripts/live/backtest_scalping.py \
    --episodes 100 \
    --quick-exit-mode force_close
```

**목적**: 강제 청산 모드의 영향 확인

---

## 📝 주의사항

### 1. 모델 파일 확인

백테스팅 전 다음 파일들이 존재하는지 확인:
- ✅ 정책 모델 (.pt)
- ✅ AutoEncoder 임베딩 모델 (.pt)
- ✅ 데이터베이스 (.duckdb)

### 2. 메모리 사용량

AutoEncoder를 사용하므로 메모리 사용량이 높을 수 있습니다.
- **권장**: GPU 사용 (--device cuda)
- **CPU 사용 시**: 에피소드 수를 줄이거나 배치 사이즈 조정

### 3. 데이터베이스 테이블

환경은 `datasets` 테이블에서 데이터를 읽습니다.
- 테이블명이 다른 경우 코드 수정 필요
- 필요한 컬럼: 28개 특징 (등락률, 누적거래대금, 거래회전율, 체결강도, 호가 20개)

### 4. Quick Exit Mode

- **penalty_only**: 학습 효과적, 권장
- **force_close**: 과도한 거래 유발 가능, 주의

---

## 🔧 문제 해결

### 1. 모델 로드 실패

```
ERROR: Policy model not found: models/grpo_scalping@2025120/policy.pt
```

**해결**: 모델 경로 확인 및 수정

```bash
python scripts/live/backtest_scalping.py \
    --policy path/to/your/policy.pt \
    --embedding path/to/your/autoencoder.pt
```

### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결**: CPU 사용 또는 에피소드 수 감소

```bash
python scripts/live/backtest_scalping.py \
    --device cpu \
    --episodes 50
```

### 3. 데이터베이스 연결 실패

```
ERROR: Database not found: datasets_raw_all.duckdb
```

**해결**: 데이터베이스 경로 확인

```bash
python scripts/live/backtest_scalping.py \
    --db /absolute/path/to/your/database.duckdb
```

---

## 📚 관련 문서

- [GRPOScalpingEnv 환경 문서](../../ai_trader/grpo/environments/scalping_env.py)
- [AutoEncoder 모델](../../ai_trader/models/autoencoder.py)
- [DirectFeatureEnv 백테스팅](backtest_strategy.py)

---

## 🎯 다음 단계

백테스팅 후:

1. ✅ 결과 분석 및 성능 평가
2. ✅ 파라미터 튜닝
3. ✅ 실전 시뮬레이션 테스트
4. ✅ 소액 실전 투입

**성공적인 백테스팅을 기원합니다!** 🚀
