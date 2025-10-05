# 설계 문서

## 개요

이 설계는 초단위 스캘핑을 위한 임베딩 모델과 GRPO 강화학습 시스템을 구현합니다. 시스템은 두 가지 주요 컴포넌트로 구성됩니다:

1. **임베딩 모델**: 60개 이상의 매매 특징을 저차원 밀집 벡터로 변환하는 인코더
2. **GRPO 에이전트**: 그룹 상대 정책 최적화를 사용하여 스캘핑 전략을 학습하는 강화학습 에이전트

임베딩 모델은 대조 학습을 통해 시장 패턴을 학습하고, GRPO 에이전트는 이 임베딩을 관측값으로 사용하여 빠른 매매 결정을 내립니다. 3초 룰을 포함한 스캘핑 특화 보상 구조를 통해 에이전트는 빠른 진입/청산 전략을 학습합니다.

## 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    DuckDB 데이터베이스                        │
│         (datasets_norm_all.duckdb / table: datasets)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────────┬─────────────────────┐
                     │                  │                     │
                     ▼                  ▼                     ▼
          ┌──────────────────┐  ┌──────────────┐   ┌─────────────────┐
          │  임베딩 모델 훈련  │  │  GRPO 훈련   │   │  실시간 추론     │
          │  (Contrastive)   │  │  (RL Agent)  │   │  (Live Trading) │
          └──────────┬───────┘  └──────┬───────┘   └────────┬────────┘
                     │                  │                     │
                     ▼                  │                     │
          ┌──────────────────┐         │                     │
          │  훈련된 임베딩     │◄────────┴─────────────────────┘
          │  모델 체크포인트   │
          └──────────────────┘
                     │
                     ▼
          ┌──────────────────┐
          │  GRPO 정책        │
          │  체크포인트        │
          └──────────────────┘
```

### 데이터 흐름

1. **훈련 단계**:
   - DuckDB → 데이터 로더 → 정규화 → 임베딩 모델 훈련
   - 훈련된 임베딩 모델 → GRPO 환경 → GRPO 에이전트 훈련

2. **추론 단계**:
   - 실시간 데이터 → 정규화 → 임베딩 모델 → GRPO 정책 → 매매 결정

## 컴포넌트 및 인터페이스

### 1. 임베딩 모델 (`ai_trader/embedding/`)

#### 1.1 모델 아키텍처 (`models.py`)

```python
class TradingEmbeddingModel(nn.Module):
    """
    매매 데이터를 밀집 벡터로 인코딩하는 임베딩 모델
    
    아키텍처:
    - Input: (batch, seq_len, 60+) 특징
    - Conv1D layers: 지역 패턴 추출
    - Multi-head Attention: 중요 특징 포착
    - Output: (batch, embedding_dim) 벡터
    """
```


#### 1.2 대조 학습 손실 (`losses.py`)

```python
class InfoNCELoss(nn.Module):
    """
    대조 학습을 위한 InfoNCE 손실
    긍정 쌍의 유사도는 최대화, 부정 쌍의 유사도는 최소화
    """
```

#### 1.3 데이터 로더 (`data.py`)

```python
class EmbeddingDataLoader:
    """
    임베딩 모델 훈련을 위한 데이터 로더
    - DuckDB에서 데이터 로드
    - 시간 순서 기반 분할 (훈련 70%, 검증 15%, 테스트 15%)
    - 긍정/부정 쌍 생성
    """
```

#### 1.4 훈련 스크립트 (`train_embedding.py`)

```bash
# CLI 인터페이스
python -m ai_trader.embedding.train_embedding \
    --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
    --table datasets \
    --out models/embedding \
    --seq-len 60 \
    --embedding-dim 128 \
    --batch-size 128 \
    --epochs 50 \
    --lr 1e-4 \
    --device cuda
```

### 2. GRPO 강화학습 (`ai_trader/grpo/`)

#### 2.1 GRPO 환경 (`env.py`)

```python
class GRPOScalpingEnv(gym.Env):
    """
    스캘핑을 위한 GRPO 훈련 환경
    
    관측 공간: 임베딩 벡터 (embedding_dim,)
    행동 공간: Discrete(3) - 0: 보유, 1: 매수, 2: 매도
    
    보상 구조:
    - 수익: (청산가 - 진입가) / 진입가 - 거래비용
    - 거래비용: 수수료(0.015%) + 세금(0.2%) = 0.215% (매수/매도 각각 적용)
    - 총 거래비용: 0.43% (왕복)
    - 양의 보상 조건: 수익률 > 0.43%
    - 빠른 손절 룰 위반: -0.01 페널티 (시간 임계값 설정 가능, 기본값 1.5초)
    - 장기 보유 페널티: -0.001 * (보유시간 - 60초)
    """
    def __init__(self,
                 embedding_model: TradingEmbeddingModel,
                 db_path: str,
                 seq_len: int = 60,
                 transaction_cost_rate: float = 0.00215,  # 0.215%
                 quick_exit_threshold: float = 1.5,  # 빠른 손절 시간 임계값 (초)
                 quick_exit_penalty: float = 0.01):  # 빠른 손절 룰 위반 페널티
        self.transaction_cost_rate = transaction_cost_rate
        # 왕복 거래비용: 0.43%
        self.round_trip_cost = transaction_cost_rate * 2
        self.quick_exit_threshold = quick_exit_threshold
        self.quick_exit_penalty = quick_exit_penalty
```

**빠른 손절 룰 구현**:
- 매수 후 설정 가능한 시간 임계값(기본값 1.5초) 이내에 가격이 상승하지 않으면 자동 매도
- 위반 시 설정 가능한 페널티(기본값 0.01) 적용 및 카운터 증가
- 에피소드 메타데이터에 위반 횟수 기록
- CLI 인수로 임계값 및 페널티 조정 가능

#### 2.2 GRPO 알고리즘 (`grpo.py`)

```python
class GRPOTrainer:
    """
    Group Relative Policy Optimization 훈련기
    
    핵심 아이디어:
    - 에피소드를 시장 상황별로 그룹화 (변동성, 추세)
    - 그룹 내 상대 성능을 기반으로 어드밴티지 계산
    - PPO 스타일 클리핑으로 정책 업데이트
    
    그룹화 기준:
    - 변동성 분위수 (낮음/중간/높음/매우높음)
    - 추세 방향 (상승/하락/횡보)
    """
```

#### 2.3 정책 네트워크 (`policy.py`)

```python
class GRPOPolicy(nn.Module):
    """
    GRPO를 위한 정책 네트워크
    
    입력: 임베딩 벡터 (embedding_dim,)
    출력: 행동 확률 분포 (3,) - 보유/매수/매도
    """
```

#### 2.4 훈련 스크립트 (`train_grpo.py`)

```bash
# CLI 인터페이스
python -m ai_trader.grpo.train_grpo \
    --embedding-model models/embedding/checkpoint_epoch50.pt \
    --db "C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb" \
    --table datasets \
    --out models/grpo_scalping \
    --seq-len 60 \
    --episodes-per-group 12 \
    --num-groups 4 \
    --total-timesteps 1000000 \
    --quick-exit-threshold 1.5 \
    --quick-exit-penalty 0.01 \
    --device cuda
```

### 3. 추론 파이프라인 (`ai_trader/inference/`)

#### 3.1 실시간 추론 (`infer_grpo.py`)

```python
class GRPOInference:
    """
    실시간 매매를 위한 GRPO 추론 엔진
    
    최적화:
    - TorchScript 컴파일로 추론 속도 향상
    - 임베딩 캐시로 중복 계산 방지
    - 목표 지연 시간: < 10ms per sample
    """
```

## 데이터 모델

### 데이터베이스 스키마

```sql
-- 기존 테이블 (datasets_norm_all.duckdb)
TABLE datasets (
    날짜 DATE,
    종목코드 VARCHAR,
    번호 INTEGER,
    등락률 REAL,
    누적거래대금 REAL,
    거래회전율 REAL,
    체결강도 REAL,
    매도대기금액1 REAL,
    매도대기금액2 REAL,
    ... (60+ 특징),
    종목명_scalar REAL,
    시간_sin REAL,
    시간_cos REAL,
    시간_scalar2 REAL,
    INDEX idx_datasets_code_date_time (종목코드, 날짜, 번호)
)
```

### 데이터 분할 전략

- **훈련 세트**: 70% (시간 순서 기준 초기 70%)
- **검증 세트**: 15% (다음 15%)
- **테스트 세트**: 15% (마지막 15%)

시간 순서를 유지하여 미래 데이터 누출(look-ahead bias)을 방지합니다.

### 체크포인트 형식

```python
# 임베딩 모델 체크포인트
{
    'state_dict': model.state_dict(),
    'config': {
        'input_dim': 60,
        'embedding_dim': 128,
        'seq_len': 60,
        'num_heads': 4
    },
    'feature_names': ['등락률', '누적거래대금', ...],
    'normalization_stats': {
        'mean': [...],
        'std': [...]
    },
    'training_metadata': {
        'epoch': 50,
        'loss': 0.123,
        'val_loss': 0.145
    }
}

# GRPO 정책 체크포인트
{
    'state_dict': policy.state_dict(),
    'config': {
        'embedding_dim': 128,
        'hidden_dim': 256
    },
    'training_metadata': {
        'timesteps': 1000000,
        'mean_reward': 0.0234,
        'win_rate': 0.58,
        'quick_exit_violations': 1234,
        'quick_exit_threshold': 1.5,
        'quick_exit_penalty': 0.01
    }
}
```

## 오류 처리

### 1. 데이터 로딩 오류

```python
try:
    conn = duckdb.connect(db_path, read_only=True)
    data = conn.execute(query).fetchdf()
except Exception as e:
    logger.error(f"데이터베이스 연결 실패: {e}")
    raise DataLoadError(f"Cannot load data from {db_path}")
```

### 2. 모델 추론 오류

```python
try:
    embedding = self.embedding_model(sequence)
except RuntimeError as e:
    logger.error(f"임베딩 생성 실패: {e}")
    # 캐시된 이전 임베딩 사용 또는 기본값 반환
    return self.get_fallback_embedding()
```

### 3. 빠른 손절 룰 위반 처리

```python
if self.quick_exit_violations > self.max_violations:
    logger.warning(f"빠른 손절 룰 위반 횟수 초과: {self.quick_exit_violations}")
    # 에피소드 조기 종료 또는 페널티 증가
    return self.terminate_episode()
```

## 테스트 전략

### 1. 단위 테스트

```python
# tests/test_embedding_model.py
def test_embedding_output_shape():
    """임베딩 모델 출력 형태 검증"""
    model = TradingEmbeddingModel(input_dim=60, embedding_dim=128)
    x = torch.randn(32, 60, 60)  # (batch, seq_len, features)
    output = model(x)
    assert output.shape == (32, 128)

# tests/test_grpo_env.py
def test_quick_exit_rule():
    """빠른 손절 룰 동작 검증"""
    env = GRPOScalpingEnv(..., quick_exit_threshold=1.5)
    env.reset()
    env.step(1)  # 매수
    # 3초 후 가격 하락 시뮬레이션
    for _ in range(3):
        obs, reward, done, _, info = env.step(0)  # 보유
    assert env.quick_exit_violations == 1
    assert reward < 0  # 페널티 확인
    
def test_configurable_threshold():
    """설정 가능한 임계값 검증"""
    env = GRPOScalpingEnv(..., quick_exit_threshold=5, quick_exit_penalty=0.02)
    assert env.quick_exit_threshold == 5
    assert env.quick_exit_penalty == 0.02
```

### 2. 통합 테스트

```python
# tests/test_integration.py
def test_end_to_end_training():
    """전체 파이프라인 통합 테스트"""
    # 임베딩 모델 훈련
    embedding_model = train_embedding_model(...)
    
    # GRPO 훈련
    policy = train_grpo(embedding_model, ...)
    
    # 추론 테스트
    inference = GRPOInference(embedding_model, policy)
    action = inference.predict(test_sequence)
    assert action in [0, 1, 2]
```

### 3. 백테스팅

```python
# tests/test_backtest.py
def test_backtest_performance():
    """
    홀드아웃 테스트 데이터에서 성능 평가
    - 승률 > 50%
    - 샤프 비율 > 1.0
    - 최대 낙폭 < 10%
    - 평균 보유 시간 < 30초
    """
    results = run_backtest(policy, test_data)
    assert results['win_rate'] > 0.50
    assert results['sharpe_ratio'] > 1.0
    assert results['max_drawdown'] < 0.10
    assert results['avg_holding_time'] < 30
```

## 설계 결정 및 근거

### 1. 임베딩 모델에 대조 학습 사용

**결정**: InfoNCE 손실을 사용한 대조 학습

**근거**:
- 시간적으로 가까운 샘플은 유사한 시장 상황을 나타냄
- 종목별 특성을 보존하면서 일반화 가능한 표현 학습
- 라벨 없이 자기지도 학습 가능

### 2. GRPO 알고리즘 선택

**결정**: PPO 기반 그룹 상대 정책 최적화

**근거**:
- 시장 상황별로 다른 전략이 필요함 (변동성, 추세)
- 그룹 내 상대 성능 비교로 더 안정적인 학습
- PPO의 클리핑으로 정책 업데이트 안정성 확보

### 3. 빠른 손절 룰 구현

**결정**: 환경 레벨에서 강제 매도 및 페널티 적용, 시간 임계값 및 페널티 설정 가능

**근거**:
- 스캘핑의 핵심 규칙을 명시적으로 학습
- 빠른 손절을 장려하여 리스크 관리
- 에이전트가 자연스럽게 빠른 진입/청산 학습
- 다양한 시장 상황에 맞게 임계값 조정 가능 (예: 변동성 높은 시장에서는 5초, 낮은 시장에서는 2초)

### 4. 임베딩 캐시 사용

**결정**: 최근 임베딩을 메모리에 캐시

**근거**:
- 슬라이딩 윈도우로 인한 중복 계산 방지
- 실시간 추론 지연 시간 최소화 (< 10ms 목표)
- 메모리 사용량과 속도의 균형

### 5. 시간 순서 기반 데이터 분할

**결정**: 훈련 70%, 검증 15%, 테스트 15% (시간 순서 유지)

**근거**:
- 미래 데이터 누출 방지 (look-ahead bias)
- 실제 거래 시나리오와 유사한 평가
- 시간에 따른 시장 변화 반영

## TensorBoard 로깅

### 임베딩 모델 훈련

```python
# 로그 메트릭
- train/loss: 훈련 손실 (InfoNCE)
- val/loss: 검증 손실
- val/silhouette_score: 종목 클러스터링 품질
- val/temporal_coherence: 시간적 일관성
- learning_rate: 학습률
```

### GRPO 훈련

```python
# 로그 메트릭
- train/mean_reward: 평균 보상
- train/win_rate: 승률
- train/avg_holding_time: 평균 보유 시간
- train/quick_exit_violations: 빠른 손절 룰 위반 횟수
- train/policy_entropy: 정책 엔트로피
- train/kl_divergence: KL 발산
- group_0/mean_return: 그룹 0 평균 수익
- group_1/mean_return: 그룹 1 평균 수익
- group_2/mean_return: 그룹 2 평균 수익
- group_3/mean_return: 그룹 3 평균 수익
```

## 성능 최적화

### 1. 임베딩 모델 최적화

- TorchScript 컴파일로 추론 속도 향상
- Mixed precision training (FP16) 사용
- Gradient checkpointing으로 메모리 절약

### 2. 데이터 로딩 최적화

- DuckDB의 인덱스 활용 (idx_datasets_code_date_time)
- 청크 단위 스트리밍으로 메모리 효율성
- 멀티프로세싱 데이터 로더 (num_workers=4)

### 3. GRPO 훈련 최적화

- 벡터화된 환경으로 병렬 롤아웃 수집
- GPU 배치 처리로 정책 업데이트 가속
- 경험 재사용 (PPO의 multiple epochs)

## 디렉토리 구조

```
ai_trader/
├── embedding/
│   ├── __init__.py
│   ├── models.py           # TradingEmbeddingModel
│   ├── losses.py           # InfoNCELoss
│   ├── data.py             # EmbeddingDataLoader
│   └── train_embedding.py  # 훈련 CLI
├── grpo/
│   ├── __init__.py
│   ├── env.py              # GRPOScalpingEnv
│   ├── grpo.py             # GRPOTrainer
│   ├── policy.py           # GRPOPolicy
│   └── train_grpo.py       # 훈련 CLI
└── inference/
    ├── __init__.py
    └── infer_grpo.py       # GRPOInference

models/
├── embedding/
│   ├── checkpoint_epoch50.pt
│   └── tensorboard_logs/
└── grpo_scalping/
    ├── checkpoint_step1000000.pt
    └── tensorboard_logs/
```
