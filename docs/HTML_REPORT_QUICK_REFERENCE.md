# HTML 보고서 생성 빠른 참조

## 기본 사용법

### 임베딩 모델 보고서

```python
from ai_trader.reporting import generate_embedding_report

generate_embedding_report(
    output_path='models/embedding/report.html',
    training_metadata={'timestamp': '2025-10-05 14:30:00', 'epoch': 50},
    final_metrics={'val_loss': 0.267, 'silhouette_score': 0.623, 'temporal_coherence': 0.781},
    config={'embedding_dim': 128, 'seq_len': 60},
    tensorboard_dir='models/embedding/tensorboard_logs'  # 선택사항
)
```

### GRPO 에이전트 보고서

```python
from ai_trader.reporting import generate_grpo_report

generate_grpo_report(
    output_path='models/grpo/report.html',
    training_metadata={
        'timestamp': '2025-10-05 18:45:00',
        'total_timesteps': 1000000,
        'num_updates': 5000,
        'quick_exit_threshold': 1.5
    },
    final_metrics={
        'win_rate': 0.58,
        'sharpe_ratio': 1.42,
        'avg_holding_time': 24.3,
        'avg_profit_per_trade': 0.0067
    },
    config={'embedding_dim': 128, 'hidden_dim': 256},
    sample_episodes=[  # 선택사항
        {'total_profit': 0.0123, 'num_trades': 5, 'avg_holding_time': 18.5}
    ],
    tensorboard_dir='models/grpo/tensorboard_logs'  # 선택사항
)
```

### 통합 보고서

```python
from ai_trader.reporting import generate_combined_report

generate_combined_report(
    output_path='models/combined_report.html',
    embedding_metadata={'timestamp': '2025-10-05 14:30:00', 'epoch': 50},
    embedding_metrics={'val_loss': 0.267, 'silhouette_score': 0.623},
    grpo_metadata={'timestamp': '2025-10-05 18:45:00', 'total_timesteps': 1000000},
    grpo_metrics={'win_rate': 0.58, 'sharpe_ratio': 1.42},
    sample_episodes=None  # 선택사항
)
```

## 메트릭 색상 코드

### 임베딩 모델
- **검증 손실**: 🟢 <0.5 | 🟡 0.5-1.0 | 🔴 >1.0
- **Silhouette Score**: 🟢 >0.5 | 🟡 0.3-0.5 | 🔴 <0.3
- **Temporal Coherence**: 🟢 >0.7 | 🟡 0.5-0.7 | 🔴 <0.5

### GRPO 에이전트
- **승률**: 🟢 >55% | 🟡 50-55% | 🔴 <50%
- **샤프 비율**: 🟢 >1.5 | 🟡 1.0-1.5 | 🔴 <1.0
- **평균 보유 시간**: 🟢 <30초 | 🟡 30-60초 | 🔴 >60초
- **거래당 수익**: 🟢 >0.5% | 🟡 0-0.5% | 🔴 <0%

## 필수 메트릭

### 임베딩 모델
```python
final_metrics = {
    'train_loss': float,      # 훈련 손실
    'val_loss': float,        # 검증 손실
    'silhouette_score': float,  # 클러스터링 품질
    'temporal_coherence': float  # 시간적 일관성
}
```

### GRPO 에이전트
```python
final_metrics = {
    'mean_reward': float,           # 평균 보상
    'win_rate': float,              # 승률 (0-1)
    'sharpe_ratio': float,          # 샤프 비율
    'max_drawdown': float,          # 최대 낙폭 (0-1)
    'avg_holding_time': float,      # 평균 보유 시간 (초)
    'avg_profit_per_trade': float,  # 거래당 평균 수익 (0-1)
    'quick_exit_violations': int,   # 빠른 손절 위반 횟수
    'total_trades': int             # 총 거래 횟수
}
```

## 샘플 에피소드 형식

```python
sample_episodes = [
    {
        'total_profit': float,        # 총 수익 (0-1)
        'num_trades': int,            # 거래 횟수
        'avg_holding_time': float,    # 평균 보유 시간 (초)
        'sharpe_ratio': float,        # 샤프 비율
        'quick_exit_violations': int, # 빠른 손절 위반 (선택)
        'stock_code': str,            # 종목 코드 (선택)
        'start_time': str             # 시작 시간 (선택)
    }
]
```

## 예제 실행

```bash
# 전체 예제 실행
python examples/generate_html_report_example.py

# 생성된 보고서 확인
# - models/embedding/training_report.html
# - models/grpo_scalping/training_report.html
# - models/combined_training_report.html
```

## 훈련 스크립트 통합

### 임베딩 훈련 스크립트에 추가

```python
from ai_trader.reporting import generate_embedding_report

# 훈련 완료 후
report_path = generate_embedding_report(
    output_path=os.path.join(args.out, 'training_report.html'),
    training_metadata={
        'timestamp': datetime.now().isoformat(),
        'epoch': args.epochs
    },
    final_metrics={
        'train_loss': train_loss,
        'val_loss': val_loss,
        'silhouette_score': silhouette,
        'temporal_coherence': temporal_coherence
    },
    config=model.get_config(),
    tensorboard_dir=str(tensorboard_dir)
)
logger.info(f"Training report: {report_path}")
```

### GRPO 훈련 스크립트에 추가

```python
from ai_trader.reporting import generate_grpo_report

# 훈련 완료 후
report_path = generate_grpo_report(
    output_path=os.path.join(args.out, 'training_report.html'),
    training_metadata={
        'timestamp': datetime.now().isoformat(),
        'total_timesteps': trainer.total_timesteps,
        'num_updates': trainer.num_updates,
        'quick_exit_threshold': args.quick_exit_threshold,
        'quick_exit_penalty': args.quick_exit_penalty
    },
    final_metrics=final_metrics,
    config=policy.get_config(),
    sample_episodes=sample_episodes,
    tensorboard_dir=str(tensorboard_dir)
)
logger.info(f"Training report: {report_path}")
```

## 문제 해결

### 보고서가 생성되지 않음
- 출력 디렉토리 권한 확인
- 필수 메트릭이 모두 포함되었는지 확인

### 메트릭이 표시되지 않음
- 메트릭 키 이름이 정확한지 확인
- 값이 올바른 타입(float, int)인지 확인

### 에피소드가 표시되지 않음
- `sample_episodes` 파라미터가 None이 아닌지 확인
- 에피소드 딕셔너리에 필수 키가 있는지 확인

## 추가 정보

- 상세 문서: `ai_trader/reporting/README.md`
- 구현 요약: `docs/TASK_11.6_IMPLEMENTATION_SUMMARY.md`
- 예제 코드: `examples/generate_html_report_example.py`
