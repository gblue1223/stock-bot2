# HTML 보고서 생성 모듈

훈련 진행 상황, 최종 메트릭, 샘플 거래 에피소드를 요약하는 HTML 보고서를 생성합니다.

## 기능

- **임베딩 모델 보고서**: 임베딩 모델 훈련 결과를 시각화
- **GRPO 에이전트 보고서**: GRPO 훈련 결과 및 샘플 에피소드 표시
- **통합 보고서**: 임베딩과 GRPO 결과를 하나의 보고서로 통합

## 사용 방법

### 임베딩 모델 보고서

```python
from ai_trader.reporting import generate_embedding_report

training_metadata = {
    'timestamp': '2025-10-05 14:30:00',
    'epoch': 50,
}

final_metrics = {
    'train_loss': 0.234,
    'val_loss': 0.267,
    'silhouette_score': 0.623,
    'temporal_coherence': 0.781
}

config = {
    'input_dim': 60,
    'embedding_dim': 128,
    'seq_len': 60,
    'num_heads': 4
}

report_path = generate_embedding_report(
    output_path='models/embedding/report.html',
    training_metadata=training_metadata,
    final_metrics=final_metrics,
    config=config,
    tensorboard_dir='models/embedding/tensorboard_logs'
)
```

### GRPO 에이전트 보고서

```python
from ai_trader.reporting import generate_grpo_report

training_metadata = {
    'timestamp': '2025-10-05 18:45:00',
    'total_timesteps': 1000000,
    'num_updates': 5000,
    'quick_exit_threshold': 1.5,
    'quick_exit_penalty': 0.01
}

final_metrics = {
    'mean_reward': 0.0234,
    'win_rate': 0.58,
    'sharpe_ratio': 1.42,
    'max_drawdown': 0.067,
    'avg_holding_time': 24.3,
    'avg_profit_per_trade': 0.0067,
    'quick_exit_violations': 234,
    'total_trades': 1500
}

config = {
    'embedding_dim': 128,
    'hidden_dim': 256,
    'learning_rate': 0.0003
}

sample_episodes = [
    {
        'total_profit': 0.0123,
        'num_trades': 5,
        'avg_holding_time': 18.5,
        'sharpe_ratio': 1.67,
        'stock_code': '005930'
    }
]

report_path = generate_grpo_report(
    output_path='models/grpo/report.html',
    training_metadata=training_metadata,
    final_metrics=final_metrics,
    config=config,
    sample_episodes=sample_episodes,
    tensorboard_dir='models/grpo/tensorboard_logs'
)
```

### 통합 보고서

```python
from ai_trader.reporting import generate_combined_report

report_path = generate_combined_report(
    output_path='models/combined_report.html',
    embedding_metadata=embedding_metadata,
    embedding_metrics=embedding_metrics,
    grpo_metadata=grpo_metadata,
    grpo_metrics=grpo_metrics,
    sample_episodes=sample_episodes
)
```

## 보고서 내용

### 임베딩 모델 보고서

- 훈련 정보 (완료 시간, 에포크, 설정)
- 최종 메트릭 (검증 손실, Silhouette Score, Temporal Coherence)
- 모델 아키텍처 설정
- TensorBoard 로그 링크

### GRPO 에이전트 보고서

- 훈련 정보 (타임스텝, 업데이트 횟수, 빠른 손절 설정)
- 성능 메트릭 (승률, 샤프 비율, 평균 보유 시간, 거래당 수익)
- 샘플 거래 에피소드 (최대 10개)
- 하이퍼파라미터 설정
- TensorBoard 로그 링크

### 통합 보고서

- 임베딩 모델 요약
- GRPO 에이전트 요약
- 주요 메트릭 비교

## 메트릭 해석

### 임베딩 모델

- **Silhouette Score**: 종목별 클러스터링 품질 (0.5 이상 우수)
- **Temporal Coherence**: 시간적 일관성 (0.7 이상 우수)

### GRPO 에이전트

- **승률**: 수익 거래 비율 (55% 이상 우수)
- **샤프 비율**: 위험 대비 수익 (1.5 이상 우수)
- **최대 낙폭**: 최대 손실 비율 (5% 미만 우수)
- **평균 보유 시간**: 스캘핑 특성 (30초 미만 우수)

## 예제

전체 예제는 `examples/generate_html_report_example.py`를 참조하세요.

```bash
python examples/generate_html_report_example.py
```
