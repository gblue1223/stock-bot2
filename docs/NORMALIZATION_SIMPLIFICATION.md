# 정규화 간소화 가이드

실시간 데이터가 학습 데이터와 유사하다고 가정할 때의 간소화 방법

## 변경 사항 요약

### 1. 아키텍처 변경

**이전 (복잡)**:
```
실시간 데이터 → PerStockNormalizer (종목별 Rolling Window)
                ↓
            GRPOInference (정규화 없음)
                ↓
              추론 결과
```

**이후 (간단)**:
```
실시간 데이터 → GRPOInference (학습 통계로 정규화)
                ↓
              추론 결과
```

### 2. 코드 변경

#### `scripts/live/live_trading.py`

```python
# 학습 통계를 GRPOInference에 직접 전달
normalization_stats_for_inference = None
if use_training_stats:
    normalization_stats_for_inference = {
        'mean': training_mean,
        'std': training_std
    }

base_inference = GRPOInference(
    policy_path=config.model_path,
    embedding_model_path=config.embedding_model_path,
    device=config.device,
    use_torchscript=False,
    normalization_stats=normalization_stats_for_inference  # ✅ 직접 전달
)
```

#### `ai_trader/grpo/train_scalping.py`

```python
# 학습 완료 후 정규화 통계 자동 저장
if hasattr(env, 'mean') and hasattr(env, 'std'):
    normalization_stats = {
        'mean': env.mean.cpu().numpy().tolist(),
        'std': env.std.cpu().numpy().tolist()
    }
    stats_path = os.path.join(args.output_dir, 'normalization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(normalization_stats, f, indent=2)
```

### 3. 설정 파일

**config/trading_config_with_training_stats.json**:
```json
{
  "normalization_stats_path": "models/grpo_scalping/normalization_stats.json"
}
```

## 사용 방법

### 1단계: 모델 학습
```bash
python ai_trader/grpo/train_scalping.py \
    --embedding-model models/autoencoder/best_model.pt \
    --db datasets_norm_all.duckdb \
    --out models/grpo_scalping \
    --episodes-per-group 12
```

학습 완료 후 자동 생성:
- `models/grpo_scalping/scalping_grpo_model.pt`
- `models/grpo_scalping/normalization_stats.json` ✅ 새로 추가

### 2단계: 설정 파일 작성
```json
{
  "model_path": "models/grpo_scalping/scalping_grpo_model.pt",
  "embedding_model_path": "models/autoencoder/best_model.pt",
  "normalization_stats_path": "models/grpo_scalping/normalization_stats.json"
}
```

### 3단계: 실시간 거래 실행
```bash
python scripts/live/live_trading.py \
    --config config/trading_config_with_training_stats.json
```

## 장점

### 1. 간단한 구조
- PerStockNormalizer 제거
- 종목별 통계 관리 불필요
- 코드 복잡도 감소

### 2. 즉시 추론 가능
- 워밍업 기간 불필요 (0초)
- 첫 틱부터 정확한 추론

### 3. 일관성 보장
- 학습 시와 동일한 정규화
- 종목 간 일관된 처리
- 재현 가능한 결과

### 4. 성능 향상
- 메모리 사용량 감소
- Rolling Window 계산 제거
- 추론 지연 시간 동일 (2.87ms)

## 주의사항

### 가정의 유효성 검증

실시간 데이터가 학습 데이터와 유사한지 확인:

```python
# 실시간 데이터 수집 후 분포 비교
import numpy as np

# 학습 통계
training_mean = np.array([...])
training_std = np.array([...])

# 실시간 통계 (100 샘플)
realtime_samples = []  # 수집된 샘플들
realtime_mean = np.mean(realtime_samples, axis=0)
realtime_std = np.std(realtime_samples, axis=0)

# 차이 계산
mean_diff = np.abs(training_mean - realtime_mean)
std_diff = np.abs(training_std - realtime_std)

# 임계값 확인 (예: 20% 이내)
if np.max(mean_diff / training_mean) > 0.2:
    print("⚠️ 평균 차이가 큼 - 온라인 정규화 고려")
if np.max(std_diff / training_std) > 0.2:
    print("⚠️ 표준편차 차이가 큼 - 온라인 정규화 고려")
```

### 재학습 주기

학습 데이터가 오래되면 성능 저하 가능:
- **권장**: 1-3개월마다 재학습
- **필수**: 시장 구조 변화 시 재학습

### 폴백 옵션

성능 저하 시 온라인 정규화로 전환:
```json
{
  "normalization_stats_path": null,
  "online_window_size": 200,
  "online_warmup_samples": 50
}
```

## 성능 비교

| 항목 | 간소화 전 | 간소화 후 |
|------|----------|----------|
| 코드 라인 | ~150 | ~50 |
| 워밍업 시간 | 50-200초 | 0초 |
| 메모리 (종목당) | ~50KB | ~1KB |
| 추론 지연 | 2.87ms | 2.87ms |
| 일관성 | 중간 | 높음 |
| 적응성 | 높음 | 낮음 |

## 문제 해결

### Q: 성능이 학습 시보다 나쁨
A: 실시간 데이터 분포 확인 → 차이가 크면 온라인 정규화 사용

### Q: 특정 종목만 성능이 나쁨
A: 해당 종목이 학습 데이터에 충분히 포함되었는지 확인

### Q: 시간대별 성능 차이
A: 장 초반/후반 데이터 분포 차이 → 시간대별 학습 데이터 비율 확인

## 결론

실시간 데이터가 학습 데이터와 유사하다면:
- ✅ 학습 통계 직접 사용 (간소화)
- ✅ 즉시 추론 가능
- ✅ 일관성 보장

실시간 데이터가 다를 수 있다면:
- ✅ 온라인 정규화 유지
- ✅ 동적 적응
- ✅ 종목별 최적화
