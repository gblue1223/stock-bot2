# AutoEncoder 모델 평가 결과 분석

**평가 일시:** 2025-10-18 15:17:39  
**모델:** C:\Users\user\Workspace\datasets@20260109\autoencoder\model.pt  
**데이터:** C:\Users\user\Workspace\datasets@20260109\datasets_norm_all.duckdb

## 모델 구성

- **Input Dimension:** 28 features
- **Embedding Dimension:** 128
- **Hidden Dimension:** 256
- **Sequence Length:** 60
- **Num Layers:** 3
- **Model Type:** Standard AutoEncoder

## 평가 결과

### 1. 재구성 오차 (Reconstruction Error)

| 지표 | 값 | 평가 |
|------|-----|------|
| MSE (Mean) | 0.0778 | ⚠️ 개선 필요 |
| MSE (Std) | 0.0168 | - |
| MAE (Mean) | 0.1736 | ⚠️ 개선 필요 |
| MAE (Std) | 0.0129 | - |

**분석:**
- MSE가 0.05 이상으로 높은 편입니다
- 모델이 입력 데이터를 완벽하게 재구성하지 못하고 있습니다
- 정규화된 데이터 기준으로 개선이 필요합니다

**권장 사항:**
- 더 긴 학습 (epochs 증가)
- Learning rate 조정
- 모델 용량 증가 (hidden_dim 또는 num_layers 증가)

### 2. 임베딩 품질 (Embedding Quality)

| 지표 | 값 | 평가 |
|------|-----|------|
| Silhouette Score | 0.0000 | ⚠️ 평가 불가 |
| Temporal Coherence | 0.9948 | ✅ 우수 |
| Num Samples | 91 | - |
| Num Stocks | 1 | ⚠️ 단일 종목 |
| Embedding Dim | 128 | - |

**분석:**
- **Silhouette Score = 0.0:** 테스트 데이터에 단일 종목만 포함되어 클러스터링 평가 불가
  - Silhouette Score는 최소 2개 이상의 종목이 필요합니다
  - 더 많은 샘플과 다양한 종목으로 재평가 필요
  
- **Temporal Coherence = 0.9948:** 매우 우수
  - 시간적으로 가까운 샘플들의 임베딩이 매우 유사합니다
  - 모델이 시간적 패턴을 잘 학습했습니다
  - 시계열 데이터의 연속성이 잘 보존되었습니다

**권장 사항:**
- `--max-samples` 값을 늘려서 더 많은 종목 포함 (예: 10000)
- 다양한 종목에 대한 평가 필요

### 3. 임베딩 분포 (Embedding Distribution)

| 지표 | 값 | 평가 |
|------|-----|------|
| Mean L2 Norm | 0.5936 | ⚠️ 낮음 |
| Std L2 Norm | 0.0489 | ✅ 안정적 |
| Norm Range | [0.49, 0.68] | ⚠️ 낮음 |

**분석:**
- **L2 Norm이 낮음 (< 1.0):**
  - 임베딩 벡터의 크기가 작습니다
  - 학습이 충분하지 않거나 gradient vanishing 가능성
  - 적절한 범위는 1 ~ 10입니다
  
- **표준편차가 작음 (0.049):**
  - 임베딩 크기가 일관적입니다
  - 학습이 안정적으로 진행되었습니다

**권장 사항:**
- Learning rate 증가 고려
- 더 긴 학습 시간
- Gradient clipping 값 확인

## 종합 평가

### 강점
✅ **시간적 일관성 우수:** Temporal Coherence가 0.99로 매우 높음  
✅ **안정적 학습:** 임베딩 분포의 표준편차가 작아 안정적  

### 약점
⚠️ **재구성 오차 높음:** MSE 0.078로 개선 필요  
⚠️ **임베딩 크기 작음:** L2 norm이 0.59로 낮음  
⚠️ **평가 데이터 부족:** 단일 종목으로 Silhouette Score 평가 불가  

### 전체 평가: **보통 (개선 필요)**

## 개선 방안

### 1. 즉시 적용 가능
```bash
# 더 많은 샘플로 재평가
python scripts/pre/diagnosis/evaluate_autoencoder.py --max-samples 10000

# 다양한 종목 포함 확인
```

### 2. 모델 재학습 고려
```python
# 학습 파라미터 조정
config = {
    'learning_rate': 1e-4,  # 현재보다 높게
    'epochs': 100,  # 더 길게
    'hidden_dim': 512,  # 더 크게
    'num_layers': 4,  # 더 깊게
}
```

### 3. 데이터 확인
- 정규화가 올바르게 되었는지 확인
- 이상치 제거 확인
- 특징 선택 재검토

## 다음 단계

### 평가 데이터 부족 해결
```bash
# 10,000 샘플로 재평가 (더 많은 종목 포함)
python scripts/pre/diagnosis/evaluate_autoencoder.py --max-samples 10000
```

### 모델 개선 후 재평가
1. 학습 파라미터 조정
2. 모델 재학습
3. 평가 스크립트 재실행
4. 결과 비교

### GRPO 학습 진행 여부
- **현재 상태로 진행 가능:** Temporal Coherence가 우수하므로 시도 가능
- **권장:** 재구성 오차를 개선한 후 진행하면 더 좋은 결과 기대

## 참고 자료

- [평가 가이드](../EVALUATION_GUIDE.md)
- [빠른 시작](../QUICKSTART.md)
- [AutoEncoder 모델](../../../ai_trader/embedding/autoencoder_model.py)
