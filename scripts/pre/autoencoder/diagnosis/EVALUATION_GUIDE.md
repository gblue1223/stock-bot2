# AutoEncoder 모델 평가 가이드

## 개요

이 가이드는 AutoEncoder 임베딩 모델의 품질을 평가하고 문제를 진단하는 방법을 설명합니다.

## 빠른 시작

### Windows (배치 파일 사용)
```cmd
scripts\pre\diagnosis\run_evaluation.bat
```

### Python 직접 실행
```bash
python scripts/pre/autoencoder/diagnosis/evaluate_autoencoder.py
```

## 평가 프로세스

### 1. 재구성 오차 평가

**목적:** 모델이 입력 데이터를 얼마나 잘 재구성하는지 측정

**지표:**
- **MSE (Mean Squared Error)**: 평균 제곱 오차
- **MAE (Mean Absolute Error)**: 평균 절대 오차

**기준:**
- 정규화된 데이터 기준
  - MSE < 0.01: 우수
  - MSE 0.01 ~ 0.05: 양호
  - MSE > 0.05: 개선 필요

**문제 진단:**
- MSE가 너무 높음 → 모델 용량 부족 또는 학습 부족
- MSE가 너무 낮음 (< 0.001) → 과적합 가능성

### 2. 임베딩 품질 평가

#### Silhouette Score

**목적:** 종목별로 임베딩이 얼마나 잘 클러스터링되는지 측정

**범위:** -1 ~ 1 (높을수록 좋음)

**기준:**
- 0.5 이상: 우수 - 종목별 특성이 명확히 구분됨
- 0.3 ~ 0.5: 양호 - 종목별 특성이 어느 정도 구분됨
- 0.0 ~ 0.3: 보통 - 종목별 특성이 약하게 구분됨
- 0.0 미만: 불량 - 클러스터링 실패

**문제 진단:**
- Score가 낮음 → 임베딩이 종목 특성을 잘 포착하지 못함
- 해결책: 더 긴 학습, 더 큰 embedding_dim, 데이터 품질 확인

#### Temporal Coherence

**목적:** 시간적으로 가까운 샘플의 임베딩이 유사한지 측정

**범위:** 0 ~ 1 (높을수록 좋음)

**기준:**
- 0.7 이상: 우수 - 시간적 연속성이 잘 보존됨
- 0.5 ~ 0.7: 양호 - 시간적 연속성이 어느 정도 보존됨
- 0.3 ~ 0.5: 보통 - 시간적 연속성이 약함
- 0.3 미만: 불량 - 시간적 연속성 없음

**문제 진단:**
- Score가 낮음 → 임베딩이 시간적 패턴을 무시함
- 해결책: 시퀀스 길이 조정, Positional Encoding 강화

### 3. 임베딩 분포 분석

#### L2 Norm 분포

**목적:** 임베딩 벡터의 크기 분포 확인

**적절한 범위:** 1 ~ 10

**문제 진단:**
- Norm이 너무 작음 (< 0.5) → 학습 불충분, gradient vanishing
- Norm이 너무 큼 (> 20) → 학습 불안정, gradient exploding
- Norm 분산이 너무 큼 → 일부 샘플만 학습됨

**해결책:**
- Learning rate 조정
- Gradient clipping 적용
- Batch normalization 추가

### 4. 시각화 분석

#### t-SNE 시각화

**목적:** 고차원 임베딩을 2D로 투영하여 클러스터 구조 확인

**좋은 시각화:**
- 같은 종목끼리 뭉쳐있음
- 다른 종목과 명확히 분리됨
- 클러스터가 너무 겹치지 않음

**나쁜 시각화:**
- 모든 점이 뒤섞여 있음
- 클러스터가 없음
- 한 점에 모여있음

#### PCA 시각화

**목적:** 선형 구조 확인

**좋은 시각화:**
- 주요 분산 방향이 명확함
- 종목별로 어느 정도 분리됨

## 평가 결과 해석 예시

### 예시 1: 우수한 모델
```
Reconstruction MSE: 0.0082
Reconstruction MAE: 0.0671
Silhouette Score: 0.5234
Temporal Coherence: 0.7456
Mean Embedding Norm: 3.45
```

**해석:**
- 재구성 오차가 낮아 입력을 잘 복원함
- Silhouette score가 높아 종목 특성을 잘 포착함
- Temporal coherence가 높아 시간적 패턴을 잘 학습함
- Embedding norm이 적절한 범위에 있음

**결론:** 모델이 잘 학습되었으며 GRPO 학습에 사용 가능

### 예시 2: 개선이 필요한 모델
```
Reconstruction MSE: 0.0523
Reconstruction MAE: 0.1823
Silhouette Score: 0.1234
Temporal Coherence: 0.3456
Mean Embedding Norm: 15.67
```

**해석:**
- 재구성 오차가 높아 입력을 제대로 복원하지 못함
- Silhouette score가 낮아 종목 특성을 잘 구분하지 못함
- Temporal coherence가 낮아 시간적 패턴을 무시함
- Embedding norm이 너무 커서 학습이 불안정함

**결론:** 모델 재학습 필요

**개선 방안:**
1. Learning rate 낮추기 (예: 1e-4 → 1e-5)
2. Gradient clipping 적용
3. 더 긴 학습 (epochs 증가)
4. 데이터 정규화 재확인
5. 모델 구조 조정 (hidden_dim, num_layers)

## 고급 진단

### 종목별 성능 차이 확인

일부 종목에서만 성능이 낮다면:
- 해당 종목의 데이터 품질 확인
- 데이터 불균형 문제 확인
- 종목별 정규화 필요 여부 확인

### 시간대별 성능 차이 확인

특정 시간대에서만 성능이 낮다면:
- 해당 시간대의 데이터 특성 확인
- 시장 개장/마감 시간 영향 확인
- 거래량이 적은 시간대 제외 고려

### 과적합 확인

Train set과 Test set의 성능 차이가 크다면:
- Dropout 비율 증가
- 데이터 증강 적용
- 정규화 강화 (L2 regularization)

## 자동화된 평가

### 정기적 평가 스케줄링

```bash
# Windows Task Scheduler 사용
# 매일 오전 2시에 평가 실행
schtasks /create /tn "AutoEncoder Evaluation" /tr "C:\path\to\run_evaluation.bat" /sc daily /st 02:00
```

### CI/CD 통합

```yaml
# GitHub Actions 예시
- name: Evaluate AutoEncoder
  run: |
    python scripts/pre/autoencoder/diagnosis/evaluate_autoencoder.py
    python scripts/check_evaluation_results.py
```

## 문제 해결

### 메모리 부족
```bash
# 샘플 수 줄이기
python scripts/pre/autoencoder/diagnosis/evaluate_autoencoder.py --max-samples 5000

# CPU 사용
python scripts/pre/autoencoder/diagnosis/evaluate_autoencoder.py --device cpu
```

### 시각화 오류
```bash
# matplotlib backend 설정
export MPLBACKEND=Agg  # Linux/Mac
set MPLBACKEND=Agg     # Windows
```

### DuckDB 연결 오류
- 파일 경로 확인
- 읽기 권한 확인
- 다른 프로세스가 파일을 사용 중인지 확인

## 참고 자료

- [AutoEncoder 모델 구조](../../../ai_trader/embedding/autoencoder_model.py)
- [평가 메트릭 구현](../../../ai_trader/embedding/evaluation.py)
- [데이터 로더](../../../ai_trader/embedding/data.py)
