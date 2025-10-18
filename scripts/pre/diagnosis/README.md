# AutoEncoder 모델 진단 도구

AutoEncoder 임베딩 모델의 품질을 평가하고 진단하는 스크립트 모음입니다.

## 📋 빠른 링크

- **🚀 [빠른 시작](QUICKSTART.md)** - 즉시 실행 가이드
- **📊 [최종 요약](FINAL_SUMMARY.md)** - 평가 결과 요약 및 권장사항
- **📖 [상세 가이드](EVALUATION_GUIDE.md)** - 평가 지표 해석 방법
- **📈 [10K 샘플 분석](results/ANALYSIS_10K.md)** - 상세 분석 보고서

## ✅ 평가 완료

**모델 상태:** GRPO 학습 진행 가능 ✅  
**Temporal Coherence:** 0.9992 (탁월)  
**Reconstruction MSE:** 0.0452 (양호)

## 스크립트

### evaluate_autoencoder.py

AutoEncoder 모델을 다각도로 평가합니다.

**평가 항목:**
1. **재구성 오차 (Reconstruction Error)**
   - MSE (Mean Squared Error)
   - MAE (Mean Absolute Error)
   - 분포 시각화

2. **임베딩 품질 (Embedding Quality)**
   - Silhouette Score: 종목별 클러스터링 품질 (-1 ~ 1, 높을수록 좋음)
   - Temporal Coherence: 시간적 일관성 (0 ~ 1, 높을수록 좋음)

3. **임베딩 분포 분석**
   - L2 norm 통계
   - 평균/표준편차 분석

4. **시각화**
   - t-SNE 2D 임베딩 시각화
   - PCA 2D 임베딩 시각화
   - 재구성 오차 분포
   - 임베딩 norm 분포

**사용법:**

```bash
# 기본 사용 (기본 경로 사용)
python scripts/pre/diagnosis/evaluate_autoencoder.py

# 커스텀 경로 지정
python scripts/pre/diagnosis/evaluate_autoencoder.py \
    --model path/to/model.pt \
    --db path/to/database.duckdb \
    --device cuda \
    --output results/evaluation \
    --max-samples 10000

# CPU 사용
python scripts/pre/diagnosis/evaluate_autoencoder.py --device cpu
```

**파라미터:**
- `--model`: 모델 파일 경로 (기본값: `C:\Users\user\Workspace\datasets@20251013\autoencoder\model.pt`)
- `--db`: DuckDB 데이터베이스 경로 (기본값: `C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb`)
- `--device`: 디바이스 선택 (cuda/cpu, 기본값: cuda)
- `--output`: 결과 저장 디렉토리 (기본값: `scripts/pre/diagnosis/results`)
- `--max-samples`: 평가에 사용할 최대 샘플 수 (기본값: 10000)

**출력 파일:**
- `evaluation_results.json`: 전체 평가 결과 (JSON)
- `reconstruction_errors.png`: 재구성 오차 분포 그래프
- `embedding_norms.png`: 임베딩 L2 norm 분포 그래프
- `embeddings_tsne.png`: t-SNE 2D 시각화
- `embeddings_pca.png`: PCA 2D 시각화

**평가 지표 해석:**

1. **Reconstruction Error**
   - MSE/MAE가 낮을수록 좋음
   - 정규화된 데이터 기준 MSE < 0.01이면 양호

2. **Silhouette Score**
   - 0.5 이상: 우수한 클러스터링
   - 0.3 ~ 0.5: 양호한 클러스터링
   - 0.3 미만: 개선 필요

3. **Temporal Coherence**
   - 0.7 이상: 우수한 시간적 일관성
   - 0.5 ~ 0.7: 양호한 시간적 일관성
   - 0.5 미만: 개선 필요

4. **Embedding Norm**
   - 평균 norm이 너무 크거나 작으면 학습 불안정
   - 1 ~ 10 범위가 적절

## 요구사항

```bash
pip install torch numpy matplotlib seaborn scikit-learn duckdb
```

## 예제 출력

```
=============================================================
Starting Full Evaluation
=============================================================
Device: cuda
Model: C:\Users\user\Workspace\datasets@20251013\autoencoder\model.pt
Database: C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb
Output: scripts/pre/diagnosis/results

Loading model...
Model loaded: standard
  Input dim: 50
  Embedding dim: 128
  Hidden dim: 256
  Seq len: 60

Loading data (max 10000 samples)...
Data loaded successfully

Computing reconstruction error on test set...
Reconstruction Error:
  MSE: 0.008234 ± 0.003421
  MAE: 0.067123 ± 0.015234

Extracting embeddings from test set...
Extracted 5000 embeddings

Evaluating embedding quality...
Embedding Quality Metrics:
  Silhouette Score: 0.4523
  Temporal Coherence: 0.6789
  Num Samples: 5000
  Num Stocks: 150
  Embedding Dim: 128

Embedding Distribution:
  Mean L2 norm: 3.4567 ± 0.8901
  Norm range: [1.2345, 6.7890]

=============================================================
Evaluation Summary
=============================================================
Reconstruction MSE: 0.008234
Reconstruction MAE: 0.067123
Silhouette Score: 0.4523
Temporal Coherence: 0.6789
Mean Embedding Norm: 3.4567
=============================================================

Evaluation completed!
```

## 문제 해결

### CUDA Out of Memory
```bash
# 샘플 수 줄이기
python scripts/pre/diagnosis/evaluate_autoencoder.py --max-samples 5000

# CPU 사용
python scripts/pre/diagnosis/evaluate_autoencoder.py --device cpu
```

### 모델 로드 실패
- 모델 파일 경로 확인
- checkpoint에 'config' 키가 있는지 확인
- PyTorch 버전 호환성 확인

### 데이터베이스 연결 실패
- DuckDB 파일 경로 확인
- 파일 읽기 권한 확인
- 데이터베이스 파일 손상 여부 확인
