# AutoEncoder 평가 빠른 시작 가이드

## 1. 임포트 테스트

먼저 모든 의존성이 제대로 설치되어 있는지 확인:

```bash
python scripts/pre/diagnosis/test_imports.py
```

성공 시 출력:
```
✓ AutoEncoder models imported successfully
✓ Data loader imported successfully
✓ Evaluation functions imported successfully
✓ External dependencies imported successfully

✓ All imports successful!
```

## 2. 평가 실행

### 방법 1: 기본 설정으로 실행

```bash
python scripts/pre/diagnosis/evaluate_autoencoder.py
```

기본 경로:
- 모델: `C:\Users\user\Workspace\datasets@20251013\autoencoder\model.pt`
- DB: `C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb`
- 출력: `scripts/pre/diagnosis/results/`

### 방법 2: Windows 배치 파일 사용

```cmd
scripts\pre\diagnosis\run_evaluation.bat
```

### 방법 3: 커스텀 경로 지정

```bash
python scripts/pre/diagnosis/evaluate_autoencoder.py \
    --model "path/to/your/model.pt" \
    --db "path/to/your/database.duckdb" \
    --output "custom/output/dir" \
    --device cuda \
    --max-samples 10000
```

## 3. 결과 확인

평가가 완료되면 `scripts/pre/diagnosis/results/` 디렉토리에 다음 파일들이 생성됩니다:

### JSON 결과
```bash
cat scripts/pre/diagnosis/results/evaluation_results.json
```

### 시각화 파일
- `reconstruction_errors.png` - 재구성 오차 분포
- `embedding_norms.png` - 임베딩 L2 norm 분포
- `embeddings_tsne.png` - t-SNE 2D 시각화
- `embeddings_pca.png` - PCA 2D 시각화

## 4. 결과 해석

### 콘솔 출력 예시

```
=============================================================
Evaluation Summary
=============================================================
Reconstruction MSE: 0.008234
Reconstruction MAE: 0.067123
Silhouette Score: 0.4523
Temporal Coherence: 0.6789
Mean Embedding Norm: 3.4567
=============================================================
```

### 평가 기준

| 지표 | 우수 | 양호 | 개선 필요 |
|------|------|------|-----------|
| Reconstruction MSE | < 0.01 | 0.01 ~ 0.05 | > 0.05 |
| Silhouette Score | > 0.5 | 0.3 ~ 0.5 | < 0.3 |
| Temporal Coherence | > 0.7 | 0.5 ~ 0.7 | < 0.5 |
| Embedding Norm | 1 ~ 10 | 0.5 ~ 20 | < 0.5 or > 20 |

## 5. 문제 해결

### CUDA Out of Memory

```bash
# 샘플 수 줄이기
python scripts/pre/diagnosis/evaluate_autoencoder.py --max-samples 5000

# CPU 사용
python scripts/pre/diagnosis/evaluate_autoencoder.py --device cpu
```

### 모듈을 찾을 수 없음 (ModuleNotFoundError)

```bash
# 프로젝트 루트에서 실행하는지 확인
pwd  # 또는 cd

# 가상환경 활성화 확인
which python  # Linux/Mac
where python  # Windows

# 임포트 테스트 실행
python scripts/pre/diagnosis/test_imports.py
```

### 데이터베이스 연결 실패

```bash
# 파일 존재 확인
ls -l "C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb"

# 읽기 권한 확인
# 다른 프로세스가 파일을 사용 중인지 확인
```

## 6. 다음 단계

평가 결과가 양호하면:
1. GRPO 학습에 임베딩 모델 사용
2. Fine-tuning 진행
3. 실전 트레이딩 시스템에 통합

평가 결과가 불량하면:
1. 학습 파라미터 조정 (learning rate, epochs)
2. 모델 구조 조정 (hidden_dim, num_layers)
3. 데이터 전처리 재검토
4. 재학습 후 다시 평가

## 참고 문서

- [상세 평가 가이드](EVALUATION_GUIDE.md)
- [사용법 README](README.md)
- [AutoEncoder 모델 코드](../../../ai_trader/embedding/autoencoder_model.py)
