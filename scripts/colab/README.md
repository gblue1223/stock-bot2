# 🚀 Google Colab Training Notebooks

이 디렉토리는 Google Colab에서 Stock Bot 모델을 훈련하기 위한 노트북과 스크립트를 포함합니다.

## 📁 파일 구조

```
scripts/colab/
├── autoencoder_training_complete.ipynb  # AutoEncoder 훈련 노트북
├── grpo_training_complete.ipynb         # GRPO 강화학습 훈련 노트북
├── autoencoder_finetuning.ipynb         # AutoEncoder Fine-tuning 노트북
├── notebook_summary.md                  # 노트북 완성 요약
└── README.md                            # 이 파일
```

## 🔧 사용 방법

### 1. 데이터 준비

#### AutoEncoder 훈련용 데이터

전처리된 HDF5 배치 파일을 Google Drive에 업로드:

```
Google Drive/
└── MyDrive/
    └── models/
        └── stockbot/
            └── pre_training_data/
                ├── 2024_09/
                │   ├── batch_000000.h5
                │   ├── batch_000001.h5
                │   └── ...
                ├── 2024_10/
                └── ...
```

#### GRPO 훈련용 데이터

정규화된 DuckDB 데이터베이스를 Google Drive에 업로드:

```
Google Drive/
└── MyDrive/
    └── models/
        └── stockbot/
            ├── datasets_norm_all.duckdb  # GRPO 훈련용
            └── autoencoder_colab_XXXXXX/
                └── model.pt              # 훈련된 임베딩 모델
```

### 2. Colab에서 실행

#### AutoEncoder 훈련

1. Google Colab (https://colab.research.google.com/) 접속
2. 파일 > 노트북 업로드 > `autoencoder_training_complete.ipynb` 선택
3. 런타임 > 런타임 유형 변경 > GPU 선택
4. **GitHub 토큰 설정** (선택사항, 권장):
   - 🔑 아이콘 클릭 → "Add new secret"
   - Name: `GITHUB_TOKEN`
   - Value: your_personal_access_token
5. 셀을 순서대로 실행

#### GRPO 훈련

1. **먼저 AutoEncoder 훈련 완료 필요**
2. `grpo_training_complete.ipynb` 업로드
3. GPU 런타임 선택 (T4 이상 권장)
4. 노트북 내 경로 설정:
   - `EMBEDDING_MODEL_PATH`: 훈련된 AutoEncoder 모델 경로
   - `DB_PATH`: DuckDB 데이터베이스 경로
5. 셀을 순서대로 실행

**방법 2: 수동 노트북 생성 (고급 사용자용)**

1. 새 노트북 생성
2. 완전한 노트북의 각 셀을 참조하여 수동으로 구성
3. 필요에 따라 코드 수정 및 커스터마이징

### 3. 실행 단계

#### AutoEncoder 훈련 (autoencoder_training_complete.ipynb)

1. **환경 설정**: 패키지 설치 및 GPU 확인
2. **Drive 마운트**: Google Drive 연결 및 데이터 확인
3. **GitHub 클론**: 최신 프로젝트 코드 자동 다운로드
4. **모델 Import**: 프로젝트 모듈에서 모델 클래스 가져오기
5. **훈련 설정**: 하이퍼파라미터 및 옵티마이저 설정
6. **데이터 로딩**: 전처리된 데이터 로딩
7. **훈련 실행**: 메인 훈련 루프 (진행 상황 실시간 표시)
8. **결과 시각화**: 훈련 곡선 및 재구성 샘플 플롯
9. **모델 저장**: 최종 모델 및 결과 Google Drive에 저장

#### GRPO 훈련 (grpo_training_complete.ipynb)

1. **환경 설정**: PyTorch, DuckDB 등 패키지 설치
2. **Drive 마운트**: 임베딩 모델 및 데이터베이스 확인
3. **GitHub 클론**: 최신 GRPO 코드 다운로드
4. **훈련 설정**: GRPO 하이퍼파라미터 구성 (GPU 메모리에 따라 자동 조정)
5. **CLI 훈련 실행**: `train_grpo.py` CLI 스크립트 실행
6. **TensorBoard 모니터링**: 실시간 훈련 진행 상황 확인
7. **백테스트**: 훈련된 정책 성능 평가
8. **모델 저장**: 체크포인트 및 로그를 Google Drive에 저장

## ⚙️ 설정 옵션

### AutoEncoder 설정

#### GPU 메모리에 따른 설정

**T4 GPU (15GB):**

```python
CONFIG = {
    'batch_size': 4,
    'max_sequences': 2000,
    'max_batches_per_month': 50,
    'embedding_dim': 128,
    'hidden_dim': 256
}
```

**V100 GPU (32GB):**

```python
CONFIG = {
    'batch_size': 8,
    'max_sequences': 4000,
    'max_batches_per_month': 100,
    'embedding_dim': 128,
    'hidden_dim': 256
}
```

### GRPO 설정

#### GPU 메모리에 따른 설정

**T4 GPU (15GB):**

```python
CONFIG = {
    'episodes_per_group': 8,
    'num_groups': 3,
    'total_timesteps': 500000,
    'hidden_dim': 256
}
```

**V100 GPU (32GB):**

```python
CONFIG = {
    'episodes_per_group': 16,
    'num_groups': 6,
    'total_timesteps': 2000000,
    'hidden_dim': 256
}
```

#### 주요 하이퍼파라미터

```python
CONFIG = {
    'learning_rate': 3e-4,          # 학습률
    'gamma': 0.99,                  # 할인 계수
    'clip_epsilon': 0.2,            # PPO 클리핑
    'kl_target': 0.01,              # KL 발산 목표
    'quick_exit_threshold': 1.5,    # 빠른 손절 임계값 (초)
    'quick_exit_penalty': 0.01,     # 빠른 손절 페널티
    'transaction_cost_rate': 0.00215 # 거래 비용 (0.215%)
}
```

### 훈련 데이터 선택

```python
CONFIG = {
    'train_months': ['2024_09', '2024_10', '2024_11'],  # 시작: 3개월
    'val_months': ['2024_12'],
    'max_epochs': 15
}
```

### 🔄 점진적 데이터 확장

**3개월로 시작해서 나중에 더 추가하는 방법:**

**방법 1: 설정 변경 후 재훈련**
```python
# 노트북에서 CONFIG 수정
CONFIG['train_months'] = ['2024_09', '2024_10', '2024_11', '2025_01', '2025_02']
CONFIG['max_epochs'] = 20  # 더 많은 에포크
```

**방법 2: 체크포인트에서 계속 훈련**
```python
# 기존 모델 로드
checkpoint = torch.load('/content/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 새 데이터로 추가 훈련
CONFIG['train_months'].extend(['2025_01', '2025_02'])
```

**방법 3: Fine-tuning (권장)**
```python
# 낮은 학습률로 새 데이터에 적응
CONFIG['learning_rate'] = 1e-4  # 기존보다 10배 낮게
CONFIG['max_epochs'] = 5        # 적은 에포크로 빠른 적응
CONFIG['train_months'] = ['2024_09', '2024_10', '2024_11', '2025_01']
```

**장점:**
- 처음에는 빠른 훈련으로 기본 모델 확보
- 새 데이터가 준비되면 점진적으로 성능 향상
- 체크포인트 시스템으로 안전한 확장

## 📊 예상 성능

### AutoEncoder 훈련

#### 훈련 시간

- **T4 GPU**: 약 4-6시간 (15 에포크)
- **V100 GPU**: 약 2-3시간 (15 에포크)

#### 메모리 사용량

- **모델**: ~600MB
- **배치 데이터**: ~2-4GB
- **총 GPU 메모리**: ~6-8GB

#### 예상 결과

- **초기 손실**: ~1.5
- **최종 손실**: ~0.3-0.5
- **압축률**: 28차원 → 128차원 임베딩

### GRPO 훈련

#### 훈련 시간

- **T4 GPU**: 약 8-12시간 (1M timesteps)
- **V100 GPU**: 약 4-6시간 (1M timesteps)

#### 메모리 사용량

- **정책 네트워크**: ~100MB
- **환경 및 버퍼**: ~2-3GB
- **총 GPU 메모리**: ~4-6GB

#### 예상 성능 지표

- **승률 (Win Rate)**: > 50%
- **샤프 비율 (Sharpe Ratio)**: > 1.0
- **최대 낙폭 (Max Drawdown)**: < 10%
- **평균 보유 시간**: < 30초

## 🆕 새로운 기능

### GitHub 자동 클론

- 최신 프로젝트 코드를 자동으로 다운로드
- GitHub 토큰을 사용한 인증 (선택사항)
- 기존 클론이 있으면 자동 업데이트

### 향상된 모델 Import

- 프로젝트 모듈에서 직접 모델 클래스 가져오기
- Import 실패 시 자동 fallback
- 코드 중복 최소화

### 개선된 훈련 모니터링

- 실시간 진행 상황 표시
- 재구성 손실과 정규화 손실 분리 표시
- 5 에포크마다 자동 체크포인트 저장

## 🔍 모니터링

### 실시간 진행 상황

노트북에서 실시간으로 다음을 모니터링할 수 있습니다:

- 훈련/검증 손실 (재구성 + 정규화)
- 처리된 시퀀스 수
- 마스킹 비율
- 학습률 변화
- GPU 메모리 사용량

### 저장되는 파일

```
Google Drive/MyDrive/stock_bot_results/autoencoder_YYYYMMDD_HHMMSS/
├── best_model.pt              # 최고 성능 모델
├── checkpoint_epoch_*.pt      # 정기 체크포인트
├── training_history.json      # 훈련 히스토리
├── config.json               # 설정 정보
├── summary.json              # 요약 보고서
├── training_results.png      # 훈련 결과 플롯
└── reconstruction_sample.png # 재구성 샘플
```

## 🚨 주의사항

### Colab 제한사항

1. **세션 시간**: 최대 12시간 (Pro는 24시간)
2. **GPU 할당**: 사용량에 따라 제한될 수 있음
3. **디스크 공간**: 약 100GB 제한

### 권장사항

1. **정기 저장**: 5 에포크마다 체크포인트 저장
2. **메모리 관리**: 배치 크기 조절로 OOM 방지
3. **백업**: 중요한 결과는 Google Drive에 저장

## ❓ 자주 묻는 질문 (FAQ)

### Q: 3개월 데이터로 시작해서 나중에 더 추가할 수 있나요?
**A: 네, 가능합니다!** 여러 방법이 있습니다:
- **점진적 확장**: 체크포인트에서 새 데이터로 계속 훈련
- **Fine-tuning**: 낮은 학습률로 새 데이터에 적응
- **전체 재훈련**: 모든 데이터를 포함해서 처음부터 훈련

### Q: 어떤 방법이 가장 효율적인가요?
**A: Fine-tuning이 권장됩니다:**
- 기존 학습 결과를 보존하면서 새 데이터 학습
- 훈련 시간 단축 (전체 재훈련 대비 80% 절약)
- 안정적인 성능 향상

### Q: 데이터를 추가할 때 주의사항은?
**A: 다음 사항들을 확인하세요:**
- 새 데이터의 전처리 형식이 기존과 동일한지
- 검증 데이터셋도 함께 업데이트
- 학습률을 기존보다 낮게 설정 (1e-4 권장)

## 🔧 문제 해결

### GPU 메모리 부족

```python
# 배치 크기 줄이기
CONFIG['batch_size'] = 2
CONFIG['max_sequences'] = 1000

# 모델 크기 줄이기
CONFIG['hidden_dim'] = 128
CONFIG['embedding_dim'] = 64
```

### 데이터 로딩 오류

```python
# 데이터 경로 확인
DATA_PATH = '/content/drive/MyDrive/stock_bot_data/pre_training_data'
print(f"Data exists: {os.path.exists(DATA_PATH)}")

# 월별 데이터 확인
months = os.listdir(DATA_PATH)
print(f"Available months: {months}")
```

### 훈련 중단 시 재시작

```python
# 체크포인트에서 재시작
checkpoint = torch.load('/content/checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. GPU 할당 상태
2. 메모리 사용량
3. 데이터 경로
4. 패키지 버전

## 🎯 다음 단계

### AutoEncoder 훈련 완료 후

1. **모델 평가**: 임베딩 품질 분석
2. **GRPO 훈련**: 임베딩을 상태 표현으로 사용하여 강화학습 시작
3. **Fine-tuning**: 필요시 새 데이터로 추가 학습

### GRPO 훈련 완료 후

1. **백테스트**: 다양한 시장 상황에서 성능 검증
2. **하이퍼파라미터 튜닝**: 성능 개선을 위한 파라미터 조정
3. **실전 배포**: 실시간 거래 시스템에 통합

## 📝 노트북 사용 팁

### GRPO 훈련 노트북 특징

- **CLI 기반**: 직접 코드를 작성하지 않고 `train_grpo.py` CLI를 사용
- **자동 설정**: GPU 메모리에 따라 하이퍼파라미터 자동 조정
- **TensorBoard 통합**: 실시간 훈련 모니터링
- **백테스트 포함**: 훈련 후 즉시 성능 평가 가능

### 권장 워크플로우

1. **AutoEncoder 먼저 훈련** (4-6시간)
2. **임베딩 품질 확인**
3. **GRPO 훈련 시작** (8-12시간)
4. **TensorBoard로 모니터링**
5. **백테스트로 성능 검증**
6. **필요시 하이퍼파라미터 조정 후 재훈련**
