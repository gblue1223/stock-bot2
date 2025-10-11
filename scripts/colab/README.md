# 🚀 Google Colab AutoEncoder Training

이 디렉토리는 Google Colab에서 Stock Bot AutoEncoder를 훈련하기 위한 노트북과 스크립트를 포함합니다.

## 📁 파일 구조

```
scripts/colab/
├── autoencoder_training_complete.ipynb  # 완전한 Jupyter 노트북 (권장)
├── notebook_summary.md                  # 노트북 완성 요약
└── README.md                            # 이 파일
```

## 🔧 사용 방법

### 1. 데이터 준비

먼저 전처리된 데이터를 Google Drive에 업로드해야 합니다:

```
Google Drive/
└── MyDrive/
    └── stock_bot_data/
        └── pre_training_data/
            ├── 2024_09/
            │   ├── batch_000000.h5
            │   ├── batch_000001.h5
            │   └── ...
            ├── 2024_10/
            └── ...
```

### 2. Colab에서 실행

**방법 1: 완전한 Jupyter 노트북 사용 (권장)**

1. Google Colab (https://colab.research.google.com/) 접속
2. 파일 > 노트북 업로드 > `autoencoder_training_complete.ipynb` 선택
3. 런타임 > 런타임 유형 변경 > GPU 선택
4. **GitHub 토큰 설정** (선택사항, 권장):
   - 🔑 아이콘 클릭 → "Add new secret"
   - Name: `GITHUB_TOKEN`
   - Value: your_personal_access_token
5. 셀을 순서대로 실행 (모든 코드가 포함되어 있음)

**방법 2: 수동 노트북 생성 (고급 사용자용)**

1. 새 노트북 생성
2. 완전한 노트북의 각 셀을 참조하여 수동으로 구성
3. 필요에 따라 코드 수정 및 커스터마이징

### 3. 실행 단계

**완전한 노트북 실행 (autoencoder_training_complete.ipynb):**

1. **환경 설정**: 패키지 설치 및 GPU 확인
2. **Drive 마운트**: Google Drive 연결 및 데이터 확인
3. **GitHub 클론**: 최신 프로젝트 코드 자동 다운로드
4. **모델 Import**: 프로젝트 모듈에서 모델 클래스 가져오기
5. **훈련 설정**: 하이퍼파라미터 및 옵티마이저 설정
6. **데이터 로딩**: 전처리된 데이터 로딩
7. **훈련 실행**: 메인 훈련 루프 (진행 상황 실시간 표시)
8. **결과 시각화**: 훈련 곡선 및 재구성 샘플 플롯
9. **모델 저장**: 최종 모델 및 결과 Google Drive에 저장

## ⚙️ 설정 옵션

### GPU 메모리에 따른 설정

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

### 훈련 데이터 선택

```python
CONFIG = {
    'train_months': ['2024_09', '2024_10', '2024_11', '2024_12'],
    'val_months': ['2025_01'],
    'max_epochs': 20
}
```

## 📊 예상 성능

### 훈련 시간

- **T4 GPU**: 약 4-6시간 (20 에포크)
- **V100 GPU**: 약 2-3시간 (20 에포크)

### 메모리 사용량

- **모델**: ~600MB
- **배치 데이터**: ~2-4GB
- **총 GPU 메모리**: ~6-8GB

### 예상 결과

- **초기 손실**: ~1.5
- **최종 손실**: ~0.3-0.5
- **압축률**: 28차원 → 128차원 임베딩

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

훈련 완료 후:

1. **모델 평가**: 임베딩 품질 분석
2. **GRPO 준비**: 임베딩을 상태 표현으로 사용
3. **배포**: 추론 서버 구축
