# Google Colab 훈련 가이드

Google Colab Pro 환경에서 임베딩 모델을 훈련하는 방법을 안내합니다.

## 📋 목차

- [사전 준비](#사전-준비)
- [노트북 사용법](#노트북-사용법)
- [설정 가이드](#설정-가이드)
- [훈련 전략](#훈련-전략)
- [트러블슈팅](#트러블슈팅)

## 🚀 사전 준비

### 1. Google Colab Pro 구독

- **Colab Pro**: 고성능 GPU (V100, A100) 사용 가능
- **최대 실행 시간**: 24시간 연속 실행
- **GPU 메모리**: V100 (16GB), A100 (40GB)

### 2. Google Drive 준비

다음 디렉토리 구조를 Google Drive에 생성하세요:

```
MyDrive/
└── ColabData/
    ├── datasets/
    │   └── stockbot/
    │       └── datasets_norm_all.duckdb  # 정규화된 데이터셋
    └── models/
        └── stockbot/
            └── embedding/  # 모델 체크포인트 저장 위치 (자동 생성)
```

### 3. 데이터셋 업로드

로컬에서 정규화된 데이터베이스 파일을 Google Drive에 업로드:

```bash
# 로컬 → Google Drive
# 웹 인터페이스를 통해 datasets_norm_all.duckdb 업로드
# 또는 Google Drive Desktop 앱 사용
```

### 4. GitHub 저장소 준비

프로젝트 코드를 GitHub에 푸시하거나, 노트북에서 직접 코드 복사:

```bash
# 로컬에서 GitHub에 푸시
git add .
git commit -m "Add embedding training code"
git push origin main
```

## 📓 노트북 사용법

### 1. 노트북 열기

1. Google Colab 접속: https://colab.research.google.com/
2. `파일` → `노트북 업로드`
3. `embedding_training_colab_pro.ipynb` 선택

또는 GitHub에서 직접 열기:
```
https://colab.research.google.com/github/YOUR_USERNAME/stock-bot2/blob/main/scripts/colab/embedding_training_colab_pro.ipynb
```

### 2. 런타임 설정

1. `런타임` → `런타임 유형 변경`
2. **하드웨어 가속기**: GPU
3. **GPU 유형**: V100 또는 A100 (Colab Pro)
4. 저장

### 3. 셀 실행

노트북의 셀을 순서대로 실행:

1. **환경 설정**: GPU 확인, Drive 마운트, 패키지 설치
2. **경로 설정**: 데이터베이스 및 출력 경로 지정
3. **훈련 실행**: 월별 순차 훈련
4. **시각화**: TensorBoard로 결과 확인
5. **평가**: 모델 성능 평가

## ⚙️ 설정 가이드

### 경로 설정

노트북의 "2️⃣ 설정 및 경로 지정" 섹션에서 수정:

```python
# Google Drive 경로
DRIVE_ROOT = "/content/drive/MyDrive/ColabData"

# 데이터베이스 경로
DB_PATH = f"{DRIVE_ROOT}/datasets/stockbot/datasets_norm_all.duckdb"

# 출력 디렉토리
OUTPUT_BASE = f"{DRIVE_ROOT}/models/stockbot/embedding"
```

### 훈련 하이퍼파라미터

GPU 메모리와 훈련 시간에 맞게 조정:

```python
# 배치 크기 (GPU 메모리에 따라)
BATCH_SIZE = 256  # V100: 128-256, A100: 256-512

# 에포크 수
INITIAL_EPOCHS = 30  # 초기 훈련
FINETUNE_EPOCHS = 15  # Fine-tuning

# 학습률
INITIAL_LR = 1e-4
FINETUNE_LR = 5e-5

# 최대 샘플 수 (메모리 제한)
MAX_SAMPLES = 5000000  # 500만 샘플
```

### 데이터 필터링

훈련할 기간 설정:

```python
YEAR = 2024
START_MONTH = 9   # 9월부터
END_MONTH = 12    # 12월까지
```

## 📊 훈련 전략

### 월별 순차 훈련

각 월의 데이터로 순차적으로 훈련:

1. **9월 (초기 훈련)**
   - 30 epochs
   - Learning rate: 1e-4
   - 처음부터 훈련

2. **10월 (Fine-tuning)**
   - 15 epochs
   - Learning rate: 5e-5
   - 9월 모델에서 시작

3. **11월, 12월 (Fine-tuning)**
   - 동일한 방식으로 계속

### 메모리 최적화

GPU 메모리가 부족한 경우:

```python
# 옵션 1: 배치 크기 줄이기
BATCH_SIZE = 128  # 256 → 128

# 옵션 2: 샘플 수 제한
MAX_SAMPLES = 3000000  # 300만 샘플

# 옵션 3: 임베딩 차원 줄이기
EMBEDDING_DIM = 64  # 128 → 64
```

### 훈련 시간 예상

V100 GPU 기준:

- **초기 훈련 (30 epochs)**: 약 6-8시간
- **Fine-tuning (15 epochs)**: 약 3-4시간
- **전체 4개월**: 약 18-24시간

## 📈 모니터링

### TensorBoard

실시간 훈련 모니터링:

```python
# 특정 월의 로그 보기
%tensorboard --logdir /content/drive/MyDrive/ColabData/models/stockbot/embedding_2024_09/tensorboard_logs

# 모든 월 비교
%tensorboard --logdir /content/drive/MyDrive/ColabData/models/stockbot/embedding
```

### GPU 사용률 확인

```python
# GPU 상태 확인
!nvidia-smi

# 5초마다 갱신
!watch -n 5 nvidia-smi
```

### 메모리 사용량

```python
# 시스템 메모리
!free -h

# GPU 메모리
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## 🔧 트러블슈팅

### GPU OOM (Out of Memory)

**증상**: `CUDA out of memory` 오류

**해결 방법**:
```python
# 1. 배치 크기 줄이기
BATCH_SIZE = 64  # 또는 32

# 2. 캐시 정리
import torch
torch.cuda.empty_cache()

# 3. 런타임 재시작
# 런타임 → 런타임 다시 시작
```

### 데이터베이스 연결 오류

**증상**: `Cannot connect to database` 오류

**해결 방법**:
```python
# 1. Drive 마운트 확인
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 2. Google Drive 경로 확인
!ls -lh /content/drive/MyDrive/ColabData/datasets/stockbot/

# 3. 데이터베이스 파일 존재 확인
import os
print(f"DB exists: {os.path.exists(DB_PATH)}")

# 4. 경로 수정
DB_PATH = "/content/drive/MyDrive/ColabData/datasets/stockbot/datasets_norm_all.duckdb"

**증상**: Colab 세션 종료 또는 연결 끊김

**해결 방법**:
```python
# 1. 마지막 체크포인트 찾기
import glob
import os

checkpoints = glob.glob(f"{OUTPUT_BASE}_2024_09/*.pt")
if checkpoints:
    latest = max(checkpoints, key=os.path.getctime)
    print(f"Latest checkpoint: {latest}")

# 2. 해당 체크포인트에서 재개
# train_finetune() 함수에서 --resume 옵션 사용
```

### 느린 훈련 속도

**증상**: 에포크당 시간이 너무 오래 걸림

**해결 방법**:
```python
# 1. GPU 사용 확인
!nvidia-smi  # GPU 사용률이 낮으면 문제

# 2. 워커 수 조정
NUM_WORKERS = 4  # 2 → 4

# 3. 검증 주기 늘리기
VAL_EVERY = 5  # 3 → 5 (검증을 덜 자주)

# 4. 샘플 수 줄이기 (테스트용)
MAX_SAMPLES = 1000000  # 빠른 테스트
```

### Drive 용량 부족

**증상**: 체크포인트 저장 실패

**해결 방법**:
```python
# 1. Drive 용량 확인
!df -h /content/drive

# 2. 불필요한 체크포인트 삭제
# 중간 체크포인트만 유지하고 나머지 삭제

# 3. 체크포인트 간격 늘리기
CHECKPOINT_INTERVAL = 10  # 5 → 10
```

## 💡 유용한 팁

### 1. 백그라운드 실행

Colab 탭을 닫아도 계속 실행:
- Colab Pro는 백그라운드 실행 지원
- 단, 24시간 제한 있음

### 2. 알림 설정

훈련 완료 시 알림 받기:
```python
# 이메일 알림 (선택사항)
from google.colab import auth
# Gmail API 설정 필요
```

### 3. 체크포인트 관리

```python
# 최고 성능 모델만 유지
# validation loss가 가장 낮은 체크포인트 찾기
import glob
import torch

checkpoints = glob.glob(f"{OUTPUT_BASE}_2024_09/*.pt")
best_checkpoint = None
best_val_loss = float('inf')

for ckpt in checkpoints:
    data = torch.load(ckpt)
    val_loss = data['training_metadata']['val_loss']
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_checkpoint = ckpt

print(f"Best checkpoint: {best_checkpoint}")
print(f"Best val loss: {best_val_loss}")
```

### 4. 로컬로 다운로드

```python
# 특정 체크포인트만 다운로드
from google.colab import files
files.download(best_checkpoint)

# 또는 Google Drive에서 직접 다운로드 (권장)
```

## 📚 참고 자료

- [Colab Pro 가이드](https://colab.research.google.com/signup)
- [TensorBoard 사용법](https://www.tensorflow.org/tensorboard)
- [프로젝트 README](../../README.md)
- [GCP 훈련 가이드](../../docs/GCP_TRAINING_GUIDE.md)

## 🆘 도움이 필요하신가요?

- GitHub Issues: 버그 리포트 및 기능 요청
- 프로젝트 문서: `docs/` 디렉토리 참조
