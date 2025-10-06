# 🚀 Colab Pro 빠른 시작 가이드

5분 안에 Google Colab Pro에서 임베딩 모델 훈련을 시작하세요!

## ⚡ 빠른 시작 (5분)

### 1단계: Colab 노트북 열기 (1분)

1. Google Colab 접속: https://colab.research.google.com/
2. `파일` → `노트북 업로드`
3. `embedding_training_colab_pro.ipynb` 업로드

### 2단계: 런타임 설정 (30초)

1. `런타임` → `런타임 유형 변경`
2. **하드웨어 가속기**: GPU
3. **GPU 유형**: V100 또는 A100
4. 저장

### 3단계: Google Drive 준비 (2분)

Google Drive에 다음 구조 생성:

```
MyDrive/
└── ColabData/
    ├── datasets/
    │   └── stockbot/
    │       └── datasets_norm_all.duckdb  ← 여기에 데이터셋 업로드
    └── models/
        └── stockbot/
            └── embedding/  ← 자동 생성됨
```

### 4단계: 노트북 실행 (1분)

1. 첫 번째 셀부터 순서대로 실행 (`Shift + Enter`)
2. Google Drive 마운트 승인
3. 경로 설정 확인:
   ```python
   DRIVE_ROOT = "/content/drive/MyDrive/ColabData"
   ```

### 5단계: 훈련 시작! 🎉

"3️⃣ 월별 훈련 실행" 섹션의 셀 실행:
```python
# 자동으로 9월~12월 순차 훈련
```

## 📝 체크리스트

시작하기 전에 확인하세요:

- [ ] Google Colab Pro 구독 활성화
- [ ] Google Drive에 `datasets_norm_all.duckdb` 업로드 완료 (`내 드라이브/ColabData/datasets/stockbot/`)
- [ ] 런타임 유형: GPU (V100/A100)
- [ ] Drive 여유 공간: 최소 50GB

## ⚙️ 기본 설정 (권장)

노트북의 설정 셀에서 다음 값 확인:

```python
# 데이터 기간
YEAR = 2024
START_MONTH = 9
END_MONTH = 12

# 배치 크기 (GPU에 따라)
BATCH_SIZE = 256  # V100: 128-256, A100: 256-512

# 최대 샘플
MAX_SAMPLES = 5000000  # 500만 샘플
```

## 🎯 예상 소요 시간

**V100 GPU 기준**:
- 9월 (초기 훈련): ~6-8시간
- 10월 (Fine-tuning): ~3-4시간
- 11월 (Fine-tuning): ~3-4시간
- 12월 (Fine-tuning): ~3-4시간
- **전체**: ~18-24시간

**A100 GPU 기준**:
- 전체: ~10-15시간 (약 40% 빠름)

## 📊 훈련 모니터링
### TensorBoard로 실시간 확인

노트북의 "4️⃣ TensorBoard 시각화" 섹션 실행:

```python
# 특정 월의 로그 보기
%tensorboard --logdir /content/drive/MyDrive/ColabData/models/stockbot/embedding
```

확인할 메트릭:
- **train/loss**: 훈련 손실 (감소해야 함)
- **val/loss**: 검증 손실 (감소해야 함)
- **val/temporal_coherence**: 시간적 일관성 (증가해야 함)

### GPU 사용률 확인

```python
!nvidia-smi
```

정상 상태:
- GPU Utilization: 80-100%
- Memory Usage: 10-15GB (V100 16GB 기준)

## 🔧 자주 묻는 질문 (FAQ)

### Q1: GPU 메모리 부족 오류가 발생해요

**A**: 배치 크기를 줄이세요:
```python
BATCH_SIZE = 128  # 또는 64
```

### Q2: 훈련이 너무 느려요

**A**: 다음을 확인하세요:
1. GPU 사용률: `!nvidia-smi` (80% 이상이어야 함)
2. 워커 수 증가: `NUM_WORKERS = 4`
3. 검증 주기 증가: `VAL_EVERY = 5`

### Q3: Colab 세션이 끊겼어요

**A**: 마지막 체크포인트에서 재개:
```python
# 마지막 체크포인트 찾기
import glob
checkpoints = glob.glob(f"{OUTPUT_BASE}_2024_09/*.pt")
latest = max(checkpoints, key=os.path.getctime)
print(f"Resume from: {latest}")
```

### Q4: 데이터베이스를 찾을 수 없어요

**A**: 경로를 확인하세요:
```python
import os
DB_PATH = "/content/drive/MyDrive/ColabData/datasets/stockbot/datasets_norm_all.duckdb"
print(f"Exists: {os.path.exists(DB_PATH)}")

# 파일 목록 확인
!ls -lh /content/drive/MyDrive/ColabData/datasets/stockbot/
```

### Q5: Drive 용량이 부족해요

**A**: 체크포인트 간격을 늘리세요:
```python
CHECKPOINT_INTERVAL = 10  # 5 → 10 (에포크마다 저장 횟수 감소)
```

## 💡 프로 팁

### 1. 백그라운드 실행
- Colab Pro는 탭을 닫아도 24시간 동안 실행
- 정기적으로 확인하여 오류 체크

### 2. 비용 절약
- 필요한 월만 훈련 (예: 9-10월만)
- 테스트는 적은 샘플로: `MAX_SAMPLES = 1000000`

### 3. 최고 성능 모델 찾기
```python
# validation loss가 가장 낮은 체크포인트
import torch
checkpoints = glob.glob(f"{OUTPUT_BASE}_2024_*/*.pt")
best = min(checkpoints, key=lambda x: torch.load(x)['training_metadata']['val_loss'])
print(f"Best model: {best}")
```

### 4. 로컬로 다운로드
```python
# Google Drive에서 직접 다운로드 (권장)
# 또는 Colab에서:
from google.colab import files
files.download(best_checkpoint)
```

## 📱 모바일에서 모니터링

1. Google Drive 앱 설치
2. `StockBot/models/embedding_2024_XX/` 폴더 확인
3. 체크포인트 파일 생성 확인
4. TensorBoard 로그는 PC에서 확인

## 🆘 문제 해결

### 오류가 발생하면:

1. **런타임 재시작**
   - `런타임` → `런타임 다시 시작`

2. **캐시 정리**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Drive 다시 마운트**
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```

4. **로그 확인**
   - 노트북 출력에서 오류 메시지 확인
   - TensorBoard에서 이상 패턴 확인

## 📚 다음 단계

훈련 완료 후:

1. **모델 평가**
   - 노트북의 "5️⃣ 모델 평가" 섹션 실행

2. **GRPO 훈련**
   - 임베딩 모델을 사용하여 강화학습 에이전트 훈련
   - `docs/GRPO_TRAINING.md` 참조

3. **백테스팅**
   - 실제 데이터로 성능 검증
   - `docs/GRPO_BACKTESTING.md` 참조

## 🎓 추가 학습

- [상세 가이드](README.md): 전체 설정 및 고급 옵션
- [트러블슈팅](README.md#트러블슈팅): 일반적인 문제 해결
- [프로젝트 문서](../../docs/): 전체 시스템 이해

---

**준비되셨나요?** 노트북을 열고 첫 번째 셀부터 실행하세요! 🚀
