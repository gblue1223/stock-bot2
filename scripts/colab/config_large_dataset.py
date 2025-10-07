import os
from pathlib import Path

"""
대용량 데이터셋 (월별 5천만 개) 훈련을 위한 최적화된 설정

데이터 규모:
- 전체: 4억 개
- 월별: 약 5천만 개
- 현재 문제: 50만 개만 사용 → 과적합 발생

해결 전략:
1. 샘플 수 대폭 증가
2. 배치 크기 증가
3. Temperature 조정
4. Regularization 강화
5. 학습률 스케줄링
"""

# ========================================
# 데이터 설정
# ========================================

# 샘플 수 (메모리에 따라 조정)
# Colab Pro (25GB RAM): 500만 ~ 1000만
# Colab Pro+ (52GB RAM): 1000만 ~ 2000만
# A100 80GB 최적화: 5천만 샘플 전체 사용 가능
MAX_SAMPLES = 50000000  # 5천만 샘플 (100%, 월별 전체 데이터)

# 데이터 기간
YEAR = 2024
START_MONTH = 9
END_MONTH = 12

# ========================================
# 모델 아키텍처
# ========================================

SEQ_LEN = 60
EMBEDDING_DIM = 128
NUM_HEADS = 4

# Regularization
DROPOUT = 0.2  # Dropout 추가 (과적합 방지)
WEIGHT_DECAY = 1e-4  # L2 regularization

# ========================================
# 훈련 하이퍼파라미터
# ========================================

# 배치 크기 (GPU 메모리에 따라)
# V100 (16GB): 512 ~ 768
# A100 (40GB): 1024 ~ 2048
# A100 (80GB): 4096 ~ 8192 (최적)
BATCH_SIZE = 4096  # A100 80GB 최적화

# 에포크 수 (A100 80GB 최적화)
INITIAL_EPOCHS = 15  # 대용량 데이터에 맞게 증가
FINETUNE_EPOCHS = 10  # Fine-tuning도 충분히

# 학습률 (NaN 방지를 위해 매우 보수적으로 설정)
INITIAL_LR = 5e-5  # 1e-4 → 5e-5 (더 안정적)
FINETUNE_LR = 1e-5  # 5e-5 → 1e-5 (Fine-tuning)

# 학습률 스케줄러
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"  # "cosine" or "step"
LR_WARMUP_EPOCHS = 2  # Warmup 기간

# ========================================
# 대조 학습 설정
# ========================================

# Temperature (중요!)
# 낮을수록 엄격한 학습, 높을수록 부드러운 학습
TEMPERATURE = 0.10  # 0.07 → 0.10 (과적합 방지)

# 시간 임계값
POSITIVE_TIME_THRESHOLD = 10  # 초
NEGATIVE_TIME_THRESHOLD = 60  # 초

# Hard Negative Mining
USE_HARD_NEGATIVES = True  # 어려운 negative 샘플 우선 선택
HARD_NEGATIVE_RATIO = 0.5  # 50%는 hard negative

# ========================================
# 데이터 로더 설정
# ========================================

NUM_WORKERS = 16  # A100 80GB: 더 많은 워커로 데이터 로딩 가속
PIN_MEMORY = True

# ========================================
# 체크포인트 및 검증
# ========================================

CHECKPOINT_INTERVAL = 3  # A100: 더 적게 저장 (속도 우선)
VAL_EVERY = 2  # 2 에포크마다 검증 (속도 향상)

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5  # 5 에포크 동안 개선 없으면 중단
EARLY_STOPPING_MIN_DELTA = 0.001  # 최소 개선 폭

# ========================================
# 메모리 최적화
# ========================================

# Gradient Accumulation (메모리 부족 시)
USE_GRADIENT_ACCUMULATION = False
ACCUMULATION_STEPS = 2  # 2 스텝마다 업데이트

# Mixed Precision Training
USE_AMP = True  # Automatic Mixed Precision (메모리 절약 + 속도 향상)

# ========================================
# 출력 설정
# ========================================

DRIVE_ROOT = "/content/drive/MyDrive/ColabData"
DB_PATH = f"{DRIVE_ROOT}/datasets/stockbot/datasets_norm_all.duckdb"
OUTPUT_BASE = f"{DRIVE_ROOT}/models/stockbot/embedding"

# 디렉토리 생성
os.makedirs(OUTPUT_BASE, exist_ok=True)

# 데이터베이스 파일 존재 확인
if os.path.exists(DB_PATH):
    print(f"✓ Database file found: {os.path.getsize(DB_PATH) / 1e9:.2f} GB")
else:
    print(f"⚠️  Database file not found!")
    print(f"   Please upload datasets_norm_all.duckdb to:")
    print(f"   {DB_PATH}")
    
# ========================================
# 설정 요약 출력
# ========================================

def print_config():
    """설정 요약 출력"""
    print("=" * 80)
    print("대용량 데이터셋 훈련 설정")
    print("=" * 80)
    print(f"샘플 수: {MAX_SAMPLES:,} (월별 5천만 개 중 {MAX_SAMPLES/30000000*100:.1f}%)")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"에포크: {INITIAL_EPOCHS} (초기) / {FINETUNE_EPOCHS} (Fine-tuning)")
    print(f"학습률: {INITIAL_LR} (초기) / {FINETUNE_LR} (Fine-tuning)")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Dropout: {DROPOUT}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Early Stopping: {USE_EARLY_STOPPING}")
    print(f"Mixed Precision: {USE_AMP}")
    print("=" * 80)
    
    # 예상 메모리 사용량
    samples_per_epoch = MAX_SAMPLES * 0.7  # Train split
    batches_per_epoch = samples_per_epoch / BATCH_SIZE
    print(f"\n예상 훈련 통계:")
    print(f"- 훈련 샘플: {samples_per_epoch:,.0f}")
    print(f"- 배치 수/에포크: {batches_per_epoch:,.0f}")
    print(f"- 총 배치 수: {batches_per_epoch * INITIAL_EPOCHS:,.0f}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
