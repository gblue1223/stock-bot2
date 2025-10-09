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
MAX_CHUNKS = 10000000

# 데이터 기간
YEAR = 2024
START_MONTH = 9
END_MONTH = 12

# ========================================
# 모델 아키텍처
# ========================================

# 속도 최적화: 시퀀스 길이 감소 (가장 효과적!)
SEQ_LEN = 30  # 60 → 30 (계산량 50% 감소)
EMBEDDING_DIM = 128
NUM_HEADS = 4

# Regularization
DROPOUT = 0.2  # Dropout 추가 (과적합 방지)
WEIGHT_DECAY = 1e-4  # L2 regularization

# ========================================
# 훈련 하이퍼파라미터
# ========================================

# 배치 크기 (GPU 메모리에 따라)
# V100 (16GB): 8192 ~ 12288
# A100 (40GB): 16384 ~ 24576
# A100 (80GB): 32768 ~ 65536 (최적)
# 중요: 배치 크기가 클수록 GPU 활용도 향상!
BATCH_SIZE = 16384  # A100 40GB 기준 (메모리 확인 후 32768로 증가 가능)

# 에포크 수 (속도 우선)
INITIAL_EPOCHS = 3  # 15 → 10 (속도 우선)
FINETUNE_EPOCHS = 2   # 10 → 5

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

# 시간 임계값 (시간_scalar는 표준화된 값이므로 조정 필요)
# 시간_scalar는 장 시작 후 경과 시간(초)을 표준화한 값
# 표준화 범위가 대략 [-2, 2]라고 가정하면:
# - 0.05 ≈ 수 초 차이
# - 0.3 ≈ 수십 초 차이
# - 1.0 ≈ 수 분 차이
POSITIVE_TIME_THRESHOLD = 0.05  # 표준화된 시간 단위 (매우 가까운 시간)
NEGATIVE_TIME_THRESHOLD = 0.5   # 표준화된 시간 단위 (충분히 먼 시간)

# Hard Negative Mining
USE_HARD_NEGATIVES = True  # 어려운 negative 샘플 우선 선택
HARD_NEGATIVE_RATIO = 0.5  # 50%는 hard negative

# ========================================
# 데이터 로더 설정
# ========================================

# Colab 환경 최적화: CPU 코어가 2개뿐이므로 num_workers=0이 최적!
# multiprocessing 오버헤드가 오히려 성능 저하 유발
NUM_WORKERS = 0  # Colab: 단일 프로세스가 가장 빠름!
PIN_MEMORY = True

# ========================================
# 체크포인트 및 검증
# ========================================

CHECKPOINT = "/content/drive/MyDrive/models/embedding_2024_09"
CHECKPOINT_INTERVAL = 1  # 청크마다 저장 (Colab 세션 끊김 대비)
VAL_EVERY = 1  # 매 에포크 검증 (학습 모니터링)
LOG_INTERVAL = 100  # 로그 출력 간격 (50 → 100, I/O 더욱 감소)

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
    print(f"샘플 수: {MAX_SAMPLES:,} (월별 5천만 개 중 {MAX_SAMPLES/50000000*100:.1f}%)")
    print(f"워커 수: {NUM_WORKERS}")
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
