#!/bin/bash
# 원격 Vast.ai 인스턴스에서 실행되는 설정 스크립트
# 직접 실행하지 말 것 - setup_and_train.sh 에서 호출됨

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

CHECKPOINT_ID="$1"
DATASET_ID="$2"
WORKSPACE="/workspace"

log_info "🔧 Vast.ai 인스턴스 설정 시작..."
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# 시스템 패키지
log_info "📦 시스템 패키지 업데이트 중..."
apt-get update -qq
apt-get install -y -qq wget curl unzip python3-pip git

# Python 의존성
log_info "🐍 Python 의존성 설치 중..."
pip install -q --break-system-packages gdown duckdb pandas pyarrow python-dotenv stable-baselines3 tensorboard torch torchvision gymnasium scikit-learn

# Google Drive 다운로드 함수
download_from_gdrive() {
    local file_id="$1"
    local output="$2"
    local label="$3"

    log_info "📥 $label 다운로드 중..."
    gdown "https://drive.google.com/uc?id=$file_id" -O "$output"
    if [ -f "$output" ]; then
        log_success "$label 완료: $(ls -lh "$output" | awk '{print $5}')"
    else
        log_error "$label 다운로드 실패"
        exit 1
    fi
}

mkdir -p models/grpo_scalping_v10/checkpoints logs runs

CHECKPOINT_DEST="models/grpo_scalping_v10/checkpoints/checkpoint_iter770.pt"
DATASET_DEST="datasets_raw_09_11.duckdb"

if [ -f "$CHECKPOINT_DEST" ]; then
    log_warning "체크포인트 이미 존재 - 스킵: $CHECKPOINT_DEST"
else
    download_from_gdrive "$CHECKPOINT_ID" "$CHECKPOINT_DEST" "체크포인트"
fi

if [ -f "$DATASET_DEST" ]; then
    log_warning "데이터셋 이미 존재 - 스킵: $DATASET_DEST"
else
    download_from_gdrive "$DATASET_ID" "$DATASET_DEST" "데이터셋"
fi

# 환경 변수
cat > .env << 'ENVEOF'
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=/workspace
DB_PATH=/workspace/datasets_raw_09_11.duckdb
MODEL_DIR=/workspace/models
LOGS_DIR=/workspace/logs
ENVEOF

# GPU 확인
log_info "🖥️ GPU 상태 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    log_success "GPU 사용 가능"
else
    log_warning "GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다."
fi

log_success "🎉 인스턴스 설정 완료!"
