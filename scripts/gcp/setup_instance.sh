#!/bin/bash
# GCP 인스턴스 환경 설정 스크립트
# 필요한 패키지와 의존성을 설치합니다.

set -e  # 에러 발생 시 스크립트 중단

# =============================================================================
# 설정 변수
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-ai-trader-training}"

# =============================================================================
# 함수 정의
# =============================================================================

print_header() {
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

setup_remote_environment() {
    print_header "원격 환경 설정 중..."
    
    # 설정 스크립트를 원격으로 실행
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="bash -s" << 'REMOTE_SCRIPT'

set -e

echo "=========================================="
echo "시스템 패키지 업데이트"
echo "=========================================="
sudo apt-get update
sudo apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    zip \
    unzip

echo ""
echo "=========================================="
echo "NVIDIA 드라이버 확인"
echo "=========================================="
nvidia-smi || echo "Warning: nvidia-smi 실행 실패. GPU 드라이버를 확인하세요."

echo ""
echo "=========================================="
echo "Python 환경 확인"
echo "=========================================="
python3 --version
pip3 --version

echo ""
echo "=========================================="
echo "작업 디렉토리 생성"
echo "=========================================="
mkdir -p ~/ai-trader
cd ~/ai-trader

echo ""
echo "=========================================="
echo "PyTorch 및 CUDA 확인"
echo "=========================================="
python3 << 'PYTHON_CHECK'
import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)
PYTHON_CHECK

echo ""
echo "✓ 환경 설정 완료!"

REMOTE_SCRIPT

    echo ""
    echo "✓ 원격 환경 설정이 완료되었습니다!"
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    print_header "GCP 인스턴스 환경 설정"
    
    echo "인스턴스: $INSTANCE_NAME"
    echo "프로젝트: $PROJECT_ID"
    echo "Zone: $ZONE"
    echo ""
    
    setup_remote_environment
    
    echo ""
    print_header "설정 완료"
    echo "다음 단계: ./upload_and_train.sh 를 실행하여 훈련을 시작하세요."
}

main "$@"
