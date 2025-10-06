#!/bin/bash
# GCP Compute Engine 인스턴스 생성 스크립트
# GPU 인스턴스를 생성하여 딥러닝 훈련을 수행합니다.

set -e  # 에러 발생 시 스크립트 중단

# =============================================================================
# 설정 변수
# =============================================================================

# 프로젝트 설정
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"

# 인스턴스 설정
INSTANCE_NAME="${INSTANCE_NAME:-ai-trader-training}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"  # 4 vCPU, 15GB RAM
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"  # T4 GPU (가성비 좋음)
GPU_COUNT="${GPU_COUNT:-1}"

# 디스크 설정
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-standard}"

# 이미지 설정 (Deep Learning VM with PyTorch)
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# 네트워크 설정
NETWORK_TIER="STANDARD"  # PREMIUM 또는 STANDARD

# =============================================================================
# 함수 정의
# =============================================================================

print_header() {
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        echo "Error: gcloud CLI가 설치되어 있지 않습니다."
        echo "설치 방법: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

check_project() {
    if [ "$PROJECT_ID" = "your-project-id" ]; then
        echo "Error: GCP_PROJECT_ID 환경 변수를 설정해주세요."
        echo "예시: export GCP_PROJECT_ID=my-project-123"
        exit 1
    fi
}

print_config() {
    print_header "인스턴스 생성 설정"
    echo "프로젝트 ID: $PROJECT_ID"
    echo "Zone: $ZONE"
    echo "인스턴스 이름: $INSTANCE_NAME"
    echo "머신 타입: $MACHINE_TYPE"
    echo "GPU 타입: $GPU_TYPE"
    echo "GPU 개수: $GPU_COUNT"
    echo "부트 디스크 크기: $BOOT_DISK_SIZE"
    echo "이미지: $IMAGE_FAMILY"
    echo ""
}

create_instance() {
    print_header "GCP 인스턴스 생성 중..."
    
    gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --maintenance-policy=TERMINATE \
        --provisioning-model=STANDARD \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type="$BOOT_DISK_TYPE" \
        --boot-disk-device-name="$INSTANCE_NAME" \
        --network-tier="$NETWORK_TIER" \
        --metadata="install-nvidia-driver=True" \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --tags=ai-training
    
    echo ""
    echo "✓ 인스턴스 생성 완료!"
}

wait_for_instance() {
    print_header "인스턴스 준비 대기 중..."
    
    echo "인스턴스가 부팅되고 SSH 접속이 가능해질 때까지 기다립니다..."
    sleep 30
    
    local max_attempts=20
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "연결 시도 $attempt/$max_attempts..."
        
        if gcloud compute ssh "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --command="echo 'SSH connection successful'" &> /dev/null; then
            echo "✓ SSH 연결 성공!"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    echo "Warning: SSH 연결 시간 초과. 수동으로 확인해주세요."
    return 1
}

print_instance_info() {
    print_header "인스턴스 정보"
    
    gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="table(name,zone,machineType,status,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)"
    
    echo ""
    echo "SSH 접속 명령어:"
    echo "  gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE"
    echo ""
    echo "다음 단계:"
    echo "  1. ./setup_instance.sh 를 실행하여 환경을 설정하세요."
    echo "  2. ./upload_and_train.sh 를 실행하여 훈련을 시작하세요."
    echo ""
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    print_header "GCP 훈련 인스턴스 생성"
    
    # 사전 검사
    check_gcloud
    check_project
    
    # 설정 출력
    print_config
    
    # 사용자 확인
    read -p "위 설정으로 인스턴스를 생성하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    # 인스턴스 생성
    create_instance
    
    # 인스턴스 준비 대기
    wait_for_instance
    
    # 인스턴스 정보 출력
    print_instance_info
    
    echo "✓ 모든 작업이 완료되었습니다!"
}

# 스크립트 실행
main "$@"
