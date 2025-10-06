#!/bin/bash
# GCP 인스턴스 삭제 스크립트
# 훈련이 완료된 후 인스턴스를 삭제하여 비용을 절감합니다.

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

check_instance_exists() {
    if ! gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" &> /dev/null; then
        echo "Error: 인스턴스 '$INSTANCE_NAME'를 찾을 수 없습니다."
        exit 1
    fi
}

backup_results() {
    print_header "결과 백업 확인"
    
    echo "인스턴스를 삭제하기 전에 훈련 결과를 백업했는지 확인하세요!"
    echo ""
    echo "백업 명령어:"
    echo "  gcloud compute scp --recurse \\"
    echo "    $INSTANCE_NAME:~/ai-trader/models/embedding \\"
    echo "    ./models/ \\"
    echo "    --project=$PROJECT_ID \\"
    echo "    --zone=$ZONE"
    echo ""
    
    read -p "결과를 백업했습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "먼저 결과를 백업하세요."
        exit 0
    fi
}

delete_instance() {
    print_header "인스턴스 삭제 중..."
    
    gcloud compute instances delete "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --quiet
    
    echo ""
    echo "✓ 인스턴스가 삭제되었습니다!"
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    print_header "GCP 인스턴스 삭제"
    
    echo "인스턴스: $INSTANCE_NAME"
    echo "프로젝트: $PROJECT_ID"
    echo "Zone: $ZONE"
    echo ""
    
    # 인스턴스 존재 확인
    check_instance_exists
    
    # 백업 확인
    backup_results
    
    # 최종 확인
    echo ""
    echo "⚠️  경고: 이 작업은 되돌릴 수 없습니다!"
    read -p "정말로 인스턴스를 삭제하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    # 인스턴스 삭제
    delete_instance
    
    echo ""
    print_header "완료"
    echo "인스턴스가 성공적으로 삭제되었습니다."
    echo "GCP 콘솔에서 확인하세요: https://console.cloud.google.com/compute/instances"
}

main "$@"
