#!/bin/bash
# 전체 훈련 워크플로우 실행 스크립트
# 인스턴스 생성부터 훈련 완료까지 자동으로 실행합니다.

set -e  # 에러 발생 시 스크립트 중단

# =============================================================================
# 설정
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# 함수 정의
# =============================================================================

print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
    echo ""
}

print_step() {
    echo ""
    echo ">>> $1"
    echo ""
}

check_config() {
    print_header "설정 확인"
    
    if [ -z "$GCP_PROJECT_ID" ] || [ "$GCP_PROJECT_ID" = "your-project-id" ]; then
        echo "Error: GCP_PROJECT_ID가 설정되지 않았습니다."
        echo ""
        echo "다음 중 하나를 선택하세요:"
        echo ""
        echo "1. 환경 변수 설정:"
        echo "   export GCP_PROJECT_ID=your-project-id"
        echo "   export DB_PATH=/path/to/database.duckdb"
        echo ""
        echo "2. 설정 파일 사용:"
        echo "   cp config.example.sh config.sh"
        echo "   # config.sh 편집"
        echo "   source config.sh"
        echo ""
        exit 1
    fi
    
    if [ -z "$DB_PATH" ]; then
        echo "Error: DB_PATH가 설정되지 않았습니다."
        echo "   export DB_PATH=/path/to/database.duckdb"
        exit 1
    fi
    
    if [ ! -f "$DB_PATH" ]; then
        echo "Error: 데이터베이스 파일을 찾을 수 없습니다: $DB_PATH"
        exit 1
    fi
    
    echo "✓ 프로젝트 ID: $GCP_PROJECT_ID"
    echo "✓ Zone: ${GCP_ZONE:-us-central1-a}"
    echo "✓ 데이터베이스: $DB_PATH"
    echo "✓ 인스턴스 이름: ${INSTANCE_NAME:-ai-trader-training}"
    echo ""
}

show_workflow() {
    print_header "훈련 워크플로우"
    
    echo "다음 단계가 순차적으로 실행됩니다:"
    echo ""
    echo "1. ✨ 인스턴스 생성 (약 2-3분)"
    echo "2. 🔧 환경 설정 (약 1분)"
    echo "3. 📤 코드 및 데이터 업로드 (약 2-5분, 데이터 크기에 따라)"
    echo "4. 🚀 훈련 시작 (백그라운드)"
    echo ""
    echo "예상 소요 시간: 약 5-10분 (훈련 시간 제외)"
    echo ""
}

estimate_costs() {
    print_header "예상 비용"
    
    local machine_type="${MACHINE_TYPE:-n1-standard-4}"
    local gpu_type="${GPU_TYPE:-nvidia-tesla-t4}"
    local epochs="${EPOCHS:-50}"
    
    echo "설정:"
    echo "  머신: $machine_type"
    echo "  GPU: $gpu_type"
    echo "  에포크: $epochs"
    echo ""
    
    # 간단한 비용 추정
    local hourly_cost="0.54"  # T4 기본 설정
    if [ "$gpu_type" = "nvidia-tesla-v100" ]; then
        hourly_cost="2.86"
    elif [ "$gpu_type" = "nvidia-tesla-p100" ]; then
        hourly_cost="1.84"
    fi
    
    # 훈련 시간 추정 (매우 대략적)
    local estimated_hours=3
    if [ "$epochs" -gt 50 ]; then
        estimated_hours=5
    fi
    if [ "$epochs" -gt 100 ]; then
        estimated_hours=8
    fi
    
    local total_cost=$(echo "$hourly_cost * $estimated_hours" | bc)
    
    echo "예상 비용:"
    echo "  시간당: \$$hourly_cost"
    echo "  예상 훈련 시간: 약 ${estimated_hours}시간"
    echo "  예상 총 비용: \$$total_cost"
    echo ""
    echo "⚠️  실제 비용은 다를 수 있습니다."
    echo ""
}

confirm_start() {
    echo "계속하시겠습니까?"
    read -p "(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
}

run_workflow() {
    # Step 1: 인스턴스 생성
    print_step "Step 1/4: 인스턴스 생성"
    bash "$SCRIPT_DIR/create_instance.sh"
    
    # Step 2: 환경 설정
    print_step "Step 2/4: 환경 설정"
    bash "$SCRIPT_DIR/setup_instance.sh"
    
    # Step 3: 코드 및 데이터 업로드
    print_step "Step 3/4: 코드 및 데이터 업로드"
    bash "$SCRIPT_DIR/upload_and_train.sh"
    
    # Step 4: 완료
    print_step "Step 4/4: 완료!"
}

show_next_steps() {
    print_header "다음 단계"
    
    echo "훈련이 백그라운드에서 실행 중입니다! 🎉"
    echo ""
    echo "모니터링 방법:"
    echo "  ./monitor_training.sh"
    echo ""
    echo "비용 확인:"
    echo "  ./check_costs.sh"
    echo ""
    echo "훈련 완료 후:"
    echo "  1. ./download_results.sh  # 결과 다운로드"
    echo "  2. ./delete_instance.sh   # 인스턴스 삭제 (중요!)"
    echo ""
    echo "SSH 접속:"
    echo "  gcloud compute ssh ${INSTANCE_NAME:-ai-trader-training} \\"
    echo "    --project=$GCP_PROJECT_ID \\"
    echo "    --zone=${GCP_ZONE:-us-central1-a}"
    echo ""
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    cd "$SCRIPT_DIR"
    
    print_header "🚀 GCP 훈련 자동 실행"
    
    # 설정 확인
    check_config
    
    # 워크플로우 설명
    show_workflow
    
    # 비용 추정
    estimate_costs
    
    # 사용자 확인
    confirm_start
    
    # 시작 시간 기록
    local start_time=$(date +%s)
    
    # 워크플로우 실행
    run_workflow
    
    # 종료 시간 계산
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    # 완료 메시지
    print_header "✅ 모든 작업 완료"
    echo "소요 시간: ${minutes}분 ${seconds}초"
    echo ""
    
    # 다음 단계 안내
    show_next_steps
}

# 스크립트 실행
main "$@"
