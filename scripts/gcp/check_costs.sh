#!/bin/bash
# 비용 확인 스크립트
# 현재 실행 중인 인스턴스의 예상 비용을 확인합니다.

set -e  # 에러 발생 시 스크립트 중단

# =============================================================================
# 설정 변수
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-ai-trader-training}"

# =============================================================================
# 가격 정보 (us-central1-a 기준, 2024년 기준)
# =============================================================================

declare -A MACHINE_PRICES=(
    ["n1-standard-1"]="0.0475"
    ["n1-standard-2"]="0.0950"
    ["n1-standard-4"]="0.1900"
    ["n1-standard-8"]="0.3800"
    ["n1-highmem-2"]="0.1184"
    ["n1-highmem-4"]="0.2368"
    ["n1-highmem-8"]="0.4736"
)

declare -A GPU_PRICES=(
    ["nvidia-tesla-k80"]="0.45"
    ["nvidia-tesla-p4"]="0.60"
    ["nvidia-tesla-t4"]="0.35"
    ["nvidia-tesla-p100"]="1.46"
    ["nvidia-tesla-v100"]="2.48"
)

# =============================================================================
# 함수 정의
# =============================================================================

print_header() {
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

check_instance_status() {
    print_header "인스턴스 상태 확인"
    
    if ! gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" &> /dev/null; then
        echo "Error: 인스턴스 '$INSTANCE_NAME'를 찾을 수 없습니다."
        exit 1
    fi
    
    local status=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="value(status)")
    
    echo "인스턴스: $INSTANCE_NAME"
    echo "상태: $status"
    echo ""
    
    if [ "$status" != "RUNNING" ]; then
        echo "인스턴스가 실행 중이 아닙니다. 비용이 발생하지 않습니다."
        exit 0
    fi
}

get_instance_info() {
    local machine_type=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="value(machineType.basename())")
    
    local gpu_info=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="value(guestAccelerators[0].acceleratorType.basename())")
    
    local gpu_count=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="value(guestAccelerators[0].acceleratorCount)")
    
    echo "$machine_type|$gpu_info|$gpu_count"
}

calculate_costs() {
    print_header "비용 계산"
    
    local info=$(get_instance_info)
    local machine_type=$(echo "$info" | cut -d'|' -f1)
    local gpu_type=$(echo "$info" | cut -d'|' -f2)
    local gpu_count=$(echo "$info" | cut -d'|' -f3)
    
    echo "머신 타입: $machine_type"
    echo "GPU 타입: $gpu_type"
    echo "GPU 개수: $gpu_count"
    echo ""
    
    # 머신 비용
    local machine_cost=${MACHINE_PRICES[$machine_type]:-0}
    
    # GPU 비용
    local gpu_cost=0
    if [ -n "$gpu_type" ] && [ "$gpu_type" != "" ]; then
        local gpu_unit_cost=${GPU_PRICES[$gpu_type]:-0}
        gpu_cost=$(echo "$gpu_unit_cost * ${gpu_count:-1}" | bc)
    fi
    
    # 총 비용
    local total_hourly=$(echo "$machine_cost + $gpu_cost" | bc)
    local total_daily=$(echo "$total_hourly * 24" | bc)
    local total_weekly=$(echo "$total_daily * 7" | bc)
    local total_monthly=$(echo "$total_daily * 30" | bc)
    
    echo "비용 분석 (USD):"
    echo "  머신 비용: \$$machine_cost/시간"
    if [ "$gpu_cost" != "0" ]; then
        echo "  GPU 비용: \$$gpu_cost/시간 (${gpu_count}개)"
    fi
    echo ""
    echo "총 예상 비용:"
    echo "  시간당: \$$total_hourly"
    echo "  일일 (24시간): \$$total_daily"
    echo "  주간 (7일): \$$total_weekly"
    echo "  월간 (30일): \$$total_monthly"
    echo ""
}

get_uptime() {
    print_header "실행 시간 확인"
    
    local creation_time=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="value(creationTimestamp)")
    
    echo "생성 시간: $creation_time"
    
    # 현재 시간과 생성 시간의 차이 계산
    local creation_epoch=$(date -d "$creation_time" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$creation_time" +%s 2>/dev/null || echo "0")
    local current_epoch=$(date +%s)
    local uptime_seconds=$((current_epoch - creation_epoch))
    
    if [ "$uptime_seconds" -gt 0 ]; then
        local hours=$((uptime_seconds / 3600))
        local minutes=$(((uptime_seconds % 3600) / 60))
        
        echo "실행 시간: ${hours}시간 ${minutes}분"
        echo ""
        
        # 현재까지 발생한 비용 계산
        local info=$(get_instance_info)
        local machine_type=$(echo "$info" | cut -d'|' -f1)
        local gpu_type=$(echo "$info" | cut -d'|' -f2)
        local gpu_count=$(echo "$info" | cut -d'|' -f3)
        
        local machine_cost=${MACHINE_PRICES[$machine_type]:-0}
        local gpu_cost=0
        if [ -n "$gpu_type" ] && [ "$gpu_type" != "" ]; then
            local gpu_unit_cost=${GPU_PRICES[$gpu_type]:-0}
            gpu_cost=$(echo "$gpu_unit_cost * ${gpu_count:-1}" | bc)
        fi
        
        local hourly_cost=$(echo "$machine_cost + $gpu_cost" | bc)
        local hours_decimal=$(echo "scale=2; $uptime_seconds / 3600" | bc)
        local total_cost=$(echo "$hourly_cost * $hours_decimal" | bc)
        
        echo "현재까지 발생한 비용: \$$total_cost"
    fi
    
    echo ""
}

show_cost_saving_tips() {
    print_header "비용 절감 팁"
    
    echo "1. 훈련 완료 후 즉시 인스턴스 삭제:"
    echo "   ./delete_instance.sh"
    echo ""
    echo "2. Preemptible VM 사용 (최대 80% 절감):"
    echo "   export USE_PREEMPTIBLE=true"
    echo "   ./create_instance.sh"
    echo ""
    echo "3. 저렴한 Zone 선택:"
    echo "   export GCP_ZONE=us-central1-a"
    echo ""
    echo "4. 필요한 만큼만 훈련:"
    echo "   export EPOCHS=20  # 작은 에포크로 시작"
    echo ""
    echo "5. 인스턴스 일시 중지 (비용 절감, 디스크 비용만 발생):"
    echo "   gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo "6. 예산 알림 설정:"
    echo "   https://console.cloud.google.com/billing/budgets"
    echo ""
}

check_billing() {
    print_header "청구 정보"
    
    echo "GCP 콘솔에서 실시간 비용 확인:"
    echo "  https://console.cloud.google.com/billing"
    echo ""
    echo "프로젝트별 비용 확인:"
    echo "  https://console.cloud.google.com/billing/reports"
    echo ""
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    print_header "GCP 인스턴스 비용 확인"
    
    echo "프로젝트: $PROJECT_ID"
    echo "Zone: $ZONE"
    echo ""
    
    # 인스턴스 상태 확인
    check_instance_status
    
    # 비용 계산
    calculate_costs
    
    # 실행 시간 및 누적 비용
    get_uptime
    
    # 비용 절감 팁
    show_cost_saving_tips
    
    # 청구 정보
    check_billing
    
    print_header "완료"
    echo "⚠️  훈련 완료 후 반드시 인스턴스를 삭제하세요!"
    echo "   ./delete_instance.sh"
}

main "$@"
