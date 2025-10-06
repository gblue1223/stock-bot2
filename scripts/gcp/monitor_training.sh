#!/bin/bash
# 훈련 모니터링 스크립트
# 원격 인스턴스의 훈련 진행 상황을 모니터링합니다.

set -e  # 에러 발생 시 스크립트 중단

# =============================================================================
# 설정 변수
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-ai-trader-training}"
REMOTE_DIR="~/ai-trader"

# =============================================================================
# 함수 정의
# =============================================================================

print_header() {
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

show_menu() {
    clear
    print_header "훈련 모니터링 메뉴"
    echo "1) 훈련 로그 실시간 보기 (tail -f)"
    echo "2) GPU 사용률 확인 (nvidia-smi)"
    echo "3) tmux 세션 연결"
    echo "4) 디스크 사용량 확인"
    echo "5) 훈련 프로세스 확인"
    echo "6) TensorBoard 포트 포워딩"
    echo "7) 체크포인트 목록 보기"
    echo "8) 종료"
    echo ""
    read -p "선택 (1-8): " choice
    echo ""
}

view_training_log() {
    print_header "훈련 로그 실시간 보기"
    echo "종료하려면 Ctrl+C를 누르세요."
    echo ""
    
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="tail -f $REMOTE_DIR/training.log"
}

check_gpu_usage() {
    print_header "GPU 사용률 확인"
    
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="nvidia-smi"
    
    echo ""
    read -p "계속하려면 Enter를 누르세요..."
}

attach_tmux() {
    print_header "tmux 세션 연결"
    echo "세션에서 나가려면: Ctrl+B, D"
    echo ""
    
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="tmux attach -t training"
}

check_disk_usage() {
    print_header "디스크 사용량 확인"
    
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="df -h && echo '' && du -sh $REMOTE_DIR/*"
    
    echo ""
    read -p "계속하려면 Enter를 누르세요..."
}

check_training_process() {
    print_header "훈련 프로세스 확인"
    
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="ps aux | grep train_embedding | grep -v grep"
    
    echo ""
    read -p "계속하려면 Enter를 누르세요..."
}

setup_tensorboard() {
    print_header "TensorBoard 포트 포워딩"
    
    echo "TensorBoard를 시작합니다..."
    echo ""
    echo "1. 이 터미널은 열어두세요."
    echo "2. 브라우저에서 http://localhost:6006 을 여세요."
    echo "3. 종료하려면 Ctrl+C를 누르세요."
    echo ""
    
    # 백그라운드에서 TensorBoard 시작
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="pkill -f tensorboard; tensorboard --logdir=$REMOTE_DIR/models/embedding/tensorboard_logs --host=0.0.0.0 --port=6006 &" &
    
    sleep 3
    
    # 포트 포워딩
    echo "포트 포워딩 시작..."
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        -- -L 6006:localhost:6006 -N
}

list_checkpoints() {
    print_header "체크포인트 목록"
    
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="ls -lh $REMOTE_DIR/models/embedding/*.pt 2>/dev/null || echo '체크포인트가 아직 없습니다.'"
    
    echo ""
    read -p "계속하려면 Enter를 누르세요..."
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    while true; do
        show_menu
        
        case $choice in
            1)
                view_training_log
                ;;
            2)
                check_gpu_usage
                ;;
            3)
                attach_tmux
                ;;
            4)
                check_disk_usage
                ;;
            5)
                check_training_process
                ;;
            6)
                setup_tensorboard
                ;;
            7)
                list_checkpoints
                ;;
            8)
                echo "종료합니다."
                exit 0
                ;;
            *)
                echo "잘못된 선택입니다."
                sleep 2
                ;;
        esac
    done
}

main "$@"
